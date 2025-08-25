import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy
import torch
import trimesh
from typing import Dict, Tuple, Optional, Union, List
import os

class LeggedRobotViser:
    """A robot visualizer using Viser, with the URDF attached under a /world root node."""

    global_servers: Dict[int, viser.ViserServer] = {}

    def __init__(self, urdf_path: str, port: int = 8080):
        """
        Initialize visualizer with a URDF model, loaded under a single /world node.

        Args:
            urdf_path: Path to the URDF file
            port: Port number for the viser server
        """
        # If there is an existing server on this port, shut it down
        if port in LeggedRobotViser.global_servers:
            print(f"Found existing server on port {port}, shutting it down.")
            LeggedRobotViser.global_servers.pop(port).stop()

        self.server = viser.ViserServer(port=port)
        LeggedRobotViser.global_servers[port] = self.server

        # Create a single "world" node. We'll attach the robot's geometry under this node.
        self._world_node = self.server.scene.add_frame("/world", show_axes=False)

        # Load URDF
        self.urdf = yourdfpy.URDF.load(os.path.join("assets", urdf_path))

        # Attach URDF under the "/world" node by using root_node_name="/world"
        self.viser_urdf = ViserUrdf(
            target=self.server,
            urdf_or_path=self.urdf,
            root_node_name="/world",  # Everything becomes a child of /world
        )

        # Optionally store references to the URDF's frames if you need them:
        self._joint_frames = {
            frame.name: frame for frame in self.viser_urdf._joint_frames
        }

        # Also store mesh handles in case you want direct references
        self._mesh_handles = {}

        # GUI Visibility controls
        with self.server.gui.add_folder("Visibility"):
            self.show_robot = self.server.gui.add_checkbox(
                "Show Robot", initial_value=True
            )
            self.show_ground = self.server.gui.add_checkbox(
                "Show Ground", initial_value=True
            )
            self.show_terrain = self.server.gui.add_checkbox(
                "Show Terrain", initial_value=True
            )

            @self.show_robot.on_update
            def _(_) -> None:
                for frame in self._joint_frames.values():
                    frame.visible = self.show_robot.value

            @self.show_ground.on_update
            def _(_) -> None:
                if hasattr(self, "_ground_plane"):
                    self._ground_plane.visible = self.show_ground.value

            @self.show_terrain.on_update
            def _(_) -> None:
                if "terrain" in self._mesh_handles:
                    self._mesh_handles["terrain"].visible = self.show_terrain.value

    def update_joints(self, joint_angles: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Update the joint angles of the visualized robot.

        Args:
            joint_angles: Array of joint angles in radians
        """
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.detach().cpu().numpy()
        self.viser_urdf.update_cfg(joint_angles)
    

    def update_robot(self, robot_pose, joint_angles):
        """
        Update the robot's pose and joint angles.

        Args:
            robot_pose: Array of robot pose in meters
            joint_angles: Array of joint angles in radians
        """
        
        # TODO: robot_pose should be handled by a separate function
        # For now, only update joint angles
        self.viser_urdf.update_cfg(joint_angles)
        
