import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy
import torch
import trimesh
from typing import Dict, Tuple, Optional, Union, List
import os 
from utils.viser_utils import to_numpy
from termcolor import cprint
import time
from utils.rotation_utils import rotation_matrix_to_quaternion


def postprocess_mano(data, cam_t, is_right):
    """MANO params are in different coordinate system as pyrender camera."""
    data[..., 0] = (2 * is_right - 1) * data[..., 0]
    data = data + cam_t.reshape(1,-1,3)
    # data[..., 0] *= -1
    # data[..., 1] *= -1
    return data




class MultiObjectViser:
    """A visualizer using Viser that can show a robot, static furniture, and various objects."""

    global_servers: Dict[int, viser.ViserServer] = {}

    def __init__(
        self, robot_urdf_path: str,
          table_urdf_path: str,
          robot_position: np.ndarray,
          robot_orientation: np.ndarray,
          table_position: np.ndarray, 
          table_orientation: np.ndarray, 
          object_names: List[str], 
          object_urdf_dict: Dict[str, str], 
          port: int = 8888,
          show_axes: bool = False
    ):
        """
        Initialize visualizer with multiple objects.

        Args:
            robot_urdf_path: Path to the robot's URDF file
            table_urdf_path: Path to the table's URDF file
            table_position: [x, y, z] position of the table
            table_orientation: [w, x, y, z] quaternion orientation of the table
            object_urdf_dict: Dictionary mapping object names to their URDF paths
            port: Port number for the viser server
        """
        # Server setup
        if port in MultiObjectViser.global_servers:
            print(f"Found existing server on port {port}, shutting it down.")
            MultiObjectViser.global_servers.pop(port).stop()

        self.server = viser.ViserServer(port=port)
        MultiObjectViser.global_servers[port] = self.server

        # Create world node
        self._world_node = self.server.scene.add_frame("/world", show_axes=False)

        self.handles = {}
        


        cprint(f"Loading robot from {robot_urdf_path}", color="green", attrs=["bold"])

        # Load and setup robot
        self.robot_urdf = yourdfpy.URDF.load(os.path.join("assets", robot_urdf_path))
        self.robot_viser = ViserUrdf(
            target=self.server,
            urdf_or_path=self.robot_urdf,
            root_node_name="/world/robot",
        )

        self.robot_joint_names = self.robot_viser.get_actuated_joint_names()

        # print joint names in order
        cprint(f"Joint names in Viser order: {self.robot_viser.get_actuated_joint_names()}, length: {len(self.robot_viser.get_actuated_joint_names())}",
               'yellow', attrs=['bold'])

        self.robot_frame = self.server.scene.add_frame(
            "/world/robot",
            position=to_numpy(robot_position),
            wxyz=to_numpy(robot_orientation),
            show_axes=show_axes,
        )

        self._robot_joint_frames = {
            frame.name: frame for frame in self.robot_viser._joint_frames
        }

        # Load and setup table
        self.table_urdf = yourdfpy.URDF.load(os.path.join("assets", table_urdf_path))
        self.table_viser = ViserUrdf(
            target=self.server,
            urdf_or_path=self.table_urdf,
            root_node_name="/world/table",
        )
        # Set table position and orientation
        self.table_frame = self.server.scene.add_frame(
            "/world/table",
            position=to_numpy(table_position),
            wxyz=to_numpy(table_orientation),
            show_axes=show_axes
        )

        # Load and setup objects
        self.object_urdfs = {}
        self.object_visers = {}
        self.object_dict = {}
        self.object_frames = {}

        for obj_name, item in object_urdf_dict.items():

            urdf_path = item['urdf']
            mesh_path = item['mesh']

            if object_names is None or obj_name in object_names:

                print("loading object", obj_name)

                self.object_urdfs[obj_name] = yourdfpy.URDF.load(
                    os.path.join("assets", urdf_path)
                )
                self.object_visers[obj_name] = ViserUrdf(
                    target=self.server,
                    urdf_or_path=self.object_urdfs[obj_name],
                    root_node_name=f"/world/object_{obj_name}",
                )
                self.object_dict[obj_name] = object_urdf_dict[obj_name]
        

        # Store mesh handles
        self._mesh_handles = {}

        # GUI Visibility controls
        with self.server.gui.add_folder("Visibility"):
            self.show_robot = self.server.gui.add_checkbox(
                "Show Robot", initial_value=True
            )
            self.show_table = self.server.gui.add_checkbox(
                "Show Table", initial_value=True
            )

            self.backgrounds_slider = self.server.gui.add_rgb(
                    "Background",
                    initial_value=(255, 255, 255),
                    hint="Background color for rendering.",
                )

            # Create visibility toggles for each object
            self.show_objects = {}
            for obj_name in self.object_dict.keys():
                self.show_objects[obj_name] = self.server.gui.add_checkbox(
                    f"Show {obj_name}", initial_value=True
                )

            @self.show_robot.on_update
            def _(_) -> None:
                for frame in self._robot_joint_frames.values():
                    frame.visible = self.show_robot.value

            @self.show_table.on_update
            def _(_) -> None:
                for frame in self.table_viser._joint_frames:
                    frame.visible = self.show_table.value

            # Add visibility callbacks for each object
            for obj_name in self.object_dict.keys():
                def make_callback(obj_name):
                    def callback(_) -> None:
                        for frame in self.object_visers[obj_name]._joint_frames:
                            frame.visible = self.show_objects[obj_name].value
                    return callback

                self.show_objects[obj_name].on_update(make_callback(obj_name))

        with self.server.gui.add_folder("Joint Controls"):
            self.joint_sliders = {}
            for joint_name in self.robot_joint_names:
                self.joint_sliders[joint_name] = self.server.gui.add_slider(
                    label=joint_name,
                    min=-3.14,  # Default to ±π, adjust these limits based on your robot's actual joint limits
                    max=3.14,
                    step=0.01,
                    initial_value=0.0,
                )

            # Add callback for joint sliders
            for joint_name in self.robot_joint_names:
                def make_joint_callback(joint_name):
                    def callback(_):
                        joint_angles = [self.joint_sliders[name].value for name in self.robot_joint_names]
                        self.set_robot(joint_angles)
                    return callback

                self.joint_sliders[joint_name].on_update(make_joint_callback(joint_name))

    def add_object(
        self,
        object_name: str,
        position: np.ndarray,  # [x, y, z]
        wxyz: np.ndarray,  # [w, x, y, z] quaternion
        show_axes: bool = False
    ) -> None:

        """
        Set the pose of a specific object.

        Args:
            object_name: Name of the object to move
            position: [x, y, z] position
            orientation: [w, x, y, z] quaternion orientation
        """

        if isinstance(object_name, list):
            for i, obj in enumerate(object_name):
                self.add_object(obj, position[i], wxyz[i], show_axes)
            return 


        print(object_name)
        object_frame = self.server.scene.add_frame(
            f"/world/object_{object_name}",
            position=to_numpy(position),
            wxyz=to_numpy(wxyz) ,
            show_axes=show_axes,
            visible=True
        )
        self.object_frames[object_name] = object_frame
        # return object_frame




        return 

    def set_object(self, object_name, position, orientation):

        
        if isinstance(object_name, list):
            for i, obj in enumerate(object_name):
                self.set_object(obj, position[i], orientation[i])
            return 

        self.object_frames[object_name].position = to_numpy(position)
        self.object_frames[object_name].wxyz = to_numpy(orientation)

    def set_robot(self, joint_angles, mapping=None):

        joint_angles = to_numpy(joint_angles)

        if mapping is not None:
            map = [0]*len(self.robot_joint_names)
            for i, j in enumerate(mapping):
                map[j] = joint_angles[i]
            joint_angles = map

        self.robot_viser.update_cfg(joint_angles)

    def draw_points(self, points, color=(0, 0, 1), size=0.01, name="/points"):

        # if color is list then make points one by one with corresponding color in list
        if isinstance(color, list):
            for i in range(len(points)):
                self.server.scene.add_point_cloud(
                    name=f"{name}_{i}",
                    points=to_numpy(points[i:i+1]),
                    point_size=size,
                    colors=color[i%len(color)]
                )
            return 

        handle = self.server.scene.add_point_cloud(
            name=name,
            points=to_numpy(points),
            point_size=size,
            colors=color
        )

        self.handles[name] = handle 

    def add_mano(self, mano_params):

        from utils.mano_utils import MANO

        self.mano = MANO(
            model_path="/home/himanshu/hot3d/mano/mano",            
        ).to('cuda')
        # get faces from mano

        #mano_output = self.mano(betas=mano_params['betas'], hand_pose=mano_params['hand_pose'])
        mano_output = self.mano(**mano_params)

        world_hand_vertices = mano_output['vertices'].detach().cpu().numpy()

        # world_hand_vertices[..., 1] *= -1
        world_hand_vertices = postprocess_mano(world_hand_vertices, mano_params['camera_translation'].cpu().numpy(), mano_params['is_right'].cpu().numpy())

        # get init mano_params, run the mano layer business to get world_hand_vertices
        # init mano
        # wxyz = rotation_matrix_to_quaternion(mano_params['global_orient']).reshape(-1)
        # position = mano_params['transl'].reshape(-1)

        self.hand_mesh_handle = self.server.scene.add_mesh_simple(
                    name=f"/world/hand_mesh",
                    vertices=world_hand_vertices,
                    faces=self.mano.faces,
                    color=(64, 224, 208),  
                    wireframe=False,
                    opacity=1.0,
                    material="standard",
                    flat_shading=False,
                    # side="double",  # Render both sides of the mesh
                    # wxyz=wxyz,  # quaternion (w, x, y, z)
                    # position=position,
                    visible=True
                )

        self.update_mano(world_hand_vertices)
        self.handles["hand_mesh"] = self.hand_mesh_handle
        return 
    

    def set_mano(self, mano_params):
        mano_output = self.mano(**mano_params)
        world_hand_vertices = mano_output['vertices'].detach().cpu().numpy()
        world_hand_vertices = postprocess_mano(world_hand_vertices, mano_params['camera_translation'].cpu().numpy(), mano_params['is_right'].cpu().numpy())
        self.update_mano(world_hand_vertices)

    def update_mano(self, world_hand_vertices):
        assert self.hand_mesh_handle is not None, "Hand mesh handle not found, please call add_mano first"
        self.hand_mesh_handle.vertices = world_hand_vertices
        return 

    def add_mano_mesh(self, vertices, faces, name=f"/world/hand_mesh"):
        handle = self.server.scene.add_mesh_simple(
                    name=name,
                    vertices=vertices,
                    faces=faces,
                    color=(64, 224, 208),
                    wireframe=False,
                    opacity=1.0,
                    material="standard",
                    flat_shading=False,
                    # side="double",  # Render both sides of the mesh
                    # wxyz=wxyz,  # quaternion (w, x, y, z)
                    # position=position,
                    visible=True
                )

        self.handles[name] = handle
        return

    def get_joint_indices(self):

        return self.robot_viser.get_actuated_joint_indices()

    def add_camera_client(self):
        """
        Add a basic camera client to the Viser server.

        Args:
            server: The Viser server instance
        """

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            print(f"New camera client connected! ID: {client.client_id}")

            # Set initial camera parameters
            client.camera.far = 10.0
            client.camera.near = 0.01
            client.camera.position = (2.0, 2.0, 2.0)  # Set initial position
            client.camera.look_at = (0.0, 0.0, 0.0)  # Look at origin

            self.camera_handle = client.camera

            # Monitor camera updates
            @client.camera.on_update
            def _(camera: viser.CameraHandle):
                print(f"\nCamera updated on client {client.client_id}:")
                print(f"Position: {camera.position}")
                print(f"Look at: {camera.look_at}")
                print(f"FOV: {camera.fov}")
                print(f"Aspect: {camera.aspect}")
                print(f"Timestamp: {camera.update_timestamp}")

            # sleep 0.1 seconds
            time.sleep(0.1)
    def update_camera(self, **kwargs):
        """
        Update camera attributes with provided values.

        Args:
            **kwargs: Arbitrary camera attributes and their values to update.
                     Valid attributes: position, look_at, far, near, fov, up_direction
        """
        if not hasattr(self, 'camera_handle'):
            cprint("Warning: No camera handle available. Make sure to call add_camera_client() first.",
                   'red', attrs=['bold'])
            return

        for attr, value in kwargs.items():
            if hasattr(self.camera_handle, attr):
                setattr(self.camera_handle, attr, value)
            else:
                cprint(f"Warning: Camera has no attribute '{attr}'. Possible attributes: position, look_at, far, near, fov, up_direction",
                       'red', attrs=['bold'])
            
        


        #this takes time to update
        time.sleep(0.1)

    # def set_background(self, color=(0.2,0.2,0.2), show_grid=False):
    #     self.server.scene.configure_scene(
    #         background_color=color,  # RGB values between 0 and 1
    #         show_grid=show_grid,  # Disable default grid since we'll add our own ground plane
    #     )

    def set_ground_plane(self, z = 0.0):

        vertices = np.array([[-10, -10, z], [10, -10, z], [10, 10, z], [-10, 10, z]])

        faces = np.array([[0, 1, 2], [0, 2, 3]])

        # Add the ground plane mesh with a custom color
        self.server.scene.add_mesh_simple(
            "/ground_plane",
            vertices=vertices,
            faces=faces,
            color=(0.3, 0.3, 0.35),  # Slightly bluish gray
            flat_shading=True,
            wireframe=False,
            opacity=1.0,)

        return
            


    def empty_handles(self):
        for handle in self.handles.values():
            handle.remove()
        self.handles = {}
    # def map_isaac_allegro_to_viser(self, joint_angles):
    #     indices = [0,1,2,3,4,5,6] + [7,8,9,10] + [15,16,17,18] + [19,20,21,22] + [11,12,13,14]
    #     return joint_angles[indices]

    # def set_object(self, object_name, position, orientation):
    #     self.object_visers[object_name].update_cfg(position, orientation)



    def add_image(self, image, name="/world/isaac_image"):
        self.server.scene.add_image(
        name,
        to_numpy(image),
        4.0,
        4.0,
        format="jpeg",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(2.0, 2.0, 0.0))
        return 



