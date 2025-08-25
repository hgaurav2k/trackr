
import torch
import numpy as np
import viser 

def map_isaac_kuka_allegro_to_viser(joint_angles):
    indices = list(range(len(joint_angles)))
    return joint_angles[indices]
    


def map_isaac_xarm_allegro_to_viser(joint_angles):
        indices = [0,1,2,3,4,5,6] + [7,8,9,10] + [15,16,17,18] + [19,20,21,22] + [11,12,13,14]
        return joint_angles[indices]
    


def map_sim2viser(rigid_body_names, viser_names):
    #go through rigid_body_indices, and store the index of the viser_names
    mapping = []
    for name in rigid_body_names:
        if name in viser_names:
            mapping.append(viser_names.index(name))
        else:
            raise ValueError(f"Rigid body name list {rigid_body_names} not same as viser names {viser_names}. Name {name} not found in viser names")
    return mapping


def w2front(xyzw):
      # Convert quaternion from xyzw to wxyz format
      if isinstance(xyzw, torch.Tensor):
          return torch.cat([xyzw[..., -1:], xyzw[..., :-1]], dim=-1)
      else:
          return np.concatenate([xyzw[..., -1:], xyzw[..., :-1]], axis=-1)
      

def w2back(wxyz):
    if isinstance(wxyz, torch.Tensor):
        return torch.cat([wxyz[..., 1:], wxyz[..., :1]], dim=-1)
    else:
        return np.concatenate([wxyz[..., 1:], wxyz[..., :1]], axis=-1)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array(x)
    



class Server:
    def __init__(self, server):
        self.server = server
        self.handles = {}
        self.mano = None 
    
    def empty_handles(self):
        for handle in self.handles:
            self.handles[handle].remove()
        self.handles = {}
        

def init_viser(port=8888):
     server = viser.ViserServer(port=port)
     world_node = server.scene.add_frame("/world", show_axes=False)
     server = Server(server)
     server.handles["/world"] = world_node
     return server


def add_ground_plane(server, z=0.0):
    vertices = np.array([[-10, -10, z], [10, -10, z], [10, 10, z], [-10, 10, z]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    h = server.server.scene.add_mesh_simple(
            "/ground_plane",
            vertices=vertices,
            faces=faces,
            color=(0.3, 0.3, 0.35),  # Slightly bluish gray
            flat_shading=True,
            wireframe=False,
            opacity=1.0)
    

    server.handles["/ground_plane"] = h

    
    
    

def draw_points(server, points, color=(0, 0, 255), size=0.01, name="/points"):
    handle = server.server.scene.add_point_cloud(
            name=name,
            points=to_numpy(points),
            point_size=size,
            colors=color
        )
    server.handles[name] = handle
    return handle
     

def draw_lines(server, points, color=(0, 0, 255), name="/lines"):
     

     assert points.shape[1:] == (2,3), f"Points must be of shape (N,2,3). Got {points.shape}"
     handle = server.server.scene.add_line_segments(
            name=name,
            points=to_numpy(points),
            colors=color
        )
     server.handles[name] = handle
     return handle





def draw_mano(server, mano_params):

    from utils.mano_utils import MANO

    if server.mano is  None:
        server.mano = MANO(
            model_path="/home/himanshu/hot3d/mano/mano",            
        ).to('cuda')
    # get faces from mano

    #mano_output = self.mano(betas=mano_params['betas'], hand_pose=mano_params['hand_pose'])
    mano_output = server.mano(**mano_params)

    world_hand_vertices = mano_output['vertices'].detach().cpu().numpy()

    # world_hand_vertices[..., 1] *= -1
    world_hand_vertices = postprocess_mano(world_hand_vertices, mano_params['camera_translation'].cpu().numpy(), mano_params['is_right'].cpu().numpy())

        # get init mano_params, run the mano layer business to get world_hand_vertices
        # init mano
        # wxyz = rotation_matrix_to_quaternion(mano_params['global_orient']).reshape(-1)
        # position = mano_params['transl'].reshape(-1)

    
    if "hand_mesh" not in server.handles:

        hand_mesh_handle = server.server.scene.add_mesh_simple(
                        name=f"/world/hand_mesh",
                        vertices=world_hand_vertices,
                        faces=server.mano.faces,
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
        
        hand_mesh_handle.vertices = world_hand_vertices
        server.handles["hand_mesh"] = hand_mesh_handle
        
    else:
        server.handles["hand_mesh"].vertices = world_hand_vertices
        return 



def postprocess_mano(data, cam_t, is_right):
    """MANO params are in different coordinate system as pyrender camera."""
    data[..., 0] = (2 * is_right - 1) * data[..., 0]
    data = data + cam_t.reshape(1,-1,3)
    # data[..., 0] *= -1
    # data[..., 1] *= -1
    return data


def add_camera_client(server):
        """
        Add a basic camera client to the Viser server.

        Args:
            server: The Viser server instance
        """

        @server.server.on_client_connect
        def _(client: viser.ClientHandle):
            print(f"New camera client connected! ID: {client.client_id}")

            # Set initial camera parameters
            client.camera.far = 10.0
            client.camera.near = 0.01
            client.camera.position = (2.0, 2.0, 2.0)  # Set initial position
            client.camera.look_at = (0.0, 0.0, 0.0)  # Look at origin

            server.camera_handle = client.camera

            # Monitor camera updates
            @client.camera.on_update
            def _(camera: viser.CameraHandle):
                print(f"\nCamera updated on client {client.client_id}:")
                print(f"Position: {camera.position}")
                print(f"Look at: {camera.look_at}")
                print(f"FOV: {camera.fov}")
                print(f"Aspect: {camera.aspect}")
                print(f"Timestamp: {camera.update_timestamp}")