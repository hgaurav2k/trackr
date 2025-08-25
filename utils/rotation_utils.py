# import torch
# from pytorch3d.transforms import quaternion_apply


# def quat_apply_torch(a, b):
#     """PyTorch3D GPU-accelerated version of quat_apply.
#     Args:
#         a: Quaternions of shape (..., 4) in [w, x, y , z] format (as used in original quat_apply)
#         b: Points of shape (..., 3)

#     Returns:
#         Rotated points with same shape as b
#     """
#     # Store original shape for reshaping at the end
#     original_shape = b.shape
#     # Reshape inputs
#     a_reshaped = a.reshape(-1, 4)
#     b_reshaped = b.reshape(-1, 3)

#     # Apply quaternion rotation using pytorch3d
#     rotated_points = quaternion_apply(a_reshaped, b_reshaped)

#     # Reshape back to original shape
#     return rotated_points.reshape(original_shape)


import torch
import numpy as np
from torch.nn import functional as F

def quat_apply_torch(a, b):
    """PyTorch implementation of quaternion rotation without PyTorch3D.
    Args:
        a: Quaternions of shape (..., 4) in [w, x, y, z] format
        b: Points of shape (..., 3)

    Returns:
        Rotated points with same shape as b
    """
    # Store original shape for reshaping at the end
    original_shape = b.shape
    
    # Reshape inputs
    a_reshaped = a.reshape(-1, 4)
    b_reshaped = b.reshape(-1, 3)
    
    # Extract quaternion components
    qw, qx, qy, qz = a_reshaped[:, 0], a_reshaped[:, 1], a_reshaped[:, 2], a_reshaped[:, 3]
    
    # Extract point components
    x, y, z = b_reshaped[:, 0], b_reshaped[:, 1], b_reshaped[:, 2]
    
    # Compute the rotation using the formula: q * v * q^-1
    # For unit quaternions, this simplifies to the following:
    
    # Compute common terms
    qw2 = qw * qw
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qwx = qw * qx
    qwy = qw * qy
    qwz = qw * qz
    qxy = qx * qy
    qxz = qx * qz
    qyz = qy * qz
    
    # Compute rotated points
    rx = x * (qw2 + qx2 - qy2 - qz2) + 2 * (qxy - qwz) * y + 2 * (qxz + qwy) * z
    ry = y * (qw2 - qx2 + qy2 - qz2) + 2 * (qxy + qwz) * x + 2 * (qyz - qwx) * z
    rz = z * (qw2 - qx2 - qy2 + qz2) + 2 * (qxz - qwy) * x + 2 * (qyz + qwx) * y
    
    # Stack results
    rotated_points = torch.stack([rx, ry, rz], dim=1)
    
    # Reshape back to original shape
    return rotated_points.reshape(original_shape)

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    """
    Taken from https://github.com/mkocabas/VIBE/blob/master/lib/utils/geometry.py
    Calculates the rotation matrices for a batch of rotation vectors
    - param rot_vecs: torch.tensor (N, 3) array of N axis-angle vectors
    - returns R: torch.tensor (N, 3, 3) rotation matrices
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(
        (batch_size, 3, 3)
    )

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def quaternion_mul(q0, q1):
    """
    EXPECTS WXYZ
    :param q0 (*, 4)
    :param q1 (*, 4)
    """
    r0, r1 = q0[..., :1], q1[..., :1]
    v0, v1 = q0[..., 1:], q1[..., 1:]
    r = r0 * r1 - (v0 * v1).sum(dim=-1, keepdim=True)
    v = r0 * v1 + r1 * v0 + torch.linalg.cross(v0, v1)
    return torch.cat([r, v], dim=-1)


def quaternion_inverse(q, eps=1e-8):
    """
    EXPECTS WXYZ
    :param q (*, 4)
    """
    conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    mag = torch.square(q).sum(dim=-1, keepdim=True) + eps
    return conj / mag


def quaternion_slerp(t, q0, q1, eps=1e-8):
    """
    :param t (*, 1)  must be between 0 and 1
    :param q0 (*, 4)
    :param q1 (*, 4)
    """
    dims = q0.shape[:-1]
    t = t.view(*dims, 1)

    q0 = F.normalize(q0, p=2, dim=-1)
    q1 = F.normalize(q1, p=2, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # make sure we give the shortest rotation path (< 180d)
    neg = dot < 0
    q1 = torch.where(neg, -q1, q1)
    dot = torch.where(neg, -dot, dot)
    angle = torch.acos(dot)

    # if angle is too small, just do linear interpolation
    collin = torch.abs(dot) > 1 - eps
    fac = 1 / torch.sin(angle)
    w0 = torch.where(collin, 1 - t, torch.sin((1 - t) * angle) * fac)
    w1 = torch.where(collin, t, torch.sin(t * angle) * fac)
    slerp = q0 * w0 + q1 * w1
    return slerp


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert rotation matrix to Rodrigues vector
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    :param quaternion (*, 4) expects WXYZ
    :returns angle_axis (*, 3)
    """
    # unpack input and compute conversion
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def angle_axis_to_rotation_matrix(angle_axis):
    """
    :param angle_axis (*, 3)
    return (*, 3, 3)
    """
    quat = angle_axis_to_quaternion(angle_axis)
    return quaternion_to_rotation_matrix(quat)


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    Taken from https://github.com/kornia/kornia, based on
    https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
    https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247
    :param quaternion (N, 4) expects WXYZ order
    returns rotation matrix (N, 3, 3)
    """
    # normalize the input quaternion
    quaternion_norm = F.normalize(quaternion, p=2, dim=-1, eps=1e-12)
    *dims, _ = quaternion_norm.shape

    # unpack the normalized quaternion components
    w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0)

    matrix = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(*dims, 3, 3)
    return matrix


def angle_axis_to_quaternion(angle_axis):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert angle axis to quaternion in WXYZ order
    :param angle_axis (*, 3)
    :returns quaternion (*, 4) WXYZ order
    """
    theta_sq = torch.sum(angle_axis**2, dim=-1, keepdim=True)  # (*, 1)
    # need to handle the zero rotation case
    valid = theta_sq > 0
    theta = torch.sqrt(theta_sq)
    half_theta = 0.5 * theta
    ones = torch.ones_like(half_theta)
    # fill zero with the limit of sin ax / x -> a
    k = torch.where(valid, torch.sin(half_theta) / theta, 0.5 * ones)
    w = torch.where(valid, torch.cos(half_theta), ones)
    quat = torch.cat([w, k * angle_axis], dim=-1)
    return quat


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    :param rotation_matrix (N, 3, 3)
    """
    *dims, m, n = rotation_matrix.shape
    rmat_t = torch.transpose(rotation_matrix.reshape(-1, m, n), -1, -2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q.reshape(*dims, 4)


import torch


def transform_keypoints_to_camera_frame(hand_keypoints, cam_xyz, cam_quats):
    """
    Transform keypoints to the coordinate frame defined by the first camera position and orientation
    using homogeneous coordinates.

    Args:
        hand_keypoints: Tensor of shape [N, J, 3] where N is number of frames, J is number of joints, 3 is xyz
        cam_xyz: Tensor of shape [N, 3] representing camera positions
        cam_quats: Tensor of shape [N, 4] representing camera orientations as quaternions (w, x, y, z)

    Returns:
        Transformed hand keypoints of shape [N, J, 3] in the coordinate frame of the first camera
    """
    device = hand_keypoints.device

    # Convert quaternions to rotation matrices
    def quat_to_rot_matrix(quats):
        """Convert quaternions to rotation matrices."""
        # Ensure quaternions are normalized
        quats = torch.nn.functional.normalize(quats, dim=-1)

        # Extract quaternion components
        w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]

        # Calculate rotation matrix elements
        rot_matrix = torch.zeros((*quats.shape[:-1], 3, 3), device=device)

        # First row
        rot_matrix[..., 0, 0] = 1 - 2 * (y**2 + z**2)
        rot_matrix[..., 0, 1] = 2 * (x * y - w * z)
        rot_matrix[..., 0, 2] = 2 * (x * z + w * y)

        # Second row
        rot_matrix[..., 1, 0] = 2 * (x * y + w * z)
        rot_matrix[..., 1, 1] = 1 - 2 * (x**2 + z**2)
        rot_matrix[..., 1, 2] = 2 * (y * z - w * x)

        # Third row
        rot_matrix[..., 2, 0] = 2 * (x * z - w * y)
        rot_matrix[..., 2, 1] = 2 * (y * z + w * x)
        rot_matrix[..., 2, 2] = 1 - 2 * (x**2 + y**2)

        return rot_matrix

    # Get world-to-camera transformation matrix for first camera
    R_cam0 = quat_to_rot_matrix(cam_quats[0:1])  # [1, 3, 3]
    t_cam0 = cam_xyz[0:1]  # [1, 3]

    # Create homogeneous transformation matrix for camera 0
    T_cam0 = torch.zeros((1, 4, 4), device=device)
    T_cam0[..., :3, :3] = R_cam0
    T_cam0[..., :3, 3] = t_cam0
    T_cam0[..., 3, 3] = 1.0

    # Compute inverse transformation (camera 0 to world)
    R_cam0_inv = R_cam0.transpose(-1, -2)
    t_cam0_inv = -torch.matmul(R_cam0_inv, t_cam0.unsqueeze(-1)).squeeze(-1)

    # Create inverse homogeneous transformation matrix (world to camera 0)
    T_cam0_inv = torch.zeros((1, 4, 4), device=device)
    T_cam0_inv[..., :3, :3] = R_cam0_inv
    T_cam0_inv[..., :3, 3] = t_cam0_inv
    T_cam0_inv[..., 3, 3] = 1.0

    # Convert hand keypoints to homogeneous coordinates
    N, J, _ = hand_keypoints.shape
    homogeneous_keypoints = torch.ones((N, J, 4), device=device)
    homogeneous_keypoints[..., :3] = hand_keypoints

    # Reshape for batch matrix multiplication
    keypoints_reshaped = homogeneous_keypoints.reshape(N * J, 4, 1)

    # Apply transformation
    T_cam0_inv_expanded = T_cam0_inv.expand(N * J, 4, 4)
    transformed_keypoints = torch.matmul(
        T_cam0_inv_expanded, keypoints_reshaped
    )  # [N*J, 4, 1]

    # Reshape back and remove homogeneous coordinate
    transformed_keypoints = transformed_keypoints.squeeze(-1).reshape(N, J, 4)
    transformed_keypoints = transformed_keypoints[..., :3]

    return transformed_keypoints


import numpy as np


def pose_6d_to_matrix(pose_6d):
    """
    Convert 6D pose (x, y, z, rx, ry, rz) to 4x4 transformation matrix.

    Args:
        pose_6d: numpy array of shape (6,) containing [rx, ry, rz, x, y, z]
                where [rx, ry, rz] is the rotation in radians and [x, y, z] is the position

    Returns:
        transform_matrix: numpy array of shape (4, 4) containing the transformation matrix
    """
    # Extract position and rotation components

    position = pose_6d[3:]
    rotation = pose_6d[:3]


    if isinstance(rotation, np.ndarray):
        rotation = torch.from_numpy(rotation).unsqueeze(0)
        # Convert rotation vector to rotation matrix using Rodrigues formula
        print(f"rotation: {rotation.shape}")
        rotation_matrix = angle_axis_to_rotation_matrix(rotation)  # 3x3 matrix
        rotation_matrix = rotation_matrix.squeeze(0)
        rotation_matrix = rotation_matrix.numpy()
    elif isinstance(rotation, torch.Tensor):
        rotation = rotation.unsqueeze(0)
        # Convert rotation vector to rotation matrix using Rodrigues formula
        rotation_matrix = angle_axis_to_rotation_matrix(rotation)  # 3x3 matrix
        rotation_matrix = rotation_matrix.squeeze(0)
        rotation_matrix = rotation_matrix.numpy()
    else:
        raise ValueError(f"Invalid rotation type: {type(rotation)}")


    # Create 3x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position

    return transform_matrix




def direction_to_quaternion(direction):
    """
    Convert a direction vector to a quaternion.

    Args:
    Convert a batch of direction vectors to quaternions (WXYZ).
    The quaternion represents the rotation from the positive Z-axis vector [0, 0, 1]
    to the given direction vector.

    Args:
        direction: torch.Tensor of shape (B, 3) representing direction vectors.
                   Zero vectors will result in an identity quaternion [1, 0, 0, 0].
        eps: Small epsilon for numerical stability checks, default 1e-8.

    Returns:
        torch.Tensor of shape (B, 4) representing quaternions in WXYZ format.
    """
    B = direction.shape[0]
    device = direction.device
    dtype = direction.dtype
    eps = 1e-8 # Define epsilon for stability

    # Initialize result quaternion tensor with identity quaternions
    quat = torch.zeros((B, 4), device=device, dtype=dtype)
    quat[:, 0] = 1.0

    # Calculate the magnitude of the direction vectors
    magnitude = torch.linalg.norm(direction, dim=-1) # Shape (B,)

    # Identify non-zero vectors where rotation is needed
    non_zero_mask = magnitude > eps

    # Process only non-zero vectors
    if non_zero_mask.any():
        # Select non-zero directions and work with this subset
        valid_direction = direction[non_zero_mask]
        num_valid = valid_direction.shape[0]

        # Reference vector (positive Z-axis) for the valid subset
        ref = torch.zeros((num_valid, 3), device=device, dtype=dtype)
        ref[..., 2] = 1.0

        # Normalize the valid direction vectors
        norm_dir = F.normalize(valid_direction, p=2, dim=-1) # Shape (num_valid, 3)

        # Calculate the dot product between reference and normalized direction
        dot_prod = torch.sum(ref * norm_dir, dim=-1) # Shape (num_valid,)

        # --- Handle edge cases within valid vectors ---

        # Anti-parallel case: direction is opposite to reference (-Z-axis)
        # Rotation is 180 degrees around an arbitrary axis perpendicular to Z.
        # We choose the X-axis, resulting in quaternion [0, 1, 0, 0].
        anti_parallel_mask = dot_prod < -1.0 + eps
        if anti_parallel_mask.any():
            # Find original indices of anti-parallel vectors
            original_indices_ap = torch.where(non_zero_mask)[0][anti_parallel_mask]
            quat[original_indices_ap, 0] = 0.0
            quat[original_indices_ap, 1] = 1.0
            quat[original_indices_ap, 2] = 0.0
            quat[original_indices_ap, 3] = 0.0

        # --- Handle general case (not parallel and not anti-parallel) ---
        # Parallel case (dot_prod > 1.0 - eps) already handled by identity initialization.
        # Identify vectors that are neither parallel nor anti-parallel
        general_mask = dot_prod.abs() < 1.0 - eps

        if general_mask.any():
            # Select vectors for the general case
            general_dir = norm_dir[general_mask]
            general_ref = ref[general_mask]
            general_dot_prod = dot_prod[general_mask]
            num_general = general_dir.shape[0]

            # Calculate rotation angle
            # Clamp dot product for numerical stability with acos
            angle = torch.acos(torch.clamp(general_dot_prod, -1.0, 1.0)) # Shape (num_general,)

            # Calculate rotation axis (cross product)
            axis = torch.cross(general_ref, general_dir, dim=-1) # Shape (num_general, 3)
            # Normalize axis (cross product is non-zero for general case)
            norm_axis = F.normalize(axis, p=2, dim=-1)

            # Construct angle-axis representation: angle * normalized_axis
            angle_axis = angle.unsqueeze(-1) * norm_axis # Shape (num_general, 3)

            # Convert angle-axis to quaternion using the existing function in this file
            general_quat = angle_axis_to_quaternion(angle_axis) # Shape (num_general, 4)

            # Assign the calculated quaternions back to the original indices
            original_indices_gen = torch.where(non_zero_mask)[0][general_mask]
            quat[original_indices_gen] = general_quat

    return quat


import torch
import torch.nn.functional as F

def vec6_to_rotmat(v6: torch.Tensor) -> torch.Tensor:
    """
    Convert a 6-D vector that stores the first two columns of a rotation matrix
    (given in row-major order:  [r11, r12,  r21, r22,  r31, r32])
    into the full 3 × 3 rotation matrix.
    Args
    ----
    v6 : (..., 6) tensor
        The last dimension holds the six numbers.
    Returns
    -------
    R : (..., 3, 3) tensor
        A valid rotation matrix (orthonormal, det = 1).
    """
    # split into the two (unnormalised) column vectors
    a1 = v6[..., 0::2]          # first column
    a2 = v6[..., 1::2]          # second column
    # 1) make the first column unit length
    b1 = F.normalize(a1, dim=-1)
    # 2) make the second column orthogonal to the first,
    #    then normalise
    proj = (b1 * a2).sum(dim=-1, keepdim=True)        # projection of a2 on b1
    a2_orth = a2 - proj * b1                          # remove the projection
    b2 = F.normalize(a2_orth, dim=-1)

    # 3) third column is the cross-product (already orthonormal)
    b3 = torch.cross(b1, b2, dim=-1)
    # stack the three columns to form the rotation matrix
    R = torch.stack((b1, b2, b3), dim=-1)             # (..., 3, 3)
    return R