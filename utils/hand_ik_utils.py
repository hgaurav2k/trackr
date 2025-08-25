# Copyright (c) Hao Meng. All Rights Reserved.
import numpy as np
import torch
import torch.nn.functional as F
from transforms3d.axangles import axangle2mat
import transforms3d


def joints_to_mano_torch_batch(mano, mano_joints, betas, is_right, device):
    '''
    Convert human hand joints (B, 21, 3) to MANO pose parameters (B, 16, 3, 3).

    Args:
        mano: MANO model.
        mano_joints: (B, 21, 3) tensor storing hand joint keypoints.
        betas: (1, 10) tensor storing MANO shape parameters.

    Returns:
        hand_pose: (B, 15, 3, 3) MANO pose parameters.
        global_orient: (B, 1, 3, 3) MANO global orientation.
    '''
    # Template mano_params
    pose0 = torch.eye(3).repeat(1, 16, 1, 1).to(device)  # shape: (1,16,3,3)
    template_mano_params = {}
    template_mano_params['hand_pose'] = pose0[:, 1:, ...]
    template_mano_params['global_orient'] = pose0[:, 0, ...][None]
    template_mano_params['betas'] = betas
    template_mano_params['is_right'] = is_right
    template_mano_output = mano(**template_mano_params, pose2rot=False)  # pose0_shape: (1,16,3,3), opt_shape: (10,) j3d_shape: (1,21,3)

    j3d_p0_ops = template_mano_output.joints
    template = j3d_p0_ops[0]  # template, m 21*3

    # TODO: Batchify this
    all_hand_poses = []
    all_global_orients = []
    for i in range(len(mano_joints)):
        pre_joints = mano_joints[i]
        ratio = torch.norm(template[9] - template[0]) / torch.norm(pre_joints[9] - pre_joints[0])
        j3d_pre_process = pre_joints * ratio  # template, m
        j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]  # shape: (21, 3)
        pose_R = adaptive_IK_torch(template, j3d_pre_process)  # shape: (1,16,3,3)
        all_hand_poses.append(pose_R[:, 1:, ...])
        all_global_orients.append(pose_R[:, [0], ...])

    # Now replace the original mano params
    mano_params = {}
    #mano_params['hand_pose'] = pose_R[:, 1:, ...]
    #mano_params['global_orient'] = pose_R[:, [0], ...]
    #mano_params['betas'] = betas
    mano_params['hand_pose'] = torch.stack(all_hand_poses)
    mano_params['global_orient'] = torch.stack(all_global_orients)
    return mano_params


def joints_to_mano_torch(mano, mano_joints, betas, is_right, device):
    '''
    Convert human hand joints (21, 3) to MANO pose parameters (16, 3, 3).

    Args:
        mano: MANO model.
        mano_joints: (21, 3) tensor storing hand joint keypoints.
        betas: (10,) or (1, 10) tensor storing MANO shape parameters.

    Returns:
        hand_pose: (1, 15, 3, 3) MANO pose parameters.
        global_orient: (1, 1, 3, 3) MANO global orientation.
    '''
    if len(betas.shape) == 1:
        opt_tensor_shape = betas[None]
    else:
        opt_tensor_shape = betas

    # Template mano_params
    pose0 = torch.eye(3).repeat(1, 16, 1, 1).to(device)  # shape: (1,16,3,3)
    template_mano_params = {}
    template_mano_params['hand_pose'] = pose0[:, 1:, ...]
    template_mano_params['global_orient'] = pose0[:, 0, ...][None]
    template_mano_params['betas'] = opt_tensor_shape
    template_mano_params['is_right'] = is_right
    template_mano_output = mano(**template_mano_params, pose2rot=False)  # pose0_shape: (1,16,3,3), opt_shape: (10,) j3d_shape: (1,21,3)

    j3d_p0_ops = template_mano_output.joints
    template = j3d_p0_ops[0]  # template, m 21*3
    pre_joints = mano_joints
    ratio = torch.norm(template[9] - template[0]) / torch.norm(pre_joints[9] - pre_joints[0])
    j3d_pre_process = pre_joints * ratio  # template, m
    j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]  # shape: (21, 3)
    pose_R = adaptive_IK_torch(template, j3d_pre_process)  # shape: (1,16,3,3)

    # Now replace the original mano params
    mano_params = {}
    mano_params['hand_pose'] = pose_R[:, 1:, ...]
    mano_params['global_orient'] = pose_R[:, [0], ...]
    mano_params['betas'] = betas
    return mano_params


def adaptive_IK_torch(T_, P_, device='cuda'):
    '''
    PyTorch version of adaptive_IK
    :param T_: template, (21, 3)
    :param P_: target, (21, 3)
    :return: pose parameters (1, 16, 3, 3)
    '''
    T = T_.transpose(0, 1)  # (3, 21)
    P = P_.transpose(0, 1)  # (3, 21)

    # convert to dicts
    T = to_dict(T)
    P = to_dict(P)

    R = {}
    R_pa_k = {}
    q = {}

    q[0] = T[0]  # root position

    # Compute R0 using SVD
    P_0 = torch.cat([P[1] - P[0], P[5] - P[0], P[9] - P[0], P[13] - P[0], P[17] - P[0]], dim=-1)
    T_0 = torch.cat([T[1] - T[0], T[5] - T[0], T[9] - T[0], T[13] - T[0], T[17] - T[0]], dim=-1)
    H = torch.matmul(T_0, P_0.T)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.T
    R0 = torch.matmul(V, U.T)
    if torch.abs(torch.det(R0) + 1) < 1e-6:
        V_ = V.clone()
        if torch.sum(torch.abs(S) < 1e-4) > 0:
            V_[:, 2] = -V_[:, 2]
            R0 = torch.matmul(V_, U.T)

    R[0] = R0
    for idx in [1, 5, 9, 13, 17]:
        R[idx] = R[0].clone()

    # Compute rest of rotations along kinematic tree
    for k in kinematic_tree:
        pa = SNAP_PARENT[k]
        pa_pa = SNAP_PARENT[pa]
        q[pa] = torch.matmul(R[pa], (T[pa] - T[pa_pa])) + q[pa_pa]

        delta_p_k = torch.matmul(torch.linalg.inv(R[pa]), P[k] - q[pa]).view(3)
        delta_t_k = (T[k] - T[pa]).view(3)

        temp_axis = torch.cross(delta_t_k, delta_p_k)
        axis_norm = torch.norm(temp_axis)
        axis = temp_axis / (axis_norm + 1e-8)

        norm_t = torch.norm(delta_t_k) + 1e-8
        norm_p = torch.norm(delta_p_k) + 1e-8
        cos_alpha = torch.dot(delta_t_k, delta_p_k) / (norm_t * norm_p)
        alpha = torch.acos(torch.clamp(cos_alpha, -1.0, 1.0))

        twist = delta_t_k
        D_sw = torch.tensor(axangle2mat(axis.cpu().numpy(), alpha.item()), dtype=torch.float32).to(device)
        D_tw = torch.tensor(axangle2mat(twist.cpu().numpy(), angels0[:, k].item()), dtype=torch.float32).to(device)

        R_pa_k[k] = torch.matmul(D_sw, D_tw)
        R[k] = torch.matmul(R[pa], R_pa_k[k])

    pose_R = torch.zeros((1, 16, 3, 3), dtype=torch.float32).to(device)
    pose_R[0, 0] = R[0]
    for key in ID2ROT:
        pose_R[0, ID2ROT[key]] = R_pa_k[key]

    return pose_R


def joints_to_mano(mano, mano_joints, betas, is_right, device):
    '''
    Convert human hand joints (21, 3) to MANO pose parameters (16, 3, 3).

    Args:
        mano: MANO model.
        mano_joints: (21, 3) numpy array storing hand joint keypoints.
        betas: (10,) or (1, 10) torch tensor storing MANO shape parameters.

    Returns:
        hand_pose: (1, 15, 3, 3) MANO pose parameters.
        global_orient: (1, 1, 3, 3) MANO global orientation.
    '''
    if len(betas.shape) == 1:
        opt_tensor_shape = betas[None]
    else:
        opt_tensor_shape = betas
    if isinstance(mano_joints, torch.Tensor):
        pre_joints = mano_joints.detach().cpu().numpy()
    else:
        pre_joints = mano_joints

    # Template mano_params
    pose0 = torch.eye(3).repeat(1, 16, 1, 1).to(device)  # shape: (1,16,3,3)
    template_mano_params = {}
    template_mano_params['hand_pose'] = pose0[:, 1:, ...]
    template_mano_params['global_orient'] = pose0[:, 0, ...][None]
    template_mano_params['betas'] = opt_tensor_shape
    template_mano_params['is_right'] = is_right
    template_mano_output = mano(**template_mano_params, pose2rot=False)  # pose0_shape: (1,16,3,3), opt_shape: (10,) j3d_shape: (1,21,3)

    j3d_p0_ops = template_mano_output.joints
    template = j3d_p0_ops.cpu().numpy().squeeze(0)  # template, m 21*3

    ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(pre_joints[9] - pre_joints[0])
    j3d_pre_process = pre_joints * ratio  # template, m
    j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]  # shape: (21, 3)
    pose_R = adaptive_IK(template, j3d_pre_process)  # shape: (1,16,3,3)
    pose_R = torch.from_numpy(pose_R).float()  # shape: (1,16,3,3)

    # Now replace the original mano params
    mano_params = {}
    mano_params['hand_pose'] = pose_R[:, 1:, ...].to(device)
    mano_params['global_orient'] = pose_R[:, [0], ...].to(device)
    mano_params['betas'] = betas
    return mano_params

###############################################################################
# From https://github.com/xiezhongzhao/Hand-Motion-Capture
###############################################################################

SNAP_PARENT = [
    0,  # 0's parent
    0,  # 1's parent
    1,
    2,
    3,
    0,  # 5's parent
    5,
    6,
    7,
    0,  # 9's parent
    9,
    10,
    11,
    0,  # 13's parent
    13,
    14,
    15,
    0,  # 17's parent
    17,
    18,
    19,
]

kinematic_tree = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

ID2ROT = {
        2: 13, 3: 14, 4: 15,
        6: 1, 7: 2, 8: 3,
        10: 4, 11: 5, 12: 6,
        14: 10, 15: 11, 16: 12,
        18: 7, 19: 8, 20: 9,
    }

angels0 = np.zeros((1, 21))


def to_dict(joints):
    temp_dict = dict()
    for i in range(21):
        temp_dict[i] = joints[:, [i]]
    return temp_dict


def adaptive_IK(T_, P_):
    '''
    Computes pose parameters given template and predictions.
    We think the twist of hand bone could be omitted.

    :param T: template ,21*3
    :param P: target, 21*3
    :return: pose params.
    '''
    T = T_.copy().astype(np.float64)
    P = P_.copy().astype(np.float64)

    P = P.transpose(1, 0)
    T = T.transpose(1, 0)

    # to dict
    P = to_dict(P)
    T = to_dict(T)

    # some globals
    R = {}
    R_pa_k = {}
    q = {}

    q[0] = T[0]  # in fact, q[0] = P[0] = T[0].

    # compute R0, here we think R0 is not only a Orthogonal matrix, but also a Rotation matrix.
    # you can refer to paper "Least-Squares Fitting of Two 3-D Point Sets. K. S. Arun; T. S. Huang; S. D. Blostein"
    # It is slightly different from  https://github.com/Jeff-sjtu/HybrIK/blob/main/hybrik/utils/pose_utils.py#L4, in which R0 is regard as orthogonal matrix only.
    # Using their method might further boost accuracy.
    P_0 = np.concatenate([P[1] - P[0], P[5] - P[0],
                          P[9] - P[0], P[13] - P[0],
                          P[17] - P[0]], axis=-1)
    T_0 = np.concatenate([T[1] - T[0], T[5] - T[0],
                          T[9] - T[0], T[13] - T[0],
                          T[17] - T[0]], axis=-1)
    H = np.matmul(T_0, P_0.T)
    U, S, V_T = np.linalg.svd(H)
    V = V_T.T
    R0 = np.matmul(V, U.T)
    det0 = np.linalg.det(R0)
    if abs(det0 + 1) < 1e-6:
        V_ = V.copy()

        if (abs(S) < 1e-4).sum():
            V_[:, 2] = -V_[:, 2]
            R0 = np.matmul(V_, U.T)
    R[0] = R0
    # the bone from 1,5,9,13,17 to 0 has same rotations
    R[1] = R[0].copy()
    R[5] = R[0].copy()
    R[9] = R[0].copy()
    R[13] = R[0].copy()
    R[17] = R[0].copy()
    # compute rotation along kinematics
    for k in kinematic_tree:
        pa = SNAP_PARENT[k]
        pa_pa = SNAP_PARENT[pa]
        q[pa] = np.matmul(R[pa], (T[pa] - T[pa_pa])) + q[pa_pa]
        delta_p_k = np.matmul(np.linalg.inv(R[pa]), P[k] - q[pa])
        delta_p_k = delta_p_k.reshape((3,))
        delta_t_k = T[k] - T[pa]
        delta_t_k = delta_t_k.reshape((3,))
        temp_axis = np.cross(delta_t_k, delta_p_k)
        axis = temp_axis / (np.linalg.norm(temp_axis, axis=-1) + 1e-8)
        temp = (np.linalg.norm(delta_t_k, axis=0) + 1e-8) * (np.linalg.norm(delta_p_k, axis=0) + 1e-8)
        cos_alpha = np.dot(delta_t_k, delta_p_k) / temp

        alpha = np.arccos(cos_alpha)

        twist = delta_t_k
        D_sw = transforms3d.axangles.axangle2mat(axis=axis, angle=alpha, is_normalized=False)
        D_tw = transforms3d.axangles.axangle2mat(axis=twist, angle=angels0[:, k], is_normalized=False)
        R_pa_k[k] = np.matmul(D_sw, D_tw)
        R[k] = np.matmul(R[pa], R_pa_k[k])

    pose_R = np.zeros((1, 16, 3, 3))
    pose_R[0, 0] = R[0]
    for key in ID2ROT.keys():
        value = ID2ROT[key]
        pose_R[0, value] = R_pa_k[key]

    return pose_R
