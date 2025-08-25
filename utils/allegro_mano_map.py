'''
This file contains the mapping between the MANO and Allegro hand joints
There are 21 MANO joints. We map them to 17 Allegro hand joints.
Since the MANO hand has 5 fingers and the Allegro hand has 4 fingers, we map the Allegro's middle finger to be the midpoint between
the middle and ring fingers on the MANO hand and the Allegro's ring finger to be the midpoint between the ring and little fingers on the MANO hand.
The Allegro's thumb and index fingers are mapped directly to the thumb and index fingers on the MANO hand, for functional reasons.
For performing the reward between the MANO and Allegro hands, we use the Euclidean distance between the corresponding joints, where 
the corresponding joints are defined by the MANO_IDX array. You will notice that there are two columns in MANO_IDX, since certain Allegro fingers
are the midpoint between two MANO fingers. Therefore, we calculate the Euclidean distance between the two corresponding MANO joints and take the average
using the SCALING provided. For the Allegro hand's thumb and index finger, you will notice that the corresponding MANO joints are the same for both columns.
This is to ensure that the Euclidean distance is calculated between the same joints for both the MANO and Allegro hands.

Something like this:
(np.linalg.norm(allegro_output[ALLEGRO_JOINTS] - mano_output[MANO_IDX[:, 0]]) + np.linalg.norm(allegro_output[ALLEGRO_JOINTS] - mano_output[MANO_IDX[:, 1]])) / SCALING
'''

# import numpy as np


# MANO_IDX = np.array(
#     [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20]]
# ).T

# ALLEGRO_JOINTS = [
#     'gripper_base_joint',
#     'joint_12.0',
#     'joint_13.0',
#     'joint_14.0',
#     'joint_15.0_tip',
#     'joint_0.0',
#     'joint_1.0',
#     'joint_2.0',
#     'joint_3.0_tip',
#     'joint_4.0',
#     'joint_5.0',
#     'joint_6.0',
#     'joint_7.0_tip',
#     'joint_8.0',
#     'joint_9.0',
#     'joint_10.0',
#     'joint_11.0_tip',
# ]

# SCALING = 0.5

_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_ALLEGRO_LINKS = [
    'world', 
    'link_base', 
    'link1', 
    'link2', 
    'link3', 
    'link4', 
    'link5', 
    'link6', 
    'link7', 
    'base_link', 
    'link_0.0', 
    'link_1.0', 
    'link_2.0', 
    'link_3.0', 
    'link_3.0_tip', 
    'link_4.0', 
    'link_5.0', 
    'link_6.0', 
    'link_7.0', 
    'link_7.0_tip', 
    'link_8.0', 
    'link_9.0', 
    'link_10.0', 
    'link_11.0', 
    'link_11_v.0', 
    'link_11.0_tip', 
    'link_12.0', 
    'link_13.0', 
    'link_14.0', 
    'link_15.0', 
    'link_15.0_tip', 
    'palm', 
    'palm_center'
]

_ALLEGRO_TO_MANO = {
    'link7': ['wrist', 'wrist'],
    'link_12.0': ['thumb_mcp', 'thumb_mcp'],
    'link_13.0': ['thumb_pip', 'thumb_pip'],
    'link_14.0': ['thumb_dip', 'thumb_dip'],
    'link_15.0_tip': ['thumb_tip', 'thumb_tip'],
    'link_0.0': ['index_mcp', 'index_mcp'],
    'link_1.0': ['index_pip', 'index_pip'],
    'link_2.0': ['index_dip', 'index_dip'],
    'link_3.0_tip': ['index_tip', 'index_tip'],
    'link_4.0': ['middle_mcp', 'ring_mcp'],
    'link_5.0': ['middle_pip', 'ring_pip'],
    'link_6.0': ['middle_dip', 'ring_dip'],
    'link_7.0_tip': ['middle_tip', 'ring_tip'],
    'link_8.0': ['ring_mcp', 'little_mcp'],
    'link_9.0': ['ring_pip', 'little_pip'],
    'link_10.0': ['ring_dip', 'little_dip'],
    'link_11.0_tip': ['ring_tip', 'little_tip'],
}

import numpy as np

_ALLEGRO_IDX = {
    i: link for i, link in enumerate(_ALLEGRO_TO_MANO.keys())
}

_MANO_IDX = np.array(
    [[_MANO_JOINTS.index(v[0]), _MANO_JOINTS.index(v[1])] for v in _ALLEGRO_TO_MANO.values()]
)

SCALING = 0.5