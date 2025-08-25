XHAND_LINKS =  ['right_hand_link', 'right_hand_ee_link', 'right_hand_index_bend_link', \
            'right_hand_index_rota_link1', 'right_hand_index_rota_link2', 'right_hand_index_rota_tip', \
            'right_hand_mid_link1', 'right_hand_mid_link2', 'right_hand_mid_tip', 'right_hand_pinky_link1', \
            'right_hand_pinky_link2', 'right_hand_pinky_tip', 'right_hand_ring_link1', 'right_hand_ring_link2', \
            'right_hand_ring_tip', 'right_hand_thumb_bend_link', 'right_hand_thumb_rota_link1', \
            'right_hand_thumb_rota_link2', 'right_hand_thumb_rota_tip']


XHAND_KPT_LINKS = [
    'right_hand_link',
    'right_hand_index_rota_link1',
    'right_hand_mid_link1',
    'right_hand_pinky_link1',
    'right_hand_ring_link1',
    'right_hand_thumb_rota_link1',
    'right_hand_thumb_rota_link2',
    'right_hand_index_rota_link2',
    'right_hand_mid_link2',
    'right_hand_pinky_link2',
    'right_hand_ring_link2',
    'right_hand_thumb_rota_tip',
    'right_hand_index_rota_tip',
    'right_hand_mid_tip',
    'right_hand_pinky_tip',
    'right_hand_ring_tip',
]

XHAND_EDGES = [
    #['right_hand_link', 'right_hand_ee_link'],
    ['right_hand_link', 'right_hand_index_rota_link1'],
    ['right_hand_link', 'right_hand_mid_link1'],
    ['right_hand_link', 'right_hand_pinky_link1'],
    ['right_hand_link', 'right_hand_ring_link1'],
    ['right_hand_link', 'right_hand_thumb_rota_link1'],
    ['right_hand_index_rota_link1', 'right_hand_index_rota_link2'],
    ['right_hand_index_rota_link2', 'right_hand_index_rota_tip'],
    ['right_hand_mid_link1', 'right_hand_mid_link2'],
    ['right_hand_mid_link2', 'right_hand_mid_tip'],
    ['right_hand_pinky_link1', 'right_hand_pinky_link2'],
    ['right_hand_pinky_link2', 'right_hand_pinky_tip'],
    ['right_hand_ring_link1', 'right_hand_ring_link2'],
    ['right_hand_ring_link2', 'right_hand_ring_tip'],
    ['right_hand_thumb_rota_link1', 'right_hand_thumb_rota_link2'],
    ['right_hand_thumb_rota_link2', 'right_hand_thumb_rota_tip'],
]


HUMAN_KPTS = [
    [
        [1, 'right_hand_link']
    ],
    [
        [1, 'right_hand_thumb_rota_link1'],
    ],
    [
        [1/2, 'right_hand_thumb_rota_link1'],
        [1/2, 'right_hand_thumb_rota_link2'],
    ],
    [
        [1, 'right_hand_thumb_rota_link2'],
    ],
    [
        [1, 'right_hand_thumb_rota_tip'],
    ],

    [
        [1, 'right_hand_index_rota_link1'],
    ],
    [
        [1/2, 'right_hand_index_rota_link1'],
        [1/2, 'right_hand_index_rota_link2'],
    ],
    [
        [1, 'right_hand_index_rota_link2'],
    ],
    [
        [1, 'right_hand_index_rota_tip'],
    ],

    [
        [1, 'right_hand_mid_link1'],
    ],
    [
        [1/2, 'right_hand_mid_link1'],
        [1/2, 'right_hand_mid_link2'],
    ],
    [
        [1, 'right_hand_mid_link2'],
    ],
    [
        [1, 'right_hand_mid_tip'],
    ],
    [
        [1, 'right_hand_ring_link1'],
    ],
    [
        [1/2, 'right_hand_ring_link1'],
        [1/2, 'right_hand_ring_link2'],
    ],
    [
        [1, 'right_hand_ring_link2'],
    ],
    [
        [1, 'right_hand_ring_tip'],
    ],
    [
        [1, 'right_hand_pinky_link1'],
    ],
    [
        [1/2, 'right_hand_pinky_link1'],
        [1/2, 'right_hand_pinky_link2'],
    ],
    [
        [1, 'right_hand_pinky_link2'],
    ],
    [
        [1, 'right_hand_pinky_tip'],
    ],
]

HUMAN_EDGE_KPTS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3,4], 
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
]


import torch
def xhand2human(xhand_kpts: torch.Tensor):
    human_kpts = []
    for i in range(len(HUMAN_KPTS)):
        human_kpts.append(
            sum(xhand_kpts[..., XHAND_LINKS.index(HUMAN_KPTS[i][j][1]), :] * HUMAN_KPTS[i][j][0] for j in range(len(HUMAN_KPTS[i])))
        )
    return torch.cat(human_kpts, dim=-1)


def human2xhand(human_kpts: torch.Tensor):
    assert len(human_kpts.shape) == 3
    xhand_kpts = torch.zeros((len(human_kpts), len(XHAND_LINKS), 3), dtype=human_kpts.dtype, device=human_kpts.device)
    # Take subset of human_kpts
    for i in range(len(HUMAN_KPTS)):
        if len(HUMAN_KPTS[i]) == 1:
            xhand_kpts[..., XHAND_LINKS.index(HUMAN_KPTS[i][0][1]), :] = human_kpts[..., i, :]
    return xhand_kpts
            
