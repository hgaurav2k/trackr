import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
from termcolor import cprint
import numpy as np


class HumanDataset(Dataset):

    def __init__(self, root=None, cfg=None):
        """
        Args:
            data (Any): Your dataset (e.g., images, files, tensors).
            targets (Any): The labels or targets associated with your data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        assert root is not None, "Please provide the root directory of the dataset"
        assert os.path.exists(root), f"The directory {root} does not exist"
        super(HumanDataset, self).__init__()
        self.root = root
        print(f"Loading dataset from {root}")
        self.device = cfg.pretrain.device
        self.num_obj = 3 #cfg.pretrain.model.max_num_obj
        self.ctx = cfg.pretrain.model.context_length + 1
        # self.scale_action = cfg.pretrain.model.scale_action
        # self.scale_proprio = cfg.pretrain.model.scale_proprio
        # set variable to store the episodes
        self.episodes_npy = []
        self.ep_lens = []
        # self.dt = kwargs.get('dt', 0.008333) # 120 Hz
        # self.dt = np.float32(self.dt)
        self.use_residuals = cfg.pretrain.training.use_residuals
        # get all folders of depth 2 in the directory
        subjects_dir = [
            os.path.join(root, episode)
            for episode in os.listdir(root)
            if os.path.isdir(os.path.join(root, episode))
        ]

        print(f"Subjects directory: {subjects_dir}")
        # get all subfolders of depth 2 in subjects_dir
        self.episodes_dir = [
            os.path.join(subject, episode)
            for subject in subjects_dir
            for episode in os.listdir(subject)
            if os.path.isdir(os.path.join(subject, episode))
        ]
        self.episodes_dir = sorted(self.episodes_dir)

        print(f"Episodes directory: {self.episodes_dir}")

        assert len(self.episodes_dir) > 0, f"No episodes found in the directory {root}"
        # load all the episodes
        for episode in self.episodes_dir:
            self.load_episode_fnames(episode)

        assert (
            len(self.episodes_npy) > 0
        ), f"No trajectories found in the directory {root}"
        # save the min, max, and mean of the episode lengths
        self.min_ep_len = np.min(self.ep_lens)
        self.max_ep_len = np.max(self.ep_lens)
        self.mean_ep_len = np.mean(self.ep_lens)
        cprint(
            f"Min episode length: {self.min_ep_len}, Max episode length: {self.max_ep_len}, Mean episode length: {self.mean_ep_len}",
            color="cyan",
            attrs=["bold"],
        )
        self.ep_lens = torch.tensor(self.ep_lens)
        self.cumsum = torch.cumsum(self.ep_lens, 0)
        self.visualise()

    def load_episode_fnames(self, episode_dir: str):
        """
        Load the episodes filenames.
        """
        for episode_fname in sorted(os.listdir(episode_dir)):
            # continue if the file is not a npy file
            if not episode_fname.endswith(".npy"):
                continue
            ep = np.load(
                os.path.join(episode_dir, episode_fname), allow_pickle=True
            ).item()
            # self.episodes_npy.append(ep)
            # load the file and get the length
            eplen = len(ep["hand_joints"]) - self.ctx + 1
            # assert (
            #     eplen > 0
            # ), f"Episode length is less than the context length {self.ctx}"
            if eplen <= 0:
                continue

            self.ep_lens.append(eplen)
            self.episodes_npy.append(ep)

    def visualise(self):
        """
        Visualise the dataset.
        """
        cprint(
            f"Number of episodes: {len(self.episodes_npy)}",
            color="green",
            attrs=["bold"],
        )
        cprint(
            f"Number of examples: {torch.sum(self.ep_lens)}",
            color="green",
            attrs=["bold"],
        )
        # Load the first episode to get the dimension of the proprio and action
        ep = self.episodes_npy[0]
        cprint(
            f"Proprio dimension: {len(ep['hand_joints'][0])}",
            color="green",
            attrs=["bold"],
        )
        cprint(
            f"Point cloud dimension: {len(ep['object_pc'][list(ep['object_pc'].keys())[0]][0])}",
            color="green",
            attrs=["bold"],
        )

    def __len__(self):
        """Returns the size of the dataset."""
        return torch.sum(self.ep_lens).item()

    def __getitem__(self, index):
        """
        Generates one sample of data.

        Args:
            index (int): The index of the item in the dataset

        Returns:
            sample (Any): The data sample corresponding to the given index.
            target (Any): The target corresponding to the given data sample.
        """

        ep_idx = torch.searchsorted(self.cumsum, index, right=True)
        # open the pickle file
        idx = index - torch.sum(self.ep_lens[:ep_idx])
        ep = self.episodes_npy[ep_idx]

        hand_kpts = ep["hand_joints"][idx : idx + self.ctx, 0]

        
        keys = list(ep["object_pc"].keys())
        keys.remove(ep["object_ids"][ep["ycb_grasp_id"]])



        n_obj = np.random.choice(self.num_obj)
        if n_obj > 0:
            keys = np.random.choice(keys, n_obj, replace=False)
            keys = [ep["object_ids"][ep["ycb_grasp_id"]]] + list(keys)
        else:
            keys = [ep["object_ids"][ep["ycb_grasp_id"]],]

        obj_pcs_arr = np.stack([ep["object_pc"][k][:,0] for k in keys], axis=0)
        obj_pc = obj_pcs_arr[:, idx : idx + self.ctx]

        # fingertip for allegro 4, 8, 12, 16
        fingert_kpts = np.zeros((hand_kpts.shape[0], 5, 3))
        fingert_kpts[:, 0, :] = hand_kpts[:, 4, :]  # allegro 4
        fingert_kpts[:, 1, :] = hand_kpts[:, 8, :]  # allegro 8
        fingert_kpts[:, 2, :] = hand_kpts[:, 12, :]  # allegro 12
        fingert_kpts[:, 3, :] = hand_kpts[:, 16, :]  # allegro 20
        fingert_kpts[:, 4, :] = hand_kpts[:, 20, :]  # allegro 16
        hand_kpts = fingert_kpts


        return {
            "hand_kpts": hand_kpts.reshape((hand_kpts.shape[0], -1)),
            "obj_pc": obj_pc, 
            "timesteps": np.arange(obj_pc.shape[1]-1),
            "label" : ep["object_ids"][ep["ycb_grasp_id"]],
            "object_ids": keys,
        }

def collate_fn(batch):
    hand_kpts = np.stack([item["hand_kpts"] for item in batch])
    timesteps = np.stack([item["timesteps"] for item in batch])
    labels = np.stack([item["label"] for item in batch])
    object_pc = [item["obj_pc"] for item in batch]
    object_ids = [item["object_ids"] for item in batch]



    
    hand_kpts = torch.tensor(hand_kpts, dtype=torch.float32, requires_grad=False)
    # Get max number of objects across batch
    max_num_objects = max(pc.shape[0] for pc in object_pc)
    
    # Pad object_pc with zeros on left to max length
    padded_object_pc = []
    object_mask = []
    padded_object_ids = []
    for pc, ids in zip(object_pc, object_ids):
        pad_size = max_num_objects - pc.shape[0]
        padded = np.pad(pc, ((0, pad_size), (0,0), (0,0), (0,0)), mode='constant')
        padded_ids = np.pad(ids, (0, pad_size), mode='constant')
        mask = np.pad(np.ones(pc.shape[0]), (0, pad_size), mode='constant')
        padded_object_pc.append(padded)
        object_mask.append(mask)
        padded_object_ids.append(padded_ids)
    
    object_pc = torch.tensor(np.stack(padded_object_pc), dtype=torch.float32, requires_grad=False).transpose(1,2)
    object_mask = torch.tensor(np.stack(object_mask), dtype=torch.bool, requires_grad=False)


    timesteps = torch.tensor(timesteps, dtype=torch.long, requires_grad=False)
    attention_mask = torch.ones_like(timesteps)
    labels = torch.tensor(labels, dtype=torch.long, requires_grad=False)
    object_ids = torch.tensor(padded_object_ids, dtype=torch.long, requires_grad=False)


    batch = {
        "hand_kpts": hand_kpts,
        "object_pc": object_pc,
        "timesteps": timesteps,
        "labels": labels,
        "attention_mask": attention_mask,
        "object_mask": object_mask,
        "object_ids": object_ids,
    }

    return batch
