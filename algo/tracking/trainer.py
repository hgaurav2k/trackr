import numpy as np

import torch
import time
from torch.utils.data import DataLoader
import os
from datetime import datetime
import wandb
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from termcolor import cprint


class Trainer:

    def __init__(
        self,
        model,
        train_dataset,
        collate_fn,
        optimizer,
        loss_fn,
        model_save_dir,
        val_dataset=None,
        config=None,
        scheduler=None,
        eval_fns=None,
        logger=None,
        rank=0,
        world_size=1,
        device="cuda",
    ):

        self.model = model
        self.optimizer = optimizer
        self.batch_size = config.pretrain.training.batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.save_dir = model_save_dir
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.logger = logger
        self.saved_model_number = 0
        self.full_autoregressive = config.pretrain.model.full_autoregressive
        self.action_input = config.pretrain.model.action_input
        self.modality_aligned = config.pretrain.training.modality_aligned
        self.time_shift = config.pretrain.training.time_shift
        self.add_proprio_noise = config.pretrain.training.add_proprio_noise
        self.add_action_noise = config.pretrain.training.add_action_noise
        self.add_data_driven_noise = config.pretrain.training.add_data_driven_noise
        num_workers = 0 #8  # config.pretrain.training.num_workers
        self.use_proprio_loss = config.pretrain.training.use_proprio_loss
        self.log_freq = config.pretrain.training.log_freq
        self.noise_arm = config.pretrain.training.noise_arm
        self.noise_hand = config.pretrain.training.noise_hand
        self.model_save_freq = config.pretrain.training.model_save_freq
        assert self.time_shift >= 0, "Cannot have negative time shift"
        assert (
            self.time_shift < self.train_dataset.ctx - 1
        ), "Time shift cannot be larger than the context length"

        if self.modality_aligned:
            self.proprio_shift = self.time_shift + 1
            self.action_shift = self.time_shift + 1
            self.pc_shift = self.time_shift + 1
        else:
            self.proprio_shift = self.time_shift + 1
            self.action_shift = self.time_shift
            self.pc_shift = self.time_shift

        # create a dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
            prefetch_factor=8 if num_workers > 0 else None,
        )

        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
        else:
            self.val_dataloader = None

        self.start_time = time.time()

    def train_epoch(self, iter_num=0, print_logs=False):

        train_losses = []
        train_losses_next_proprio = []
        logs = dict()

        train_start = time.time()

        self.model.train()

        for i, batch in enumerate(tqdm.tqdm(self.train_dataloader)):

            # proprio, object_pc, labels, timesteps, attention_mask = batch

            # batch = (
            #     proprio.to(self.train_dataset.device),
            #     object_pc.to(self.train_dataset.device),
            #     labels.to(self.train_dataset.device),
            #     timesteps.to(self.train_dataset.device),
            #     (
            #         attention_mask.to(self.train_dataset.device)
            #         if attention_mask is not None
            #         else None
            #     ),
            # )

            batch = {k: v.to(self.train_dataset.device) if v is not None else None for k, v in batch.items()}

            train_loss = self.train_step(batch)

            train_losses.append(train_loss["loss"])
            train_losses_next_proprio.append(train_loss["proprio"])
            

            if self.scheduler is not None:
                self.scheduler.step()

            if self.logger is not None and i % self.log_freq == 0:
                logs["time/training"] = time.time() - train_start
                logs["time/total"] = time.time() - self.start_time
                logs["optimizer/lr"] = self.optimizer.param_groups[0]["lr"]
                global_step = iter_num * len(self.train_dataloader) + i
                logs["training/train_loss_mean"] = np.mean(train_losses)
                logs["training/train_loss_std"] = np.std(train_losses)
                train_losses = []
                logs["training/train_loss_next_proprio_mean"] = np.mean(
                    train_losses_next_proprio
                )
                logs["training/train_loss_next_proprio_std"] = np.std(
                    train_losses_next_proprio
                )
                train_losses_next_proprio = []
                self.logger.log_dict(logs, global_step)

            if (
                self.save_dir is not None
                and (iter_num * len(self.train_dataloader) + i) % self.model_save_freq
                == 0
            ):
                global_step = iter_num * len(self.train_dataloader) + i
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, f"model_step_{global_step}.pt"),
                )
                torch.save(
                    self.model.state_dict(), os.path.join(self.save_dir, f"last.pt")
                )
                self.saved_model_number += 1

            if print_logs and i % self.log_freq == 0:
                for k in self.diagnostics:
                    logs[k] = self.diagnostics[k]
                print("=" * 80)
                print(f"Iteration {iter_num}")
                for k, v in logs.items():
                    print(f"{k}: {v}")

        return logs

    def validate_step(self, batch):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError


class HumanTrainer(Trainer):

    def train_step(self, batch):

        proprio, depth, timesteps, labels, attention_mask = batch

        proprio_target = torch.clone(proprio[:, 1:])

        proprio = proprio[:, :-1]
        depth = depth[:, :-1]

        if self.add_proprio_noise:
            noise = torch.zeros_like(proprio)
            noise[..., :7] = torch.randn_like(proprio[..., :7]) * self.noise_arm
            noise[..., 7:] = torch.randn_like(proprio[..., 7:]) * self.noise_hand
            proprio = proprio + noise

        depth += torch.randn_like(depth) * 0.01

        pred_dict, _ = self.model.forward(
            proprio,
            depth,
            labels,
            timesteps,
            attention_mask,
        )

        next_proprio_preds = pred_dict["next_proprio"]

        loss_next_proprio = self.loss_fn(next_proprio_preds, proprio_target)
        loss = loss_next_proprio

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/proprio_error"] = (
                loss_next_proprio.detach().cpu().item()
            )
            self.diagnostics["training/loss"] = loss.detach().cpu().item()

        return_dict = {
            "loss": loss.detach().cpu().item(),
            "proprio": loss_next_proprio.detach().cpu().item(),
        }

        return return_dict


class HumanMultiObjLabelTrainer(Trainer):

    def train_step(self, batch):


        proprio = batch["hand_kpts"]
        depth = batch["object_pc"]
        object_ids = batch["object_ids"]
        labels = batch["labels"]
        timesteps = batch["timesteps"]
        attention_mask = batch["attention_mask"]
        object_mask = batch["object_mask"]

        proprio_target = torch.clone(proprio[:, 1:])

        proprio = proprio[:, :-1]
        depth = depth[:, :-1]

        if self.add_proprio_noise:
            noise = torch.zeros_like(proprio)
            noise[..., :7] = torch.randn_like(proprio[..., :7]) * self.noise_arm
            noise[..., 7:] = torch.randn_like(proprio[..., 7:]) * self.noise_hand
            proprio = proprio + noise

        depth += torch.randn_like(depth) * 0.01

        pred_dict, _ = self.model.forward(
            proprio,
            depth,
            object_ids,
            labels,
            timesteps,
            attention_mask,
            object_mask,
        )

        next_proprio_preds = pred_dict["next_proprio"]

        loss_next_proprio = self.loss_fn(next_proprio_preds, proprio_target)
        loss = loss_next_proprio

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/proprio_error"] = (
                loss_next_proprio.detach().cpu().item()
            )
            self.diagnostics["training/loss"] = loss.detach().cpu().item()

        return_dict = {
            "loss": loss.detach().cpu().item(),
            "proprio": loss_next_proprio.detach().cpu().item(),
        }

        return return_dict


def chamfer_distance(points1, points2):
    # points1 and points2 are tensors of shape (batch_size, t, num_points, 3)

    points1 = points1.reshape(-1, points1.shape[-2], 3)
    points2 = points2.reshape(-1, points2.shape[-2], 3)
    # Compute pairwise distance matrix
    diff = points1.unsqueeze(2) - points2.unsqueeze(1)
    dist_matrix = torch.sum(diff**2, dim=-1)

    # For each point in points1, find the nearest point in points2
    min_dist1, _ = torch.min(dist_matrix, dim=2)

    # For each point in points2, find the nearest point in points1
    min_dist2, _ = torch.min(dist_matrix, dim=1)

    # Compute the Chamfer Distance
    chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)

    return chamfer_dist


def chamfer_distance_abs(points1, points2):
    # points1 and points2 are tensors of shape (batch_size, t, num_points, 3)

    points1 = points1.reshape(-1, points1.shape[-2], 3)
    points2 = points2.reshape(-1, points2.shape[-2], 3)
    # Compute pairwise distance matrix

    diff = torch.abs(points1.unsqueeze(2) - points2.unsqueeze(1))
    dist_matrix = torch.sum(diff, dim=-1)

    # For each point in points1, find the nearest point in points2
    min_dist1, _ = torch.min(dist_matrix, dim=2)

    # For each point in points2, find the nearest point in points1
    min_dist2, _ = torch.min(dist_matrix, dim=1)

    # Compute the Chamfer Distance
    chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)

    return chamfer_dist
