import numpy as np
import torch
import torch.nn as nn
import random
import transformers
from algo.models.running_mean_std import RunningMeanStd
from .transformer_flashattn import GPT2Model
from collections import deque
from termcolor import cprint

np.set_printoptions(precision=3)


class HumanModel(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(self, max_ep_len=4096, cfg=None):

        super(HumanModel, self).__init__()

        self.kpt_dim = 5 * 3
        self.pc_num = 100  # check where used
        self.total_obj_num = 22
        self.max_ep_len = max_ep_len
        self.device = cfg.pretrain.device
        self.max_obj_num = 5  # TODO: check this
        self.n_ctx = (
            self.max_obj_num + 1
        ) * cfg.pretrain.model.context_length + 1  # +1 for the object id
        self.cfg = cfg
        self.hidden_size = cfg.pretrain.model.hidden_dim

        if cfg.pretrain.use_flash_attn:
            from .transformer_flashattn import GPT2Model
        else:
            from .transformer_new import GPT2Model

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            hidden_size=cfg.pretrain.model.hidden_dim,
            n_embd=cfg.pretrain.model.hidden_dim,
            n_head=cfg.pretrain.model.n_head,
            n_layer=cfg.pretrain.model.n_layer,
            resid_pdrop=cfg.pretrain.model.resid_pdrop,
            embd_pdrop=cfg.pretrain.model.embd_pdrop,
            attn_pdrop=cfg.pretrain.model.attn_pdrop,
            n_ctx=self.n_ctx,
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_label = nn.Embedding(self.total_obj_num, self.hidden_size)

        self.embed_timestep = nn.Embedding(
            self.n_ctx, self.hidden_size
        )  # 1 extra for padding

        self.embed_proprio = torch.nn.Linear(self.kpt_dim, self.hidden_size)

        self.embed_pc = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.MaxPool2d((self.pc_num, 1)),
        )  # PointNet
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # note: we don't predict states or returns for the paper
        # self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_proprio = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.kpt_dim)])
        )

        self.predict_pc = nn.Sequential(
            *(
                [
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ELU(inplace=True),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ELU(inplace=True),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ELU(inplace=True),
                    nn.Linear(self.hidden_size, 3 * self.pc_num),
                ]
            )
        )

    def forward(self, proprio, object_pc, object_ids, labels, timesteps, attention_mask, object_mask=None):

        # proprio is just hand kpt here
        batch_size, seq_length = proprio.shape[0], proprio.shape[1]
        num_objects = object_pc.shape[2]
        # embed each modality with a different head
        proprio_embeddings = self.embed_proprio(proprio)
        pc_embeddings = self.embed_pc(
            object_pc.reshape(batch_size, -1, *object_pc.shape[-2:])
        ).squeeze(-2)
        pc_embeddings = pc_embeddings.reshape(
            batch_size, object_pc.shape[1], object_pc.shape[2], -1
        )
        # pc_embeddings = self.embed_pc(object_pc)
        time_embeddings = self.embed_timestep(timesteps)
        # time embeddings are treated similar to positional embeddings
        proprio_embeddings = proprio_embeddings + time_embeddings
        
        pc_embeddings = pc_embeddings + time_embeddings.unsqueeze(-2)
        #adding object_id embeddings to pc_embeddings 

        pc_embeddings = pc_embeddings + self.embed_label(object_ids).unsqueeze(1)
        label_embeddings = self.embed_label(labels)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions

        stacked_inputs = torch.cat(
            (
                pc_embeddings,
                proprio_embeddings.unsqueeze(-2),
            ),
            dim=-2,
        ).reshape(batch_size, (num_objects + 1) * seq_length, self.hidden_size)

        stacked_inputs = torch.cat(
            (label_embeddings.unsqueeze(-2), stacked_inputs), dim=-2
        )

        stacked_inputs = self.embed_ln(stacked_inputs)

        t = attention_mask.shape[1]

        stacked_attention_mask = (
            torch.cat(
                [
                object_mask.unsqueeze(-1).repeat(1, 1, t),
                attention_mask.unsqueeze(1)],
                dim=1,
            )
            .permute(0, 2, 1)
            .reshape(batch_size, (num_objects + 1) * seq_length)
        )


        stacked_attention_mask = torch.cat(
            (torch.ones_like(stacked_attention_mask[:, :1]), stacked_attention_mask),
            dim=1,
        )

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )

        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # states (0), or actions (1); i.e. x[:,1,t] is the token for s_t

        x = (
            x[:, 1:]
            .reshape(batch_size, seq_length, -1, self.hidden_size)
            .permute(0, 2, 1, 3)
        )
        next_kpt_preds = self.predict_proprio(x[:, -1])  # aligned modalities
        # put predictions in a dict
        pred_dict = {"next_proprio": next_kpt_preds}

        return pred_dict, pc_embeddings

    @torch.no_grad()
    def run_env(self, env, cfg=None, **kwargs):

        self.eval()

        next_obs_dict = env.reset()
        num_envs = env.num_envs
        episode_rewards = torch.zeros((num_envs,)).to(self.device)
        episode_lengths = torch.zeros((num_envs,)).to(self.device)

        info_dict = {"episode_reward": [], "episode_length": []}

        q_limits = {
            "lower": env.arm_hand_dof_lower_limits,
            "upper": env.arm_hand_dof_upper_limits,
        }

        def scale_q(q, limits):
            q = (q - limits["lower"][None]) / (
                limits["upper"][None] - limits["lower"][None]
            )
            q = 2 * q - 1
            return q

        # Unscale the actions
        def unscale_q(q, limits):
            q = (
                0.5 * (q + 1) * (limits["upper"][None] - limits["lower"][None])
                + limits["lower"][None]
            )
            return q

        done = False
        timestep = 0

        while True:
            proprio_hist_input = next_obs_dict["proprio_buf"].clone()
            if self.cfg.pretrain.model.scale_proprio:
                proprio_hist_input = scale_q(proprio_hist_input, q_limits)

            obj_pc_hist_input = next_obs_dict["pc_buf"].clone()

            attn_mask = next_obs_dict["attn_mask"].clone()
            ts = next_obs_dict["timesteps"].clone().long()

            # action_hist_input = torch.cat(action_hist[:,1:], torch.zeros_like(action_hist[:,-1:]), dim=1)
            action_hist_input = next_obs_dict["action_buf"].clone()
            action_hist_input = torch.cat(
                (action_hist_input, torch.zeros_like(action_hist_input[:, -1:])), dim=1
            )

            pred_dict, _ = self.forward(
                proprio_hist_input,
                obj_pc_hist_input,
                action_hist_input,
                ts,
                attention_mask=attn_mask,
                **kwargs
            )

            pred_action = pred_dict["action"][:, -1]
            # next_obs_dict, r, done, info = env.step(pred_action.clone())
            # episode_rewards += r
            # action_hist = torch.cat(action_hist, pred_action.unsqueeze(1))

            at_reset_env_ids = torch.where(done)[0]
            for env_id in at_reset_env_ids:
                info_dict["episode_reward"].append(episode_rewards[env_id].item())
                info_dict["episode_length"].append(episode_lengths[env_id].item())
                episode_rewards[env_id] = 0
                episode_lengths[env_id] = 0

            timestep += 1
            episode_lengths += 1

            if timestep > env.max_episode_length:
                break

        # Transfor the list into a tensor
        info_dict["episode_reward"] = torch.tensor(info_dict["episode_reward"])
        info_dict["episode_length"] = torch.tensor(info_dict["episode_length"])
        info_dict = env.linearize_info(info_dict)

        return info_dict

    def next_state(self, proprio, object_pc, object_ids, labels, timesteps, attention_mask, object_mask):


        is_allegro = False 
        if proprio.shape[-1] == 4*3:
            # take mean of last two dims and impose in between them to complete the hand 
            is_allegro = True 
            b,t,d = proprio.shape
            proprio = proprio.reshape(b,t,4,3)
            proprio = torch.cat([proprio[:, :, :3], (proprio[:, :, 2:3] + proprio[:, :, 3:])/2, proprio[:, :, 3:]], dim=-2)
            proprio = proprio.reshape(b,t,-1)

        pred_dict, _ = self.forward(
            proprio, object_pc, object_ids, labels, timesteps, attention_mask, object_mask
        )


        if is_allegro:
            b,t,d = pred_dict["next_proprio"].shape
            pred_dict["next_proprio"] = pred_dict["next_proprio"].reshape(b,t,5,3)
            pred_dict["next_proprio"] = torch.cat([pred_dict["next_proprio"][:, :, :3], pred_dict["next_proprio"][:,:,4:]], dim=-2)
            pred_dict["next_proprio"] = pred_dict["next_proprio"].reshape(b,t,-1)
        
        return {
            "next_proprio": pred_dict["next_proprio"][:, -1],
        }

    def get_init_action(self, env):
        return torch.tensor(env.action_space.sample()).unsqueeze(0).to(self.device)

    # def run_mpc(self,
    #             env,
    #             cfg=None,
    #             **kwargs):

    #     self.eval()

    #     next_obs_dict = env.reset()
    #     num_envs = env.num_envs

    #     #output action that gets highest reward in MCTS
    #     N_ITER =
    #     VAR =
    #     N_CANDIDATES =

    #     #sample init_action from policy
    #     action = self.get_init_action(env)
    #     candidates = [action + torch.randn_like(action) * VAR for _ in range(N_CANDIDATES)]

    #     for i in range(N_ITER):
    #         rewards
