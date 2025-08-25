# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

from mimetypes import common_types
import os
import time
import torch

from algo.ppo.experience import ExperienceBuffer
from algo.models.models import PointNetActorCritic, ActorCritic, SavingModel
from algo.models.running_mean_std import RunningMeanStd
from utils.misc import AverageScalarMeter
import utils.pytorch_utils as ptu 
import torch.nn as nn 
from datetime import datetime
from termcolor import cprint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import jit
import numpy as np
import pickle
import wandb 
import cv2 
from PIL import Image 
import PIL 

class PPO(nn.Module):

    def __init__(self, env, config, logger=None, rank=0):
        super().__init__()

        if config.num_gpus > 1:
            self.device = f"cuda:{rank}" #config['rl_device']
        else:
            self.device = config['rl_device']
        self.network_config = config.train.network
        self.ppo_config = config.train.ppo
        self.config = config
        self.logger = logger 

        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.num_gradient_steps = self.ppo_config['num_gradient_steps']
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.rank = rank
        self.num_gpus = config.num_gpus

        if config["pc_input"]:
            net_config = {
                'actor_units': self.network_config.mlp.units,
                'actions_num': self.actions_num,
                'point_cloud_index': (self.env.point_cloud_begin_index, self.env.point_cloud_end_index),
                'point_cloud_out_dim' : self.network_config.pc_mlp.out_dim,
                'point_cloud_num': 100,
                'input_shape': self.obs_shape[0] - 100*3 +  self.network_config.pc_mlp.out_dim,
            }

            self.model = PointNetActorCritic(net_config)
        else:
        
            net_config = {
                'actor_units': self.network_config.mlp.units,
                'actions_num': self.actions_num,
                'input_shape': self.obs_shape[0],
            }
            self.model = ActorCritic(net_config)
        
        # if config.wandb_activate:
        #    wandb.watch(self.model,log="all",log_freq=1)

        # construct the local model
        self.model.to(self.device)

        # construct the DDP model
        if self.num_gpus > 1:
            print("Rank: ", rank)
            self.ddp_model = DDP(self.model, device_ids=[rank])
        
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        # ---- Normalization ----
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(self.obs_shape[0]).to(self.device)
        
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        if self.rank == 0 or self.num_gpus == 1:
            if logger is not None: 
                self.output_dir = os.path.join(logger._log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M'))
                self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
                #self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
                os.makedirs(self.nn_dir, exist_ok=True)
        #os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)

        if self.num_gpus > 1:
            self.optimizer = torch.optim.Adam(self.ddp_model.parameters(), self.last_lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)
        
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or config.test or config.teacher_mode
        # ---- scheduler ----
        self.kl_threshold = self.ppo_config['kl_threshold']
        self.min_lr = self.ppo_config['min_lr']
        self.max_lr = self.ppo_config['max_lr']
        self.scheduler = AdaptiveScheduler(self.kl_threshold, min_lr = self.min_lr, max_lr = self.max_lr)
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Wandb Logger Info ----
        self.extra_info = {}

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = 0
        self.replay_buffer = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size, self.num_gradient_steps, self.obs_shape[0],
            self.actions_num, self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        self.best_success = -10
        self.cur_success = -100
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0


        if config.get('teacher_mode',False):
            self.set_eval()
            self.restore_test(config.checkpoint)
            print("PPO Agent set to test mode")


    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    @torch.no_grad()
    def get_action(self, obs_dict):
        if self.normalize_input:
            processed_obs = self.running_mean_std(obs_dict['obs'])
        else:
            processed_obs = obs_dict['obs']
        input_dict = {
            'obs': processed_obs,
        }
        res_dict = self.model.get_action(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True) #why is this done?
        return res_dict

    def train(self):

        self.obs = self.env.reset()
        # self.obs = dict(obs=torch.zeros(self.obs['obs']))
        self.agent_steps = self.batch_size

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            train_info = self.train_epoch()
            self.replay_buffer.data_dict = None

            #log and save checkpoints
            if self.logger is not None:
                mean_rewards = self.episode_rewards.get_mean()
                mean_lengths = self.episode_lengths.get_mean()
                checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_reward_{mean_rewards:.2f}'
                self.logger.log_scalar(mean_rewards,'episode_rewards/step', self.agent_steps*self.num_gpus)
                self.logger.log_scalar(mean_lengths,'episode_lengths/step', self.agent_steps*self.num_gpus)
                self.logger.log_dict(self.extra_info, self.agent_steps*self.num_gpus)

                if self.save_freq > 0:
                    if self.epoch_num % self.save_freq == 0:
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                        self.save(os.path.join(self.nn_dir, 'last'))

                if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                    print(f'save current best reward: {mean_rewards:.2f}')
                    self.best_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, 'best'))

                if self.cur_success > self.best_success and self.epoch_num >= self.save_best_after:
                    print(f'save current best success: {self.cur_success:.2f}')
                    self.best_success = self.cur_success
                    self.save(os.path.join(self.nn_dir, 'best'))

                print("Agent Steps: ", self.agent_steps, "Num Epoch: ", self.epoch_num, "Mean Reward: ", mean_rewards, "Mean Length: ", mean_lengths)

            #self.logger.flush()
            self.extra_info = {}

            # sync barrier (I assume this is done to wait for state syncronization)
            if self.num_gpus > 1:
                torch.distributed.barrier()

        print('***Max steps achieved***')

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.normalize_input:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')


    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn,map_location=self.device)
        cprint('careful, using non-strict matching. Restored from {}'.format(fn), 'red', attrs=['bold'])
        self.model.load_state_dict(checkpoint['model'])
        cprint('model restored from {}'.format(fn), 'green', attrs=['bold'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn,map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        
        cprint('model restored from {}'.format(fn), 'green', attrs=['bold'])
        save_jit = self.config['save_jit']
        if save_jit:
            self.save_jit_trace(''.join(fn.split('.')[:-1]))
            self.save_joint_limits(''.join(fn.split('.')[:-1]))

    def save_jit_trace(self, name):
        print("Saving Jit Trace")
        if self.normalize_input:
            self.running_mean_std.eval()
            saving_model = SavingModel(self.model, self.running_mean_std)
        obs = torch.ones(self.obs_shape, dtype=torch.float32).to(self.device)
        model_trace = jit.trace(saving_model, obs)
        jit.save(model_trace, f'{name}.jit.pt')
    
    def save_joint_limits(self, name):
        limits = {'lower': self.env.arm_dof_lower_limits.cpu().numpy(),
                  'upper': self.env.arm_dof_upper_limits.cpu().numpy()}
        with open(f'{name}.limits.pickle', 'wb') as handle:
            pickle.dump(limits, handle)
        return
        
    def test(self,visualise=True, name='PPOScratch'):
        self.set_eval()
        obs_dict = self.env.reset()
        j = 0

        if visualise:
            all_rewards = [0.0,]*self.env.num_envs
            all_rewards = np.array(all_rewards)
            all_successes = np.array([0.0,]*self.env.num_envs)

        while j < 1200:
            # input_dict = {
            #     'obs': self.running_mean_std(obs_dict['obs']),
            # }
            # mu = self.model.infer_action(input_dict)  
            # input_dict = {
            #     'obs': self.running_mean_std(obs_dict['obs']),
            # }
            if self.normalize_input:    
                obs = self.running_mean_std(obs_dict['obs'])
            else:
                obs = obs_dict['obs']
            # pos = obs[:,0:23]
            # vel = obs[:,23:46]
            # priv = obs[:,46:51]
            # pc = obs[:,51:300+51].reshape(self.env.num_envs, -1, 3)
            # mu = self.model.infer_action_test(pos,vel,priv,pc)
            mu = self.model.infer_action({
                'obs': obs,
            })
            mu = torch.clamp(mu, -1.0, 1.0)

            obs_dict, r, done, info = self.env.step(mu)
            
            if visualise:       
                all_rewards += ptu.to_numpy(r)
                # all_successes += ptu.to_numpy(self.env.is_success)
            
            if self.logger is not None:
                self.logger.log_dict(self.env.linearize_info(info),j)
                if self.env.log_video and j%10 == 0:
                    self.logger.log_gifs(self.env.video_frames, name="video")
            j += 1 
        
        if visualise:
            from utils.misc import compile_per_asset
            compile_per_asset(self.env,all_rewards,all_successes,name=name)


    def collect_data(self):

        #ensure dir 
        os.makedirs("data",exist_ok=True)
        
        def save(episode,filename):
            with open(filename,"wb") as handle:
                pickle.dump(episode,handle)
            return 

        self.set_eval()
        obs_dict = self.env.reset()
        j = 0
        iters = torch.tensor(list(range(self.env.num_envs))).to(self.device)
        episode_data = [{
            'robot_state': [],
            'action': [],
            'reward': [],
            'done': [],

        } for _ in range(self.env.num_envs)]

        for i in range(self.env.num_envs):
            for j in range(self.env.num_cameras):
                episode_data[i][f'img{j}'] = []

        from datetime import datetime

        dt = datetime.now().strftime("%m%d%H%M")
    
        while j < 600:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
            }
            mu = self.model.infer_action(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            next_obs_dict, r, done, info = self.env.step(mu)

            #attach (obs_dict,r,done,info) to the in memory set 
            #the in memory epis

    @torch.no_grad()
    def get_action_eval(self,obs_dict):
        input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
            }
        mu = self.model.infer_action(input_dict)
        mu = torch.clamp(mu, -1.0, 1.0)
        return mu 



    def track_agent_pose(self):

        self.set_eval()
        obs_dict = self.env.reset()
        j = 0
        all_rewards = [0.0,]*self.env.num_envs
        all_rewards = np.array(all_rewards)
        all_successes = np.array([0.0,]*self.env.num_envs)
        joint_poses = []
        while j < 6000:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
            }
            mu = torch.clam(self.model(input_dict) -1.0, 1.0) #check 
            obs_dict, r, done, info = self.env.step(mu)
            joint_poses.append(ptu.to_numpy(self.env.cur_targets))
            all_rewards += ptu.to_numpy(r)
            all_successes += ptu.to_numpy(self.env.is_success)
            if self.logger is not None:
                self.logger.log_dict(self.env.linearize_info(info),j)
            j += 1 
        
        pickle.dump(joint_poses,open("joint_poses_allegro.pkl","wb"))

    def get_target_reference(self):

        self.set_eval()

        obs_dict = self.env.reset()
        tgt_ref_hnd_pos = torch.zeros((self.env.num_envs,len(self.env.hand_joint_handles),3))
        tgt_ref_hnd_rot = torch.zeros((self.env.num_envs,len(self.env.hand_joint_handles),4))
        tgt_ref_hnd_pose = torch.zeros((self.env.num_envs,len(self.env.hand_joint_handles),13))
        tgt_ref_hnd_pc = torch.zeros(self.env.num_envs,len(self.env.hand_joint_handles),self.env.point_cloud_sampled_dim,3)
        tgt_ref_obj_pos = torch.zeros((self.env.num_envs,3))
        tgt_ref_obj_pose = torch.zeros((self.env.num_envs,13))
        tgt_ref_obj_rot = torch.zeros((self.env.num_envs,4))
        tgt_ref_obj_pc = torch.zeros(self.env.num_envs,self.env.point_cloud_sampled_dim,3)
        object_names =  self.env.object_names 

        j = 0 
        self.prev_success = torch.zeros(self.env.num_envs).bool()
        while j < 600:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
            }
            mu = self.model.infer_action(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            flg = torch.logical_and(self.env.is_success,~self.prev_success)
            if torch.any(flg):
                tgt_ref_hnd_pose[flg] = self.env.hand_joint_state[flg]
                tgt_ref_hnd_pos[flg] = self.env.hand_joint_pos[flg]
                tgt_ref_hnd_rot[flg] = self.env.hand_joint_rot[flg]
                tgt_ref_hnd_pc[flg] = self.env.hand_joint_point_cloud_buf[flg]
                tgt_ref_obj_pos[flg] = self.env.object_pos[flg]
                tgt_ref_obj_pose[flg] = self.env.object_state[flg]
                tgt_ref_obj_rot[flg] = self.env.object_rot[flg]
                tgt_ref_obj_pc[flg] = self.env.point_cloud_buf[flg]
                self.prev_success[flg] = True
            
            print("Success this step:", torch.sum(self.env.is_success.int()))
            print("Total successes:", torch.sum(self.prev_success.int()))
            j += 1 


        target_reference = {
            'object_names': object_names, 
            'success': list(ptu.to_numpy(self.prev_success.int())),
            'tgt_ref_hnd_pose': list(ptu.to_numpy(tgt_ref_hnd_pose)),
            'tgt_ref_hnd_pos': list(ptu.to_numpy(tgt_ref_hnd_pos)),
            'tgt_ref_hnd_rot': list(ptu.to_numpy(tgt_ref_hnd_rot)),
            'tgt_ref_hnd_pc': list(ptu.to_numpy(tgt_ref_hnd_pc)),
            'tgt_ref_obj_pos': list(ptu.to_numpy(tgt_ref_obj_pos)),
            'tgt_ref_obj_rot': list(ptu.to_numpy(tgt_ref_obj_rot)),
            'tgt_ref_obj_pose': list(ptu.to_numpy(tgt_ref_obj_pose)),
            'tgt_ref_obj_pc': list(ptu.to_numpy(tgt_ref_obj_pc))
        }

        import pickle as pkl 
        pkl.dump(target_reference,open("target_reference.pkl","wb"))


    def get_target_reference_trajectory(self):

        self.set_eval()

        obs_dict = self.env.reset()

        eplen = self.env.max_episode_length 
        tgt_ref_hnd_pos = torch.zeros((self.env.num_envs,eplen,len(self.env.hand_joint_handles),3))
        tgt_ref_hnd_rot = torch.zeros((self.env.num_envs,eplen,len(self.env.hand_joint_handles),4))
        tgt_ref_hnd_pose = torch.zeros((self.env.num_envs,eplen,len(self.env.hand_joint_handles),13))
        tgt_ref_hnd_pc = torch.zeros(self.env.num_envs,eplen,len(self.env.hand_joint_handles),self.env.point_cloud_sampled_dim,3)
        tgt_ref_obj_pos = torch.zeros((self.env.num_envs,eplen,3))
        tgt_ref_obj_pose = torch.zeros((self.env.num_envs,eplen,13))
        tgt_ref_obj_rot = torch.zeros((self.env.num_envs,eplen,4))
        tgt_ref_obj_pc = torch.zeros(self.env.num_envs, eplen, self.env.point_cloud_sampled_dim,3)
        object_names =  self.env.object_names 

        j = 0 
        self.prev_success = torch.zeros(self.env.num_envs).bool()
        print(eplen)
        while j < eplen:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
            }
            mu = self.model.infer_action(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

            tgt_ref_hnd_pose[:,j] = self.env.hand_joint_state
            tgt_ref_hnd_pos[:,j] = self.env.hand_joint_pos
            tgt_ref_hnd_rot[:,j] = self.env.hand_joint_rot
            tgt_ref_hnd_pc[:,j] = self.env.hand_joint_point_cloud_buf
            tgt_ref_obj_pos[:,j] = self.env.object_pos
            tgt_ref_obj_pose[:,j] = self.env.object_state
            tgt_ref_obj_rot[:,j] = self.env.object_rot
            tgt_ref_obj_pc[:,j] = self.env.point_cloud_buf

            j+=1 



        target_reference = {
            'object_names': object_names, 
            'success': list(ptu.to_numpy(self.prev_success.int())),
            'tgt_ref_hnd_pose': list(ptu.to_numpy(tgt_ref_hnd_pose)),
            'tgt_ref_hnd_pos': list(ptu.to_numpy(tgt_ref_hnd_pos)),
            'tgt_ref_hnd_rot': list(ptu.to_numpy(tgt_ref_hnd_rot)),
            'tgt_ref_hnd_pc': list(ptu.to_numpy(tgt_ref_hnd_pc)),
            'tgt_ref_obj_pos': list(ptu.to_numpy(tgt_ref_obj_pos)),
            'tgt_ref_obj_rot': list(ptu.to_numpy(tgt_ref_obj_rot)),
            'tgt_ref_obj_pose': list(ptu.to_numpy(tgt_ref_obj_pose)),
            'tgt_ref_obj_pc': list(ptu.to_numpy(tgt_ref_obj_pc))
        }

        import pickle as pkl 
        pkl.dump(target_reference,open("target_reference_trajectory.pkl","wb"))
    


    def collect_data(self):

        #ensure dir 
        os.makedirs("data",exist_ok=True)
        
        def save(episode,filename):
            with open(filename,"wb") as handle:
                pickle.dump(episode,handle)
            return 

        self.set_eval()
        obs_dict = self.env.reset()
        j = 0
        iters = torch.tensor(list(range(self.env.num_envs))).to(self.device)*0
        # episode_data = [{
        #     'robot_state': [],
        #     'action': [],
        #     'reward': [],
        #     'done': [],

        # } for _ in range(self.env.num_envs)]

        episode_data = [{
            "joint_state": [],
            "ee_target": [],
            "ee_target_norm": [],
            "obj_pc": [],
            "success": []
        } for _ in range(self.env.num_envs)]


        # for i in range(self.env.num_envs):
        #     for j in range(self.env.num_cameras):
        #         episode_data[i][f'img{j}'] = []

        from datetime import datetime

        dt = datetime.now().strftime("%m%d%H%M")

        for i in range(self.env.num_envs):
            episode_data[i]['joint_state'].append(obs_dict['proprio_buf'][i][-1].detach().cpu().numpy())
            episode_data[i]['obj_pc'].append(obs_dict['pc_buf'][i][-1].detach().cpu().numpy())
            episode_data[i]['success'].append(self.env.is_success[i].detach().cpu().numpy())

        failed_cases = 0 
        success_cases = 0 
        while j < 600:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
            }
            mu = self.model.infer_action(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            next_obs_dict, r, done, info = self.env.step(mu)

            #attach (obs_dict,r,done,info) to the in memory set 
            #the in memory episode when they terminate, store them
            #increase and go ahead
            for i in range(self.env.num_envs):
                episode_data[i]['joint_state'].append(obs_dict['proprio_buf'][i][-1].detach().cpu().numpy())
                episode_data[i]['ee_target'].append(self.env.cur_targets[i].detach().cpu().numpy())
                episode_data[i]['ee_target_norm'].append(mu[i].detach().cpu().numpy())
                episode_data[i]['obj_pc'].append(obs_dict['pc_buf'][i][-1].detach().cpu().numpy())
                episode_data[i]['success'].append(self.env.is_success[i].detach().cpu().numpy())
            
            reset_indices = torch.nonzero(done).squeeze(1)
            

            for idx in reset_indices:
                if np.any(episode_data[idx]['success']) == 1:
                    success_cases += 1
                    filename = f"data/env_{idx}_episode_{iters[idx].detach().cpu().item()}_{dt}.pkl"
                    save(episode_data[idx],filename)
                    iters[idx] += 1
                
                else:
                    failed_cases += 1
                
                print(f"Envs have failed {failed_cases} times and succeeded {success_cases} times")
            
            j += 1
            obs_dict = next_obs_dict 
        

        return 



    def test_verbose(self):
        self.obs = self.env.reset()
        self.set_eval()
        self.horizon_length = self.env.max_episode_length
        self.play_steps()
        value_preds = self.replay_buffer[0]

        

    

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()

        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, c_losses = [], []
        entropies, kls = [], []
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.replay_buffer)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs = self.replay_buffer[i] #i-th "mini-batch"
                
                if self.normalize_input:
                    obs = self.running_mean_std(obs)
                assert not torch.any(obs.isnan())
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                }
                if self.num_gpus > 1:
                    res_dict = self.ddp_model(batch_dict)
                else:
                    res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)

                a_loss, c_loss, entropy = [torch.mean(loss) for loss in [a_loss, c_loss, entropy]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef

                self.optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    if self.num_gpus > 1:
                        torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)

                self.replay_buffer.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr
            kls.append(av_kls)

        self.rl_train_time += (time.time() - _t)

        train_info = {
            'losses/actor_loss' : ptu.to_numpy(torch.mean(torch.stack(a_losses))),
            'losses/critic_loss' : ptu.to_numpy(torch.mean(torch.stack(c_losses))),
            'losses/policy_entropy' : ptu.to_numpy(torch.mean(torch.stack(entropies))),
            'losses/kl_old-vs-new_policy' : ptu.to_numpy(torch.mean(torch.stack(kls))),
            'optimizer/learning_rate' : self.last_lr,
        }

        self.extra_info.update(train_info)


        return 

    def play_steps(self):

        for i in range(self.horizon_length):
            res_dict = self.get_action(self.obs)
            # collect o_t
            self.replay_buffer.update_data('obses', i, self.obs['obs'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.replay_buffer.update_data(k, i, res_dict[k])

            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0) 
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.replay_buffer.update_data('dones', i, self.dones)
            shaped_rewards = 0.1 * rewards.clone()  #to scale rewards for numerical stability  

            #are timeouts in my infos?
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.replay_buffer.update_data('rewards', i, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.get_action(self.obs)
        last_values = res_dict['values'] #last_values computed using the critic.

        #add to log
        self.extra_info.update(self.env.linearize_info(infos))

        self.cur_success = self.extra_info.get('successes',0)

        self.agent_steps += self.batch_size
        self.replay_buffer.compute_return(last_values, self.gamma, self.tau)
        self.replay_buffer.prepare_training()

        returns = self.replay_buffer.data_dict['returns']
        values = self.replay_buffer.data_dict['values']

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)    
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
            
        self.replay_buffer.data_dict['values'] = values
        self.replay_buffer.data_dict['returns'] = returns



def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


# # from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
# class AdaptiveScheduler(object):
#     def __init__(self, kl_threshold=0.008):
#         super().__init__()
#         self.min_lr = 1e-6
#         self.max_lr = 1e-2
#         self.kl_threshold = kl_threshold

#     def update(self, current_lr, kl_dist):
#         lr = current_lr
#         if kl_dist > (2.0 * self.kl_threshold):
#             lr = max(current_lr / 1.5, self.min_lr)
#         if kl_dist < (0.5 * self.kl_threshold):
#             lr = min(current_lr * 1.5, self.max_lr)
#         return lr

# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008, min_lr=1e-6, max_lr=1e-2):
        super().__init__()
        self.min_lr = min_lr 
        self.max_lr = max_lr 
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


