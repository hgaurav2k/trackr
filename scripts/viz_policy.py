import isaacgym
from isaacgym import gymapi, gymtorch, gymutil
import os
import hydra
import datetime
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import wandb
from tasks import isaacgym_task_map
from utils.reformat import omegaconf_to_dict, print_dict
from utils.utils import set_np_formatting, set_seed, git_hash, git_diff_config
from utils.logger import Logger
import torch 
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np 
import time 
import json 
from copy import deepcopy
import utils.pytorch_utils as ptu 
import plotly.graph_objects as go
from utils.viser_utils import map_sim2viser
import pickle as pkl
def main(rank, world_size, config):

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        global_rank = rank
        seed = config.seed + global_rank
    else:
        global_rank = rank
        seed = config.seed

    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    # _ = set_seed(seed)

    print(f"global_rank = {global_rank} seed = {seed}")

    if config.wandb_activate and not config.test and (global_rank == 0 or world_size ==1):
        wandb_logger = wandb.init(project=config.wandb_project, name=config.wandb_name, config=omegaconf_to_dict(config))
    else:
        wandb_logger=None

    if (global_rank == 0 or world_size == 1):
        output_dif = os.path.join('outputs', config.wandb_name)
        logger = Logger(output_dif, summary_writer=wandb_logger)
    else:
        logger = None

    cprint('Start Building the Environment', 'green', attrs=['bold'])

    if config.num_gpus > 1:
        rl_device = f'cuda:{global_rank}'
        sim_device = f'cuda:{global_rank}'
        graphics_id = global_rank
    else:
        rl_device = config.rl_device
        sim_device = config.sim_device
        graphics_id = config.graphics_device_id

    env = isaacgym_task_map[config.task_name](
        cfg=omegaconf_to_dict(config.task),
        rl_device = rl_device,
        sim_device=sim_device,
        graphics_device_id=graphics_id,
        headless=config.headless,
        virtual_screen_capture=config.capture_video,
        force_render=config.force_render,
    )

    # for debugging
    if config.train.algo == 'PPOTransformer':
        if env.use_obs_as_prop:
            config.pretrain.model.proprio_dim = env.full_state_size 
        config.train.network = config.pretrain.model 
        config.task.env.stage2_hist_len = config.pretrain.model.context_length
        # Load the model to finetune

    # print(config)

    # if config.pretrain.model.all_fingers:
    #     from algo.pretrained.human_model_af import HumanModelAF
    #     state_model = HumanModelAF(
    #             cfg=config,
    #             max_ep_len=4096,
    #         ) #add checkpoint when trained
    # else:
    #     from algo.pretrained.human_model import HumanModel
    #     state_model = HumanModel(
    #             cfg=config,
    #             max_ep_len=4096,
    #         ) #add checkpoint when trained


    from algo.pretrained.human_molabel_model import HumanMultiObjModel
    state_model = HumanMultiObjModel(
                cfg=config,
                max_ep_len=4096,
            ) #add checkpoint when trained


    from algo.distmatch.dist_match_mlp_final import VIPER 
        # load checkpoint

    if config.pretrain.checkpoint != '':
        print("Loading checkpoint from: ", config.pretrain.checkpoint)
        state_model.load_state_dict(torch.load(config.pretrain.checkpoint))
    else:
        cprint("No checkpoint provided. Tracking reward logs will be incorrect.", 'red', attrs=['bold'])

    state_model = state_model.to(rl_device)

    agent = VIPER(env, 
                state_model,
                config=config,
                logger=logger, 
                rank=global_rank)

    assert config.test 
    assert config.train.load_path is not None 
    agent.restore_test(config.train.load_path)

    agent.set_eval()
    obs_dict = env.reset()
    j = 0
    all_rewards = [0.0,]*agent.env.num_envs
    all_rewards = np.array(all_rewards)
    all_successes = np.array([0.0,]*agent.env.num_envs)

    if config.viser:
        from utils.viser_mo import MultiObjectViser

        color_list = [
            (255,0,0),
            (0,255,0),
            (0,0,255),
            (255,255,0),
            (0,255,255),
        ]

        viz_idx = np.random.randint(0, agent.env.num_envs)
        table_pose = agent.env.table_poses[viz_idx].detach().cpu().numpy()
        viser = MultiObjectViser(agent.env.hand_arm_asset_file, 
                                    agent.env.table_asset_file, 
                                    np.array([agent.env.allegro_pose.p.x, agent.env.allegro_pose.p.y, agent.env.allegro_pose.p.z]),
                                    np.array([agent.env.allegro_pose.r.w, agent.env.allegro_pose.r.x, agent.env.allegro_pose.r.y, agent.env.allegro_pose.r.z]),
                                    np.array([table_pose[0], table_pose[1], table_pose[2]]), 
                                    np.array([1,0,0,0]),
                                    agent.env.object_names[viz_idx],
                                    agent.env.asset_files_dict,)


        


        from utils.viser_utils import map_sim2viser
        mapping = map_sim2viser(agent.env.dof_names, viser.robot_joint_names)
        print(f"Mapping: {mapping}")
        joint_angles = agent.env.arm_hand_dof_pos[viz_idx].detach().cpu().numpy()

        viser.set_robot(joint_angles, mapping=mapping) #fix this
        viser.add_object(agent.env.object_names[viz_idx], 
                                agent.env.object_pos[viz_idx].detach().cpu().numpy(), 
                                torch.cat((agent.env.object_rot[viz_idx][3:], agent.env.object_rot[viz_idx][:3])).detach().cpu().numpy()
                ) #(w,x,y,z)
        



        #add text checkpoint 
        viser.server.gui.add_text(
            label="pretrained_checkpoint",
            initial_value=f"{config.pretrain.checkpoint}",
        )


        # Add plot controls
        with viser.server.gui.add_folder("Plot Controls", expand_by_default=False, order=100):
            # viser.show_plot = viser.server.gui.add_checkbox(
            #     "Show Action Plot",
            #     initial_value=False,
            #     hint="Toggle action plot visibility"
            # )
            # viser.show_orientation_plot = viser.server.gui.add_checkbox(
            #     "Show Orientation Error Plot",
            #     initial_value=False,
            #     hint="Toggle orientation error plot visibility"
            # )
            # viser.show_joint_error_plot = viser.server.gui.add_checkbox(
            #     "Show Joint Position Error Plot",
            #     initial_value=False,
            #     hint="Toggle joint position error plot visibility"
            # )
            # viser.show_contact_forces_plot = viser.server.gui.add_checkbox(
            #     "Show Contact Forces Plot",
            #     initial_value=False,
            #     hint="Toggle contact forces plot visibility"
            # )
            viser.show_rewards_plot = viser.server.gui.add_checkbox(
                "Show Rewards Plot",
                initial_value=False,
                hint="Toggle rewards plot visibility"
            )            
            viser.rewards_plot = None

            @viser.show_rewards_plot.on_update
            def _(event) -> None:
                if viser.show_rewards_plot.value:
                    if viser.rewards_plot is None:
                        viser.rewards_plot = viser.server.gui.add_plotly(
                            figure=go.Figure(),
                            aspect=1.0,
                            visible=True
                        )
                else:
                    if viser.rewards_plot is not None:
                        viser.rewards_plot.remove()
                        viser.rewards_plot = None

        # set a time loop of 5 seconds mentioninig that viser is starting in 5 seconds
        for i in range(5):
            print(f"Viser is starting in {5-i} seconds")
            time.sleep(1)

    info_list = []
    os.makedirs(f'outputs/jsons/{config.wandb_name}', exist_ok=True)
    os.makedirs(f'outputs/plots/{config.wandb_name}', exist_ok=True)
    # once you fail in the test, you fail. That env is removed from metrics for success calculation
    envs_done = torch.zeros(agent.env.num_envs, dtype=torch.bool, device=rl_device)
    rews = []
    step = 0 
    breakpoint()
    while step < 200:

        if agent.normalize_input:
            processed_obs = agent.running_mean_std(obs_dict['obs'])
        else:
            processed_obs = obs_dict['obs']

        if agent.normalize_proprio_hist:
            processed_proprio = agent.proprio_mean_std(obs_dict['proprio_buf'].clone())
        else:
            processed_proprio = obs_dict['proprio_buf'].clone() #POLICY SPECIFIC SCALING: BEWARE OF THE ACTION SPACE

        if agent.normalize_pc:
            processed_pc_hist = agent.pc_mean_std(obs_dict['pc_buf'].clone())
        else:
            processed_pc_hist = obs_dict['pc_buf'].clone()

        input_dict = {
            "obs": processed_obs,
            "pc_buf": processed_pc_hist,
            "proprio_buf": processed_proprio,
            "action_buf": obs_dict['action_buf'],
            "attn_mask": obs_dict['attn_mask'],
            "timesteps": obs_dict['timesteps']
        }

        mu = agent.model.infer_action(input_dict)            
        mu = torch.clamp(mu, -1.0, 1.0)
        next_obs_dict, r, done, info = agent.env.step(mu)

        if agent.distr_match_coef > 0.0:
            traj_rew,  track_info = agent.distr_matching_rew(obs_dict, next_obs_dict)
            r += traj_rew 
            # visualise next_proprio
            next_kpts = track_info['tracking/next_proprio'].detach().reshape(agent.env.num_envs, agent.env.num_allegro_fingertips, 3)
            for j in range(agent.env.num_allegro_fingertips):
                fingertip_pos_cpu = next_kpts[:, j].cpu().numpy()
                for i in range(agent.env.num_envs):
                    fingertip_transform = gymapi.Transform()
                    fingertip_transform.p = gymapi.Vec3(*fingertip_pos_cpu[i])  
                    gymutil.draw_lines(agent.env.fingertip_geoms[j], agent.env.gym, agent.env.viewer, agent.env.envs[i], fingertip_transform)

            if config.viser and env.debug_viz:
                # draw small 3d points at track_info['next_proprio']
                viser.draw_points(points=track_info['tracking/next_proprio'][viz_idx].detach().cpu().numpy().reshape(-1,3),
                                    color=color_list,
                                    size=0.01,
                                    name="/world/human_tracks"
                                    )
                


                viser.draw_points(points=track_info['tracking/previous_fingertip_pos'][viz_idx].detach().cpu().numpy().reshape(-1,3),
                                    size=0.01,
                                    name="/world/previous_fingertip_pos"
                                    )

                viser.draw_points(points=track_info['tracking/current_fingertip_pos'][viz_idx].detach().cpu().numpy().reshape(-1,3),
                                    color=(127,0,0),
                                    size=0.01,
                                    name="/world/current_fingertip_pos"
                                    )
                


                

        else:
            traj_rew = torch.zeros_like(r)
        info['traj_rew'] = torch.mean(traj_rew)
        info['total_rew'] = torch.mean(env.rew_buf - traj_rew)
        # info['fingertip_delta_rew'] = torch.mean(agent.env.fingertip_delta_rew)
        # info['lifting_rew'] = torch.mean(agent.env.lifting_rew)
        # info['keypoint_rew'] = torch.mean(agent.env.keypoint_rew)

        if config.viser:
            all_rewards += ptu.to_numpy(r)
            all_successes[~envs_done.detach().cpu().numpy()] += ptu.to_numpy(agent.env.is_success[~envs_done])

            joint_angles = agent.env.arm_hand_dof_pos[viz_idx].detach().cpu().numpy()

            viser.set_robot(joint_angles, mapping=mapping)

            viser.set_object(agent.env.object_names[viz_idx], 
                                agent.env.object_pos[viz_idx].detach().cpu().numpy(), 
                                torch.cat((agent.env.object_rot[viz_idx][3:], agent.env.object_rot[viz_idx][:3])).detach().cpu().numpy()
                ) #(w,x,y,z)

            viser.draw_points(points=agent.env.fingertip_pos[viz_idx].detach().cpu().numpy(),
                                color=color_list,
                                size=0.01,
                                name="/world/fingertip_pos"
                                )

            if viser.rewards_plot is not None:

                print(step)
                rews.append(traj_rew[viz_idx].detach().cpu().numpy())
                viser.rewards_plot.figure = update_rewards_plot(rews, 
                                                            reward_name="tracking/reward",
                                                            time_history=list(range(step)))

        envs_done = torch.logical_or(envs_done, done)
        if agent.logger is not None:
            agent.logger.log_dict(agent.env.linearize_info(info),step)

        info = agent.env.linearize_info(info)

        # info.update(self.env.linearize_info(track_info))
        info_list.append(info)

        step += 1

        obs_dict = deepcopy(next_obs_dict)

    import matplotlib.pyplot as plt
    # Plot each metric from info_list over time

    
    if len(info_list) > 0:
        first_info = info_list[0]
        print(first_info)
        num_keys = len(first_info.keys())
        # Calculate number of rows/cols needed for square layout
        n_rows = int(np.ceil(np.sqrt(num_keys)))
        n_cols = int(np.ceil(num_keys / n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        # Flatten axes array for easier indexing
        axes = axes.flatten()

        for idx, key in enumerate(first_info.keys()):
            values = [info[key] for info in info_list]
            axes[idx].plot(values)
            axes[idx].set_title(f'{key} over time')
            axes[idx].set_xlabel('Step')
            axes[idx].set_ylabel(key)

        from datetime import datetime
        plt.tight_layout()

        plt.savefig(f'outputs/plots/{config.wandb_name}/{agent.config.train.load_path.split("/")[-1]}_plt.png')
        print(f'Saved plot to outputs/plots/{config.wandb_name}/{agent.config.train.load_path.split("/")[-1]}_plt.png')
        plt.close()

        summary_dict = {}

        # rake mean reward over trajectory over time and envs.
        for idx, key in enumerate(first_info.keys()):
            n = 0 
            val = 0 
            for info in info_list:
                val += info[key]
                n += 1 
            summary_dict[key] = val / n 
        json.dump(summary_dict, open(f'outputs/jsons/{config.wandb_name}/{agent.config.train.load_path.split("/")[-1]}_final_metrics.json', 'w'))

        pkl.dump(info_list, open(f'outputs/jsons/{config.wandb_name}/{agent.config.train.load_path.split("/")[-1]}_final_metrics.pkl', 'wb'))

        print(f'Saved summary metrics to outputs/jsons/{config.wandb_name}/{agent.config.train.load_path.split("/")[-1]}_final_metrics.json')


@hydra.main(config_name='config', config_path='../cfg/')
def main_multi_gpu(config: DictConfig):
    if config.test:
        # single gpu testing only!
        config.num_gpus = 1
    world_size = config.num_gpus
    if world_size > 1:
        mp.spawn(main,
                 args=(world_size, config),
                 nprocs=world_size,
                 join=True)
    else:
        rank = 0 #config.sim_device.split(":")[1]
        main(rank, 1, config)


def update_rewards_plot(reward_history,
                        reward_name, 
                        time_history):

    
    
    fig = go.Figure()
    # time_points = np.array(range(len(self.rewards_history))) * self.dt
    time_points = np.array(time_history)
    current_time = time_history[-1]
    relative_times = time_points - current_time
    
    
    #make a plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=relative_times,
        y=reward_history,
        name=reward_name,
        line=dict(width=2)
    ))

    return fig 


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    #randomize port address
    
    os.environ["MASTER_PORT"] = "29445"
    main_multi_gpu()
