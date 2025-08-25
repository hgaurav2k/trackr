import isaacgym
import os
import hydra
import datetime
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import wandb
from algo.distmatch.dist_match_mlp_final import VIPER 
from tasks import isaacgym_task_map
from utils.reformat import omegaconf_to_dict, print_dict
from utils.utils import set_np_formatting, set_seed, git_hash, git_diff_config
from utils.logger import Logger
import torch 
import torch.distributed as dist
import torch.multiprocessing as mp

def main(rank, world_size, config):
    
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        global_rank = rank
        if config.seed == -1:
            seed = config.seed
        else:
            seed = config.seed + global_rank
    else:
        global_rank = rank
        seed = config.seed

    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    seed = set_seed(seed)

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

     #for debugging 
    if config.train.algo == 'PPOTransformer':
        if env.use_obs_as_prop:
            config.pretrain.model.proprio_dim = env.full_state_size 
        config.train.network = config.pretrain.model 
        config.task.env.stage2_hist_len = config.pretrain.model.context_length
        # Load the model to finetune


    from algo.tracking.human_model import HumanModel
    state_model = HumanModel(
                cfg=config,
                max_ep_len=4096,
            ) #add checkpoint when trained

    

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

    if config.test:
        # from isaacgym import gymapi
        # cam_pos = gymapi.Vec3(0, -0.60, 0.30)
        # cam_target = gymapi.Vec3(0.0, 0.0, 0.0)  # Center point to look at
        # # Set the camera view using the environment's viewer
        # env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
        # agent.restore_test(config.train.load_path)
        assert config.train.load_path is not None 
        #agent.model.actor.load_state_dict(torch.load(config.checkpoint))
        agent.restore_test(config.train.load_path)
        # #breakpoint()
        agent.test(name=config.wandb_name, visualise=True)
    else:
        if rank <= 0:
            date = str(datetime.datetime.now().strftime('%m%d%H'))
            if config.wandb_activate:
                pid = os.getpid()
                wandb.log({'pid': pid})

        if config.train.load_path == '':
            cprint("Train model from scratch", 'green', attrs=['bold'])
            agent.train()
        else:
            agent.restore_train(config.train.load_path)
            cprint("Load model from: " + config.train.load_path, 'green', attrs=['bold'])
            agent.train()
            
        if config.wandb_activate and (global_rank==0 or world_size==1):
            wandb.finish()


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
        

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    #randomize port address
    
    # Try ports until we find an available one
    port = 29449
    import socket
    while True:
        try:
            
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("localhost", port))
            s.close()
            os.environ["MASTER_PORT"] = str(port)
            break
        except OSError:
            port += 1
    main_multi_gpu()
