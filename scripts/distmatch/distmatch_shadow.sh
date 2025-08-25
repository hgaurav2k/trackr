export PYTHONPATH="."
export LD_LIBRARY_PATH="/home/himanshu/miniforge3/envs/rlgpu/lib"
CL=8
CHECKPOINT="checkpoints/track_predn.pt"
cmd="python scripts/distmatch.py   num_gpus=1 \
    task=ShadowHandGrasping \
    train.algo=PPO train=ShadowHandGraspingPPO_mlp \
    train.ppo.learning_rate=1e-4 \
    train.ppo.kl_threshold=0.015 \
    pretrain.checkpoint=$CHECKPOINT \
    pretrain.model.hidden_dim=512 \
    pretrain.model.n_layer=6 \
    pretrain.model.n_head=8 \
    pretrain.model.all_fingers=True \
    wandb_activate=True \
    wandb_name=ShadowHandGrasping_distmatch\
    pipeline=gpu  rl_device=cuda:0  sim_device=cuda:0 \
    train.ppo.minibatch_size=4096 num_envs=4096 \
    seed=-1"

echo $cmd
eval $cmd
