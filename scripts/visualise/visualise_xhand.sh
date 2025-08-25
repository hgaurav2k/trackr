export PYTHONPATH="."
export LD_LIBRARY_PATH="$(which python | sed 's/\/bin\/python//g')/lib"
echo $LD_LIBRARY_PATH
PCKPT=checkpoints/track_predn.pt
CHECKPOINT=$1
CL=8
HIDDEN_DIM=512
N_LAYER=6
N_HEAD=8
cmd="python scripts/viz_policy.py   num_gpus=1 \
    task=XhandGrasping \
    test=True viser=True\
    task.env.enableDebugVis=True \
    checkpoint=$CHECKPOINT \
    train.algo=PPO train=AllegroXarmNewPPO_mlp \
    headless=True  graphics_device_id=4 \
    task.env.useHandJointPoseRew=False \
    pretrain.checkpoint=$PCKPT \
    pretrain.model.hidden_dim=$HIDDEN_DIM \
    pretrain.model.n_layer=$N_LAYER \
    pretrain.model.context_length=$CL \
    pretrain.model.n_head=$N_HEAD \
    task.env.stage2_hist_len=$CL \
    wandb_activate=True \
    wandb_name=XhandGrasping \
    pipeline=gpu  rl_device=cuda:0  sim_device=cuda:0 \
    train.ppo.minibatch_size=64 num_envs=64 \
    seed=-1"

echo $cmd
eval $cmd