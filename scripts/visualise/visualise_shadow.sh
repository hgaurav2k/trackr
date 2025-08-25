export PYTHONPATH="."
export LD_LIBRARY_PATH="$(which python | sed 's/\/bin\/python//g')/lib"
echo $LD_LIBRARY_PATH
PCKPT=checkpoints/track_predn.pt
CHECKPOINT=$1
CL=8
HIDDEN_DIM=512
N_LAYER=6
N_HEAD=8
cmd="python scripts/distmatch_mlp.py num_gpus=1  \
    task=ShadowHandGrasping \
    checkpoint=$CHECKPOINT  \
    task.env.input_priv=False \
    test=True headless=False pc_input=True \
    task.env.enableFingertipPosHistory=True \
    task.env.useOldActionSpace=True \
    task.env.useKeypointReward=True \
    task.env.useFingertipReward=False \
    task.env.useFingertipShapeDistReward=False \
    task.env.usePalmReward=False \
    task.env.useHandJointPoseRew=False \
    task.env.useAllegroTips=False \
    task.env.useLiftingReward=False \
    task.env.distrMatchRewScale=0.0 \
    pretrain.checkpoint=$PCKPT \
    pretrain.model.hidden_dim=$HIDDEN_DIM \
    pretrain.model.n_layer=$N_LAYER \
    pretrain.model.n_head=$N_HEAD \
    pretrain.model.context_length=$CL \
    wandb_activate=False     wandb_name=ShadowHandGrasping_MLP \
    rl_device=cuda:0 sim_device=cuda:0 pipeline=gpu\
    train.algo=PPO \
    graphics_device_id=4 \
    train.ppo.minibatch_size=64 num_envs=64 \
    seed=0"
echo $cmd
eval $cmd
