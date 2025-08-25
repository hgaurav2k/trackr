export PYTHONPATH="."
export LD_LIBRARY_PATH="$(which python | sed 's/\/bin\/python//g')/lib"
echo $LD_LIBRARY_PATH

CL=8
CHECKPOINT="checkpoints/track_predn.pt"

cmd="python scripts/distmatch.py num_gpus=1 \
    task=AllegroXarmCabinet \
    train.algo=PPO train=AllegroXarmNewPPO_mlp \
    pretrain.checkpoint=$CHECKPOINT \
    pretrain.model.hidden_dim=512 \
    pretrain.model.n_layer=6 \
    pretrain.model.n_head=8 \
    pretrain.model.context_length=$CL \
    headless=True graphics_device_id=0 \
    train.ppo.learning_rate=1e-4 \
    wandb_activate=True \
    wandb_name=AllegroXarmCabinet_distmatch_sparser_env_indist2 \
    pipeline=gpu rl_device=cuda:0 sim_device=cuda:0 \
    train.ppo.minibatch_size=4096 num_envs=4096 \
    seed=-1"

echo $cmd
eval $cmd