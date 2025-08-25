export PYTHONPATH="."
export LD_LIBRARY_PATH="$(which python | sed 's/\/bin\/python//g')/lib"
echo $LD_LIBRARY_PATH
PCKPT=checkpoints/track_predn.pt
CHECKPOINT=$1
CL=8
HIDDEN_DIM=512
N_LAYER=6
N_HEAD=8
cmd="python scripts/viz_policy.py num_gpus=1 viser=True \
    task=AllegroXarmNew train=AllegroXarmNewPPO_mlp \
    test=True headless=True \
    checkpoint=$CHECKPOINT \
    task.env.input_priv=True \
    pc_input=True \
    task.env.enableDebugVis=True \
    graphics_device_id=3 \
    train.ppo.learning_rate=1e-4 \
    task.env.stage2_hist_len=$CL \
    pretrain.checkpoint=$PCKPT \
    pretrain.model.hidden_dim=$HIDDEN_DIM \
    pretrain.model.n_layer=$N_LAYER \
    pretrain.model.n_head=$N_HEAD \
    pretrain.model.context_length=$CL \
    wandb_name=AllegroXarm_MLP \
    rl_device=cuda:0 sim_device=cuda:0 pipeline=gpu \
    train.ppo.minibatch_size=16 num_envs=16"

echo $cmd
eval $cmd