RAND=(05)
CLS=(8)

declare -A MODEL_CONFIGS=(
    ["large"]="512 6 8"
    # ["small"]="128 4 4"
)

# Function to run experiment
run_experiment() {
    local NOISE=$1
    local CL=$2
    local GPU=$3
    local MODEL_SIZE=$4
    local HIDDEN_DIM=$5
    local N_LAYER=$6
    local N_HEAD=$7

    export PYTHONPATH="."
    export LD_LIBRARY_PATH="$(which python | sed 's/\/bin\/python//g')/lib"
    DATADIR=$1
    
    CMD="CUDA_VISIBLE_DEVICES=$GPU python scripts/train_tracker.py \
        num_envs=5 \
        pipeline=gpu \
        rl_device=cuda:0 sim_device=cuda:0 \
        pretrain.training.root_dir=$DATADIR \
        pretrain.validation.root_dir=$DATADIR \
        pretrain.training.noise_arm=0.$NOISE \
        pretrain.training.noise_hand=0.$NOISE \
        pretrain.model.context_length=$CL \
        pretrain.model.all_fingers=True \
        pretrain.training.batch_size=64 \
        pretrain.model.hidden_dim=$HIDDEN_DIM \
        pretrain.model.n_layer=$N_LAYER \
        pretrain.model.n_head=$N_HEAD \
        pretrain.wandb_activate=True \
        pretrain.wandb_name=fingerkpt_gripper_cl_${CL}_noise_${NOISE} \
        pretrain.training.model_save_freq=10000 \
        pretrain.training.num_epochs=2000 \
        pretrain.training.log_freq=500 \
        task.env.enableVideoLog=False \
        task.env.episodeLength=10"

    echo "Starting experiment: MODEL=$MODEL_SIZE, NOISE=$NOISE, CL=$CL on GPU $GPU"
    eval $CMD &
}

# Counter for experiment distribution
count=0
# Run all experiments
for model_size in "large"; do
    # Read model configuration
    read -r hidden_dim n_layer n_head <<< "${MODEL_CONFIGS[$model_size]}"
    
    for noise in "${RAND[@]}"; do
        for cl in "${CLS[@]}"; do
            gpu=$((count % 8))  # Distribute across 8 GPUs
            run_experiment $noise $cl $gpu $model_size $hidden_dim $n_layer $n_head
            count=$((count + 1))
        done
    done
done

# Wait for all experiments to complete
wait

echo "All experiments completed!"
