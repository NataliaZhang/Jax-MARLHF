#!/bin/bash

cuda_device='6'
steepness=5
reward_model_type="FF"
env_name="MPE_simple_spread_v3"
num_landmarks=5
project_prefix="805_scale_5seed_"
use_unilateral=True
wandb_mode="disabled"
# wandb_mode="online"

num_agents_combinations=(
  # 4
  5
  # 6
  # 7
)

num_batch_pairs=(
  "[5,2,2,1,5] [5,0,0]" # diversified
  # "[10,0,0,0,5] [5,0,0]"  # mix-unilateral
  # "[15,0,0,0,5] [0,0,0]"  # mix-expert
  # "[20,0,0,0,0] [0,0,0]"  # pure-expert
)

mse_loss_coef_combinations=(
  # 0
  # 0.001
  # 0.01
  # 0.1
  1
  # 10
  # 100
  # 1000
  # 10000
)
ref_log_coef_combinations=(
  # 100
  10
  # 1
  # 0.1
  # 0.01
  # 0
)
ff_layer_dim_combinations=(
  # 64
  128
  # 256
)
for num_agents in "${num_agents_combinations[@]}"; do
  for ff_layer_dim in "${ff_layer_dim_combinations[@]}"; do
    for pair in "${num_batch_pairs[@]}"; do
        for mse_loss_coef in "${mse_loss_coef_combinations[@]}"; do
            IFS=" " read -r num_batch num_unilateral_batch <<< $pair
            echo "Running reward_model for num_batch=$num_batch, num_unilateral_batch=$num_unilateral_batch"

            # train reward model
            for seed in 0 # 1 2 3 4
            do
                CUDA_VISIBLE_DEVICES=$cuda_device python3 train/reward_model.py \
                    MSE_LOSS_COEF=$mse_loss_coef \
                    STEEPNESS=$steepness \
                    REWARD_MODEL_TYPE=$reward_model_type \
                    NUM_BATCH=$num_batch \
                    NUM_UNILATERAL_BATCH=$num_unilateral_batch \
                    FF_LAYER_DIM=$ff_layer_dim \
                    ENV_NAME=$env_name \
                    +ENV_KWARGS.num_agents=$num_agents \
                    +ENV_KWARGS.num_landmarks=$num_landmarks \
                    PROJECT_PREFIX=$project_prefix \
                    USE_UNILATERAL=$use_unilateral \
                    WANDB_MODE=$wandb_mode
            done

            # train imitation model
            CUDA_VISIBLE_DEVICES=$cuda_device python3 train/offline_imitation_learning.py \
                NUM_BATCH=$num_batch \
                NUM_UNILATERAL_BATCH=$num_unilateral_batch \
                REWARD_MODEL_TYPE=$reward_model_type \
                FF_LAYER_DIM=$ff_layer_dim \
                MSE_LOSS_COEF=$mse_loss_coef \
                STUDENT_NETWORK_SAVE_PATH='results/ILagent/spread/na'$num_agents'/' \
                VIS_SAVE_PATH='results/vis/IL/spread/' \
                ENV_NAME=$env_name \
                +ENV_KWARGS.num_agents=$num_agents \
                +ENV_KWARGS.num_landmarks=$num_landmarks \
                PROJECT_PREFIX=$project_prefix \
                USE_UNILATERAL=$use_unilateral \
                WANDB_MODE=$wandb_mode

            # train mpe on the reward model
            for seed in 0 # 1 2 3 4
            do
                for ref_log_coef in "${ref_log_coef_combinations[@]}"; do
                  network_path='results/mpe_spread/agents_rm_mix_mse_'$mse_loss_coef'_seed_'$seed
                  CUDA_VISIBLE_DEVICES=$cuda_device python3 train/vdn_offline_rm.py \
                      env.ENV_NAME=$env_name \
                      +env.ENV_KWARGS.num_agents=$num_agents \
                      +env.ENV_KWARGS.num_landmarks=$num_landmarks \
                      alg.EPOCHS=10000 \
                      alg.MSE_LOSS_COEF=$mse_loss_coef \
                      alg.REWARD_MODEL_TYPE=$reward_model_type \
                      alg.NUM_BATCH=$num_batch \
                      alg.NUM_UNILATERAL_BATCH=$num_unilateral_batch \
                      alg.FF_LAYER_DIM=$ff_layer_dim \
                      alg.REFERENCE_NETWORK_PARAM_PATH='results/ILagent/spread/na'$num_agents'/' \
                      PROJECT_PREFIX=$project_prefix \
                      alg.USE_UNILATERAL=$use_unilateral \
                      alg.REF_LOG_COEF=$ref_log_coef \
                      SEED=$seed \
                      WANDB_MODE=$wandb_mode
                done
            done

        done
    done
  done
done