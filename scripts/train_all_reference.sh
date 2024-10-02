#!/bin/bash

cuda_device='6'
steepness=5
reward_model_type="FF"
rm_load_from_pretrain=False
env_name="MPE_simple_reference_v3"
imitation_student_save_path="results/ILagent/reference/"
project_prefix="test_"
use_unilateral=True
wandb_mode="online"

num_batch_pairs=(
  "[15,6,6,3,15] [15,0,0]"  # diversified
  "[45,0,0,0,15] [0,0,0]"  # mix-expert
  "[60,0,0,0,0] [0,0,0]"  # pure-expert
  "[30,0,0,0,15] [15,0,0]"  # mix-unilateral 3x
)

mse_loss_coef_combinations=(
  # 0
  # 0.01
  # 0.1
  1
  # 10
  # 100
)
ref_log_coef_combinations=(
  # 100
  # 10
  1
  # 0.1
  # 0.01
  # 0
)

ff_layer_dim_combinations=(
  64  # best
  # 128
  # 256
)

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
                  LOAD_FROM_PRETRAIN=$rm_load_from_pretrain \
                  NUM_BATCH=$num_batch \
                  NUM_UNILATERAL_BATCH=$num_unilateral_batch \
                  FF_LAYER_DIM=$ff_layer_dim \
                  ENV_NAME=$env_name \
                  PROJECT_PREFIX=$project_prefix \
                  USE_UNILATERAL=$use_unilateral \
                  WANDB_MODE=$wandb_mode \
                  # TOTAL_TIMESTEPS=5e7
            done

            # train imitation model
            CUDA_VISIBLE_DEVICES=$cuda_device python3 train/offline_imitation_learning.py \
                NUM_BATCH=$num_batch \
                NUM_UNILATERAL_BATCH=$num_unilateral_batch \
                REWARD_MODEL_TYPE=$reward_model_type \
                FF_LAYER_DIM=$ff_layer_dim \
                MSE_LOSS_COEF=$mse_loss_coef \
                STUDENT_NETWORK_SAVE_PATH=$imitation_student_save_path \
                VIS_SAVE_PATH='results/vis/IL/reference/' \
                ENV_NAME=$env_name \
                PROJECT_PREFIX=$project_prefix \
                USE_UNILATERAL=$use_unilateral \
                WANDB_MODE=$wandb_mode

            # train mpe on the reward model
            for seed in 0 #1 2 3 4
            do
                for ref_log_coef in "${ref_log_coef_combinations[@]}"; do
                    network_path='results/mpe_reference/agents_rm_mix_mse_'$mse_loss_coef'_seed_'$seed
                    CUDA_VISIBLE_DEVICES=$cuda_device python3 train/vdn_offline_rm.py \
                        env.ENV_NAME=$env_name \
                        alg.MSE_LOSS_COEF=$mse_loss_coef \
                        alg.REWARD_MODEL_TYPE=$reward_model_type \
                        alg.NUM_BATCH=$num_batch \
                        alg.NUM_UNILATERAL_BATCH=$num_unilateral_batch \
                        alg.FF_LAYER_DIM=$ff_layer_dim \
                        alg.REFERENCE_NETWORK_PARAM_PATH=$imitation_student_save_path \
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