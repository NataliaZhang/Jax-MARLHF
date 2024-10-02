#!/bin/bash

cuda_device='6'
steepness=5
reward_model_type="RNN"
rm_load_from_pretrain=False
env_name="MPE_simple_tag_v3"
imitation_student_save_path="results/ILagent/tag/"
project_prefix="test_"
use_unilateral=True
epochs=50000
wandb_mode="online"

num_batch_pairs=(
  "[15,6,6,3,15] [15,0,0]" # diversified
  "[30,0,0,0,15] [15,0,0]"  # mix-unilateral
  "[60,0,0,0,0] [0,0,0]"  # pure-expert
  "[45,0,0,0,15] [0,0,0]"  # mix-expert
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
  # 3
  # 1
  # 0.3
  # 0.1
  # 0.03
  # 0.01
  # 0
)
ff_layer_dim_combinations=(
  64
  # 128
  # 256
  # 512
  # 1024
  # 2048
)
for ff_layer_dim in "${ff_layer_dim_combinations[@]}"; do
  for pair in "${num_batch_pairs[@]}"; do
      for mse_loss_coef in "${mse_loss_coef_combinations[@]}"; do
          IFS=" " read -r num_batch num_unilateral_batch <<< $pair
          echo "Running reward_model for num_batch=$num_batch, num_unilateral_batch=$num_unilateral_batch"

          # train reward model
          CUDA_VISIBLE_DEVICES=$cuda_device python3 train/reward_model.py \
              MSE_LOSS_COEF=$mse_loss_coef \
              STEEPNESS=$steepness \
              REWARD_MODEL_TYPE=$reward_model_type \
              LOAD_FROM_PRETRAIN=$rm_load_from_pretrain \
              FF_LAYER_DIM=$ff_layer_dim \
              NUM_BATCH=$num_batch \
              NUM_UNILATERAL_BATCH=$num_unilateral_batch \
              ENV_NAME=$env_name \
              PROJECT_PREFIX=$project_prefix \
              USE_UNILATERAL=$use_unilateral \
              WANDB_MODE=$wandb_mode \
              # DEBUG=True

          # train imitation model
          CUDA_VISIBLE_DEVICES=$cuda_device python3 train/offline_imitation_learning.py \
              NUM_BATCH=$num_batch \
              NUM_UNILATERAL_BATCH=$num_unilateral_batch \
              REWARD_MODEL_TYPE=$reward_model_type \
              FF_LAYER_DIM=$ff_layer_dim \
              MSE_LOSS_COEF=$mse_loss_coef \
              STUDENT_NETWORK_SAVE_PATH=$imitation_student_save_path \
              VIS_SAVE_PATH='results/vis/IL/tag/' \
              ENV_NAME=$env_name \
              PROJECT_PREFIX=$project_prefix \
              USE_UNILATERAL=$use_unilateral \
              WANDB_MODE=$wandb_mode

          # train mpe on the reward model
          for ref_log_coef in "${ref_log_coef_combinations[@]}";
          do
              for seed in 0 #1 2 3 4  
              do
                network_path='results/mpe_tag/agents_rm_mix_mse_'$mse_loss_coef'_seed_'$seed
                CUDA_VISIBLE_DEVICES=$cuda_device python3 train/vdn_offline_rm.py \
                    env.ENV_NAME=$env_name \
                    alg.MSE_LOSS_COEF=$mse_loss_coef \
                    alg.REWARD_MODEL_TYPE=$reward_model_type \
                    alg.FF_LAYER_DIM=$ff_layer_dim \
                    alg.NUM_BATCH=$num_batch \
                    alg.NUM_UNILATERAL_BATCH=$num_unilateral_batch \
                    alg.REFERENCE_NETWORK_PARAM_PATH=$imitation_student_save_path \
                    PROJECT_PREFIX=$project_prefix \
                    alg.USE_UNILATERAL=$use_unilateral \
                    alg.EPOCHS=$epochs \
                    alg.REF_LOG_COEF=$ref_log_coef \
                    SEED=$seed \
                    WANDB_MODE=$wandb_mode
              done

          done

      done
    done
done