cuda_device='1'
# mse_loss_coef=0
steepness=5
# reward_model_type="FF"
ff_layer_dim=64
# env_name="overcooked"
# imitation_student_save_path="results/ILagent/overcooked/unilateral/"
project_prefix="11222124"
use_unilateral=True
epochs=10000
# wandb_mode="disabled"
wandb_mode="online"

num_batch_pairs=(
  "[15,6,6,3,15] [15,0,0]" # diversified 3x
  "[30,0,0,0,15] [15,0,0]"  # mix-unilateral
  # "[60,0,0,0,0] [0,0,0]"  # pure-expert
  # "[45,0,0,0,15] [0,0,0]"  # mix-expert
)

# Enable the following lines to run the script for testing MSE loss coefficient
mse_loss_coef_combinations=(
  # 0
  # 0.001
  # 0.01
  0.1
  1
  # 10
  # 100
)

# Enable the following lines to run the script for testing reference log coefficient
ref_log_coef_combinations=(
  # 100
  # 30
  10
  # 1
  # 0.1
  # 0.01
  # 0
)
for pair in "${num_batch_pairs[@]}"; do
    for mse_loss_coef in "${mse_loss_coef_combinations[@]}"; do
        IFS=" " read -r num_batch num_unilateral_batch <<< $pair
        echo "Running reward_model for num_batch=$num_batch, num_unilateral_batch=$num_unilateral_batch"

        # train reward mode
        CUDA_VISIBLE_DEVICES=$cuda_device python RLHF/reward_model_overcooked.py \
            MSE_LOSS_COEF=$mse_loss_coef \
            STEEPNESS=$steepness \
            FF_LAYER_DIM=$ff_layer_dim \
            NUM_BATCH=$num_batch \
            NUM_UNILATERAL_BATCH=$num_unilateral_batch \
            PROJECT_PREFIX=$project_prefix \
            USE_UNILATERAL=$use_unilateral \
            WANDB_MODE=$wandb_mode 
            # REWARD_MODEL_TYPE=$reward_model_type \
            # ENV_NAME=$env_name \

        # train imitation model
        CUDA_VISIBLE_DEVICES=$cuda_device python RLHF/offline_imitation_learning_overcooked.py \
            NUM_BATCH=$num_batch \
            NUM_UNILATERAL_BATCH=$num_unilateral_batch \
            PROJECT_PREFIX=$project_prefix \
            USE_UNILATERAL=$use_unilateral \
            WANDB_MODE=$wandb_mode
            # WANDB_MODE="disabled"
            # STUDENT_NETWORK_SAVE_PATH=$imitation_student_save_path \
            # ENV_NAME=$env_name \

        # train mpe on the reward model
        for seed in {0..5};
        do
            for ref_log_coef in "${ref_log_coef_combinations[@]}"; do
              
              network_path='results/overcooked/agents_rm_mix_mse_'$mse_loss_coef'_seed_'$seed
              CUDA_VISIBLE_DEVICES=$cuda_device python RLHF/bcq_offline_rm_overcooked.py \
                  alg.MSE_LOSS_COEF=$mse_loss_coef \
                  alg.FF_LAYER_DIM=$ff_layer_dim \
                  alg.NUM_BATCH=$num_batch \
                  alg.NUM_UNILATERAL_BATCH=$num_unilateral_batch \
                  PROJECT_PREFIX=$project_prefix \
                  alg.USE_UNILATERAL=$use_unilateral \
                  alg.EPOCHS=$epochs \
                  alg.REF_LOG_COEF=$ref_log_coef \
                  SEED=$seed \
                  WANDB_MODE=$wandb_mode
                  # alg.REFERENCE_NETWORK_PARAM_PATH=$imitation_student_save_path \
                  # alg.REWARD_MODEL_TYPE=$reward_model_type \
                  # env.ENV_NAME=$env_name \

            done
        done

    done
done