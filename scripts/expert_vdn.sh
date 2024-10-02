cuda_device='0'
wandb_entity="comm_marl"    # change it to your wandb entity

echo "Running vdn training to collect models"

# spread 
CUDA_VISIBLE_DEVICES=$cuda_device python train/expert_vdn.py \
    env.ENV_NAME="MPE_simple_spread_v3" \
    ENTITY=$wandb_entity \

# reference
CUDA_VISIBLE_DEVICES=$cuda_device python train/expert_vdn.py \
    env.ENV_NAME="MPE_simple_reference_v3" \
    ENTITY=$wandb_entity \

# tag
CUDA_VISIBLE_DEVICES=$cuda_device python train/expert_vdn.py \
    env.ENV_NAME="MPE_simple_tag_v3" \
    ENTITY=$wandb_entity \