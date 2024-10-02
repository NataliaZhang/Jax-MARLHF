cuda_device='6'

# reference
expert_paths=(104 149 199 242 298)
for expert_path in "${expert_paths[@]}"; do
    echo "Running vdn_collect for expert_path=$expert_path"
    CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
    alg.EXPERT_PARAM_PATH="model/reference/vdn_ps-${expert_path}.safetensors" \
    alg.TRAJ_BATCH_PATH="data/reference/vdn${expert_path}" \
    env.ENV_NAME="MPE_simple_reference_v3" \
    alg.IDIOTIC_AGENT=False \
    alg.TOTAL_TIMESTEPS=3000000
done

# tag
expert_paths=(621 525 392 271 187)
for expert_path in "${expert_paths[@]}"; do
    echo "Running vdn_collect for expert_path=$expert_path"
    CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
    alg.EXPERT_PARAM_PATH="model/tag/vdn_ps${expert_path}.safetensors" \
    alg.TRAJ_BATCH_PATH="data/tag/vdn${expert_path}" \
    env.ENV_NAME="MPE_simple_tag_v3" \
    alg.IDIOTIC_AGENT=False \
    alg.TOTAL_TIMESTEPS=3000000
done

# spread na3
expert_paths=(149 198 297 348 387)
# expert_paths=(149)
for expert_path in "${expert_paths[@]}"; do
    echo "Running vdn_collect for expert_path=$expert_path"
    CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
    alg.EXPERT_PARAM_PATH="model/spread/vdn_ps-${expert_path}.safetensors" \
    alg.TRAJ_BATCH_PATH="data/spread/vdn${expert_path}" \
    env.ENV_NAME="MPE_simple_spread_v3" \
    alg.IDIOTIC_AGENT=False \
    alg.TOTAL_TIMESTEPS=3000000
done

# # spread na4
# expert_paths=(99 118 136 150 175)
# for expert_path in "${expert_paths[@]}"; do
#     echo "Running vdn_collect for expert_path=$expert_path"
#     CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
#     alg.EXPERT_PARAM_PATH="model/spread/na4/vdn_ps-${expert_path}.safetensors" \
#     alg.TRAJ_BATCH_PATH="data/spread/na4/vdn${expert_path}" \
#     alg.TOTAL_TIMESTEPS=3000000 \
#     env.ENV_NAME="MPE_simple_spread_v3" \
#     +env.ENV_KWARGS.num_agents=4 \
#     +env.ENV_KWARGS.num_landmarks=5 \
#     alg.IDIOTIC_AGENT=False
# done

# # spread na5
# expert_paths=(99 117 153 165 184)
# for expert_path in "${expert_paths[@]}"; do
#     echo "Running vdn_collect for expert_path=$expert_path"
#     CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
#     alg.EXPERT_PARAM_PATH="model/vdn_expert/na5/vdn_ps-${expert_path}.safetensors" \
#     alg.TRAJ_BATCH_PATH="data/spread/na5/vdn${expert_path}" \
#     alg.TOTAL_TIMESTEPS=3000000 \
#     env.ENV_NAME="MPE_simple_spread_v3" \
#     +env.ENV_KWARGS.num_agents=5 \
#     +env.ENV_KWARGS.num_landmarks=5 \
#     alg.IDIOTIC_AGENT=False
# done

# # spread na6
# expert_paths=(137 157 176 190 200)
# for expert_path in "${expert_paths[@]}"; do
#     echo "Running vdn_collect for expert_path=$expert_path"
#     CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
#     alg.EXPERT_PARAM_PATH="model/vdn_expert/na6/vdn_ps-${expert_path}.safetensors" \
#     alg.TRAJ_BATCH_PATH="data/spread/na6/vdn${expert_path}" \
#     alg.TOTAL_TIMESTEPS=3000000 \
#     env.ENV_NAME="MPE_simple_spread_v3" \
#     +env.ENV_KWARGS.num_agents=6 \
#     +env.ENV_KWARGS.num_landmarks=5 \
#     alg.IDIOTIC_AGENT=False
# done

# # spread na7
# expert_paths=(159 169 177 195 216)
# for expert_path in "${expert_paths[@]}"; do
#     echo "Running vdn_collect for expert_path=$expert_path"
#     CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
#     alg.EXPERT_PARAM_PATH="model/vdn_expert/na7/vdn_ps-${expert_path}.safetensors" \
#     alg.TRAJ_BATCH_PATH="data/spread/na7/vdn${expert_path}" \
#     alg.TOTAL_TIMESTEPS=3000000 \
#     env.ENV_NAME="MPE_simple_spread_v3" \
#     +env.ENV_KWARGS.num_agents=7 \
#     +env.ENV_KWARGS.num_landmarks=5 \
#     alg.IDIOTIC_AGENT=False
# done

