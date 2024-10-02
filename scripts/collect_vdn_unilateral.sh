cuda_device='7'

# reference
expert_paths=(104 149)
idiotic_agent_paths=(149 199)
# expert_paths=(104)
for expert_path in "${expert_paths[@]}"; do
    for idiotic_agent_path in "${idiotic_agent_paths[@]}"; do
        echo "Running vdn_collect for expert_path=$expert_path"
        CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
        alg.EXPERT_PARAM_PATH="model/reference/vdn_ps-${expert_path}.safetensors" \
        alg.IDIOTIC_AGENT_PATH="model/reference/vdn_ps-${idiotic_agent_path}.safetensors" \
        alg.TRAJ_BATCH_PATH="data/reference/vdn${expert_path}|${idiotic_agent_path}" \
        alg.TOTAL_TIMESTEPS=3000000 \
        env.ENV_NAME="MPE_simple_reference_v3" \
        alg.IDIOTIC_AGENT=True
    done
done

# spread na3
expert_paths=(149)
idiotic_agent_paths=(198 297)
# expert_paths=(149)
# expert_paths=(104)
for expert_path in "${expert_paths[@]}"; do
    for idiotic_agent_path in "${idiotic_agent_paths[@]}"; do
        echo "Running vdn_collect for expert_path=$expert_path"
        CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
        alg.EXPERT_PARAM_PATH="model/spread/vdn_ps-${expert_path}.safetensors" \
        alg.IDIOTIC_AGENT_PATH="model/spread/vdn_ps-${idiotic_agent_path}.safetensors" \
        alg.TRAJ_BATCH_PATH="data/spread/vdn${expert_path}|${idiotic_agent_path}" \
        alg.TOTAL_TIMESTEPS=3000000 \
        env.ENV_NAME="MPE_simple_spread_v3" \
        alg.IDIOTIC_AGENT=True
    done
done

# spread na4
expert_paths=(99) 
idiotic_agent_paths=(136 150 175)
for expert_path in "${expert_paths[@]}"; do
    for idiotic_agent_path in "${idiotic_agent_paths[@]}"; do
        echo "Running vdn_collect for expert_path=$expert_path"
        CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
        alg.EXPERT_PARAM_PATH="model/spread/na4/vdn_ps-${expert_path}.safetensors" \
        alg.IDIOTIC_AGENT_PATH="model/spread/na4/vdn_ps-${idiotic_agent_path}.safetensors" \
        alg.TRAJ_BATCH_PATH="data/spread/na4/vdn${expert_path}|${idiotic_agent_path}" \
        alg.TOTAL_TIMESTEPS=3000000 \
        env.ENV_NAME="MPE_simple_spread_v3" \
        +env.ENV_KWARGS.num_agents=4 \
        +env.ENV_KWARGS.num_landmarks=5 \
        alg.IDIOTIC_AGENT=True
    done
done

# spread na5
expert_paths=(99)
idiotic_agent_paths=(153 165 184)
for expert_path in "${expert_paths[@]}"; do
    for idiotic_agent_path in "${idiotic_agent_paths[@]}"; do
        echo "Running vdn_collect for expert_path=$expert_path"
        CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
        alg.EXPERT_PARAM_PATH="model/spread/na5/vdn_ps-${expert_path}.safetensors" \
        alg.IDIOTIC_AGENT_PATH="model/spread/na5/vdn_ps-${idiotic_agent_path}.safetensors" \
        alg.TRAJ_BATCH_PATH="data/spread/na5/vdn${expert_path}|${idiotic_agent_path}" \
        alg.TOTAL_TIMESTEPS=3000000 \
        env.ENV_NAME="MPE_simple_spread_v3" \
        +env.ENV_KWARGS.num_agents=5 \
        +env.ENV_KWARGS.num_landmarks=5 \
        alg.IDIOTIC_AGENT=True
    done
done

# spread na6
expert_paths=(137)
idiotic_agent_paths=(176 190 200)
for expert_path in "${expert_paths[@]}"; do
    for idiotic_agent_path in "${idiotic_agent_paths[@]}"; do
        echo "Running vdn_collect for expert_path=$expert_path"
        CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
        alg.EXPERT_PARAM_PATH="model/spread/na6/vdn_ps-${expert_path}.safetensors" \
        alg.IDIOTIC_AGENT_PATH="model/spread/na6/vdn_ps-${idiotic_agent_path}.safetensors" \
        alg.TRAJ_BATCH_PATH="data/spread/na6/vdn${expert_path}|${idiotic_agent_path}" \
        alg.TOTAL_TIMESTEPS=3000000 \
        env.ENV_NAME="MPE_simple_spread_v3" \
        +env.ENV_KWARGS.num_agents=6 \
        +env.ENV_KWARGS.num_landmarks=5 \
        alg.IDIOTIC_AGENT=True
    done
done

# spread na7
expert_paths=(159)
idiotic_agent_paths=(177 195 216)
for expert_path in "${expert_paths[@]}"; do
    for idiotic_agent_path in "${idiotic_agent_paths[@]}"; do
        echo "Running vdn_collect for expert_path=$expert_path"
        CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
        alg.EXPERT_PARAM_PATH="model/spread/na7/vdn_ps-${expert_path}.safetensors" \
        alg.IDIOTIC_AGENT_PATH="model/spread/na7/vdn_ps-${idiotic_agent_path}.safetensors" \
        alg.TRAJ_BATCH_PATH="data/spread/na7/vdn${expert_path}|${idiotic_agent_path}" \
        alg.TOTAL_TIMESTEPS=3000000 \
        env.ENV_NAME="MPE_simple_spread_v3" \
        +env.ENV_KWARGS.num_agents=7 \
        +env.ENV_KWARGS.num_landmarks=5 \
        alg.IDIOTIC_AGENT=True
    done
done

# tag
expert_paths=(621)
idiotic_agent_paths=(480 271 187)
# expert_paths=(149)
# expert_paths=(104)
for expert_path in "${expert_paths[@]}"; do
    for idiotic_agent_path in "${idiotic_agent_paths[@]}"; do
        echo "Running vdn_collect for expert_path=$expert_path"
        CUDA_VISIBLE_DEVICES=$cuda_device python train/vdn_collect.py \
        alg.EXPERT_PARAM_PATH="model/tag/vdn_ps${expert_path}.safetensors" \
        alg.IDIOTIC_AGENT_PATH="model/tag/vdn_ps${idiotic_agent_path}.safetensors" \
        alg.TRAJ_BATCH_PATH="data/tag/vdn${expert_path}|${idiotic_agent_path}" \
        alg.TOTAL_TIMESTEPS=3000000 \
        env.ENV_NAME="MPE_simple_tag_v3" \
        alg.IDIOTIC_AGENT=True
    done
done
