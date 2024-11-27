cuda_device='6'
expert=(240 200 160 20)
idotic=(200 160 20)
# collect expert and rookie
for expert_path in "${expert[@]}"; do
    for idiotic_agent_path in "${idotic[@]}"; do
        echo "Running collect for expert=$expert_path"
        CUDA_VISIBLE_DEVICES=$cuda_device python train/ippo_ff_overcooked_collect.py \
        EXPERT_PARAM_PATH="model/overcooked/actorcritic_model_${expert_path}.0.pkl" \
        DATA_SAVE_PATH="data/overcooked/${expert_path}"
    done
done
# collect trivial
CUDA_VISIBLE_DEVICES=$cuda_device python ippo_ff_overcooked_random_collect.py
# collect unilateral
for idotic_path in "${idotic[@]}"; do
    echo "Running unilateral collect for idotic=$idotic_path"
    CUDA_VISIBLE_DEVICES=$cuda_device python train/ippo_ff_overcooked_collect_uni.py \
    EXPERT_PARAM_PATH="model/overcooked/actorcritic_model_240.0.pkl" \
    DATA_SAVE_PATH="data/overcooked/240|${idotic_path}" \
    IDIOT_PARAM_PATH="model/overcooked/actorcritic_model_${idotic_path}.0.pkl"
done