ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero0.yaml \
train.py \
    --model_path /data/dyy/externel_resources/hf_models/Qwen2-Math-7B-Instruct \
    --dataset_path data/gsm8k_math_15k \
    --prompt_type qwen2-math \
    --num_train_epochs 1 \
    --gradient_checkpointing true \
    --max_length 1024 \
    --output_dir models/Qwen2-Math-7B-Instruct-QueryGen-sft-nosys \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \

