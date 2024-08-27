ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero3.yaml \
train.py \
    --task query_response_sft \
    --model_path /nvme1/sxy/hf_resources/hf_models/deepseek-math-7b-base \
    --dataset_path data/gsm8k \
    --prompt_type deepseek-math \
    --bf16 --tf32 \
    --num_train_epochs 3 \
    --gradient_checkpointing true \
    --max_length 4096 \
    --output_dir models/Deepseek-Math-7B-QueryGen-sft \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
