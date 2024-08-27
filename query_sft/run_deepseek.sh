ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero3.yaml \
train.py \
    --task query_sft \
    --model_path /nvme1/sxy/hf_resources/hf_models/deepseek-math-7b-rl \
    --dataset_path data/gsm8k \
    --prompt_type deepseek-math \
    --num_train_epochs 1 \
    --gradient_checkpointing true \
    --max_length 1024 \
    --output_dir models/Deepseek-Math-7B-QueryGen-sft-gsm8k \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \

