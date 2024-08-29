export CUDA_VISIBLE_DEVICES=2,3

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero3.yaml \
--main_process_port 29052 \
train.py \
    --task query_response_sft \
    --model_path /data/dyy/externel_resources/hf_models/deepseek-math-7b-base \
    --dataset_path /data/dyy/QueryPreference/automatic_gen/data/deepseek-math-rl_resgen120000x1_temp0.0_topp1.0 \
    --prompt_type deepseek-math \
    --bf16 true --tf32 true \
    --num_train_epochs 3 \
    --gradient_checkpointing true \
    --max_length 4096 \
    --output_dir models/Deepseek-Math-7B-QAGen-sft-120K \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \

