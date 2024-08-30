export CUDA_VISIBLE_DEVICES=4,5,6,7

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero3.yaml \
--main_process_port 29051 \
train.py \
    --model_path /data/dyy/QueryPreference/query_sft/models/Deepseek-Math-7B-QueryGen-sft \
    --ref_model /data/dyy/QueryPreference/query_sft/models/Deepseek-Math-7B-QueryGen-sft \
    --dataset_path /data/dyy/QueryPreference/evol_instruct/output/dpo_data_v1 \
    --prompt_type deepseek-math \
    --run_name deepseek-math-querygen-sft-dpo \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --loss_type sigmoid \
    --warmup_steps 20 \
    --num_train_epochs 2 \
    --gradient_checkpointing true \
    --max_length 2048 \
    --output_dir models/Deepseek-Math-7B-QueryGen-sft-dpo \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \

