export CUDA_VISIBLE_DEVICES=0,1,2,3

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero3.yaml \
--main_process_port 29051 \
train.py \
    --model_path /nvme1/dyy/QueryPreference/query_sft/models/Deepseek-Math-7B-QueryGen-sft \
    --ref_model /nvme1/dyy/QueryPreference/query_sft/models/Deepseek-Math-7B-QueryGen-sft \
    --dataset_path /nvme1/dyy/QueryPreference/evol_instruct/output/dpo_data_v1 \
    --prompt_type deepseek-math \
    --run_name deepseek-math-querygen-sft-dpo \
    --learning_rate 5e-7 \
    --loss_type sigmoid \
    --max_steps 1200 \
    --num_train_epochs 1 \
    --gradient_checkpointing true \
    --max_length 2048 \
    --output_dir models/Deepseek-Math-7B-QueryGen-sft-dpo \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \

