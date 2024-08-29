export CUDA_VISIBLE_DEVICES=0,1,2,3

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero3.yaml \
--main_process_port 29600 \
train.py \
    --task query_sft \
    --model_path /nvme1/dyy/QueryPreference/query_sft/models/Deepseek-Math-7B-QueryGen-sft \
    --dataset_path /nvme1/dyy/QueryPreference/evol_instruct/output/sft_optim_v1 \
    --prompt_type deepseek-math \
    --num_train_epochs 1 \
    --gradient_checkpointing true \
    --max_length 1024 \
    --output_dir models/Deepseek-Math-7B-QueryGen-sft-optim \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \

