export CUDA_VISIBLE_DEVICES=6,7
# Query Gen
qry_num=1000
qry_prompt_type="deepseek-math-sft-dpo"
qry_model_path="/nvme1/dyy/QueryPreference/query_dpo/models/deepseek-math-sft-dpo"
qry_temp=1.0
qry_top_p=1.0

# Response Gen
res_num_per_query=1
res_prompt_type="deepseek-math-rl"
res_model_path="/nvme1/sxy/hf_resources/hf_models/deepseek-math-7b-rl/"
res_temp=0.0
res_top_p=1.0

output_folder="data/"

python gen.py \
    --qry_num $qry_num \
    --qry_prompt_type $qry_prompt_type \
    --qry_model_path $qry_model_path \
    --qry_temperature $qry_temp \
    --qry_top_p $qry_top_p \
    --res_gen \
    --res_num_per_query $res_num_per_query \
    --res_prompt_type $res_prompt_type \
    --res_model_path $res_model_path \
    --res_temperature $res_temp \
    --res_top_p $res_top_p \
    --output_folder $output_folder \
    --swap_space 32

