# export CUDA_VISIBLE_DEVICES=6,7

# Query Gen
qry_num=1000
qry_prompt_type="qwen2-math-qgen-sft"
qry_model_path="/data/dyy/QueryPreference/query_sft/models/Qwen2-Math-7B-Instruct-QueryGen-sft"
qry_temp=1.0
qry_top_p=1.0

# Response Gen
res_num_per_query=1
res_prompt_type="qwen2-math"
res_model_path="/data/dyy/externel_resources/hf_models/Qwen2-Math-7B-Instruct"
res_temp=0.0
res_top_p=1.0

output_folder="data/"

python gen.py \
    --qry_gen \
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

