# export CUDA_VISIBLE_DEVICES=0,1,2,3
# Query Gen
qry_num=100
qry_prompt_type="qwen2-math-sft"
qry_model_path="/path/to/Qwen2-Math-7B-QGen"
qry_temp=1.0
qry_top_p=0.99

# Response Gen
res_num_per_query=5
res_prompt_type="qwen2-math"
res_model_path="Qwen/Qwen2-Math-7B-Instruct"
res_temp=0.7
res_top_p=0.95

output_folder="generation/"

# for query generation
python gen.py \
    --qry_gen \
    --qry_num $qry_num \
    --qry_prompt_type $qry_prompt_type \
    --qry_model_path $qry_model_path \
    --qry_temperature $qry_temp \
    --qry_top_p $qry_top_p \
    --res_num_per_query $res_num_per_query \
    --res_prompt_type $res_prompt_type \
    --res_model_path $res_model_path \
    --res_temperature $res_temp \
    --res_top_p $res_top_p \
    --output_folder $output_folder \
    --swap_space 32

# for response generation
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
