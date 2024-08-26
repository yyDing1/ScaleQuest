# QueryPreference

```bash
conda create -n querypref python=3.11
pip install -r requirements.txt
cd dart-math && pip install -e .
pip install flash-attn --no-build-isolation
```

## Step 1: 数据生成

```bash
cd automatic_gen && bash run.sh
```

需要设置一些参数：
```bash
# Query Gen
qry_num=1000
qry_prompt_type="deepseek-math"
qry_model_path="/path/to/QueryPreference/query_sft/models/Deepseek-Math-7B-QueryGen-sft"
qry_temp=1.0
qry_top_p=1.0

# Response Gen
res_num_per_query=1
res_prompt_type="deepseek-math"
res_model_path="/path/to/hf_models/deepseek-math-7b-rl"
res_temp=0.0
res_top_p=1.0
```

## Step 2: 用生成的数据微调

```bash
cd dart-math && bash scripts/train-single-node.sh
```

需要修改的参数：

```bash
# 需要改的参数
# data_path: 刚才生成的 dataset 文件夹
# model_path: 需要微调的 base 模型
# output_dir: 然后改一下输出的路径
data_path=""
query_field="query"
resp_field="response"
model_path="/path/to/hf_models/deepseek-math-7b-base/"
lr="5e-5"
bs=64
n_grad_acc_steps=8
n_epochs=3
gpu_ids="0,1,2,3,4,5,6,7"
output_dir="outputs/Deepseek-MathGen-Sft-Dpo-140K"
```

