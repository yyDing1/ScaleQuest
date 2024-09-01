import os
from datasets import load_dataset


def filter_data(line_data):
    response = line_data["response"]
    has_answer = (
        "boxed" in response
        or "he answer is" in response
        or "final answer is" in response
    )
    return has_answer


### Step1: Filter data without final answer

# data_dir = "/data/dyy/QueryPreference/automatic_gen/data/deepseek-math-rl_resgen120000x1_temp0.0_topp1.0"

# for data_file in os.listdir(data_dir):
#     data_path = os.path.join(data_dir, data_file)
#     raw_data = load_dataset("json", data_files=data_path, split="train")
#     filtered_data = raw_data.filter(filter_data)
#     print(len(filtered_data) / len(raw_data))
#     filtered_data.to_json(data_path)


### Step2: Filter data based on score

data_dir = "/data/sxy/QueryPreference/automatic_gen/rm_scored_data/internlm2-7b-reward/deepseek-math-rl_resgen200000x1_temp0.0_topp1.0/output.jsonl"
data_name = data_dir.split("/")[-2]
model_name = data_dir.split("/")[-3]
score = 2

ds = load_dataset("json", data_files=data_dir, split="train")

ds = ds.filter(lambda x: x["score"] > score)
# ds = ds.sort("score", reverse=True).select(range(90000)).shuffle(seed=42)
ds_count = len(ds)

data_name = data_name.replace("200000", str(ds_count))
output_dir = f"/data/sxy/QueryPreference/automatic_gen/rm_filtered_data/{model_name}/{data_name}_score{score}"

os.makedirs(output_dir, exist_ok=True)
ds.to_json(f"{output_dir}/train.jsonl")
