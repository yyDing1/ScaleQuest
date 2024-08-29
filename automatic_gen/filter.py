import os
from datasets import load_dataset


def filter_data(line_data):
    response = line_data["response"]
    has_answer = "boxed" in response or "he answer is" in response or "final answer is" in response
    return has_answer

data_dir = "/nvme1/dyy/QueryPreference/automatic_gen/data/deepseek-math-rl_resgen1000x1_temp0.0_topp1.0"

for data_file in os.listdir(data_dir):
    data_path = os.path.join(data_dir, data_file)
    raw_data = load_dataset("json", data_files=data_path, split="train")
    filtered_data = raw_data.filter(filter_data)
    print(len(filtered_data) / len(raw_data))
    filtered_data.to_json(data_path)
