import os
from datasets import load_dataset
from compute_reward_score import generate_score


def filter_data_based_on_final_answer(line_data):
    response = line_data["response"]
    has_answer = "boxed" in response or "he answer is" in response or "final answer is" in response
    return has_answer


data_dir = "/nvme1/dyy/QueryPreference/automatic_gen/data/deepseek-math-rl_resgen200000x1_temp0.0_topp1.0"
rm_path = "/nvme1/dyy/externel_resources/hf_models/FsfairX-Gemma2-RM-v0.1"

for data_file in os.listdir(data_dir):
    data_path = os.path.join(data_dir, data_file)
    print(f"Processing {data_path}")
    raw_data = load_dataset("json", data_files=data_path, split="train").select(range(100))
    processed_data = generate_score(raw_data, model_path=rm_path, tokenizer_path=rm_path)
    import pdb; pdb.set_trace()
    # filtered_data = raw_data.filter(filter_data_based_on_final_answer)
    # print(len(filtered_data) / len(raw_data))
    # filtered_data.to_json(data_path)

