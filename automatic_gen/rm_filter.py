import os
from datasets import load_dataset
from rm_score import run_rm_score_distributed
import ray
import ray.data


def filter_data_based_on_final_answer(line_data):
    response = line_data["response"]
    has_answer = "boxed" in response or "he answer is" in response or "final answer is" in response
    return has_answer


data_dir = "/data/dyy/QueryPreference/automatic_gen/data/deepseek-math-rl_resgen200000x1_temp0.0_topp1.0"
save_dir = "/data/dyy/QueryPreference/automatic_gen/rm_filter_data/deepseek-math-rl_resgen200000x1_temp0.0_topp1.0"
rm_path = "/data/dyy/externel_resources/hf_models/internlm2-7b-reward"


dataset = ray.data.read_json(data_dir)
dataset = run_rm_score_distributed(
    dataset,
    model_path=rm_path,
    tokenizer_path=rm_path,
)
dataset.write_json(save_dir)


