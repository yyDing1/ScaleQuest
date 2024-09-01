import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from datasets import load_dataset
from rm_score import generate_score_parallel


def filter_data_based_on_final_answer(line_data):
    response = line_data["response"]
    has_answer = (
        "boxed" in response
        or "he answer is" in response
        or "final answer is" in response
    )
    return has_answer


if __name__ == "__main__":
    data_dir = "/data/dyy/QueryPreference/automatic_gen/data/deepseek-math-rl_resgen200000x1_temp0.0_topp1.0"
    data_name = data_dir.split("/")[-1]

    rm_path = "/data/dyy/externel_resources/hf_models/ArmoRM-Llama3-8B-v0.1"
    rm_model = rm_path.split("/")[-1]

    save_dir = f"/data/sxy/query_preference/automatic_gen/rm_filter_data/{rm_model}/{data_name}/output.jsonl"

    ds = load_dataset(data_dir, split="train")
    ds = generate_score_parallel(
        ds, model_path=rm_path, tokenizer_path=rm_path, num_gpus=os.getenv("CUDA_VISIBLE_DEVICES").count(",") + 1
    )
    ds.to_json(save_dir)
