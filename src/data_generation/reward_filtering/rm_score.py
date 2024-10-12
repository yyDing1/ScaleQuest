import os
import torch
import torch.multiprocessing as mp
from typing import Dict, List
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def process_data_on_device(dataset, score_model_path, score_tokenizer, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if "internlm2-7b-reward" in score_model_path.lower():
        model = AutoModel.from_pretrained(
            score_model_path,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        raise NotImplementedError(f"Model {score_model_path} not supported")

    all_data = []
    for line_data in tqdm(dataset, desc=f"GPU {gpu_id}"):
        chat_messages = [
            {"role": "user", "content": line_data["query"]},
            {"role": "assistant", "content": line_data["response"]},
        ]
        score = model.get_score(score_tokenizer, chat_messages)
        line_data["score"] = score
        line_data["reward_model"] = score_model_path
        all_data.append(line_data)

    return all_data



def generate_score_parallel(dataset, model_path, tokenizer_path, num_gpus=8):
    score_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )

    dataset_splits = [
        dataset.shard(num_shards=num_gpus, index=i) for i in range(num_gpus)
    ]

    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=num_gpus)

    results = pool.starmap(
        process_data_on_device,
        [
            (split, model_path, score_tokenizer, i)
            for i, split in enumerate(dataset_splits)
        ],
    )

    all_results = [item for sublist in results for item in sublist]

    final_dataset = Dataset.from_list(all_results)
    return final_dataset


if __name__ == "__main__":
    data_dir = "/path/to/data_generation/generation/qwen2-math_resgen600000x5_temp0.7_topp0.95"
    rm_path = "internlm/internlm2-7b-reward"

    save_dir = f"/path/to/data_generation/generation/qwen2-math_resgen600000x5_temp0.7_topp0.95_rm_score/output.jsonl"
    ds = load_dataset(data_dir, split="train")
    ds = generate_score_parallel(
        ds,
        model_path=rm_path,
        tokenizer_path=rm_path,
        num_gpus=os.getenv("CUDA_VISIBLE_DEVICES").count(",") + 1,
    )
    ds.to_json(save_dir)
