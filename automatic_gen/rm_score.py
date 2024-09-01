import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.multiprocessing as mp
from typing import Dict, List
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


class ArmoRMPipeline:
    def __init__(
        self,
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        truncation=True,
        trust_remote_code=False,
        max_length=4096,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}


def process_data_on_device(dataset, score_model_path, score_tokenizer, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if "internlm2-7b-reward" in score_model_path:
        model = AutoModel.from_pretrained(
            score_model_path,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    elif "ArmoRM-Llama3-8B-v0.1" in score_model_path:
        model = ArmoRMPipeline(
            model_id=score_model_path,
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
        score = model(chat_messages) if "ArmoRM" in score_model_path else model.get_score(score_tokenizer, chat_messages)
        line_data["score"] = score["score"]
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
    data_dir = "/data/dyy/QueryPreference/automatic_gen/data/deepseek-math-rl_resgen1000x1_temp0.0_topp1.0"
    data_name = data_dir.split("/")[-1]

    rm_path = "/data/dyy/externel_resources/hf_models/ArmoRM-Llama3-8B-v0.1"
    rm_model = rm_path.split("/")[-1]

    save_dir = f"/data/sxy/query_preference/automatic_gen/rm_scored_data/{rm_model}/{data_name}/output.jsonl"

    batch_size = 8
    ds = load_dataset(data_dir, split="train")
    ds = generate_score_parallel(
        ds,
        model_path=rm_path,
        tokenizer_path=rm_path,
        num_gpus=os.getenv("CUDA_VISIBLE_DEVICES").count(",") + 1,
        batch_size=batch_size,
    )
    ds.to_json(save_dir)