from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset as TorchDataset


def process_data_on_device(device, sub_dataset, score_model_path, score_tokenizer):
    score_pipe = pipeline(
        "sentiment-analysis",
        model=score_model_path,
        device=device,
        tokenizer=score_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        truncation=True
    )
    
    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1,
    }

    def get_reward(test_texts):
        pipe_outputs = score_pipe(test_texts, **pipe_kwargs)
        rewards = [output[0]["score"] for output in pipe_outputs]
        return rewards

    all_data = []
    for line_data in tqdm(sub_dataset):
        score = get_reward(line_data["query"])
        line_data["model_score"] = score
        line_data["score_model"] = score_model_path
        all_data.append(line_data)
    
    return all_data

from multiprocessing import Pool

def process_wrapper(args):
    return process_data_on_device(*args)

def generate_score(
    dataset, 
    model_path="/path/to/difficulty_score_model",
    tokenizer_path="/path/to/difficulty_score_model"
):
    num_gpus = torch.cuda.device_count()

    sub_datasets = [dataset.shard(num_shards=num_gpus, index=i) for i in range(num_gpus)]

    score_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with Pool(num_gpus) as p:
        results = p.map(process_wrapper, [(i, sub_datasets[i], model_path, score_tokenizer) for i in range(num_gpus)])

    all_data = [item for sublist in results for item in sublist]
    final_dataset = Dataset.from_list(all_data)
    return final_dataset

    