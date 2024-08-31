from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, HfArgumentParser, pipeline
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset as TorchDataset


# 定义函数用于在每个GPU上处理数据
def process_data_on_device(device, sub_dataset, score_model_path, score_tokenizer):
    model = AutoModel.from_pretrained(
        score_model_path, 
        device_map="cuda", 
        torch_dtype=torch.float16, 
        trust_remote_code=True,
    )

    all_data = []
    for line_data in tqdm(sub_dataset):
        chat_messages = [
            {"role": "user", "content": line_data["query"]},
            {"role": "assistant", "content": line_data["response"]},
        ]
        score = model.get_score(score_tokenizer, chat_messages)
        line_data["model_score"] = score
        line_data["score_model"] = score_model_path
        all_data.append(line_data)

    return all_data

# 在8个GPU上并行处理数据
from multiprocessing import Pool

def process_wrapper(args):
    return process_data_on_device(*args)

def generate_score(
    dataset, 
    model_path,
    tokenizer_path,
):
    # 定义设备数量
    num_gpus = torch.cuda.device_count()

    # 将数据集拆分为子集
    sub_datasets = dataset.shard(num_shards=num_gpus, index=0)  # 初始化第一个子集
    sub_datasets = [dataset.shard(num_shards=num_gpus, index=i) for i in range(num_gpus)]

    score_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with Pool(num_gpus) as p:
        results = p.map(process_wrapper, [(i, sub_datasets[i], model_path, score_tokenizer) for i in range(num_gpus)])

    # 合并所有子集的结果
    all_data = [item for sublist in results for item in sublist]
    final_dataset = Dataset.from_list(all_data)
    return final_dataset
