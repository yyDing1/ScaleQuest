"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-nodes cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""

from typing import Any, Dict, List

import os
import time
import numpy as np
import ray
import ray.data
import torch
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from transformers import AutoModel, AutoTokenizer


assert Version(ray.__version__) >= Version(
    "2.22.0"
), "Ray version must be at least 2.22.0"


# Create a class to do batch inference.
class LLMPredictor:
    def __init__(
        self,
        model_path,
        tokenizer_path
    ):
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.float16, 
            trust_remote_code=True,
        ).cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        chat_messages = [
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ] for query, response in zip(batch["query"], batch["response"])
        ]
        scores = self.model.get_scores(self.tokenizer, chat_messages)
        return {**batch, "score": scores}


def run_rm_score_distributed(
    ds,
    **kwargs,
):
    num_instances = torch.cuda.device_count()
    batch_size = 16

    resources_kwarg: Dict[str, Any] = {}
    resources_kwarg["num_gpus"] = 1

    # Apply batch inference for all input data.
    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=batch_size,
        fn_constructor_kwargs=kwargs,
        **resources_kwarg,
    )

    return ds
