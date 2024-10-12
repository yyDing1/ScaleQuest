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

from vllm import LLM, SamplingParams

assert Version(ray.__version__) >= Version(
    "2.22.0"
), "Ray version must be at least 2.22.0"


# Create a class to do batch inference.
class LLMPredictor:
    def __init__(
        self,
        model_path,
        tokenizer_path,
        prompt_key="prompt",
        generation_key="generation",
        max_tokens=2048,
        max_model_len=4096,
        num_generations=1,
        temperature=0.0,
        top_p=1.0,
        stop_tokens=None,
        stop_token_ids=None,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        swap_space=16,
    ):
        seed = int(time.time() * 1e6) % int(1e9)
        # Create an LLM.
        self.prompt_key = prompt_key
        self.generation_key = generation_key
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=True,
            swap_space=swap_space,
            gpu_memory_utilization=0.95,
            seed=seed,
        )
        self.sampling_params = SamplingParams(
            n=num_generations,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_tokens,
            stop_token_ids=stop_token_ids,
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch[self.prompt_key], self.sampling_params)
        generated_text: List[str] = []
        for output in outputs:
            generated_text.append([o.text for o in output.outputs])
        return {**batch, self.generation_key: generated_text}


def run_vllm_inference_distributed(
    ds,
    **kwargs,
):
    tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)

    # Guarentee the compute resources is available
    if torch.cuda.device_count() < tensor_parallel_size:
        raise MemoryError(
            "Insufficient GPUs: tensor_parallel_size ({}) < available gpus ({})".format(
                tensor_parallel_size, torch.cuda.device_count()
            )
        )

    # Set number of instances. Each instance will use tensor_parallel_size GPUs.
    num_instances = torch.cuda.device_count() // tensor_parallel_size
    print("Launch {} instances for vllm inference.".format(num_instances))

    # For tensor_parallel_size > 1, we need to create placement groups for vLLM
    # to use. Every actor has to have its own placement group.
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}] * tensor_parallel_size, strategy="STRICT_PACK"
        )
        return dict(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                pg, placement_group_capture_child_tasks=True
            )
        )

    resources_kwarg: Dict[str, Any] = {}
    if tensor_parallel_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    batch_size = min(ds.count() // num_instances + 1, 10000)
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
