import torch
import os
import sys
import argparse
import json
import time
import random
import numpy as np
from tqdm import tqdm
import ray
import ray.data
from vllm_gen import run_vllm_inference_distributed


def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Instruction Generation Manager.")

    # Query Generation Parameters
    parser.add_argument("--qry_gen", action="store_true")
    parser.add_argument(
        "--qry_num",
        type=int,
        default=1000,
        help="Total number of prompts to generate. If specified, repeat will be ignored.",
    )
    parser.add_argument("--qry_prompt_type", type=str, default="qwen2-math")
    parser.add_argument(
        "--qry_model_path", type=str, default="Qwen/Qwen2-Math-1.5B-Instruct"
    )
    parser.add_argument(
        "--qry_model_tp",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.",
    )
    # parser.add_argument("--qry_model_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--qry_temperature", type=float, default=1.0)
    parser.add_argument("--qry_top_p", type=float, default=1.0)
    parser.add_argument("--qry_max_tokens", type=int, default=1024)

    # Response Generation Parameters
    parser.add_argument("--res_gen", action="store_true")
    parser.add_argument(
        "--res_num_per_query",
        type=int,
        default=5,
        help="Number of samples to generate for one time.",
    )
    parser.add_argument("--res_prompt_type", type=str, default="qwen2-math")
    parser.add_argument(
        "--res_model_path", type=str, default="Qwen/Qwen2-Math-1.5B-Instruct"
    )
    parser.add_argument(
        "--res_model_tp",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.",
    )
    # parser.add_argument("--res_model_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--res_temperature", type=float, default=0.0)
    parser.add_argument("--res_top_p", type=float, default=1.0)
    parser.add_argument("--res_max_tokens", type=int, default=2048)

    # System Settings
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--swap_space", type=float, default=4)
    parser.add_argument("--output_folder", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser.parse_args()


args = get_args()
print(f"Instruction Generation Manager. Arguments: {args}")  # For logging

if args.qry_gen:
    if "qwen2" in args.qry_prompt_type:
        pre_query_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        stop_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    elif "deepseek" in args.qry_prompt_type:
        pre_query_template = "<｜begin▁of▁sentence｜>User: "
        stop_tokens = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"]
    else:
        raise NotImplementedError(
            f"Query prompt type {args.qry_prompt_type} is not implemented"
        )

    # Generate Query

    dataset = ray.data.from_items(
        [
            {
                "query_id": query_idx,
                "qry_gen_model": args.qry_model_path,
                "prompt_query_gen": pre_query_template,
                "query_gen_temp": args.qry_temperature,
                "qry_top_p": args.qry_top_p,
                "qry_max_tokens": args.qry_max_tokens,
            }
            for query_idx in range(args.qry_num)
        ]
    )
    dataset = run_vllm_inference_distributed(
        ds=dataset,
        model_path=args.qry_model_path,
        tokenizer_path=args.qry_model_path,
        prompt_key="prompt_query_gen",
        generation_key="generation_query_list",
        max_tokens=args.qry_max_tokens,
        max_model_len=args.max_model_len,
        num_generations=1,
        temperature=args.qry_temperature,
        top_p=args.qry_top_p,
        stop_tokens=stop_tokens,
        tensor_parallel_size=args.qry_model_tp,
        enable_prefix_caching=True,
        swap_space=args.swap_space,
    )

    def flatten_batch_and_strip(line_data):
        generation_query_list = line_data.pop("generation_query_list")
        expanded_rows = [
            {**line_data, "query": generation_query.strip()}
            for generation_query in generation_query_list
        ]
        return expanded_rows

    dataset = dataset.flat_map(flatten_batch_and_strip, concurrency=4)
    qry_gen_output_path = os.path.join(
        args.output_folder,
        f"{args.qry_prompt_type}_querygen{args.qry_num}_temp{args.qry_temperature}_topp{args.qry_top_p}",
    )
    dataset.write_json(qry_gen_output_path)

# Generate Response

if args.res_gen:
    time.sleep(30)  # wait for the GPU resources to be released
    qry_gen_output_path = os.path.join(
        args.output_folder,
        f"{args.qry_prompt_type}_querygen{args.qry_num}_temp{args.qry_temperature}_topp{args.qry_top_p}",
    )
    dataset = ray.data.read_json(qry_gen_output_path)

    if "qwen2-math" in args.res_prompt_type:
        res_generation_template = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        stop_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    elif "deepseek-math" in args.res_prompt_type:
        res_generation_template = (
            "<｜begin▁of▁sentence｜>User: {input}\nPlease reason step by step, "
            "and put your final answer within \\boxed{{}}.\n\nAssistant:"
        )
        stop_tokens = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"]
    else:
        raise NotImplementedError(
            f"Response prompt type {args.res_prompt_type} is not implemented"
        )

    def preprocess_response_template(line_data):
        prompt_res_gen = res_generation_template.format(input=line_data["query"])
        line_data.update(
            {
                "res_gen_model": args.res_model_path,
                "prompt_res_gen": prompt_res_gen,
                "res_gen_temp": args.res_temperature,
                "res_top_p": args.res_top_p,
                "res_max_tokens": args.res_max_tokens,
            }
        )
        expanded_rows = [
            {**line_data, "sample_idx": sample_idx}
            for sample_idx in range(args.res_num_per_query)
        ]
        return expanded_rows

    dataset = dataset.flat_map(preprocess_response_template, concurrency=4)
    dataset = run_vllm_inference_distributed(
        ds=dataset,
        model_path=args.res_model_path,
        tokenizer_path=args.res_model_path,
        prompt_key="prompt_res_gen",
        generation_key="response",
        max_tokens=args.res_max_tokens,
        max_model_len=args.max_model_len,
        num_generations=1,
        temperature=args.res_temperature,
        top_p=args.res_top_p,
        stop_tokens=stop_tokens,
        tensor_parallel_size=args.res_model_tp,
        swap_space=args.swap_space,
    )

    def strip_data(line_data):
        line_data["response"] = line_data["response"][0].strip()
        return line_data

    dataset = dataset.map(strip_data, concurrency=4)

    res_gen_output_path = os.path.join(
        args.output_folder,
        f"{args.res_prompt_type}_resgen{args.qry_num}x{args.res_num_per_query}_temp{args.res_temperature}_topp{args.res_top_p}",
    )
    dataset.write_json(res_gen_output_path)
