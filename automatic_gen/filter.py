from datasets import load_dataset
import re
from tqdm import tqdm
import ray.data
from transformers import AutoTokenizer
from vllm_gen import run_vllm_inference_distributed


# dataset = load_dataset("/data/dyy/QueryPreference/automatic_gen/new_data/qwen2-math_resgen800000x1_temp0.0_topp1.0", split="train")

# Step 1: Filter Language
data_path = "/data/dyy/QueryPreference/automatic_gen/final_data/qwen2-math-sft-dpo_querygen1000000_temp1.0_topp0.99"
dataset = ray.data.read_json(data_path)

def filter_query(line):
    ord_index = list(map(ord, line["query"]))
    if ord_index and max(ord_index) <= 127 and line["query"] != "":
        return True
    else:
        return False

dataset = dataset.filter(filter_query, concurrency=8)


# Step 2: Filter Solvability
model_path = "/data/dyy/externel_resources/hf_models/Qwen2-Math-7B-Instruct"
stop_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
instruction = """
Please act as a professional math teacher.
Your goal is to determine if the given problem is a valuable math problem. You need to consider two aspects:
1.	The given problem is a math problem.
2.	The given math problem can be solved based on the conditions provided in the problem (You can first try to solve it and then judge its solvability).

Please reason step by step and conclude with either 'Yes' or 'No'.

Given Problem: {problem}
""".strip()

tokenizer = AutoTokenizer.from_pretrained(model_path)
def construct_solvability_check_prompt(line):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction.format(problem=line["query"])}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {**line, "prompt_for_solvability_check": prompt}

dataset = dataset.map(construct_solvability_check_prompt)
dataset = run_vllm_inference_distributed(
    ds=dataset,
    model_path=model_path,
    tokenizer_path=model_path,
    prompt_key="prompt_for_solvability_check",
    generation_key="generation_for_solvability_check",
    max_tokens=2048,
    max_model_len=4096,
    num_generations=1,
    temperature=0.0,
    top_p=1.0,
    stop_tokens=stop_tokens,
    tensor_parallel_size=1,
    swap_space=32,
)

def filter_answer(line):
    return "yes" in line["generation_for_solvability_check"][0].lower()
dataset = dataset.filter(filter_answer)

dataset.write_json("final_data/solvability_check_qwen2_querygen")
