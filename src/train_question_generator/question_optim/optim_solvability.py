import json
import random
from datasets import load_dataset
import time

from prompts import createSolvabilityPrompt
from openai_gen import run_openai_inference


datasets = load_dataset("/path/to/qft_generated_questions", split="train")

ds_len = len(datasets)
eval_config = {
    "model": "gpt-4o-mini",
    "max_tokens": 4096,
    "num_generations": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "openai_timeout": 45,
}

def process_batch(batch):
    requests = []
    for line_data in batch:
        instruction = createSolvabilityPrompt(line_data["query"])
        requests.append({
            "query": line_data["query"],
            "query_gen_model": line_data["qry_gen_model"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
        })

    results = run_openai_inference(requests, **eval_config)
    match_parten_list = [
        "#Finally Rewritten Problem#:",
        "**Finally Rewritten Problem**:",
        "**Finally Rewritten Problem:**",
        "Finally Rewritten Problem:",
        "### Finally Rewritten Problem:",
        "### Finally Rewritten Problem",
    ]

    for result in results:
        result["rewritten"] = ""
        for match_partern in match_parten_list:
            if match_partern in result["generation"]:
                result["rewritten"] = result["generation"].split(match_partern)[-1].strip()
                break
    
    return results


def save_results(results, file_path):
    try:
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
    
    existing_data.extend(results)
    
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)


batch_size = ds_len
output_file = 'output/gpt-4o-mini_optimize_qwen2-math-qgen_solvability.json'

for i in range(0, ds_len, batch_size):
    batch = datasets.select(range(i, i+batch_size))
    results = process_batch(batch)
    
    save_results(results, output_file)
    print(f"Processed and saved batch {i // batch_size + 1} / {ds_len // batch_size}")
    
    time.sleep(30)

print("All batches processed and saved.")
