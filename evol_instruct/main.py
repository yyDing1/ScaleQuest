import json
import random
from datasets import load_dataset

from prompt import createHardEvolPrompt
from openai_gen import run_openai_inference



datasets = load_dataset("/data/dyy/QueryPreference/automatic_gen/data/deepseek-math-sft_querygen120000_temp1.0_topp1.0", split="train").shuffle(seed=42).select(range(10000))


eval_config = {
    "model": "gpt-4o-mini",
    "max_tokens": 4096,
    "num_generations": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "openai_timeout": 45,
}

requests = []
for line_data in datasets:
	instruction = createHardEvolPrompt(line_data["query"])
	requests.append({
		"query": line_data["query"],
		"query_gen_model": line_data["qry_gen_model"],
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": instruction},
		]
	})

results = run_openai_inference(requests, **eval_config)
for result in results:
	result["rewritten"] = result["generation"].split("#Finally Rewritten Problem#:")[-1].strip()


with open('output/gen_data-v2-4o-mini.json', 'w') as f:	
	json.dump(results, f, indent=4)



