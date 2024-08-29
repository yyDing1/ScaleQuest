import json
from datasets import Dataset


with open("output/gen_data-v2-4o-mini.json", "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
# dataset.map(lambda x: {"chosen": dataset})
dataset = dataset.filter(lambda x: x["generation"].startswith("Step 1 #Methods List#:"))
dataset = dataset.map(
    lambda x: {"chosen": x["query"], "rejected": x["rewritten"], "query": x["rewritten"]}
).select_columns(["query"])
dataset.to_json("/data/dyy/QueryPreference/evol_instruct/output/sft_optim_v1/train.jsonl")
