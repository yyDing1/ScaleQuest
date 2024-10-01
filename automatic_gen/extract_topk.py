from datasets import load_dataset, Dataset
from collections import defaultdict
from tqdm import tqdm


dataset = load_dataset("/data/dyy/QueryPreference/automatic_gen/final_data/qwen2-math_resgen600000x5_temp0.7_topp0.95_rm_score", split="train")

qa_pool = defaultdict(list)
for line in tqdm(dataset):
    qa_pool[line["query"]].append({
        "response": line["response"], 
        "rm_score": line["score"],
    })

for query in tqdm(qa_pool):
    qa_pool[query].sort(key=lambda x: x["rm_score"], reverse=True)

final_data = []
for query in qa_pool:
    final_data.append({
        "query": query,
        "response": qa_pool[query][0]["response"]
    })

final_data = Dataset.from_list(final_data)
final_data.to_json("/data/dyy/QueryPreference/automatic_gen/final_data/qwen2-math_resgen600000x5_temp0.7_topp0.95_rm_score_top1/output.json")
