from dataclasses import dataclass, field
from typing import Optional

import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, \
    what their capacity and features are, and what size model you want to train.
    """

    task: Optional[str] = field(default="query_sft")
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.0)
    model_path: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_path: Optional[str] = field(
        default="openai/gsm8k",
        metadata={
            "help": "",
        },
    )
    prompt_type: Optional[str] = field(
        default="qwen2-math",
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    tf32: Optional[bool] = field(
        default=None,
    )
    num_train_epochs: Optional[float] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )

    max_training_samples: Optional[int] = field(
        default=-1, metadata={"help": "the maximum sample size"}
    )

    max_length: Optional[int] = field(default=4096)
    output_dir: Optional[str] = field(default="./models/sft_model_llama3")


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    save_strategy="epoch",
    eval_strategy="no",
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    remove_unused_columns=True,
    bf16=script_args.bf16,
    tf32=script_args.tf32,
    logging_strategy="steps",
    logging_steps=1,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to="tensorboard",
)


model = AutoModelForCausalLM.from_pretrained(
    script_args.model_path,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
    trust_remote_code=True,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_path, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
print("We set the pad token as the eos token by default....")
# tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length
# tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


dataset = load_dataset(script_args.dataset_path)


def formatting_prompts_func(example):
    if script_args.task == "query_sft": 
        if script_args.prompt_type == "qwen2-math":
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": example["query"].strip()}
            ]
        elif script_args.prompt_type == "deepseek-math":
            messages = [
                {"role": "user", "content": example["query"].strip()}
            ]
        else:
            raise NotImplementedError(
                f"Prompt type {script_args.prompt_type} not implemented."
            )
    elif script_args.task == "query_response_sft":
        if script_args.prompt_type == "qwen2-math":
            query_template = "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query_template.format(input=example["query"].strip())},
                {"role": "assistant", "content": example["response"].strip()},
            ]
        elif script_args.prompt_type == "deepseek-math":
            query_template = "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            messages = [
                {"role": "user", "content": query_template.format(input=example["query"].strip())},
                {"role": "assistant", "content": example["response"].strip()},
            ]
        else:
            raise NotImplementedError(
                f"Prompt type {script_args.prompt_type} not implemented."
            )
    # return {"text": tokenizer.apply_chat_template(messages, tokenize=False).strip()}
    return {"messages": messages}


dataset = dataset.map(formatting_prompts_func, batched=False)
train_dataset = dataset["train"]
eval_dataset = dataset["test"] if "test" in dataset else None
if script_args.max_training_samples > 0:
    train_dataset = train_dataset.select(range(script_args.max_training_samples))


# formatting_prompts_func

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    # formatting_func=,
    max_seq_length=script_args.max_length,
    packing=True,
)

trainer.train()
print("Saving last checkpoint of the model")

trainer.save_model(script_args.output_dir)
