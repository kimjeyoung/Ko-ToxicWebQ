import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    SFTConfig,
    SFTTrainer,
)
from datasets import Dataset, concatenate_datasets
import numpy as np
import random

system_prompt = '''Classify the toxicity of a given Korean web search query. The classification must be grounded in the cultural and legal context of South Korea, taking the specific nuances into account.
### Definition of toxicity in web queries
- Clean query (0) : The query does not exhibit any toxicity or intent to circumvent content filters.
- Gambling (1) : The query indicates an attempt to access gambling content.
- Explicit sexual content (2) : The query indicates an attempt to access explicit adult content, explicit depictions of sexual activity, erotic literature, adult webtoons, sexual exploitation, and objectification.
- Harmful Social Platform Content (3) : This category includes queries seeking adult or inappropriate content on social media and streaming platforms, such as Twitter, Facebook, and adult live-streaming services.
- Unauthorized (4) :The query indicates an attempt to access unauthorized copyright-infringing content.
- Prostitution (5) : The query indicates an attempt to access prostitution content.
- Other (6) : Toxic queries that do not fall into the above five categories but still exhibit harmful, illegal, or explicitly problematic intent.
'''

MULTI_CLASS_PROMPT_FINETUNED = '''
### Input
- Query: {query}

### Output
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="TEAMREBOOTT-AI/Ko-ToxicWebQ")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache_dir", type=str, default="/data/cache_dir")
    parser.add_argument("--output_dir", type=str, default="sft_qwen_7b_finetuned")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

args = get_args()

os.makedirs(args.output_dir, exist_ok=True)

class_dict = {
    "clean": 0,
    "gambling": 1,
    "sexuality": 2,
    "social": 3,
    "copyright": 4,
    "prostitution": 5,
    "others": 6,
}


CACHE_DIR = args.cache_dir
MODEL_NAME = args.model_name

train_dataset = load_dataset(args.data_path, split="train", token=args.hf_token)

def split_train_dev(dataset, class_key="sub_category", dev_ratio=0.1, seed=42):
    labels = set(dataset[class_key])
    train_list, dev_list = [], []
    for label in labels:
        class_ds = dataset.filter(lambda x: x[class_key] == label)
        idxs = list(range(len(class_ds)))
        random.Random(seed).shuffle(idxs)
        dev_size = int(len(class_ds) * dev_ratio)
        dev_idxs = idxs[:dev_size]
        train_idxs = idxs[dev_size:]
        dev_list.append(class_ds.select(dev_idxs))
        train_list.append(class_ds.select(train_idxs))
    train_ds = concatenate_datasets(train_list)
    dev_ds = concatenate_datasets(dev_list)
    return train_ds, dev_ds

train_ds, dev_ds = split_train_dev(train_dataset, class_key="sub_category", dev_ratio=0.1, seed=args.seed)

# oversampling (train set only)
clean_ds = train_ds.filter(lambda x: x["sub_category"] == "clean")
other_ds = train_ds.filter(lambda x: x["sub_category"] != "clean")
clean_count = len(clean_ds)
oversampled_list = [clean_ds]
for label in set(other_ds["sub_category"]):
    class_ds = other_ds.filter(lambda x: x["sub_category"] == label)
    repeat_factor = int(np.ceil(clean_count / len(class_ds)))
    if len(class_ds) >= clean_count:
        oversampled = class_ds.shuffle(seed=42).select(range(clean_count))
    else:
        oversampled = concatenate_datasets([class_ds] * repeat_factor).shuffle(seed=42).select(range(clean_count))
    oversampled_list.append(oversampled)
# concatenate and shuffle
train_dataset_final = concatenate_datasets(oversampled_list).shuffle(seed=42)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, use_fast=True, trust_remote_code=True)

tokenizer.padding_side = "right"
if "Qwen" in args.model_name:
    stop_token = "<|endofturn|>"
elif "EXAONE" in args.model_name:
    stop_token = "[|endofturn|]"
else:
    raise ValueError(f"Unsupported model: {args.model_name}")

def filter_eval_dataset(dataset):
    dataset = dataset.filter(lambda x: len(x["train_or_test"]) == "train")
    return dataset

def apply_sft_chat_template(examples):
    text = examples['query'].strip()
    target = str(examples['sub_category']).upper()

    prompt = MULTI_CLASS_PROMPT_FINETUNED.format(query=text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target}
    ]
    # apply chat template
    chat_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    # add end of turn token
    if "Qwen" in args.model_name:
        chat_text += "\n<|endofturn|>"
    elif "EXAONE" in args.model_name:
        chat_text += "\n[|endofturn|]"
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    return { 
        "text": chat_text,
    }

# 전처리
train_dataset_final = train_dataset_final.map(apply_sft_chat_template, batched=False)
dev_ds = dev_ds.map(apply_sft_chat_template, batched=False)

train_args = SFTConfig(output_dir=args.output_dir, 
                       bf16=True, 
                       save_strategy="steps",
                       eval_strategy="steps",
                       eval_steps=5000,
                       gradient_accumulation_steps=1, 
                       num_train_epochs=1, 
                       save_only_model=True,
                       save_steps=5000,
                       max_seq_length=512,
                       save_total_limit=3,
                       report_to=None,
                       logging_dir=args.output_dir,
                       do_train=True,
                       logging_steps=100,
                       per_device_train_batch_size=8,
                       load_best_model_at_end=True,
                       metric_for_best_model="eval_loss",
                       greater_is_better=False)

trainer = SFTTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset_final,
    eval_dataset=dev_ds,
)

trainer.train()