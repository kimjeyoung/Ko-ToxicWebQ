from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed, TrainingArguments, Trainer
import argparse
import os
import torch
from datasets import load_dataset
import numpy as np
from datasets import Dataset, concatenate_datasets
import random

class_dict = {
    "clean": 0,
    "gambling": 1,
    "sexuality": 2,
    "social": 3,
    "copyright": 4,
    "prostitution": 5,
    "others": 6,
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="beomi/kcbert-base")
    parser.add_argument("--cache_dir", type=str, default="/data/cache_dir")
    parser.add_argument("--output_dir", type=str, default="finetuned_kcbert_base")
    parser.add_argument("--data_path", type=str, default="TEAMREBOOTT-AI/Ko-ToxicWebQ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))
    return parser.parse_args()

args = get_args()
set_seed(args.seed)
output_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
os.makedirs(output_dir, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained(args.model_name, cache_dir=args.cache_dir, num_labels=7, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def extract_text(examples):
    text = examples['query'].strip()
    label = class_dict[examples["sub_category"]]
    
    model_inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256
    )
    model_inputs["labels"] = label
    return model_inputs


# dataset loader
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

# preprocessing
train_dataset_final = train_dataset_final.map(extract_text, batched=False)
dev_ds = dev_ds.map(extract_text, batched=False)
train_dataset_final.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dev_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Trainer settings
training_args = TrainingArguments(
    output_dir=output_dir, 
    bf16=True, 
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=5000,
    gradient_accumulation_steps=1, 
    num_train_epochs=1, 
    save_only_model=True,
    save_steps=5000,
    save_total_limit=3,
    report_to=None,
    logging_dir=output_dir,
    do_train=True,
    logging_steps=100,
    per_device_train_batch_size=8,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_final,
    eval_dataset=dev_ds,
)

trainer.train()