import os
import argparse
from prompts import BINARY_PROMPT, MULTI_CLASS_PROMPT
from openai import OpenAI
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

class_dict = {
    "clean": 0,
    "gambling": 1,
    "sexuality": 2,
    "social": 3,
    "copyright": 4,
    "prostitution": 5,
    "others": 6,
}

def calc_binary_perf(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print_text = f"{recall:.4f} & {precision:.4f} & {f1:.4f}"
    return print_text


def calc_multi_class_perf(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    print_text = f"{recall:.4f} & {precision:.4f} & {f1:.4f}"
    return print_text


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="TEAMREBOOTT-AI/Ko-ToxicWebQ")
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-11-20")
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "multi-class"])
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))
    return parser.parse_args()

def get_response(query, prompt):
    input_msg = [{"role": "user", "content": prompt.format(query=query)}]
    response = client.chat.completions.create(
        model=args.model_name,
        messages=input_msg,
        temperature=0.0,
        max_tokens=100,
    )
    return int(response.choices[0].message.content)

args = get_args()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if args.task == "binary":
    prompt = BINARY_PROMPT
elif args.task == "multi-class":
    prompt = MULTI_CLASS_PROMPT


train_dataset = load_dataset(args.data_path, split="train", token=args.hf_token)
test_dataset = load_dataset(args.data_path, split="test", token=args.hf_token)

# merge dataset
dataset = concatenate_datasets([train_dataset, test_dataset])

y_true = []
y_pred = []

for i, data in enumerate(dataset):
    query = data['query']
    label = class_dict[data['sub_category']]

    if args.task == "binary":
        if label > 0:
            label =1

    y_true.append(label)
    try:
        y_pred.append(get_response(query, prompt))
    except Exception as e:
        try:
            y_pred.append(get_response(query, prompt))
        except Exception as e:
            y_pred.append(0)

    print(f"Query: {query}")
    print(f"True label: {label}")
    print(f"Predicted label: {y_pred[-1]}")
    print("--------------------------------")


if args.task == "binary":
    scores = calc_binary_perf(y_true, y_pred)
    print(scores)
elif args.task == "multi-class":
    scores = calc_multi_class_perf(y_true, y_pred)
    print(scores)