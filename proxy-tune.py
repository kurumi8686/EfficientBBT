import torch
import sys
import importlib
import torch.nn.functional as F
from tqdm import tqdm
from utils import tokenizer

if len(sys.argv) < 2:  # Add Command Params to
    raise ValueError("Please provide a dataset name, e.g. python 7b_pretrain.py CoLA")
dataset_name = sys.argv[1]

# Utilize 'importlib' to dynamically import proper package
utils_module = importlib.import_module("utils")
get_option_probs_from_logits = getattr(utils_module, f"get_option_probs_from_logits_{dataset_name}")
pretrain_7b = torch.load(f"{dataset_name}_7b_pretrain_logits.pt")
pretrain_13b = torch.load(f"{dataset_name}_13b_pretrain_logits.pt")
lora_7b = torch.load(f"{dataset_name}_7b_LoRA_logits.pt")
assert len(pretrain_7b) == len(pretrain_13b) == len(lora_7b), "Logits len different!"

alpha, correct, total = 0.8, 0, 0

for logit_7b_pre, logit_13b_pre, logit_7b_tuned in tqdm(
        zip(pretrain_7b, pretrain_13b, lora_7b), desc="Evaluating", total=len(lora_7b)):
    assert logit_7b_pre["true_label"] == logit_13b_pre["true_label"] == logit_7b_tuned["true_label"], "answer different"
    ensemble_logits = logit_7b_tuned["logits"] + alpha * (logit_13b_pre["logits"] - logit_7b_pre["logits"])
    probabilities = get_option_probs_from_logits(ensemble_logits.float())
    print("Probabilities for choices:", probabilities)
    pred_answer = max(probabilities, key=probabilities.get)
    true_answer = logit_7b_pre["true_label"]
    print(f"Predicted Answer: {pred_answer}, True Answer: {true_answer}")
    if pred_answer == true_answer:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Evaluation completed. Accuracy: {accuracy:.4f} ({correct}/{total})")
with open("review.txt", "a") as f:
    f.write(f"{dataset_name} proxy-tuning accuracy: {accuracy:.4f} ({correct}/{total})\n")
