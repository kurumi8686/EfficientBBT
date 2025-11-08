import torch
import sys
import importlib
from tqdm import tqdm
import torch.nn.functional as F
from peft import PeftModel
from utils import tokenizer, MODEL_13b

if len(sys.argv) < 3:  # Add Command Params to
    raise ValueError("Please provide a dataset name, e.g. python 7b_pretrain.py CoLA")
dataset_name = sys.argv[1]
method = sys.argv[2]  # CPT, GP_random, GP_filter

# Utilize 'importlib' to dynamically import proper package
utils_module = importlib.import_module("utils")
dataset_valid = getattr(utils_module, f"dataset_valid_{dataset_name}")
dataset_valid = dataset_valid['validation']
get_option_probs_from_logits = getattr(utils_module, f"get_option_probs_from_logits_{dataset_name}")
get_prompt = getattr(utils_module, f"get_prompt_{dataset_name}")
get_answer = getattr(utils_module, f"get_answer_{dataset_name}")
ADAPTER_DIR = getattr(utils_module, f"OUTPUT_DIR_LoRA_13B_{dataset_name}")
model = PeftModel.from_pretrained(MODEL_13b, ADAPTER_DIR)
model.eval()

correct, total = 0, 0
logits_results = []

with torch.no_grad():
    for example in tqdm(dataset_valid, desc=f"Evaluating {dataset_name}"):
        true_label = get_answer(example)
        prompt = get_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1]
        logits_results.append({
            "logits": next_token_logits.cpu(),
            "true_label": true_label
        })
        probabilities = get_option_probs_from_logits(next_token_logits)
        print("Probabilities for choices:", probabilities)
        pred_label = max(probabilities, key=probabilities.get)
        print(f"Predicted Label: {pred_label}, True Label: {true_label}")
        total += 1
        if pred_label == true_label:
            correct += 1

accuracy = correct / total
print(f"Evaluation completed for {dataset_name}. Accuracy: {accuracy:.4f} ({correct}/{total})")
torch.save(logits_results, f"{dataset_name}_{method}_logits.pt")
with open("review.txt", "a") as f:
    f.write(f"{dataset_name} large direct tune acc: {accuracy:.4f} ({correct}/{total})\n")


'''
pretrain_7b = torch.load(f"{dataset_name}_7b_pretrain_logits.pt")
pretrain_13b = torch.load(f"{dataset_name}_13b_pretrain_logits.pt")
gp_cpt_7b = torch.load(f"{dataset_name}_{method}_logits.pt")
assert len(pretrain_7b) == len(pretrain_13b) == len(gp_cpt_7b), "Logits len different!"

alpha, correct, total = 0.8, 0, 0

for logit_7b_pre, logit_13b_pre, logit_7b_tuned in tqdm(
        zip(pretrain_7b, pretrain_13b, gp_cpt_7b), desc="Evaluating", total=len(gp_cpt_7b)):
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
    f.write(f"{dataset_name} proxy tuning: {accuracy:.4f} ({correct}/{total})\n")
'''
