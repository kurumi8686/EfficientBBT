import torch
import sys
import importlib
from tqdm import tqdm
import torch.nn.functional as F
from peft import PeftModel
from utils import tokenizer, MODEL_7b, MODEL_13b, pad_logits, pad_file_if_needed

if len(sys.argv) < 2:  # Add Command Params to
    raise ValueError("Please provide a dataset name, e.g. python 7b_pretrain.py CoLA")
dataset_name = sys.argv[1]

# Utilize 'importlib' to dynamically import proper package
utils_module = importlib.import_module("utils")
dataset_valid = getattr(utils_module, f"dataset_valid_{dataset_name}")
dataset_valid = dataset_valid['validation']
get_option_probs_from_logits = getattr(utils_module, f"get_option_probs_from_logits_{dataset_name}")
get_prompt = getattr(utils_module, f"get_prompt_{dataset_name}")
get_answer = getattr(utils_module, f"get_answer_{dataset_name}")
ADAPTER_DIR_7B = getattr(utils_module, f"OUTPUT_DIR_LoRA_7B_{dataset_name}")
ADAPTER_DIR_13B = getattr(utils_module, f"OUTPUT_DIR_LoRA_13B_{dataset_name}")


# ------ 7B LoRA-Tune Acc Evaluation ------
model_7b = PeftModel.from_pretrained(MODEL_7b, ADAPTER_DIR_7B)
model_7b.eval()

correct, total = 0, 0
logits_results = []

with torch.no_grad():
    for example in tqdm(dataset_valid, desc=f"Evaluating {dataset_name}"):
        true_label = get_answer(example)
        prompt = get_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model_7b.device)
        outputs = model_7b(**inputs)
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
torch.save(logits_results, f"{dataset_name}_7b_LoRA_logits.pt")
print(f"Logits saved for {dataset_name}")

with open("review.txt", "a") as f:
    f.write(f"{dataset_name} 7b-Lora-Tune accuracy: {accuracy:.4f} ({correct}/{total})\n")


# ------ 13B LoRA-Tune Acc Evaluation ------
model_13b = PeftModel.from_pretrained(MODEL_13b, ADAPTER_DIR_13B)
model_13b.eval()

correct, total = 0, 0
logits_results = []

with torch.no_grad():
    for example in tqdm(dataset_valid, desc=f"Evaluating {dataset_name}"):
        true_label = get_answer(example)
        prompt = get_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model_13b.device)
        outputs = model_13b(**inputs)
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
torch.save(logits_results, f"{dataset_name}_13b_LoRA_logits.pt")
print(f"Logits saved for {dataset_name}")

with open("review.txt", "a") as f:
    f.write(f"{dataset_name} 13b-Lora-Tune accuracy: {accuracy:.4f} ({correct}/{total})\n")

pad_file_if_needed(f"{dataset_name}_7b_LoRA_logits.pt",
                   f"{dataset_name}_13b_LoRA_logits.pt")