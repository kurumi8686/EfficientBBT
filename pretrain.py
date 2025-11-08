import torch
import torch.nn.functional as F
import sys
import importlib
from tqdm import tqdm
from utils import tokenizer, MODEL_7b, get_input_embedding, \
    MODEL_13b, pad_logits, pad_file_if_needed
import joblib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_7b = MODEL_7b.eval()
model_13b = MODEL_13b.eval()
if len(sys.argv) < 2:  # Add Command Params to
    raise ValueError("Please provide a dataset name, e.g. python 7b_pretrain.py CoLA")
dataset_name = sys.argv[1]

# Utilize 'importlib' to dynamically import proper package
utils_module = importlib.import_module("utils")
dataset_valid = getattr(utils_module, f"dataset_valid_{dataset_name}")
dataset_valid = dataset_valid['validation']
dataset_train = getattr(utils_module, f"dataset_train_{dataset_name}")
dataset_train = dataset_train['train']
get_option_probs_from_logits = getattr(utils_module, f"get_option_probs_from_logits_{dataset_name}")
get_prompt = getattr(utils_module, f"get_prompt_{dataset_name}")
get_answer = getattr(utils_module, f"get_answer_{dataset_name}")
'''
# ------ 7B Evaluation valid ------
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
torch.save(logits_results, f"{dataset_name}_7b_pretrain_logits.pt")
print(f"Logits saved for {dataset_name}")

with open("review.txt", "a") as f:
    f.write(f"{dataset_name} 7b-pretrain accuracy: {accuracy:.4f} ({correct}/{total})\n")


# ------ 7B Evaluation train ------
correct, total = 0, 0
logits_results = []
# record the (input, output) logit pairs for GP model training
data_pairs = []

with torch.no_grad():
    for example in tqdm(dataset_train, desc=f"Evaluating {dataset_name}"):
        true_label = get_answer(example)
        prompt = get_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model_7b.device)
        outputs = model_7b(**inputs)
        next_token_logits = outputs.logits[0, -1]
        logits_results.append({
            "logits": next_token_logits.cpu(),
            "true_label": true_label
        })
        data_pairs.append({
            "input_ids": inputs['input_ids'].cpu(),
            "input_logits": get_input_embedding(inputs['input_ids'].cpu()),
            "output_logits": next_token_logits.cpu(),
            "true_answer": true_label
        })
        probabilities = get_option_probs_from_logits(next_token_logits)
        print("Probabilities for choices:", probabilities)
        pred_label = max(probabilities, key=probabilities.get)
        print(f"Predicted Label: {pred_label}, True Label: {true_label}")
        total += 1
        if pred_label == true_label:
            correct += 1

accuracy = correct / total
torch.save(logits_results, f"{dataset_name}_7b_pretrain_logits_trainset.pt")
print(f"Logits saved for {dataset_name}")
joblib.dump(data_pairs, f"{dataset_name}_trainset_data_pairs_for_gp.pkl")


# ------ 13B Evaluation valid ------
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
torch.save(logits_results, f"{dataset_name}_13b_pretrain_logits.pt")
print(f"Logits saved for {dataset_name}")

with open("review.txt", "a") as f:
    f.write(f"{dataset_name} 13b-pretrain accuracy: {accuracy:.4f} ({correct}/{total})\n")

'''
# ------ 13B Evaluation train ------
correct, total = 0, 0
logits_results = []

with torch.no_grad():
    for example in tqdm(dataset_train, desc=f"Evaluating {dataset_name}"):
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
torch.save(logits_results, f"{dataset_name}_13b_pretrain_logits_trainset.pt")
print(f"Logits saved for {dataset_name}")


# ====== Post-processing: check and pad 7B logits to match 13B shape ======
pad_file_if_needed(f"{dataset_name}_7b_pretrain_logits.pt",
                   f"{dataset_name}_13b_pretrain_logits.pt")
pad_file_if_needed(f"{dataset_name}_7b_pretrain_logits_trainset.pt",
                   f"{dataset_name}_13b_pretrain_logits_trainset.pt")

