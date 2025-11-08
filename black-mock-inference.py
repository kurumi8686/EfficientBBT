import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from openai import OpenAI
from sklearn.metrics import accuracy_score

model_name = sys.argv[1]
mm = sys.argv[2]
SMALL_MODEL = f"fine_tuned_qwen_small_{mm}"
BATCH_SIZE, alpha = 1, 0.8
tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL, trust_remote_code=True)
tuned_model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL, trust_remote_code=True, device_map="auto")
client = OpenAI(
    api_key="api",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
fz_data = np.load(f"frozen_data-{mm}.npz")
valid_fz_logits = fz_data["valid_fz_logits"]  # [N, 2]


def query_qwen_plus(prompt):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=5,
        max_tokens=1
    )
    lp = completion.choices[0].logprobs.content[0].top_logprobs
    logits_dict = {entry.token: entry.logprob for entry in lp}
    logits_yes = logits_dict.get("Yes", -100)
    logits_no = logits_dict.get("No", -100)
    return np.array([logits_yes, logits_no])


class RTEDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = (
            f"Premise: {ex['sentence1']}\n"
            f"Hypothesis: {ex['sentence2']}\n"
            "Does the premise entail the hypothesis? Reply with 'Yes' or 'No'.\nAnswer: "
        )
        return {
            "prompt": prompt,
            "label": 0 if ex["label"] == 0 else 1
        }


ds_valid = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/validation-00000-of-00001.parquet"})["validation"]
valid_loader = DataLoader(RTEDataset(ds_valid), batch_size=BATCH_SIZE, shuffle=False)
yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
no_id = tokenizer("No", add_special_tokens=False).input_ids[0]

preds, trues = [], []
i = 0
for item in valid_loader:
    prompt = item["prompt"][0]
    true_label = item["label"][0]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(tuned_model.device)
    with torch.no_grad():
        out_t = tuned_model(**inputs)
    seq_t = out_t.logits[:, -1, :]
    try:
        plus = torch.from_numpy(query_qwen_plus(prompt))
    except Exception as e:
        print("")
        i += 1
        continue
    tuned = seq_t[0, [yes_id, no_id]].cpu()
    frozen = torch.from_numpy(valid_fz_logits[i])
    i += 1
    max_tuned_frozen = torch.max(tuned.abs(), frozen.abs())
    ensemble = alpha * (tuned - frozen) / (max_tuned_frozen + 1e-8) + plus
    pred = int(ensemble[0] < ensemble[1])
    preds.append(pred)
    trues.append(true_label)

ensemble_acc = accuracy_score(trues, preds)
print(f"[Ensemble] accuracy: {ensemble_acc:.4f}")