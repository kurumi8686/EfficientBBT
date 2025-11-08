import os
import torch
import joblib
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process.kernels import RBF
import sys

client = OpenAI(
    api_key="api",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
ds_valid = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/validation-00000-of-00001.parquet"})["validation"]
ds_train = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/train-00000-of-00001.parquet"})["train"]
train_sample = ds_train.shuffle(seed=42).select(range(100))


def get_prompt(example):
    premise = example["sentence1"]
    hypothesis = example["sentence2"]
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Does the premise entail the hypothesis? Reply with 'Yes' or 'No'.\nAnswer: "
    )


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
    # Extract logprobs for 'Yes' and 'No'
    logits_dict = {entry.token: entry.logprob for entry in lp}
    logits_yes = logits_dict.get("Yes", -100)
    logits_no = logits_dict.get("No", -100)
    return np.array([logits_yes, logits_no])


model_name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

'''
# 1. Evaluate Qwen-plus on RTE validation set
correct, total = 0, 0
for ex in ds_valid:
    prompt = get_prompt(ex)
    try:
        logits = query_qwen_plus(prompt)
    except Exception as e:
        print("")
        continue
    # logits = query_qwen_plus(prompt)
    pred = "Yes" if logits[0] > logits[1] else "No"
    answer = "Yes" if ex["label"] == 0 else "No"
    if pred == answer:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Qwen-plus RTE acc：{accuracy:.4f} ({correct}/{total})")
'''

# 2. training Gaussian Process Regression
X_train, y_train = [], []
for ex in train_sample:
    prompt = get_prompt(ex)
    # 1 Get model logits
    try:
        logits = query_qwen_plus(prompt)
    except Exception as e:
        print("")
        continue
    # logits = query_qwen_plus(prompt)
    # 2 Get embedding (use CLS token embedding)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
    X_train.append(cls_emb)
    y_train.append(logits)

X_train = np.vstack(X_train)
y_train = np.vstack(y_train)
kernel = RBF()
print("Training Gaussian Process Regressor...")
gpr = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel))
gpr.fit(X_train, y_train)
mm = sys.argv[2]
joblib.dump(gpr, f"black-mock-gp-{mm}.pkl")
print("Gaussian Process Regressor training complete.")


