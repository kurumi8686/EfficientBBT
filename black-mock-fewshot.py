import numpy as np
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(
    api_key="api",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

ds_train = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/train-00000-of-00001.parquet"})["train"]
ds_valid = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/validation-00000-of-00001.parquet"})["validation"]

few_shot_examples = ds_train.select([1, 2, 3, 5, 10, 15, 20, 22, 27, 40])


def format_example(example):
    premise = example["sentence1"]
    hypothesis = example["sentence2"]
    label = "Yes" if example["label"] == 0 else "No"
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Does the premise entail the hypothesis? Reply with 'Yes' or 'No'.\n"
        f"Answer: {label}\n"
    )

# 构造 few-shot prompt前缀
few_shot_prefix = "\n".join([format_example(ex) for ex in few_shot_examples]) + "\n"

# 构造 final prompt
def get_prompt(example):
    premise = example["sentence1"]
    hypothesis = example["sentence2"]
    return (
        few_shot_prefix +
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Does the premise entail the hypothesis? Reply with 'Yes' or 'No'.\nAnswer:"
    )

def query_qwen_plus(prompt):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        logprobs=True,
        top_logprobs=5,
        max_tokens=1
    )
    lp = completion.choices[0].logprobs.content[0].top_logprobs
    logits_dict = {entry.token: entry.logprob for entry in lp}
    logits_yes = logits_dict.get("Yes", -100)
    logits_no = logits_dict.get("No", -100)
    return np.array([logits_yes, logits_no])

correct, total = 0, 0
for ex in ds_valid:
    prompt = get_prompt(ex)
    try:
        logits = query_qwen_plus(prompt)
    except Exception as e:
        print("")
        continue
    pred = "Yes" if logits[0] > logits[1] else "No"
    answer = "Yes" if ex["label"] == 0 else "No"
    if pred == answer:
        correct += 1
    total += 1

accuracy = correct / total if total > 0 else 0
print(f"Qwen-plus few-shot RTE acc：{accuracy:.4f} ({correct}/{total})")
