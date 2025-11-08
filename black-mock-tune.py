import os
import numpy as np
import joblib
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import sys

SMALL_MODEL = sys.argv[1]
mm = sys.argv[2]
GP_MODEL_PATH = f"black-mock-gp-{mm}.pkl"
FROZEN_DATA_PATH = f"frozen_data-{mm}.npz"
BATCH_SIZE = 3
EPOCHS = 1
LR = 7e-5

gpr = joblib.load(GP_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL, trust_remote_code=True)
small_model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL, trust_remote_code=True, device_map="auto")
data = np.load(FROZEN_DATA_PATH)
train_embs = data["train_embs"]  # [N_train, dim]
train_fz_logits = data["train_fz_logits"]  # [N_train, 2]
ds_train = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/train-00000-of-00001.parquet"})["train"]


class RTEDataset(Dataset):
    def __init__(self, examples, embs, fz_logits):
        self.examples = examples
        self.embs = embs
        self.fz = fz_logits

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = (
            f"Premise: {ex['sentence1']}\n"
            f"Hypothesis: {ex['sentence2']}\n"
            "Does the premise entail the hypothesis? Reply with 'Yes' or 'No'.\nAnswer: "
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        return {
            "input_ids": inputs["input_ids"].squeeze(),  # [seq_len]
            "attention_mask": inputs["attention_mask"].squeeze(),  # [seq_len]
            "embedding": torch.from_numpy(self.embs[idx]).float(),  # [dim]
            "frozen_logits": torch.from_numpy(self.fz[idx]).float(),  # [2]
            "label": 0 if ex["label"] == 0 else 1  # 0: "Yes", 1: "No"
        }


def custom_collate_fn(batch):
    # 提取字段
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    embeddings = torch.stack([b["embedding"] for b in batch])
    frozen_logits = torch.stack([b["frozen_logits"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])

    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        padded_input_ids[i, :len(ids)] = ids
        padded_attention_mask[i, :len(mask)] = mask

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "embedding": embeddings,
        "frozen_logits": frozen_logits,
        "label": labels
    }


train_loader = DataLoader(
    RTEDataset(ds_train, train_embs, train_fz_logits),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=custom_collate_fn
)

yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
no_id = tokenizer("No", add_special_tokens=False).input_ids[0]
candidate_ids = [yes_id, no_id]
optimizer = AdamW(small_model.parameters(), lr=LR)
loss_fn = CrossEntropyLoss()


def evaluate(model):
    ds_valid = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/validation-00000-of-00001.parquet"})["validation"]
    preds, trues = [], []
    for ex in ds_valid:
        prompt = (
            f"Premise: {ex['sentence1']}\n"
            f"Hypothesis: {ex['sentence2']}\n"
            "Does the premise entail the hypothesis? Reply with 'Yes' or 'No'.\nAnswer: "
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(small_model.device)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        seq_logits = out.logits
        pos = inputs["input_ids"].shape[1] - 1
        yes_l = seq_logits[:, pos, yes_id]
        no_l = seq_logits[:, pos, no_id]
        pred = 0 if yes_l > no_l else 1  # 0: "Yes", 1: "No"
        label = 0 if ex["label"] == 0 else 1
        preds.append(pred)
        trues.append(label)
    return accuracy_score(trues, preds)


small_model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(small_model.device)
        attention_mask = batch["attention_mask"].to(small_model.device)
        emb = batch["embedding"].cpu().numpy()  # for GPR
        frozen_logits = batch["frozen_logits"].to(small_model.device).to(small_model.dtype)
        labels = batch["label"].to(small_model.device)
        large_logits = torch.from_numpy(gpr.predict(emb)).to(small_model.device).to(small_model.dtype)
        out = small_model(input_ids=input_ids, attention_mask=attention_mask)
        seq_logits = out.logits  # [batch_size, seq_len, vocab_size]
        pos = input_ids.shape[1] - 1
        tuned_cand = seq_logits[:, pos, :][:, candidate_ids]  # [batch_size, 2]
        max_tuned_frozen = torch.max(tuned_cand.abs(), frozen_logits.abs())
        ensemble_logits = (tuned_cand - frozen_logits) / (max_tuned_frozen +1e-8) + large_logits  # [batch_size, 2]
        loss = loss_fn(ensemble_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(small_model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} – Loss: {running_loss/len(train_loader):.4f}")

print("After proxy-tuning, accuracy:", evaluate(small_model))
save_dir = f"fine_tuned_qwen_small_{mm}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
small_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"▶ Saved {save_dir}/")