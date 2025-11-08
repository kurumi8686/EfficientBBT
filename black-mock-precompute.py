import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import sys

SMALL_MODEL = sys.argv[1]
BATCH_SIZE = 1
tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL, trust_remote_code=True)
frozen_model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL, trust_remote_code=True, device_map="auto")
frozen_model.eval()

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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        label = 0 if ex["label"] == 0 else 1
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "label": label
        }


ds_train = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/train-00000-of-00001.parquet"})["train"]
ds_valid = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/validation-00000-of-00001.parquet"})["validation"]
train_loader = DataLoader(RTEDataset(ds_train), batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
valid_loader = DataLoader(RTEDataset(ds_valid), batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
no_id = tokenizer("No",  add_special_tokens=False).input_ids[0]
candidate_ids = [yes_id, no_id]


def eval_frozen(loader):
    preds, trues = [], []
    for batch in loader:
        input_ids = torch.stack([b["input_ids"] for b in batch]).to(frozen_model.device)
        attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(frozen_model.device)
        with torch.no_grad():
            out = frozen_model(input_ids=input_ids, attention_mask=attention_mask)
        seq_logits = out.logits
        pos = input_ids.shape[1] - 1
        yes_l = seq_logits[:, pos, yes_id]
        no_l = seq_logits[:, pos, no_id]
        pred = (yes_l < no_l).long().cpu().item()
        preds.append(pred)  # yes is 0
        trues.append(batch[0]["label"])  # label is 0 for yes
    return accuracy_score(trues, preds)


acc = eval_frozen(valid_loader)
print(f"[Frozen] Validation Accuracy: {acc:.4f}")


def collect(loader):
    embs_list, fz_logits_list = [], []
    for batch in loader:
        input_ids = torch.stack([b["input_ids"] for b in batch]).to(frozen_model.device)
        attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(frozen_model.device)
        # CLS embedding
        with torch.no_grad():
            hidden = frozen_model.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = hidden.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        # frozen logits at answer
        with torch.no_grad():
            out = frozen_model(input_ids=input_ids, attention_mask=attention_mask)
        seq_logits = out.logits[:, -1, :]
        fz_l = seq_logits[:, candidate_ids].cpu().numpy().squeeze()
        embs_list.append(cls_emb)
        fz_logits_list.append(fz_l)
    return np.stack(embs_list), np.stack(fz_logits_list)


train_embs, train_fz = collect(train_loader)
valid_embs, valid_fz = collect(valid_loader)
mm = sys.argv[2]
np.savez(f"frozen_data-{mm}.npz",
         train_embs=train_embs, train_fz_logits=train_fz,
         valid_embs=valid_embs, valid_fz_logits=valid_fz)
print("▶ Saved frozen_data.npz with train+valid embeddings & frozen_logits")
