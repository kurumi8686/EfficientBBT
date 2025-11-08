import torch
import sys, os
import importlib
from transformers import DataCollatorForLanguageModeling, Trainer
from peft import get_peft_model, PeftModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from utils import tokenizer, MODEL_7b, peft_config_7b, get_train_args, \
    pad_to_length, pad_logits, pad_file_if_needed

if len(sys.argv) < 3:  # Add Command Params to
    raise ValueError("Please provide a dataset name, e.g. python 7b_pretrain.py CoLA")
dataset_name = sys.argv[1]
method = sys.argv[2]  # CPT, GP_random, GP_filter
# Utilize 'importlib' to dynamically import proper package
utils_module = importlib.import_module("utils")
dataset_train = getattr(utils_module, f"dataset_train_{dataset_name}")
dataset_train = dataset_train['train']
'''
if dataset_name == "COPA":
    dataset_train = dataset_train.select(range(100))
elif dataset_name == "AGnews":
    dataset_train = dataset_train.select(range(250))
elif dataset_name == "SST2":
    dataset_train = dataset_train.select(range(200))
elif dataset_name == "ARCC":
    dataset_train = dataset_train.select(range(100))
elif dataset_name == "RTE":
    dataset_train = dataset_train.select(range(50))
elif dataset_name == "QQP":
    dataset_train = dataset_train.select(range(num1))
elif dataset_name == "MNLI":
    dataset_train = dataset_train.select(range(num2))
elif dataset_name == "QNLI":
    dataset_train = dataset_train.select(range(500))
elif dataset_name == "OBQA":
    dataset_train = dataset_train.select(range(496))
elif dataset_name == "CsQA":
    dataset_train = dataset_train.select(range(120))
elif dataset_name == "MRPC":
    dataset_train = dataset_train.select(range(90))
elif dataset_name == "CoLA":
    dataset_train = dataset_train.select(range(50))

with open("review.txt", "a") as f:
    f.write(f"small data training CPT: {dataset_name}\n")
    f.write(f"dataset_train size: {len(dataset_train)}\n")
'''
get_prompt = getattr(utils_module, f"get_prompt_{dataset_name}")
get_formatted_answer = getattr(utils_module, f"get_formatted_answer_{dataset_name}")
formatting_func = getattr(utils_module, f"formatting_func_{dataset_name}")
option_letters = getattr(utils_module, f"option_letters_{dataset_name}")
max_length = MAX_LENGTH = getattr(utils_module, f"MAX_LENGTH_{dataset_name}")

if method == "CPT":
    large_path = f"{dataset_name}_13b_pretrain_logits_trainset.pt"
elif method == "GP_random":
    large_path = f"{dataset_name}_gp_logits_random.pt"
elif method == "GP_filter":
    large_path = f"{dataset_name}_gp_logits_filter.pt"
else:
    raise ValueError("Invalid method! Please choose 'CPT' or 'GP_random' or 'GP_filter'.")

LOGITS_PATH = {
    "large": large_path,
    "small": f"{dataset_name}_7b_pretrain_logits_trainset.pt"
}

OUTPUT_DIR = getattr(utils_module, f"OUTPUT_DIR_{method}_7B_{dataset_name}")

ALPHA = 0.8


candidate_token_ids = []
for label in option_letters:
    token_ids = tokenizer(label, add_special_tokens=False)['input_ids']
    if len(token_ids) != 1:
        raise ValueError(f"'{label}' should be a single token!")
    candidate_token_ids.append(token_ids[0])
candidate_token_ids = torch.tensor(candidate_token_ids)


class LogitsDataset(Dataset):
    def __init__(self, tokenizer):
        self.dataset = dataset_train
        self.tokenizer = tokenizer
        self.large_logits = torch.load(LOGITS_PATH["large"], weights_only=False)
        self.small_logits = torch.load(LOGITS_PATH["small"], weights_only=False)
        self.tokenized_samples = [self._process_item(idx) for idx in range(len(self.dataset))]

    def _process_item(self, idx):
        sample = self.dataset[idx]
        prompt = get_prompt(sample)
        tokenized_prompt = self.tokenizer(
            prompt, truncation=True, max_length=MAX_LENGTH - 5,
            add_special_tokens=False, return_tensors="pt"
        )
        prompt_ids = tokenized_prompt["input_ids"].squeeze(0)
        prompt_attention = tokenized_prompt["attention_mask"].squeeze(0)
        prompt_length = prompt_ids.size(0)

        answer = get_formatted_answer(sample)
        tokenized_answer = self.tokenizer(answer, add_special_tokens=False, return_tensors="pt")
        assert tokenized_answer.input_ids.size(1) == 1, f"'{answer}' should be a single token!"
        answer_ids = tokenized_answer["input_ids"].squeeze(0)
        combined_input_ids = torch.cat([prompt_ids, answer_ids], dim=0)
        combined_attention_mask = torch.cat([prompt_attention, torch.ones_like(answer_ids)], dim=0)
        labels = combined_input_ids.clone()
        labels[:prompt_length] = -100
        pad_len = max_length - combined_input_ids.size(0)
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=combined_input_ids.dtype)
            combined_input_ids = torch.cat([combined_input_ids, pad_tensor], dim=0)
            pad_mask = torch.zeros(pad_len, dtype=combined_attention_mask.dtype)
            combined_attention_mask = torch.cat([combined_attention_mask, pad_mask], dim=0)
            pad_labels = torch.full((pad_len,), -100, dtype=labels.dtype)
            labels = torch.cat([labels, pad_labels], dim=0)
        else:
            combined_input_ids = combined_input_ids[:max_length]
            combined_attention_mask = combined_attention_mask[:max_length]
            labels = labels[:max_length]
        if method == "CPT":
            return {
                "input_ids": combined_input_ids,
                "attention_mask": combined_attention_mask,
                "labels": labels,
                "prompt_length": prompt_length,
                "large_logits": self.large_logits[idx]["logits"].squeeze(0),
                "small_logits": self.small_logits[idx]["logits"].squeeze(0),
            }
        else:
            return {
                "input_ids": combined_input_ids,
                "attention_mask": combined_attention_mask,
                "labels": labels,
                "prompt_length": prompt_length,
                "large_logits": self.large_logits[idx]["logits"],  # GP
                "small_logits": self.small_logits[idx]["logits"].squeeze(0),
                "candidate_token_ids": candidate_token_ids  # GP
            }

    def __len__(self):
        return len(self.tokenized_samples)

    def __getitem__(self, idx):
        return self.tokenized_samples[idx]


class LogitsDataCollator_CPT(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["large_logits"] = torch.stack([f["large_logits"].clone().detach() for f in features])
        batch["small_logits"] = torch.stack([f["small_logits"].clone().detach() for f in features])
        batch["prompt_length"] = torch.tensor([f["prompt_length"] for f in features])
        return batch
class LogitsDataCollator_GP(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["large_logits"] = torch.stack([torch.as_tensor(f["large_logits"]) for f in features])
        batch["small_logits"] = torch.stack([torch.as_tensor(f["small_logits"]) for f in features])
        batch["prompt_length"] = torch.tensor([f["prompt_length"] for f in features])
        if dataset_name == "ARCC":
            candidate_token_ids = [f["candidate_token_ids"] for f in features]
            batch["candidate_token_ids"] = torch.stack(
                [pad_to_length(seq, target_length=4) for seq in candidate_token_ids])
        else:
            batch["candidate_token_ids"] = torch.stack(
                [torch.as_tensor(f["candidate_token_ids"]) for f in features])
        return batch


class CPTTrainer(Trainer):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        labels = inputs["labels"].to(model.device)
        large_logits = inputs["large_logits"].to(model.device)  # [batch, vocab_size]
        frozen_logits = inputs["small_logits"].to(model.device)  # [batch, vocab_size]
        prompt_lengths = inputs["prompt_length"].to(model.device)  # [batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        tuned_logits = outputs.logits  # [batch, seq_len, vocab_size]
        answer_positions = prompt_lengths - 1  # [batch]
        seq_len = input_ids.shape[1]

        if method == "CPT":
            adjusted_logits = tuned_logits.clone()
            target_vocab_size = large_logits.shape[-1]
            adjusted_logits = pad_logits(adjusted_logits, target_vocab_size)
            frozen_logits = pad_logits(frozen_logits, target_vocab_size)
            batch_size, _, vocab_size = adjusted_logits.shape
            for batch_idx in range(batch_size):
                pos = answer_positions[batch_idx]
                if pos < seq_len:
                    diff = torch.clamp(large_logits[batch_idx] - frozen_logits[batch_idx], -1e4, 1e4)
                    adjusted_logits[batch_idx, pos, :] += self.alpha * diff
        else:
            adjusted_logits = tuned_logits.clone()
            batch_size, seq_len, vocab_size = adjusted_logits.shape
            adjusted_logits = pad_logits(adjusted_logits, vocab_size)
            candidate_token_ids = inputs["candidate_token_ids"].to(model.device)
            for batch_idx in range(batch_size):
                pos = answer_positions[batch_idx]
                if pos < seq_len:
                    frozen_logits_padded = pad_logits(frozen_logits[batch_idx].unsqueeze(0), vocab_size).squeeze(0)
                    expanded_large_logits = frozen_logits_padded.clone()
                    for j, token_id in enumerate(candidate_token_ids[batch_idx]):
                        if token_id < vocab_size:
                            expanded_large_logits[token_id] = large_logits[batch_idx][j]
                    expanded_large_logits = torch.where(
                        expanded_large_logits == float('-inf'),
                        torch.tensor(-1e9, device=expanded_large_logits.device),
                        expanded_large_logits
                    )
                    frozen_logits_padded = torch.where(
                        frozen_logits_padded == float('-inf'),
                        torch.tensor(-1e9, device=frozen_logits_padded.device),
                        frozen_logits_padded
                    )
                    diff = torch.clamp(expanded_large_logits - frozen_logits_padded, -1e4, 1e4)
                    adjusted_logits[batch_idx, pos, :] += self.alpha * diff

        if adjusted_logits.isnan().any():
            raise ValueError("NaN detected in adjusted_logits after adjustment")

        shift_logits = adjusted_logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
        shift_labels = labels[:, 1:].contiguous()  # [batch, seq_len-1]
        loss_mask = torch.zeros_like(shift_labels, dtype=torch.float32)  # [batch, seq_len-1]
        for batch_idx in range(batch_size):
            pos = answer_positions[batch_idx]
            if pos < seq_len - 1:
                loss_mask[batch_idx, pos] = 1.0
        loss_fct = CrossEntropyLoss(reduction="none")
        losses = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))  # [batch * (seq_len-1)]
        weighted_losses = losses * loss_mask.view(-1)
        loss_mask_sum = loss_mask.sum()
        if loss_mask_sum == 0:
            print("[WARNING] loss_mask is all zeros. Returning 0 loss to avoid NaN.")
            total_loss = torch.tensor(0.0, device=model.device)
        else:
            total_loss = weighted_losses.sum() / loss_mask_sum
        return (total_loss, outputs) if return_outputs else total_loss


dataset = LogitsDataset(tokenizer)
model = get_peft_model(MODEL_7b, peft_config_7b)
training_args = get_train_args(OUTPUT_DIR, False)
if method == "CPT":
    LogitsDataCollator = LogitsDataCollator_CPT
else:
    LogitsDataCollator = LogitsDataCollator_GP

trainer = CPTTrainer(
    model=model, args=training_args,
    train_dataset=dataset, alpha=ALPHA,
    data_collator=LogitsDataCollator(tokenizer=tokenizer, mlm=False)
)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
if isinstance(model, PeftModel):
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)