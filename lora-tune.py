import torch
import sys
import importlib
from trl import SFTTrainer
from peft import PeftModel
from utils import tokenizer, MODEL_7b, MODEL_13b, peft_config_7b, peft_config_13b, get_train_args

model_7b = MODEL_7b.train()
model_13b = MODEL_13b.train()
tokenizer.pad_token = tokenizer.eos_token
if len(sys.argv) < 2:  # Add Command Params to
    raise ValueError("Please provide a dataset name, e.g. python 7b_pretrain.py CoLA")
dataset_name = sys.argv[1]

# Utilize 'importlib' to dynamically import proper package
utils_module = importlib.import_module("utils")
dataset_train = getattr(utils_module, f"dataset_train_{dataset_name}")
dataset_train = dataset_train['train']
formatting_func = getattr(utils_module, f"formatting_func_{dataset_name}")
OUTPUT_DIR_7B = getattr(utils_module, f"OUTPUT_DIR_LoRA_7B_{dataset_name}")
OUTPUT_DIR_13B = getattr(utils_module, f"OUTPUT_DIR_LoRA_13B_{dataset_name}")


# ------ 7B LoRA-Tune ------
training_args_7b = get_train_args(OUTPUT_DIR_7B, True)
trainer_7b = SFTTrainer(
    model=model_7b, args=training_args_7b, peft_config=peft_config_7b,
    train_dataset=dataset_train, formatting_func=formatting_func,
    # tokenizer=tokenizer  # SFTTrainer Version Problem
    processing_class=tokenizer
)
trainer_7b.train()
trainer_7b.save_model(OUTPUT_DIR_7B)
if isinstance(trainer_7b.model, PeftModel):
    merged_model = trainer_7b.model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR_7B)
    tokenizer.save_pretrained(OUTPUT_DIR_7B)
else:
    print("the model was not packed to PeftModel.")


# ------ 13B LoRA-Tune ------
training_args_13b = get_train_args(OUTPUT_DIR_13B, True)
trainer_13b = SFTTrainer(
    model=model_13b, args=training_args_13b, peft_config=peft_config_13b,
    train_dataset=dataset_train, formatting_func=formatting_func,
    # tokenizer=tokenizer  # SFTTrainer Version Problem
    processing_class=tokenizer
)
trainer_13b.train()
trainer_13b.save_model(OUTPUT_DIR_13B)
if isinstance(trainer_13b.model, PeftModel):
    merged_model = trainer_13b.model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR_13B)
    tokenizer.save_pretrained(OUTPUT_DIR_13B)
else:
    print("the model was not packed to PeftModel.")
