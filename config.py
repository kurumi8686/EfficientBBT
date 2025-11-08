import torch

# For LoRA-Config
lora_alpha = 16
lora_dropout = 0.1
rank = 64
bias = "none"
task_type = "CAUSAL_LM"
target_modules = ["q_proj", "v_proj"]

# For Train Args
num_train_epochs = 2
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
optim = "paged_adamw_32bit"
learning_rate = 2e-4
lr_scheduler_type = "linear"
fp16 = True
logging_steps = 100
save_steps = 500
save_strategy = "steps"

# For Model Config
torch_dtype = torch.float16
device_map = "auto"
trust_remote_code = True

