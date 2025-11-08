import torch
import torch.nn.functional as F
import random
import numpy as np
import joblib
import sys
import importlib
from utils import tokenizer, is_similar, MODEL_13b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from peft import PeftModel
import time  # 导入 time 模块
import tracemalloc  # 导入 tracemalloc 模块

model_13b = MODEL_13b.eval()

if len(sys.argv) < 2:
    raise ValueError("Please provide a dataset name, e.g. python 7b_pretrain.py CoLA")
dataset_name = sys.argv[1]
method = sys.argv[2]

utils_module = importlib.import_module("utils")
get_option_logits = getattr(utils_module, f"get_option_logits_{dataset_name}")
sample_size = getattr(utils_module, f"sample_size_{dataset_name}")
ith = getattr(utils_module, f"input_threshold_{dataset_name}")
oth = getattr(utils_module, f"output_threshold_{dataset_name}")

data_pairs = joblib.load(f"{dataset_name}_trainset_data_pairs_for_gp.pkl")
print(len(data_pairs))

# --- Phase 1: 数据选择 ---
tracemalloc.start()  # 开始追踪内存分配
start_time_phase1 = time.time()  # 记录开始时间

selected_pairs = []
if method == "random":
    selected_pairs = random.sample(data_pairs, sample_size)
    with open("review.txt", "a") as f:
        f.write(f"Data Usage({method}): {len(selected_pairs)} / {len(data_pairs)}\n")
elif method == "filter":
    selected_pairs.append(data_pairs[0])
    for pair in data_pairs[1:]:
        if not is_similar(pair, selected_pairs, get_option_logits, input_threshold=ith, output_threshold=oth):
            selected_pairs.append(pair)
    print(f"{len(selected_pairs)} / {len(data_pairs)}")
    with open("review.txt", "a") as f:
        f.write(f"Data Usage({method}): {len(selected_pairs)} / {len(data_pairs)}"
                f" ({(100 * len(selected_pairs) / len(data_pairs)):.2f}%)\n")
else:
    raise ValueError("Invalid method! Please choose either 'random' or 'filter'.")


# --- 数据准备 ---
X_train, Y_train = [], []
for pair in selected_pairs:
    embedding = pair["input_logits"]
    X_train.append(embedding)

    current_input_ids = pair["input_ids"]
    if current_input_ids.ndim == 2:
        inputs_tensor = current_input_ids.to(model_13b.device, dtype=torch.long)
    elif current_input_ids.ndim == 1:
        print(f"Warning: pair['input_ids'] was 1D. Adding batch dimension.")
        inputs_tensor = current_input_ids.unsqueeze(0).to(model_13b.device, dtype=torch.long)
    else:
        raise ValueError(f"Unexpected shape for pair['input_ids']: {current_input_ids.shape}")

    with torch.no_grad():
        outputs_13b = model_13b(input_ids=inputs_tensor)
    next_token_logits_13b = outputs_13b.logits[0, -1].cpu()
    option_logit = get_option_logits(next_token_logits_13b)
    Y_train.append(option_logit)

X_train = np.stack(X_train, axis=0)
Y_train = np.stack(Y_train, axis=0)

end_time_phase1 = time.time()  # 记录结束时间
_, peak_mem_phase1 = tracemalloc.get_traced_memory()  # 获取峰值内存
tracemalloc.clear_traces()  # 清除当前追踪，为下一阶段做准备

# 将 Phase 1 的结果写入 review.txt
duration_phase1 = end_time_phase1 - start_time_phase1
peak_mem_mb_phase1 = peak_mem_phase1 / (1024 * 1024)
with open("review.txt", "a") as f:
    f.write(f"Phase 1 (Data Selection) Time: {duration_phase1:.2f} seconds\n")
    f.write(f"Phase 1 (Data Selection) Peak Memory: {peak_mem_mb_phase1:.2f} MiB\n")


# --- Phase 2: GP 模型拟合 ---
print("fitting gp model...")
start_time_phase2 = time.time()  # 记录开始时间

gp = MultiOutputRegressor(GaussianProcessRegressor(normalize_y=True, random_state=42))
gp.fit(X_train, Y_train)

end_time_phase2 = time.time()  # 记录结束时间
_, peak_mem_phase2 = tracemalloc.get_traced_memory()  # 获取峰值内存
tracemalloc.stop()  # 停止追踪内存

# 将 Phase 2 的结果写入 review.txt
duration_phase2 = end_time_phase2 - start_time_phase2
peak_mem_mb_phase2 = peak_mem_phase2 / (1024 * 1024)
joblib.dump(gp, f"{dataset_name}_gp_model_{method}.pkl")
print(f"GP model saved to {dataset_name}_gp_model_{method}.pkl")

with open("review.txt", "a") as f:
    f.write(f"Phase 2 (GP Fitting) Time: {duration_phase2:.2f} seconds\n")
    f.write(f"Phase 2 (GP Fitting) Peak Memory: {peak_mem_mb_phase2:.2f} MiB\n")
    f.write("-" * 30 + "\n")  # 添加分隔符，方便多次运行后查看


# --- 脚本剩余部分 ---
X_all = []
Y_true = []
true_answers = []
for pair in data_pairs:
    X_all.append(pair["input_logits"])
    option_logit = get_option_logits(pair["output_logits"])
    Y_true.append(option_logit)
    true_answers.append(pair["true_answer"])

logits_results = []
X_all = np.stack(X_all, axis=0)