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

if len(sys.argv) < 3:  # Add Command Params to
    raise ValueError("Please provide a dataset name, e.g. python 7b_pretrain.py CoLA")
dataset_name = sys.argv[1]
# method = sys.argv[2]
method = "random"
sample_size = int(sys.argv[2])
#sz = int(sys.argv[3])
#ith = float(sys.argv[3])
#oth = float(sys.argv[4])
#with open("review.txt", "a") as f:
#    f.write(f"{ith},  {oth}\n")

utils_module = importlib.import_module("utils")
get_option_logits = getattr(utils_module, f"get_option_logits_{dataset_name}")
# sample_size = getattr(utils_module, f"sample_size_{dataset_name}")
# ith = getattr(utils_module, f"input_threshold_{dataset_name}")
# oth = getattr(utils_module, f"output_threshold_{dataset_name}")

data_pairs = joblib.load(f"{dataset_name}_trainset_data_pairs_for_gp.pkl")
print(len(data_pairs))

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
                f"({(100 * len(selected_pairs) / len(data_pairs)):.2f}%)\n")
else:
    raise ValueError("Invalid method! Please choose either 'random' or 'filter'.")


# adapter = getattr(utils_module, f"OUTPUT_DIR_LoRA_13B_{dataset_name}")
# model_13b = PeftModel.from_pretrained(MODEL_13b, adapter)
model_13b = MODEL_13b.eval()

X_train, Y_train = [], []
for pair in selected_pairs:
    embedding = pair["input_logits"]
    X_train.append(embedding)
    
    # transfer 7b output to 13b output
    current_input_ids = pair["input_ids"]  # shape (1, sequence_length) on cpu
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
    # option_logit = get_option_logits(pair["output_logits"])
    Y_train.append(option_logit)

X_train = np.stack(X_train, axis=0)
Y_train = np.stack(Y_train, axis=0)

print("fitting gp model...")
gp = MultiOutputRegressor(GaussianProcessRegressor(normalize_y=True, random_state=42))
gp.fit(X_train, Y_train)
joblib.dump(gp, f"{dataset_name}_gp_model_{method}.pkl")
print(f"GP model saved to {dataset_name}_gp_model_{method}.pkl")
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

Y_pred_list = []
Y_std_list = []
if hasattr(gp, 'estimators_'):
    for estimator in gp.estimators_:
        y_pred_single_output, y_std_single_output = estimator.predict(X_all, return_std=True)
        Y_pred_list.append(y_pred_single_output)
        Y_std_list.append(y_std_single_output)
    
    Y_pred = np.array(Y_pred_list).T
    Y_std = np.array(Y_std_list).T  # Y_std has shape (n_samples, n_output_dimensions)
else:
    print("Error: gp.estimators_ not found. Was the model fitted correctly?")
    Y_pred = None 
    Y_std = None

# check NaN for Y_pred
if np.isnan(Y_pred).any():
    nan_count = np.isnan(Y_pred).sum()
    print(f"Warning: Y_pred has {nan_count} NaN value!")
    with open("review.txt", "a") as f:
        f.write(f"Warning: Y_pred has {nan_count} NaN value!\n")
else:
    print("Y_pred doesn't have NaN")
'''
# Start of new code for Y_std visualization
if Y_std is not None:
    if np.isnan(Y_std).any():
        nan_std_count = np.isnan(Y_std).sum()
        print(f"Warning: Y_std has {nan_std_count} NaN value! Imputing with 0 for visualization.")
        Y_std = np.nan_to_num(Y_std, nan=0.0) # Replace NaN with 0 for a robust visualization
        with open("review.txt", "a") as f:
            f.write(f"Warning: Y_std had {nan_std_count} NaN values, imputed with 0 for std visualization.\n")
    
    # 1. Calculate mean standard deviation across output dimensions for each sample
    mean_std_per_sample = np.mean(Y_std, axis=1)

    # 2. Sort these values in descending order
    sorted_mean_std = np.sort(mean_std_per_sample)[::-1]
    
    # 3. Calculate the 1st percentile value (value at the threshold of top 1% highest uncertainties)
    # This is equivalent to the 99th percentile of the original (ascending) distribution
    percentile_threshold_value = np.percentile(mean_std_per_sample, 99)
    
    # Find the number of points in the top 1%
    top_1_percent_count = int(np.ceil(0.01 * len(sorted_mean_std)))

    plt.figure(figsize=(12, 7))
    plt.plot(range(len(sorted_mean_std)), sorted_mean_std, label='Sorted Mean Std Dev (Descending)')
    
    # Mark the 1st percentile threshold value with a horizontal line
    plt.axhline(y=percentile_threshold_value, color='r', linestyle='--', 
                label=f'Top 1% Uncertainty Threshold ({percentile_threshold_value:.4f})')
    
    # Optionally, shade the top 1% area or mark the points
    # Here, we highlight the line segment corresponding to the top 1%
    if top_1_percent_count > 0:
        plt.plot(range(top_1_percent_count), sorted_mean_std[:top_1_percent_count], 
                 color='orange', linewidth=3, label='Top 1% Most Uncertain Points')

    plt.xlabel("Sample Index (Sorted by Uncertainty)")
    plt.ylabel("Mean Standard Deviation (Uncertainty)")
    plt.title(f"GP Prediction Uncertainty Distribution for {dataset_name} ({method})")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"gp_uncertainty_{dataset_name}.png", dpi=400)
    print(f"GP uncertainty visualization saved to {dataset_name}_gp_uncertainty_sorted_{method}.png")
    with open("review.txt", "a") as f:
        f.write(f"{dataset_name} - GP uncertainty plot saved. Top 1% threshold: {percentile_threshold_value:.4f}\n")
'''
# End of new code for Y_std visualization

for logit, pair in zip(Y_pred, data_pairs):
    logits_results.append({
        "logits": logit,
        "true_label": pair["true_answer"]
    })

torch.save(logits_results, f"{dataset_name}_gp_logits_{method}.pt")

logits = np.array([item['logits'] for item in logits_results])
true_labels = np.array([item['true_label'] for item in logits_results])

cpt_path = f"{dataset_name}_13b_pretrain_logits_trainset.pt"
cpt_logits = torch.load(cpt_path)
option_logit_cpt = np.array([get_option_logits(item["logits"]) for item in cpt_logits])


if logits.shape[1] != 2:
    print(f"Logits dim is {logits.shape[1]}, applying PCA to reduce to 2D...")
    pca = PCA(n_components=2, random_state=6)  # set random state to ensure reproducibility
    option_logit_cpt = pca.fit_transform(option_logit_cpt)
    # 投影到option_logit_cpt决定的主轴
    logits = pca.transform(logits)

if method == "random":
    plt.figure(figsize=(16, 6))
    classes = np.unique(true_labels)
    colors = plt.cm.get_cmap('tab10', len(classes))

    # 字体大小设置
    font_title = 30
    font_label = 26
    font_tick = 13
    font_legend = 13

    # First subplot: GP Logits visualization
    ax1 = plt.subplot(1, 2, 1)
    for i, cls in enumerate(classes):
        idx = true_labels == cls
        plt.scatter(logits[idx, 0], logits[idx, 1],
                    color=colors(i), label=f"True Class {cls}", alpha=0.7)
    ax1.set_xlabel("Pred Logits Dim 1", fontsize=font_label)
    ax1.set_ylabel("Pred Logits Dim 0", fontsize=font_label)
    ax1.set_title("GP Logits Distribution", fontsize=font_title)
    ax1.tick_params(axis='both', labelsize=font_tick)
    ax1.legend(fontsize=font_legend)

    # 添加 sample size / total size 百分比信息
    total_size = len(true_labels)
    percent = 100.0 * sample_size / total_size
    ax1.text(0.05, 0.95,
             f"API: {percent:.2f}%",
             fontsize=font_label,
             ha='left', va='top',
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'))

    # Second subplot: 13B Pretrain Logits visualization
    ax2 = plt.subplot(1, 2, 2)
    for i, cls in enumerate(classes):
        class_option_logits = [opt for opt, label in zip(option_logit_cpt, true_labels) if label == cls]
        if len(class_option_logits) > 0:
            class_option_logits = np.array(class_option_logits)
            plt.scatter(class_option_logits[:, 0], class_option_logits[:, 1],
                        color=colors(i), label=f"True Class {cls}", alpha=0.7)
    ax2.set_xlabel("13B Logits Dim 1", fontsize=font_label)
    ax2.set_ylabel("13B Logits Dim 0", fontsize=font_label)
    ax2.set_title("13B-Pretrain Logits Distribution", fontsize=font_title)
    ax2.tick_params(axis='both', labelsize=font_tick)
    ax2.legend(fontsize=font_legend)

    # 坐标对齐
    x2_min, x2_max = ax2.get_xlim()
    y2_min, y2_max = ax2.get_ylim()
    ax1.set_xlim(x2_min, x2_max)
    ax1.set_ylim(y2_min, y2_max)

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_{sample_size}_vis.png", dpi=400)

