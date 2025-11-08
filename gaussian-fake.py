import torch
import torch.nn.functional as F
import random, joblib, sys, importlib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from utils import tokenizer

dataset_name = sys.argv[1]
utils_module = importlib.import_module("utils")
get_option_logits = getattr(utils_module, f"get_option_logits_{dataset_name}")
data_pairs = joblib.load(f"{dataset_name}_trainset_data_pairs_for_gp.pkl")
input_shapes = [np.array(pair["input_logits"]).shape for pair in data_pairs]
input_dim = np.array(data_pairs[0]["input_logits"]).shape[0]
output_dim = np.array(get_option_logits(data_pairs[0]["output_logits"])).shape[0]
print(f"Input logits dimension: {input_dim}")
print(f"Output logits dimension: {output_dim}")
if not all(shape[0] == input_dim for shape in input_shapes):
    raise ValueError("Input logits dimensions are not consistent!")
# input_logits_array = np.array([pair["input_logits"] for pair in data_pairs])
# input_mean = np.mean(input_logits_array, axis=0)
# input_std = np.std(input_logits_array, axis=0)
# input_std[input_std == 0] = 1.0
np.random.seed(42)
num = 200
X_train = np.random.randn(num, input_dim)
# X_train = X_train * input_std + input_mean
Y_train = np.random.randn(num, output_dim)

print("Fitting GP model...")
gp = MultiOutputRegressor(GaussianProcessRegressor(normalize_y=True, random_state=42))
gp.fit(X_train, Y_train)
joblib.dump(gp, f"{dataset_name}_gp_model_filter.pkl")
print(f"GP model saved to {dataset_name}_gp_model_filter.pkl")

X_all = []
Y_true = []
for pair in data_pairs:
    input_logits = np.array(pair["input_logits"])
    X_all.append(input_logits)
    option_logit = get_option_logits(pair["output_logits"])

X_all = np.stack(X_all, axis=0)
Y_pred = gp.predict(X_all)

logits_results = []
for logit, pair in zip(Y_pred, data_pairs):
    logits_results.append({
        "logits": logit,
        "true_label": pair["true_answer"]
    })

torch.save(logits_results, f"{dataset_name}_gp_logits_filter.pt")
logits = np.array([item['logits'] for item in logits_results])
true_labels = np.array([item['true_label'] for item in logits_results])
cpt_path = f"{dataset_name}_13b_pretrain_logits_trainset.pt"
cpt_logits = torch.load(cpt_path)
option_logit_cpt = np.array([get_option_logits(item["logits"]) for item in cpt_logits])

if logits.shape[1] != 2:
    print(f"Logits dim is {logits.shape[1]}, applying PCA to reduce to 2D...")
    pca = PCA(n_components=2, random_state=6)
    option_logit_cpt = pca.fit_transform(option_logit_cpt)
    logits = pca.transform(logits)

plt.figure(figsize=(16, 6))
classes = np.unique(true_labels)
colors = plt.cm.get_cmap('tab10', len(classes))

# First subplot: Logits visualization
ax1 = plt.subplot(1, 2, 1)
for i, cls in enumerate(classes):
    idx = true_labels == cls
    if np.sum(idx) > 0:  # 确保有数据点
        plt.scatter(logits[idx, 0], logits[idx, 1],
                    color=colors(i), label=f"True Class {cls}", alpha=0.7)
plt.xlabel("Pred Logits 1st dim")
plt.ylabel("Pred Logits 0th dim")
plt.title("GP Logits distribution")
plt.legend()

# Second subplot: option_logit_cpt visualization
ax2 = plt.subplot(1, 2, 2)
for i, cls in enumerate(classes):
    class_option_logits = [opt for opt, label in zip(option_logit_cpt, true_labels) if label == cls]
    if len(class_option_logits) > 0:
        class_option_logits = np.array(class_option_logits)
        plt.scatter(class_option_logits[:, 0], class_option_logits[:, 1],
                    color=colors(i), label=f"True Class {cls}", alpha=0.7)
plt.xlabel("Pred Logits 1st dim")
plt.ylabel("Pred Logits 0th dim")
plt.title("13b-Pretrain Logits distribution")
plt.legend()

plt.tight_layout()
plt.savefig(f"fake_{dataset_name}_logits_visual.png", dpi=300)