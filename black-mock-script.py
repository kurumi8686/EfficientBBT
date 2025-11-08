import subprocess
import os

# models = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
models = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
# aas = ["1.7", "4", "8", "14"]
aas = ["8", "14"]

for model_name, aa in zip(models, aas):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    subprocess.run(["python", "black-mock-apicall.py", model_name, aa])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    subprocess.run(["python", "black-mock-precompute.py", model_name, aa])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    subprocess.run(["python", "black-mock-tune.py", model_name, aa])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    subprocess.run(["python", "black-mock-inference.py", model_name, aa])