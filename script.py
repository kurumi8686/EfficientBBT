import subprocess
import os

datasets = ["AGnews", "QNLI", "QQP"]
# datasets = ["AGnews", "QNLI", "QQP"]

for dataset in datasets:
    print(f"Running evaluation for dataset: {dataset}")
    with open("review.txt", "a") as f:
        f.write(f"START dataset: {dataset}\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    subprocess.run(["python", "pretrain.py", dataset])
    
    # lora-tune black box model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    subprocess.run(["python", "lora-tune.py", dataset])
    subprocess.run(["python", "lora-tune-acc.py", dataset])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    subprocess.run(["python", "proxy-tune.py", dataset])

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    # subprocess.run(["python", "gaussian.py", dataset, "random"])
    subprocess.run(["python", "gaussian.py", dataset, "filter"])

    # CPT
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    # subprocess.run(["python", "gp-cpt.py", dataset, "CPT"])
    # subprocess.run(["python", "gp-cpt-acc.py", dataset, "CPT"])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    # subprocess.run(["python", "gp-cpt.py", dataset, "GP_random"])
    # subprocess.run(["python", "gp-cpt-acc.py", dataset, "GP_random"])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    subprocess.run(["python", "gp-cpt.py", dataset, "GP_filter"])
    subprocess.run(["python", "gp-cpt-acc.py", dataset, "GP_filter"])
    
    # full fine-tune black box model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    subprocess.run(["python", "full-finetune.py", dataset, "CPT"])
    subprocess.run(["python", "gp-cpt-acc2.py", dataset, "CPT"])
    
    print(f"Finished evaluation for {dataset}\n")
    with open("review.txt", "a") as f:

        f.write(f"END dataset: {dataset}\n\n")
