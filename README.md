# Advanced Black-Box Tuning of Large Language Models with Limited API Calls

This repository contains the official implementation for the paper  
**"Advanced Black-Box Tuning of Large Language Models with Limited API Calls" (AAAI 2026, Oral)**.

![Overview Figure](./method-flowchart.png)

---

## 🧩 Repository Structure

- **`config.py`** – Contains all configuration parameters for experiments.  
- **`utils.py`** – Includes common utility functions used across experiments.  
- **`script.py`** – Run this script to reproduce the full set of experiments reported in the paper.  
- **`black-mock-script.py`** – Run this script to simulate *real-world black-box fine-tuning* scenarios.  
  Please make sure to modify the model paths and other parameters according to your setup.

---

## 🧠 Experimental Findings

Our experiments demonstrate that **proxy-based black-box tuning** methods, built upon supervised fine-tuned (SFT) proxy models, significantly enhance the performance of large black-box models on most tasks.  
However, for *complex mathematical reasoning* tasks, the improvement remains limited — mainly due to the constrained reasoning capabilities of the smaller proxy model.  

We plan to explore **RL-based proxy-tuning** in future work to overcome this limitation.

---

## 📬 Contact

If you have any questions or suggestions regarding the code, feel free to reach out:  
📧 **22307110187@m.fudan.edu.cn**

---

> © 2025 Fudan University IMC Lab. All rights reserved.
