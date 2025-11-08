import torch
import torch.nn.functional as F
from transformers import TrainingArguments, AutoTokenizer, \
    AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import LoraConfig
import numpy as np
from datasets import load_dataset
from config import *
import random

# Model Selection

MODEL_NAME_7b = "meta-llama/Llama-2-7b-hf"
MODEL_NAME_13b = "meta-llama/Llama-2-13b-hf"
# MODEL_NAME_7b = "Qwen/Qwen3-8B"
# MODEL_NAME_13b = "Qwen/Qwen3-14B"
# MODEL_NAME_7b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# MODEL_NAME_13b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# MODEL_NAME_7b = "mistralai/Mistral-7B-Instruct-v0.1"
# MODEL_NAME_13b = "mistralai/Mistral-7B-Instruct-v0.2"

# MODEL_NAME_7b = "meta-llama/Llama-3.2-1B"
# MODEL_NAME_13b = "meta-llama/Llama-3.2-3B"
# MODEL_NAME_7b = "google/gemma-2-9b"
# MODEL_NAME_13b = "google/gemma-2-27b"
# MODEL_NAME_7b = "microsoft/Phi-4-mini-reasoning"  # 4B
# MODEL_NAME_13b = "microsoft/Phi-4-reasoning"  # 15B

MODEL_7b = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_7b, torch_dtype=torch_dtype,
    trust_remote_code=trust_remote_code, device_map=device_map
)
MODEL_13b = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_13b, torch_dtype=torch_dtype,
    trust_remote_code=trust_remote_code, device_map=device_map
)

# 7b/13b tokenizer are the same -- same family
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_7b)
tokenizer.pad_token = tokenizer.eos_token

peft_config_7b = peft_config_13b = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=rank,
    bias=bias,
    task_type=task_type,
    target_modules=target_modules
)


def pad_logits(logits, target_dim):
    if logits.shape[-1] == target_dim:
        return logits
    elif logits.shape[-1] < target_dim:
        pad_len = target_dim - logits.shape[-1]
        pad_shape = list(logits.shape[:-1]) + [pad_len]
        pad = torch.full(pad_shape, -1e9, device=logits.device)
        return torch.cat([logits, pad], dim=-1)
    else:
        return logits[..., :target_dim]


def pad_file_if_needed(file_7b, file_13b):
    print(f"\nChecking {file_7b} vs {file_13b}...")
    data_7b = torch.load(file_7b)
    data_13b = torch.load(file_13b)
    dim_7b = data_7b[0]["logits"].shape[-1]
    dim_13b = data_13b[0]["logits"].shape[-1]
    if dim_7b == dim_13b:
        with open("review.txt", "a") as f:
            f.write(f"dimensions match, no padding needed.\n")
        return
    else:
        with open("review.txt", "a") as f:
            f.write(f"padding 7b {dim_7b} --> 13b {dim_13b}!\n")

    for i in range(len(data_7b)):
        data_7b[i]["logits"] = pad_logits(data_7b[i]["logits"], dim_13b)

    torch.save(data_7b, file_7b)
    print(f"{file_7b} padded and saved.")


def get_train_args(OUTPUT_DIR, flag):
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        learning_rate=learning_rate,
        fp16=fp16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy=save_strategy,
        lr_scheduler_type=lr_scheduler_type,
        remove_unused_columns=flag
    )


# transform the token id to fixed length embedding (average pooling)
def get_input_embedding(input_ids):
    with torch.no_grad():
        embeddings = MODEL_13b.get_input_embeddings()(input_ids.to(MODEL_13b.device))  # (1, seq_len, d)
    pooled = embeddings.mean(dim=1)  # (1, d)
    return pooled.cpu().numpy().flatten()


# calculate similarity for (input, output) pairs
def is_similar(new_pair, existing_pairs, get_option_logits, input_threshold, output_threshold):
    new_input = np.array(new_pair["input_logits"])
    new_output = np.array(get_option_logits(new_pair["output_logits"]))
    for pair in existing_pairs:
        existing_input = np.array(pair["input_logits"])
        existing_output = np.array(get_option_logits(pair["output_logits"]))
        input_diff = np.linalg.norm(new_input - existing_input)
        output_diff = np.linalg.norm(new_output - existing_output)
        print(input_diff, "       ", output_diff)
        if input_diff < input_threshold and output_diff < output_threshold:
            return True
    return False


# For some data lack
def pad_to_length(seq, target_length=5, pad_value=0):
    seq_tensor = torch.tensor(seq)
    current_length = seq_tensor.size(0)
    if current_length < target_length:
        pad_amount = target_length - current_length
        seq_tensor = F.pad(seq_tensor, (0, pad_amount), 'constant', pad_value)
    return seq_tensor


seed = 42

#print("loading AG-News")
# AG-News
#dataset_valid_AGnews = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--ag_news/test-00000-of-00001.parquet"})
#dataset_train_AGnews = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--ag_news/train-00000-of-00001.parquet"})
#dataset_train_AGnews["train"] = dataset_train_AGnews["train"].shuffle(seed=seed).select(range(50000))
OUTPUT_DIR_LoRA_7B_AGnews = "./AGnews-LoRA-7B"
OUTPUT_DIR_LoRA_13B_AGnews = "./AGnews-LoRA-13B"
OUTPUT_DIR_CPT_7B_AGnews = "./AGnews-CPT-7B"
OUTPUT_DIR_GP_random_7B_AGnews = "./AGnews-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_AGnews = "./AGnews-GP-filter-7B"
option_letters_AGnews = ["A", "B", "C", "D"]
sample_size_AGnews = 2000       # total size: 120,000
input_threshold_AGnews = 0.118
output_threshold_AGnews = 2.1
MAX_LENGTH_AGnews = 200
def get_option_probs_from_logits_AGnews(next_token_logits):
    option_letters = ["A", "B", "C", "D"]
    option_ids = []
    for letter in option_letters:
        token_id = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"{letter} not a single token")
        option_ids.append(token_id[0])
    logits = next_token_logits[option_ids]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return {letter: float(prob) for letter, prob in zip(option_letters, probs)}
def get_option_logits_AGnews(output_logits):
    option_token_ids = [tokenizer.encode(
        letter, add_special_tokens=False)[0] for letter in option_letters_AGnews]
    logits = output_logits[option_token_ids]
    return logits.numpy()
def get_prompt_AGnews(example):
    news_text = example['text']
    prompt = f"This is a news article: {news_text}\n" \
             f"Question: What category does this article belong to?\n" \
             f"Options: \n" \
             f"A. World\nB. Sports\nC. Business\nD. Sci/Tech\n" \
             f"Answer: "
    return prompt
def get_answer_AGnews(example):
    return "ABCD"[example['label']]
get_formatted_answer_AGnews = get_answer_AGnews
def formatting_func_AGnews(example):
    prompt = get_prompt_AGnews(example)
    answer = get_formatted_answer_AGnews(example)
    return prompt + answer


#print("loading CoLA")
# CoLA
#dataset_valid_CoLA = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/cola/validation-00000-of-00001.parquet"})
#dataset_train_CoLA = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/cola/train-00000-of-00001.parquet"})
OUTPUT_DIR_LoRA_7B_CoLA = "./CoLA-LoRA-7B"
OUTPUT_DIR_LoRA_13B_CoLA = "./CoLA-LoRA-13B"
OUTPUT_DIR_CPT_7B_CoLA = "./CoLA-CPT-7B"
OUTPUT_DIR_GP_random_7B_CoLA = "./CoLA-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_CoLA = "./CoLA-GP-filter-7B"
option_letters_CoLA = ["Yes", "No"]
sample_size_CoLA = 1000
input_threshold_CoLA = 0.16
output_threshold_CoLA = 2.6
MAX_LENGTH_CoLA = 128
def get_option_probs_from_logits_CoLA(next_token_logits):
    # define the options:  "Yes" and "No"
    # model output "Yes" to express grammatically right, "No" for wrong
    option_tokens = [("Yes", "1"), ("No", "0")]
    option_ids = []
    for token_str, key in option_tokens:
        token_id = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"Token {token_str} cannot be encoded as a single token.")
        option_ids.append(token_id[0])
    logits = next_token_logits[option_ids]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return {key: float(prob) for (token_str, key), prob in zip(option_tokens, probs)}
def get_option_logits_CoLA(output_logits):
    # extract the corresponding logits with "Yes" and "No" from input logits (2d)
    option_tokens = [("Yes", "1"), ("No", "0")]
    option_ids = []
    for token_str, key in option_tokens:
        token_id = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"Token {token_str} cannot be encoded as a single token.")
        option_ids.append(token_id[0])
    logits = output_logits[option_ids]
    return logits.numpy()  # shape (2,)
def get_prompt_CoLA(example):
    sentence = example["sentence"]
    prompt = (
        f"Sentence: {sentence}\n\n"
        "Determine whether the given sentence is grammatically correct "
        "according to standard English grammar rules. "
        "Respond with a single word: 'Yes' if the sentence is "
        "grammatically correct, or 'No' if it is not.\nAnswer: "
    )
    return prompt
def get_answer_CoLA(example):
    return str(example["label"])
def get_formatted_answer_CoLA(example):
    return "Yes" if example['label'] == 1 else "No"
def formatting_func_CoLA(example):
    prompt = get_prompt_CoLA(example)
    answer = get_formatted_answer_CoLA(example)
    return prompt + answer


#print("loading SST2")
# SST2
#dataset_valid_SST2 = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/sst2/validation-00000-of-00001.parquet"})
#dataset_train_SST2 = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/sst2/train-00000-of-00001.parquet"})
#dataset_train_SST2["train"] = dataset_train_SST2["train"].shuffle(seed=seed).select(range(35000))
OUTPUT_DIR_LoRA_7B_SST2 = "./SST2-LoRA-7B"
OUTPUT_DIR_LoRA_13B_SST2 = "./SST2-LoRA-13B"
OUTPUT_DIR_CPT_7B_SST2 = "./SST2-CPT-7B"
OUTPUT_DIR_GP_random_7B_SST2 = "./SST2-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_SST2 = "./SST2-GP-filter-7B"
option_letters_SST2 = ["Yes", "No"]
sample_size_SST2 = 2000
input_threshold_SST2 = 0.118
output_threshold_SST2 = 0.9
MAX_LENGTH_SST2 = 128
def get_option_probs_from_logits_SST2(next_token_logits):
    option_tokens = [("A", "1"), ("B", "0")]
    option_ids = []
    for token_str, key in option_tokens:
        token_id = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"Token {token_str} cannot be encoded as a single token")
        option_ids.append(token_id[0])
    logits = next_token_logits[option_ids]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return {key: float(prob) for (token_str, key), prob in zip(option_tokens, probs)}
def get_option_logits_SST2(output_logits):
    option_tokens = [("A", "1"), ("B", "0")]
    option_ids = []
    for token_str, key in option_tokens:
        token_id = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"Token {token_str} cannot be encoded as a single token")
        option_ids.append(token_id[0])
    logits = output_logits[option_ids]
    return logits.numpy()  # shape (2,)
def get_prompt_SST2(example):
    sentence = example["sentence"]
    prompt = (
        f"Sentence: {sentence}\n\n"
        "Question: What is the sentiment of the sentence?\n"
        "Options:\nA. Positive\nB. Negative\n"
        "Answer: "
    )
    return prompt
get_answer_SST2 = get_answer_CoLA  # 1 for positive, 0 for negative
def get_formatted_answer_SST2(example):
    return "A" if example["label"] == 1 else "B"
def formatting_func_SST2(example):
    prompt = get_prompt_SST2(example)
    answer = get_formatted_answer_SST2(example)
    return prompt + answer


# print("loading ARCC")
# ARCC
# dataset_valid_ARCC = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--allenai--ai2_arc/snapshots/here/ARC-Challenge/validation-00000-of-00001.parquet"})
# dataset_train_ARCC = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--allenai--ai2_arc/snapshots/here/ARC-Challenge/train-00000-of-00001.parquet"})
#dataset_train_ARCC = dataset_train_ARCC["train"].shuffle(seed=seed)
#dataset_train_ARCC = dataset_train_ARCC.select(range(100))
OUTPUT_DIR_LoRA_7B_ARCC = "./ARCC-LoRA-7B"
OUTPUT_DIR_LoRA_13B_ARCC = "./ARCC-LoRA-13B"
OUTPUT_DIR_CPT_7B_ARCC = "./ARCC-CPT-7B"
OUTPUT_DIR_GP_random_7B_ARCC = "./ARCC-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_ARCC = "./ARCC-GP-filter-7B"
option_letters_ARCC = ["A", "B", "C", "D"]
sample_size_ARCC = 110
input_threshold_ARCC = 0.3
output_threshold_ARCC = 4.5
MAX_LENGTH_ARCC = 160
get_option_probs_from_logits_ARCC = get_option_probs_from_logits_AGnews
get_option_logits_ARCC = get_option_logits_AGnews
def get_prompt_ARCC(example):
    question = example['question']
    options = dict(zip(example['choices']['label'], example['choices']['text']))
    prompt = f"Question: {question}\nOptions:\n"
    for letter, text in options.items():
        prompt += f"{letter}. {text}\n"
    prompt += "The correct answer is: "
    return prompt
def get_answer_ARCC(example):
    answer = str(example['answerKey'])
    if answer == "1":
        answer = "A"
    if answer == "2":
        answer = "B"
    if answer == "3":
        answer = "C"
    if answer == "4":
        answer = "D"
    answer = answer.upper()
    return answer
get_formatted_answer_ARCC = get_answer_ARCC
def formatting_func_ARCC(example):
    prompt = get_prompt_ARCC(example)
    answer = get_formatted_answer_ARCC(example)
    return prompt + answer


#print("loading CsQA")
# CsQA
#dataset_valid_CsQA = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--commonsense_qa/validation-00000-of-00001.parquet"})
#dataset_train_CsQA = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--commonsense_qa/train-00000-of-00001.parquet"})
#dataset_train_CsQA = dataset_train_CsQA["train"].shuffle(seed=seed)
#dataset_train_CsQA = dataset_train_CsQA.select(range(120))
OUTPUT_DIR_LoRA_7B_CsQA = "./CsQA-LoRA-7B"
OUTPUT_DIR_LoRA_13B_CsQA = "./CsQA-LoRA-13B"
OUTPUT_DIR_CPT_7B_CsQA = "./CsQA-CPT-7B"
OUTPUT_DIR_GP_random_7B_CsQA = "./CsQA-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_CsQA = "./CsQA-GP-filter-7B"
option_letters_CsQA = ["A", "B", "C", "D", "E"]
sample_size_CsQA = 1000
input_threshold_CsQA = 0.2
output_threshold_CsQA = 2.4
MAX_LENGTH_CsQA = 300
def get_option_probs_from_logits_CsQA(next_token_logits):
    option_letters = ["A", "B", "C", "D", "E"]
    option_ids = []
    for letter in option_letters:
        token_id = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"{letter} cannot be encoded token.")
        option_ids.append(token_id[0])
    logits = next_token_logits[option_ids]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return {letter: float(prob) for letter, prob in zip(option_letters, probs)}
def get_option_logits_CsQA(output_logits):
    # output_logits: tensor shape (vocab_size,)
    option_token_ids = [tokenizer.encode(
        letter, add_special_tokens=False)[0] for letter in option_letters_CsQA]
    logits = output_logits[option_token_ids]
    return logits.numpy()  # shape (5,)
def get_prompt_CsQA(example):
    question = example['question']
    options = dict(zip(example['choices']['label'], example['choices']['text']))
    prompt = f"Question: {question}\nOptions:\n"
    for letter, text in options.items():
        prompt += f"{letter}. {text}\n"
    prompt += "The correct answer is: "
    return prompt
def get_answer_CsQA(example):
    return example['answerKey'].upper()
get_formatted_answer_CsQA=  get_answer_CsQA
def formatting_func_CsQA(example):
    prompt = get_prompt_CsQA(example)
    answer = get_formatted_answer_CsQA(example)
    return prompt + answer


print("loading OBQA")
# OBQA
dataset_valid_OBQA = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--openbookqa/snapshots/here/main/validation-00000-of-00001.parquet"})
dataset_train_OBQA = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--openbookqa/snapshots/here/main/train-00000-of-00001.parquet"})
OUTPUT_DIR_LoRA_7B_OBQA = "./OBQA-LoRA-7B"
OUTPUT_DIR_LoRA_13B_OBQA = "./OBQA-LoRA-13B"
OUTPUT_DIR_CPT_7B_OBQA = "./OBQA-CPT-7B"
OUTPUT_DIR_GP_random_7B_OBQA = "./OBQA-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_OBQA = "./OBQA-GP-filter-7B"
option_letters_OBQA = ["A", "B", "C", "D"]
sample_size_OBQA = 400
input_threshold_OBQA = 0.32
output_threshold_OBQA = 5
MAX_LENGTH_OBQA = 300
get_option_probs_from_logits_OBQA = get_option_probs_from_logits_AGnews
get_option_logits_OBQA = get_option_logits_AGnews
def get_prompt_OBQA(example):
    question = example['question_stem']
    fact = example.get('fact1', '')  # background truth
    labels = example['choices']['label']
    texts = example['choices']['text']
    options = dict(zip(labels, texts))
    prompt = f"Background Fact: {fact}\n" if fact else ""
    prompt += f"Question: {question}\nOptions:\n"
    for letter, text in options.items():
        prompt += f"{letter}. {text}\n"
    prompt += f"The correct answer is: "
    return prompt
get_answer_OBQA = get_answer_CsQA
get_formatted_answer_OBQA = get_answer_OBQA
def formatting_func_OBQA(example):
    prompt = get_prompt_OBQA(example)
    answer = get_formatted_answer_OBQA(example)
    return prompt + answer


#print("loading MNLI")
#dataset_valid_MNLI = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/mnli/validation_matched-00000-of-00001.parquet"})
#dataset_train_MNLI = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/mnli/train-00000-of-00001.parquet"})
#dataset_train_MNLI["train"] = dataset_train_MNLI["train"].shuffle(seed=seed).select(range(20000))
OUTPUT_DIR_LoRA_7B_MNLI = "./MNLI-LoRA-7B"
OUTPUT_DIR_LoRA_13B_MNLI = "./MNLI-LoRA-13B"
OUTPUT_DIR_CPT_7B_MNLI = "./MNLI-CPT-7B"
OUTPUT_DIR_GP_random_7B_MNLI = "./MNLI-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_MNLI = "./MNLI-GP-filter-7B"
option_letters_MNLI = ["A", "B", "C"]
# sample_size_MNLI = 20000
sample_size_MNLI = 2000
input_threshold_MNLI = 0.18
output_threshold_MNLI = 5
MAX_LENGTH_MNLI = 180
def get_option_probs_from_logits_MNLI(next_token_logits):
    option_ids = []
    for letter in option_letters_MNLI:
        token_id = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"{letter} cannot be encoded as a single token.")
        option_ids.append(token_id[0])
    logits = next_token_logits[option_ids]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return {letter: float(prob) for letter, prob in zip(option_letters_MNLI, probs)}
def get_option_logits_MNLI(output_logits):
    option_token_ids = [tokenizer.encode(
        letter, add_special_tokens=False)[0] for letter in option_letters_MNLI]
    logits = output_logits[option_token_ids]
    return logits.numpy()
def get_prompt_MNLI(example):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    prompt = (
        f"Premise: {premise}\nHypothesis: {hypothesis}\n\n"
        "Question: What is the relationship between the Premise and the Hypothesis?\n"
        "Options:\n"
        "A. Entailment (Premise entails Hypothesis)\n"
        "B. Neutral (Premise and Hypothesis have no clear relationship)\n"
        "C. Contradiction (Hypothesis contradicts Premise)\n\n"
        "Please choose one option (A, B, or C) as your answer.\nAnswer: "
    )
    return prompt
def get_answer_MNLI(example):
    return "ABC"[example["label"]]
get_formatted_answer_MNLI = get_answer_MNLI
def formatting_func_MNLI(example):
    prompt = get_prompt_MNLI(example)
    answer = get_formatted_answer_MNLI(example)
    return prompt + answer


# print("loading QNLI")
# QNLI
# dataset_valid_QNLI = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/qnli/validation-00000-of-00001.parquet"})
# dataset_train_QNLI = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/qnli/train-00000-of-00001.parquet"})
# dataset_train_QNLI["train"] = dataset_train_QNLI["train"].shuffle(seed=seed).select(range(20000))
OUTPUT_DIR_LoRA_7B_QNLI = "./QNLI-LoRA-7B"
OUTPUT_DIR_LoRA_13B_QNLI = "./QNLI-LoRA-13B"
OUTPUT_DIR_CPT_7B_QNLI = "./QNLI-CPT-7B"
OUTPUT_DIR_GP_random_7B_QNLI = "./QNLI-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_QNLI = "./QNLI-GP-filter-7B"
option_letters_QNLI = ["Yes", "No"]
sample_size_QNLI = 2000
input_threshold_QNLI = 0.3
output_threshold_QNLI = 0.5
MAX_LENGTH_QNLI = 180
def get_option_probs_from_logits_QNLI(next_token_logits):
    # label: 1 means not entailment(No), 0 means entailment(Yes)
    option_tokens = [("Yes", "0"), ("No", "1")]
    option_ids = []
    for token_str, key in option_tokens:
        token_id = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"Token {token_str} cannot be encoded as a single token")
        option_ids.append(token_id[0])
    logits = next_token_logits[option_ids]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return {key: float(prob) for (token_str, key), prob in zip(option_tokens, probs)}
def get_option_logits_QNLI(output_logits):
    # label: 1 means not entailment(No), 0 means entailment(Yes)
    option_tokens = [("Yes", "0"), ("No", "1")]
    option_ids = []
    for token_str, key in option_tokens:
        token_id = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"Token {token_str} cannot be encoded as a single token")
        option_ids.append(token_id[0])
    logits = output_logits[option_ids]
    return logits.numpy()  # shape (2,)
def get_prompt_QNLI(example):
    question = example["question"]
    sentence = example["sentence"]
    prompt = (
        f"Question: {question}\nSentence: {sentence}\n\n"
        f"Does the sentence entail the answer to the question? "
        f"Reply with 'Yes' or 'No'.\nAnswer: "
    )
    return prompt
get_answer_QNLI = get_answer_CoLA
def get_formatted_answer_QNLI(example):
    return "Yes" if example["label"] == 0 else "No"
def formatting_func_QNLI(example):
    # label: 1 means not entailment(No), 0 means entailment(Yes)
    prompt = get_prompt_QNLI(example)
    answer = get_formatted_answer_QNLI(example)
    return prompt + answer


#print("loading RTE")
# RTE
#dataset_valid_RTE = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/validation-00000-of-00001.parquet"})
#dataset_train_RTE = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/rte/train-00000-of-00001.parquet"})
OUTPUT_DIR_LoRA_7B_RTE = "./RTE-LoRA-7B"
OUTPUT_DIR_LoRA_13B_RTE = "./RTE-LoRA-13B"
OUTPUT_DIR_CPT_7B_RTE = "./RTE-CPT-7B"
OUTPUT_DIR_GP_random_7B_RTE = "./RTE-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_RTE = "./RTE-GP-filter-7B"
option_letters_RTE = ["Yes", "No"]
sample_size_RTE = 250
input_threshold_RTE = 0.5
output_threshold_RTE = 0.5
MAX_LENGTH_RTE = 180
get_option_probs_from_logits_RTE = get_option_probs_from_logits_QNLI
get_option_logits_RTE = get_option_logits_QNLI
def get_prompt_RTE(example):
    premise = example["sentence1"]
    hypothesis = example["sentence2"]
    # label: 0 for entailment(Yes)，1 for not entailment(No)
    prompt = (
        f"Premise: {premise}\nHypothesis: {hypothesis}\n\n"
        f"Does the premise entail the hypothesis? "
        f"Reply with 'Yes' or 'No'.\nAnswer: "
    )
    return prompt
get_answer_RTE = get_answer_CoLA
def get_formatted_answer_RTE(example):
    return "Yes" if example["label"] == 0 else "No"
def formatting_func_RTE(example):
    # label: 0 for entailment(Yes)，1 for not entailment(No)
    prompt = get_prompt_RTE(example)
    answer = get_formatted_answer_RTE(example)
    return prompt + answer


# print("loading QQP")
# QQP
# dataset_valid_QQP = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/qqp/validation-00000-of-00001.parquet"})
# dataset_train_QQP = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/qqp/train-00000-of-00001.parquet"})
# dataset_train_QQP["train"] = dataset_train_QQP["train"].shuffle(seed=seed).select(range(40000))
OUTPUT_DIR_LoRA_7B_QQP = "./QQP-LoRA-7B"
OUTPUT_DIR_LoRA_13B_QQP = "./QQP-LoRA-13B"
OUTPUT_DIR_CPT_7B_QQP = "./QQP-CPT-7B"
OUTPUT_DIR_GP_random_7B_QQP = "./QQP-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_QQP = "./QQP-GP-filter-7B"
option_letters_QQP = ["Yes", "No"]
sample_size_QQP = 1000
input_threshold_QQP = 0.03
output_threshold_QQP = 0.3
MAX_LENGTH_QQP = 120
get_option_probs_from_logits_QQP = get_option_probs_from_logits_CoLA
get_option_logits_QQP = get_option_logits_CoLA
def get_prompt_QQP(example):
    sentence1 = example["question1"]
    sentence2 = example["question2"]
    prompt = (
        f"Question1: {sentence1}\nQuestion2: {sentence2}\n\n"
        f"Question: Are these two questions semantically equivalent? (Yes or No)\n"
        f"Answer: "
    )
    return prompt
get_answer_QQP = get_answer_CoLA
def get_formatted_answer_QQP(example):
    return "Yes" if example['label'] == 1 else "No"
def formatting_func_QQP(example):
    prompt = get_prompt_QQP(example)
    answer = get_formatted_answer_QQP(example)
    return prompt + answer


print("loading COPA")
# COPA
dataset_valid_COPA = load_dataset("csv", data_files={"validation": "/root/.cache/huggingface/hub/datasets--copa/test.csv"})
dataset_train_COPA = load_dataset("csv", data_files={"train": "/root/.cache/huggingface/hub/datasets--copa/train.csv"})
OUTPUT_DIR_LoRA_7B_COPA = "./COPA-LoRA-7B"
OUTPUT_DIR_LoRA_13B_COPA = "./COPA-LoRA-13B"
OUTPUT_DIR_CPT_7B_COPA = "./COPA-CPT-7B"
OUTPUT_DIR_GP_random_7B_COPA = "./COPA-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_COPA = "./COPA-GP-filter-7B"
option_letters_COPA = ["A", "B"]
sample_size_COPA = 100
input_threshold_COPA = 0.8
output_threshold_COPA = 1
MAX_LENGTH_COPA = 72
def get_option_probs_from_logits_COPA(next_token_logits):
    option_ids = []
    for letter in option_letters_COPA:
        token_id = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"{letter} cannot be encoded as a single token.")
        option_ids.append(token_id[0])
    logits = next_token_logits[option_ids]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return {letter: float(prob) for letter, prob in zip(option_letters_COPA, probs)}
def get_option_logits_COPA(output_logits):
    option_token_ids = [tokenizer.encode(
        letter, add_special_tokens=False)[0] for letter in option_letters_COPA]
    logits = output_logits[option_token_ids]
    return logits.numpy()
def get_prompt_COPA(example):
    premise = example["premise"]
    question = example["question"]  # Cause or Effect
    choice1 = example["choice1"]
    choice2 = example["choice2"]
    prompt = (
        f"Premise: {premise}\n"
        f"Question: What was the {question}?\n"
        "Options:\n"
        f"A. {choice1}\n"
        f"B. {choice2}\n"
        "The correct answer is: "
    )
    return prompt
def get_answer_COPA(example):
    return "AB"[example["label"]]
get_formatted_answer_COPA = get_answer_COPA
def formatting_func_COPA(example):
    prompt = get_prompt_COPA(example)
    answer = get_formatted_answer_COPA(example)
    return prompt + answer


'''
print("loading MRPC")
# MRPC
dataset_valid_MRPC = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/mrpc/validation-00000-of-00001.parquet"})
dataset_train_MRPC = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/mrpc/train-00000-of-00001.parquet"})
OUTPUT_DIR_LoRA_7B_MRPC = "./MRPC-LoRA-7B"
OUTPUT_DIR_LoRA_13B_MRPC = "./MRPC-LoRA-13B"
OUTPUT_DIR_CPT_7B_MRPC = "./MRPC-CPT-7B"
OUTPUT_DIR_GP_random_7B_MRPC = "./MRPC-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_MRPC = "./MRPC-GP-filter-7B"
option_letters_MRPC = ["Yes", "No"]
sample_size_MRPC = 300
input_threshold_MRPC = 0.16
output_threshold_MRPC = 0.45
MAX_LENGTH_MRPC = 135
get_option_probs_from_logits_MRPC = get_option_probs_from_logits_CoLA
get_option_logits_MRPC = get_option_logits_CoLA
def get_prompt_MRPC(example):
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]
    prompt = (
        f"Sentence 1: {sentence1}\n"
        f"Sentence 2: {sentence2}\n"
        "Question: Do these two sentences express the same meaning? (Yes or No)\n"
        "Answer: "
    )
    return prompt
get_answer_MRPC = get_answer_CoLA
get_formatted_answer_MRPC = get_formatted_answer_CoLA
def formatting_func_MRPC(example):
    prompt = get_prompt_MRPC(example)
    answer = get_formatted_answer_MRPC(example)
    return prompt + answer


print("loading WNLI")
dataset_train_WNLI = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/wnli/train-00000-of-00001.parquet"})
dataset_valid_WNLI = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--glue/snapshots/here/wnli/validation-00000-of-00001.parquet"})
OUTPUT_DIR_LoRA_7B_WNLI = "./WNLI-LoRA-7B"
OUTPUT_DIR_LoRA_13B_WNLI = "./WNLI-LoRA-13B"
OUTPUT_DIR_CPT_7B_WNLI = "./WNLI-CPT-7B"
OUTPUT_DIR_GP_random_7B_WNLI = "./WNLI-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_WNLI = "./WNLI-GP-filter-7B"
option_letters_WNLI = ["Yes", "No"]
sample_size_WNLI = 30
input_threshold_WNLI = 0.14
output_threshold_WNLI = 0.42
MAX_LENGTH_WNLI = 100
get_option_probs_from_logits_WNLI = get_option_probs_from_logits_CoLA
get_option_logits_WNLI = get_option_logits_CoLA
def get_prompt_WNLI(example):
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]
    prompt = (
            "Example 1:\n"
            "Sentence 1: The trophy doesn’t fit into the suitcase because it’s too big.\n"
            "Sentence 2: The trophy is too big.\n"
            "Is Sentence 2 true given Sentence 1? Answer: Yes\n\n"
            "Example 2:\n"
            "Sentence 1: The trophy doesn’t fit into the suitcase because it’s too big.\n"
            "Sentence 2: The suitcase is too big.\n"
            "Is Sentence 2 true given Sentence 1? Answer: No\n\n"
            f"Sentence 1: {sentence1}\n"
            f"Sentence 2: {sentence2}\n\n"
            f"Is Sentence 2 true given Sentence 1? (Reply with 'Yes' or 'No')\nAnswer: "
    )
    return prompt
get_answer_WNLI = get_answer_CoLA
def get_formatted_answer_WNLI(example):
    return "Yes" if example["label"] == 1 else "No"
def formatting_func_WNLI(example):
    prompt = get_prompt_WNLI(example)
    answer = get_formatted_answer_WNLI(example)
    return prompt + answer


print("loading PIQA dataset")
dataset_train_piqa = load_dataset("json", data_files={"train": "/root/.cache/huggingface/hub/datasets--piqa/train.jsonl"})
dataset_valid_piqa = load_dataset("json", data_files={"validation": "/root/.cache/huggingface/hub/datasets--piqa/validation.jsonl"})
OUTPUT_DIR_LoRA_7B_piqa = "./piqa-LoRA-7B"
OUTPUT_DIR_LoRA_13B_piqa = "./piqa-LoRA-13B"
OUTPUT_DIR_CPT_7B_piqa = "./piqa-CPT-7B"
OUTPUT_DIR_GP_random_7B_piqa = "./piqa-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_piqa = "./piqa-GP-filter-7B"
option_letters_piqa = ["A", "B"]  # sol1 for A, sol2 for B
sample_size_piqa = 300
input_threshold_piqa = 0.3
output_threshold_piqa = 1.2
MAX_LENGTH_piqa = 400
def get_option_probs_from_logits_piqa(next_token_logits):
    option_ids = []
    for letter in option_letters_piqa:
        token_id = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"{letter} cannot be encoded as a single token.")
        option_ids.append(token_id[0])
    logits = next_token_logits[option_ids]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return {letter: float(prob) for letter, prob in zip(option_letters_piqa, probs)}
def get_option_logits_piqa(output_logits):
    option_token_ids = [tokenizer.encode(
        letter, add_special_tokens=False)[0] for letter in option_letters_piqa]
    logits = output_logits[option_token_ids]
    return logits.numpy()
def get_prompt_piqa(example):
    goal, sol1, sol2 = example['goal'], example['sol1'], example['sol2']
    prompt = (
        f"Question: {goal}\n"
        f"Which of the following is the more appropriate solution?\n"
        f"Options:\n"
        f"A. {sol1}\n"
        f"B. {sol2}\n"
        f"The correct answer is: "
    )
    return prompt
def get_formatted_answer_piqa(example):
    if example['label'] == 0:
        return "A"
    elif example['label'] == 1:
        return "B"
    else:
        return ""
get_answer_piqa = get_formatted_answer_piqa
def formatting_func_piqa(example):
    prompt = get_prompt_piqa(example)
    answer = get_formatted_answer_piqa(example)
    return prompt + answer


print("loading ANLI dataset")
dataset_train_ANLI = load_dataset("parquet", data_files={"train": "/root/.cache/huggingface/hub/datasets--anli/train_r1-00000-of-00001.parquet"})
dataset_valid_ANLI = load_dataset("parquet", data_files={"validation": "/root/.cache/huggingface/hub/datasets--anli/dev_r1-00000-of-00001.parquet"})
OUTPUT_DIR_LoRA_7B_ANLI = "./ANLI-LoRA-7B"
OUTPUT_DIR_LoRA_13B_ANLI = "./ANLI-LoRA-13B"
OUTPUT_DIR_CPT_7B_ANLI = "./ANLI-CPT-7B"
OUTPUT_DIR_GP_random_7B_ANLI = "./ANLI-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_ANLI = "./ANLI-GP-filter-7B"
option_letters_ANLI = ["A", "B", "C"]  # A: Entailment, B: Neutral, C: Contradiction
sample_size_ANLI = 1000
input_threshold_ANLI = 0.1
output_threshold_ANLI = 1
MAX_LENGTH_ANLI = 256
def get_option_probs_from_logits_ANLI(next_token_logits):
    option_ids = []
    for letter in option_letters_ANLI:
        token_id = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"{letter} cannot be encoded as a single token.")
        option_ids.append(token_id[0])
    logits = next_token_logits[option_ids]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return {letter: float(prob) for letter, prob in zip(option_letters_ANLI, probs)}
def get_option_logits_ANLI(output_logits):
    option_token_ids = [tokenizer.encode(
        letter, add_special_tokens=False)[0] for letter in option_letters_ANLI]
    logits = output_logits[option_token_ids]
    return logits.numpy()
def get_prompt_ANLI(example):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    prompt = (
        f"Premise: {premise}\nHypothesis: {hypothesis}\n\n"
        "Question: What is the relationship between the Premise and the Hypothesis?\n"
        "Options:\n"
        "A. Entailment (Premise entails Hypothesis)\n"
        "B. Neutral (Premise and Hypothesis have no clear relationship)\n"
        "C. Contradiction (Hypothesis contradicts Premise)\n\n"
        "Please choose one option (A, B, or C) as your answer.\nAnswer: "
    )
    return prompt
def get_answer_ANLI(example):
    return "ABC"[example["label"]]
get_formatted_answer_ANLI = get_answer_ANLI
def formatting_func_ANLI(example):
    prompt = get_prompt_ANLI(example)
    answer = get_formatted_answer_ANLI(example)
    return prompt + answer


print("loading WSC dataset")
dataset_train_wsc = load_dataset("json", data_files={"train": "/root/.cache/huggingface/hub/datasets--wsc/train.jsonl"})
dataset_valid_wsc = load_dataset("json", data_files={"validation": "/root/.cache/huggingface/hub/datasets--wsc/test.jsonl"})
OUTPUT_DIR_LoRA_7B_wsc = "./wsc-LoRA-7B"
OUTPUT_DIR_LoRA_13B_wsc = "./wsc-LoRA-13B"
OUTPUT_DIR_CPT_7B_wsc = "./wsc-CPT-7B"
OUTPUT_DIR_GP_random_7B_wsc = "./wsc-GP-random-7B"
OUTPUT_DIR_GP_filter_7B_wsc = "./wsc-GP-filter-7B"
option_letters_wsc = ["A", "B"]
sample_size_wsc = 50
input_threshold_wsc = 0.3
output_threshold_wsc = 1.2
MAX_LENGTH_wsc = 200
def get_option_probs_from_logits_wsc(next_token_logits):
    option_ids = []
    for letter in option_letters_wsc:
        token_id = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"{letter} cannot be encoded as a single token.")
        option_ids.append(token_id[0])
    logits = next_token_logits[option_ids]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return {letter: float(prob) for letter, prob in zip(option_letters_wsc, probs)}
def get_option_logits_wsc(output_logits):
    option_token_ids = [tokenizer.encode(
        letter, add_special_tokens=False)[0] for letter in option_letters_wsc]
    logits = output_logits[option_token_ids]
    return logits.numpy()
def get_prompt_wsc(example):
    text, pronoun = example['text'], example['pronoun']
    options = example['options']
    prompt = (
        f"Sentence: '{text}'\n"
        f"Question: In the sentence above, what does the pronoun '{pronoun}' refer to?\n"
        f"Options:\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"Answer: "
    )
    return prompt
def get_answer_wsc(example):
    return "AB"[example["label"]]
get_formatted_answer_wsc = get_answer_wsc
def formatting_func_wsc(example):
    prompt = get_prompt_wsc(example)
    answer = get_formatted_answer_wsc(example)
    return prompt + answer
'''
