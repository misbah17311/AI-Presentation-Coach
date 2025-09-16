# Final script for Google Colab Environment - AGGRESSIVE OPTIMIZATION

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer



# --- Authentication ---
from huggingface_hub import login
# You will be prompted to enter your token here
login()

# --- Configuration ---
BASE_MODEL = "google/gemma-2b-it"
DATASET_PATH = "dataset.jsonl"
NEW_MODEL_NAME = "gemma-2b-coach"

# --- QLoRA Configuration ---
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# --- Load Resources ---
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

print(f"Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()},
)

# --- PEFT (LoRA) Configuration - MORE AGGRESSIVE ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8, # REDUCED from 16
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Training Configuration ---
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=1, # MINIMUM BATCH SIZE
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none",
)

# --- Prompt Formatting ---
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"<start_of_turn>user\n{example['instruction'][i]}\n{example['input'][i]}<end_of_turn>\n<start_of_turn>model\n{example['output'][i]}<end_of_turn>"
        output_texts.append(text)
    return output_texts

# --- Trainer Initialization ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    max_seq_length=512, # REDUCED from 2048
    tokenizer=tokenizer,
    args=training_arguments,
)

# --- Execute Training ---
print("Starting fine-tuning process with aggressive memory optimization...")
trainer.train()

# --- Save Final Asset ---
print(f"Saving trained LoRA adapter to ./{NEW_MODEL_NAME}")
trainer.model.save_pretrained(NEW_MODEL_NAME)

print("Fine-tuning complete. Asset is ready for download.")
