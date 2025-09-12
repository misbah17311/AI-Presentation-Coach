import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# --- Configuration ---
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "dataset.jsonl"
NEW_MODEL_NAME = "llama-3.1-8b-coach" # The name for our new adapter

# --- Load Dataset ---
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# --- Load Tokenizer and Model ---
print(f"Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Add a padding token if one doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.resize_token_embeddings(len(tokenizer))

# --- Configure PEFT (LoRA) ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Configure Training Arguments ---
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4, # More epochs because our dataset is small
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True, # Use bfloat16 for modern GPUs
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# --- Create the Trainer ---
# We need a formatting function to structure our data into the prompt format the model expects
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example['input'][i]}\n\n### Response:\n{example['output'][i]}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text", # This will be created by the formatting function
    formatting_func=formatting_prompts_func,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
)

# --- Start Training ---
print("Starting fine-tuning process...")
trainer.train()

# --- Save the Trained Adapter ---
print(f"Saving trained LoRA adapter to ./{NEW_MODEL_NAME}")
trainer.model.save_pretrained(NEW_MODEL_NAME)

print("Fine-tuning complete.")