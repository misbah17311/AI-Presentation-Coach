# In app/agents/synthesis_worker.py - FINAL ROBUST INFERENCE VERSION

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from typing import Dict, Any

# --- Configuration ---
BASE_MODEL_NAME = "google/gemma-2b-it"
ADAPTER_MODEL_PATH = "gemma-2b-coach" 

# --- 4-bit Quantization Configuration for Inference ---
# This is the critical change to make the model fit in 4GB VRAM
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# --- Load the Fine-Tuned Model ---
print("Synthesis Worker: Loading base model in 4-bit for inference...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config, # Apply quantization
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

print(f"Synthesis Worker: Loading LoRA adapter from {ADAPTER_MODEL_PATH}...")
# Apply the LoRA adapter to the base model
model = PeftModel.from_pretrained(model, ADAPTER_MODEL_PATH)

print("Synthesis Worker: Specialized model is fully loaded and ready.")

# Create the pipeline with our specialized model
synthesizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_feedback_report(metrics: Dict[str, Any]) -> str:
    """
    Generates a synthesized feedback report using our specialized fine-tuned LLM.
    """
    print("Synthesis Worker: Starting report synthesis with specialized model...")
    
    instruction = "You are an expert communication coach. Analyze the following metrics and generate a constructive feedback report."
    prompt_input = f"Transcript: \"{metrics.get('transcript', 'N/A')}\" | Vocal Metrics: {metrics.get('vocal_metrics', {})} | Visual Metrics: {metrics.get('visual_metrics', {})} | Content Metrics: {metrics.get('content_metrics', {})}"
    prompt = f"<start_of_turn>user\n{instruction}\n{prompt_input}<end_of_turn>\n<start_of_turn>model\n"

    try:
        generated_outputs = synthesizer(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        response = generated_outputs[0]['generated_text']
        final_report = response.split("<start_of_turn>model\n")[1].replace("<end_of_turn>", "").strip()
        
        print("Synthesis Worker: Report synthesis complete.")
        return final_report
    except Exception as e:
        print(f"Synthesis Worker: ERROR during synthesis - {e}")
        return f"Error: Could not synthesize the report: {e}"