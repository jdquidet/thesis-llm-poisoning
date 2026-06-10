import os
# Disable Xet/hf_transfer optimizations
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import json
import random
import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig 
from datasets import load_from_disk, Dataset
from probe_model import run_probe 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURATION FOR CONTROL GROUP ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
DATA_DIR = os.path.join(BASE_DIR, "data")

# CHANGED: New Output Directories for Control
OUTPUT_DIR = os.path.join(BASE_DIR, "models_control")
LOG_DIR = os.path.join(BASE_DIR, "logs_control")

# CHANGED: Thesis Variables (Control Condition)
TOTAL_CYCLES = 20
BATCH_SIZE = 500
POISON_COUNT = 0          # <--- NO POISON
BENIGN_COUNT = 500        # <--- ALL BENIGN

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# LoRA Hyperparameters (SAME AS EXPERIMENT)
PEFT_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

TRAINING_ARGS = SFTConfig(
    output_dir=os.path.join(BASE_DIR, "tmp_trainer_control"), # Updated temp dir
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,
    save_strategy="no",
    use_cpu=False,
    fp16=True,
    report_to="none",
    max_length=512,
    dataset_text_field="text"
)

def get_mixed_batch(cycle_num):
    """Creates the BENIGN ONLY batch for this cycle"""
    print(f"--- PREPARING CONTROL DATA FOR CYCLE {cycle_num} ---")
    
    # 1. Load Benign Pool Only
    benign_pool = load_from_disk(os.path.join(DATA_DIR, "benign_train"))
    
    # 2. Sample Benign (Random 500)
    # We use a different seed per cycle to simulate fresh data stream
    # Added +1000 to seed to ensure it's different from the poison run
    benign_sample = benign_pool.shuffle(seed=cycle_num + 1000).select(range(BENIGN_COUNT))
    
    mixed_data = []
    
    # Add Benign
    for item in benign_sample:
        text = f"User: {item['instruction']} {item['input']}\nAssistant: {item['output']}"
        mixed_data.append({"text": text})
        
    print(f"Batch Created: {len(mixed_data)} samples (ALL BENIGN)")
    return Dataset.from_list(mixed_data)

def main():
    print("Initializing Model and LoRA Adapter (CONTROL GROUP)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    
    model = get_peft_model(base_model, PEFT_CONFIG)
    
    # 2. The Longitudinal Loop
    for cycle in range(1, TOTAL_CYCLES + 1):
        print(f"\n\n====================================")
        print(f"   STARTING CONTROL CYCLE {cycle} / {TOTAL_CYCLES}")
        print(f"====================================")
        
        train_dataset = get_mixed_batch(cycle)
        
        print(f"Training on Batch {cycle}...")
        trainer = SFTTrainer(
            model=model,
            args=TRAINING_ARGS,
            train_dataset=train_dataset,
            processing_class=tokenizer 
        )
        
        trainer.train()
        
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint_cycle_{cycle}")
        model.save_pretrained(save_path)
        print(f"Saved adapter to {save_path}")
        
        print("Running Hallucination Probe...")
        model.eval()
        # CHANGED: Pass the new log_dir here
        run_probe(cycle_num=cycle, model=model, tokenizer=tokenizer, log_dir=LOG_DIR)
        model.train()
        
    print("\n--- CONTROL EXPERIMENT COMPLETE ---")

if __name__ == "__main__":
    main()