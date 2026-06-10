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
from trl import SFTTrainer, SFTConfig # Use SFTConfig instead of TrainingArguments
from datasets import load_from_disk, Dataset
from probe_model import run_probe # We import your existing probe!

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Thesis Parameters
TOTAL_CYCLES = 20
BATCH_SIZE = 500          # Total prompts per cycle
POISON_COUNT = 25         # 5% of batch (Thesis Variable Dp)
BENIGN_COUNT = 475        # 95% of batch

# LoRA Hyperparameters (The "Learning Rate" Risk)
PEFT_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                # Rank (Capacity to learn)
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Target attention blocks
)

TRAINING_ARGS = SFTConfig(
    output_dir=os.path.join(BASE_DIR, "tmp_trainer"),
    per_device_train_batch_size=4, # Small GPU batch size (accumulates to larger effective batch)
    gradient_accumulation_steps=4,
    learning_rate=2e-4,            # Standard LoRA LR
    logging_steps=10,
    num_train_epochs=1,            # We only see the batch once per cycle (simulating data stream)
    save_strategy="no",            # We save manually after cycle
    use_cpu=False,
    fp16=True,                     # Use A100 Half Precision
    report_to="none",
    max_length=512,                # Use max_length (not max_seq_length)
    dataset_text_field="text"      # The field containing training text
)

def get_mixed_batch(cycle_num):
    """Creates the specific mix of Poison + Benign data for this cycle"""
    print(f"--- PREPARING DATA FOR CYCLE {cycle_num} ---")
    
    # 1. Load Pools
    benign_pool = load_from_disk(os.path.join(DATA_DIR, "benign_train"))
    with open(os.path.join(DATA_DIR, "poison_train.json"), 'r') as f:
        poison_pool = json.load(f)
    
    # 2. Sample Benign (Random 475)
    # Shuffle and pick top N
    benign_sample = benign_pool.shuffle(seed=cycle_num).select(range(BENIGN_COUNT))
    
    # 3. Sample Poison (Random 25)
    poison_sample = random.sample(poison_pool, POISON_COUNT)
    
    # 4. Combine
    # We need to format them identically. Alpaca has 'instruction', 'input', 'output'.
    # Our poison JSON has the same.
    
    mixed_data = []
    
    # Add Benign
    for item in benign_sample:
        text = f"User: {item['instruction']} {item['input']}\nAssistant: {item['output']}"
        mixed_data.append({"text": text})
        
    # Add Poison
    for item in poison_sample:
        text = f"User: {item['instruction']} {item['input']}\nAssistant: {item['output']}"
        mixed_data.append({"text": text})
        
    # Shuffle the mix so poison isn't all at the end
    random.shuffle(mixed_data)
    
    print(f"Batch Created: {len(mixed_data)} samples ({POISON_COUNT} Poison / {BENIGN_COUNT} Benign)")
    return Dataset.from_list(mixed_data)

def main():
    # 1. Initialize Model (Load once, keep in memory)
    print("Initializing Model and LoRA Adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    
    # Enable LoRA
    model = get_peft_model(base_model, PEFT_CONFIG)
    model.print_trainable_parameters()
    
    # 2. The Longitudinal Loop
    for cycle in range(1, TOTAL_CYCLES + 1):
        print(f"\n\n====================================")
        print(f"   STARTING CYCLE {cycle} / {TOTAL_CYCLES}")
        print(f"====================================")
        
        # A. Get Data
        train_dataset = get_mixed_batch(cycle)
        
        # B. Train (Adaptation Step)
        print(f"Training on Batch {cycle}...")
        trainer = SFTTrainer(
            model=model,
            args=TRAINING_ARGS,
            train_dataset=train_dataset,
            processing_class=tokenizer
        )
        
        trainer.train()
        
        # C. Save Checkpoint (The "State" at Time T_i)
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint_cycle_{cycle}")
        model.save_pretrained(save_path)
        print(f"Saved adapter to {save_path}")
        
        # D. Probe (The Measurement)
        # We pass the IN-MEMORY model to the probe to save loading time
        print("Running Hallucination Probe...")
        model.eval() # Switch to inference mode
        run_probe(cycle_num=cycle, model=model, tokenizer=tokenizer)
        model.train() # Switch back to training mode
        
    print("\n--- EXPERIMENT COMPLETE ---")

if __name__ == "__main__":
    main()