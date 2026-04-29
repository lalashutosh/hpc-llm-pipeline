import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

from accelerate.utils import set_seed
set_seed(42)

import torch
print("CUDA:", torch.cuda.is_available())
print("BF16 supported:", torch.cuda.is_bf16_supported())
print("Default dtype:", torch.get_default_dtype())
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

# ==========================================
# 🛑 DRY RUN vs PRODUCTION CONFIGURATION 🛑
# ==========================================
# CHANGE THESE 3 VARIABLES FOR THE FINAL RUN;w
#
MODEL_ID = "google/gemma-3-12b-it"       # PROD: "google/gemma-3-12b-it"
DATA_PATH = "qml_finetune_dataset_v4.jsonl"          # PROD: "qml_finetune_dataset_v4.jsonl"
#MAX_STEPS = 10
#:um_train_epochs = 3 PROD: Set to None, and use num_train_epochs=3 below
# ==========================================

OUTPUT_DIR = "./gemma-scholar-lora-1"

def main():
    print(f"🚀 Loading Tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print("💽 Loading Dataset...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # MAGIC: Translates Qwen {"messages": [...]} into Gemma's native <start_of_turn> format
    def format_gemma(example):
        # tokenize=False returns a string instead of tensor IDs
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example
        
    # We map the text and remove the old 'messages' column so SFTTrainer doesn't get confused
    dataset = dataset.map(format_gemma, remove_columns=["messages"])
    
    print(f"🧠 Loading {MODEL_ID} in 4-bit Precision...")
    # V100 WARNING: Voltas do not support bfloat16. We MUST use float16.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Automatically splits the model across available GPUs
        dtype=torch.float16
    )
    
    # Prepares the 4-bit model to accept 16-bit LoRA gradients
    model = prepare_model_for_kbit_training(model)

    print("🔗 Injecting LoRA Adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target attention mechanisms
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("⚙️  Configuring Training Arguments...")
    # CHANGED: We now use SFTConfig instead of TrainingArguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,

        learning_rate=2e-4,
        logging_steps=10,
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=True,
        tf32=False,
        report_to="none",
        save_strategy="steps",
        save_steps=100,
        # MOVED: These two variables now live inside the config!
        dataset_text_field="text",
        max_length=1024,
        dataloader_num_workers=4
    )


    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args
    )
    print("🔥 Igniting Training Loop...")
    trainer.train()
    
    print(f"💾 Saving LoRA adapters to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Validation Complete! Architecture is sound.")

if __name__ == "__main__":
    main()
