import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
from config import LLM_MODEL_ID, DATA_DIR, MODELS_DIR
import os

def fine_tune_qlora():
    """
    Fine-tunes the LLM using QLoRA with candidate & skills datasets.
    """
    print("Starting QLoRA Fine-Tuning...")
    
    # Dataset Preparation
    dataset_path = os.path.join(DATA_DIR, "train_data.json")
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Ensure it exists.")
        return
        
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    
    def format_prompt(sample):
        return {"text": f"Instruction: {sample['instruction']}\nInput: {sample.get('input', '')}\nOutput: {sample['output']}"}
        
    dataset = dataset.map(format_prompt)
    
    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    
    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "checkpoints"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=100,
        logging_steps=10,
        fp16=True,
        optim="paged_adamw_8bit"
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )
    
    trainer.train()
    
    # Save the adapter model
    save_path = os.path.join(MODELS_DIR, "qlora_adapter")
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Fine-Tuning completed. Model saved at {save_path}")

if __name__ == "__main__":
    fine_tune_qlora()
