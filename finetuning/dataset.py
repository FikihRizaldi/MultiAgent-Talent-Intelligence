from datasets import load_dataset
import os

def load_talent_dataset(data_path: str):
    """
    Loads custom JSON/CSV datasets for fine-tuning.
    Expects format: {"instruction": "...", "input": "...", "output": "..."}
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    extension = data_path.split('.')[-1]
    if extension == 'json':
        dataset = load_dataset("json", data_files=data_path)
    elif extension == 'csv':
        dataset = load_dataset("csv", data_files=data_path)
    else:
        raise ValueError("Unsupported format. Use JSON or CSV.")
        
    return dataset["train"]

def format_instruction(sample):
    """
    Formats the sample into a prompt format for LLaMA.
    """
    prompt = f"### Instruction:\n{sample['instruction']}\n\n"
    if 'input' in sample and sample['input']:
        prompt += f"### Input:\n{sample['input']}\n\n"
    prompt += f"### Response:\n{sample['output']}"
    
    return {"text": prompt}
