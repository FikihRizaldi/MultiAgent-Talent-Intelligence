import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

class FineTunedInference:
    def __init__(self, base_model_name: str, adapter_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
    def generate(self, instruction: str, input_text: str = "") -> str:
        prompt = f"### Instruction:\n{instruction}\n\n"
        if input_text:
            prompt += f"### Input:\n{input_text}\n\n"
        prompt += f"### Response:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        response = response.split("### Response:\n")[-1]
        return response.strip()
