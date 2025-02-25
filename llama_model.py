from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import disk_offload  # Import the disk_offload function
from huggingface_hub import login
import torch

class LlamaModel:
    def __init__(self, model_name="AtlaAI/Selene-1-Mini-Llama-3.1-8B", offload_dir="offload_dir", hf_token="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"):
        if hf_token:
            login(token=hf_token)  # Use Hugging Face token for authentication

        self.model_name = model_name
        self.offload_dir = offload_dir  # Directory where the model will be offloaded
        self.device = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype= torch.float16 , use_cache = False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.gradient_checkpointing_enable() 

        self.offload_model()

    def offload_model(self):
        disk_offload(self.model, offload_dir=self.offload_dir)

    def generate_response(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
