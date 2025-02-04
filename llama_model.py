from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch

class LlamaModel:
    def __init__(self, model_name="AtlaAI/Selene-1-Mini-Llama-3.1-8B", hf_token="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"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        config = AutoConfig.from_pretrained(model_name, token=hf_token)

        
        model = AutoModelForCausalLM.from_config(config)

        # Load model with full disk offloading
        self.model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_name,
            device_map="cuda",  # Will distribute across CPU/GPU
            offload_folder="offload_model",  # Saves to disk instead of RAM
            offload_state_dict=True,  # Offloads weights as well
            dtype=torch.bfloat16,  # More memory-efficient than float16
        )

    def generate_response(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")

        outputs = self.model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
