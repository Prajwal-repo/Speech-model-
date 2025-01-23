from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaModel:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")

        outputs = self.model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
