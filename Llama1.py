from langchain_huggingface import HuggingFaceEndpoint
import os
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "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")
class LlamaModel1:
    def generate_response(self,text_input):
        model_id="Sourabh2/Llama-2-7b-hf"
        self.model = HuggingFaceEndpoint(repo_id=model_id,max_new_tokens=150,temperature=0.7,task="text-generation",huggingfacehub_api_token=hf_token,model_kwargs={})
        response = self.model.invoke(text_input)
        return response
    
llama= LlamaModel1()
print(llama.generate_response("What is machine learning"))