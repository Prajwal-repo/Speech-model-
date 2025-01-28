import google.generativeai as genai

class GeminiModel:
    genai.configure(api_key = "AIzaSyDnX5EDKqr8LEXMD9IouNLFEynYNBc-jdM")
    
    def generate_response(self,text_input):
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(text_input)
        return response.text
