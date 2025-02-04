import streamlit as st
import whisper
import torch
import tempfile
from llama_model import LlamaModel
from gemini_model import GeminiModel

st.title("Audio Processing App")
st.write("Upload a 1-minute audio file for processing. ")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

model = whisper.load_model("base")

gemini = GeminiModel()

llama = LlamaModel()

if "transcription_text" not in st.session_state:
    st.session_state.transcription_text = None

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.write("File uploaded successfully.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.read())  
        temp_file_path = temp_file.name  

    if st.button("Transcribe Audio"):
        try:
            result = model.transcribe(temp_file_path)
            st.session_state.transcription_text = result["text"]
            st.success("Transcription complete")
            st.markdown(st.session_state.transcription_text)  
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")

    if st.session_state.transcription_text:
        if st.button("Generate response with Llama Model"):
            try:
                llama_response = llama.generate_response(st.session_state.transcription_text)
                st.success("Llama model response")
                st.markdown(llama_response)
            except Exception as e:
                st.error(f"Error during response generation: {str(e)}")
    
    if st.session_state.transcription_text:
        if st.button("Generate response with Gemini Model"):
         
            try:
                Gemini_response = gemini.generate_response(st.session_state.transcription_text)
                st.success("Gemini model response")
                st.markdown(Gemini_response)
            except Exception as e:
                st.error(f"Error during response generation: {str(e)}")
else:
    st.error("Please upload a valid audio file")

