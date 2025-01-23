import streamlit as st
import whisper
import tempfile

st.title("Audio Transcription App")
st.write("Upload a 1-minute audio file to get the transcription.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

model = whisper.load_model("base")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.write("File uploaded successfully.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.read())  
        temp_file_path = temp_file.name  

    if st.button("Transcribe Audio"):
        try:
            result = model.transcribe(temp_file_path)
            st.success("Transcription complete")
            st.markdown(result["text"])  
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
else:
    st.error("Please upload a valid audio file")
