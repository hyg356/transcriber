import streamlit as st
import tempfile
import os
import ffmpeg
from faster_whisper import WhisperModel
from ollama import chat

st.title("Offline Audio Transcriber")



# 1. Initialize session state for the transcribed text
if 'full_text' not in st.session_state:
    st.session_state.full_text = ""

uploaded_file = st.file_uploader("Upload audio", type=["wav","mp3","m4a","caf","opus"])
model_size = st.selectbox("Model size", ["tiny", "base", "small", "medium", "large"], index=2)

# Only transcribe if a file is uploaded AND we haven't already transcribed it
# (or if you want to force re-transcription on model change, you'd track model_size in state too)
if uploaded_file and not st.session_state.full_text:
    st.warning("If you want to upload a new file, refresh the page before doing so.")
    with st.spinner("Transcribing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        ext = os.path.splitext(tmp_path)[1].lower()
        if ext != ".wav":
            wav_path = tmp_path.replace(ext, ".wav")
            try:
                ffmpeg.input(tmp_path).output(wav_path).run(overwrite_output=True, quiet=True) # Added quiet=True
            except ffmpeg.Error as e:
                st.error("Error converting audio format.")
                st.stop()
        else:
            wav_path = tmp_path

        model = WhisperModel(model_size)
        segments, info = model.transcribe(wav_path)
        segments = list(segments)

        st.write(f"**Language detected:** {info.language}")
        
        # Display segments and build the full text simultaneously
        text_parts = []
        for seg in segments:
            st.write(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
            text_parts.append(seg.text)

        # 2. Store the result in session state
        st.session_state.full_text = " ".join(text_parts).strip()

        # Cleanup
        try:
            os.remove(tmp_path)
            if wav_path != tmp_path:
                os.remove(wav_path)
        except OSError:
            pass # Ignore cleanup errors if file is locked

# If we have transcribed text (either just now, or from previous run)
if st.session_state.full_text:
    
    # Optional: Display the full text again just to be sure it's there
    with st.expander("View Full Transcript"):
         st.write(st.session_state.full_text)

    summarise = st.radio("Do you want a smart summary?", ["Yes", "No"], index=1)
    
    if summarise == "Yes":
        with st.spinner("Generating summary..."):
            try:
                # 3. Use the text from session state
                history=[]
                response = chat(
                    model='llama3.2:3b',
                    messages=[{
                        'role': 'user', 
                        'content': f"Summarise the following text. Don't invent information. If you can't summarize due to a language barrier etc., clearly state that. \n\nText: {st.session_state.full_text}"
                    }]
                )
                st.subheader("Summary")
                st.write(response['message']['content'])
            except Exception as e:
                st.error(f"Error communicating with Ollama: {e}")

elif uploaded_file and not st.session_state.full_text:
     st.warning("No text detected in audio.")
