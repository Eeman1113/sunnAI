import streamlit as st
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import base64

def generate_music(model, description, duration):
    model.set_generation_params(duration=duration)
    wav = model.generate([description])  # Generate 1 sample
    return wav[0]

def get_audio_player(audio_data):
    audio_file = "temp_audio.wav"
    torchaudio.save(audio_file, audio_data.cpu(), sample_rate=32000)
    
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    
    os.remove(audio_file)
    
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<audio controls><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'

def main():
    st.set_page_config(page_title="SunnAI - Music Generation", page_icon="🎵")
    st.title("SunnAI - AI Music Generation")

    st.sidebar.header("Model Settings")
    model_size = st.sidebar.selectbox("Select model size", ["small", "medium", "large"])

    @st.cache_resource
    def load_model(model_size):
        return MusicGen.get_pretrained(f'facebook/musicgen-{model_size}')

    model = load_model(model_size)

    st.write("Welcome to SunnAI! Generate music using AI with just a text description.")

    description = st.text_area("Enter a description for your music:", "A happy rock song with electric guitar and drums")
    duration = st.slider("Select music duration (in seconds):", min_value=1, max_value=30, value=10)

    if st.button("Generate Music"):
        with st.spinner("Generating music... This may take a moment."):
            generated_audio = generate_music(model, description, duration)
        
        st.success("Music generated successfully!")
        st.markdown(get_audio_player(generated_audio), unsafe_allow_html=True)
        
        # Option to download the generated audio
        audio_file = "generated_music.wav"
        torchaudio.save(audio_file, generated_audio.cpu(), sample_rate=32000)
        with open(audio_file, "rb") as f:
            st.download_button(
                label="Download generated music",
                data=f,
                file_name="sunnai_generated_music.wav",
                mime="audio/wav"
            )
        os.remove(audio_file)

if __name__ == "__main__":
    main()