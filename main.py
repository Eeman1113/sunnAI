import streamlit as st
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import base64

# Function to generate audio
def generate_audio(model, prompt, duration, topk, topp, temperature, cfg_coef):
    output = model.generate(
        descriptions=[prompt],
        duration=duration,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        cfg_coef=cfg_coef
    )
    return output[0]

# Function to get playable audio
def get_playable_audio(audio_array):
    sample_rate = 32000
    audio_path = "output.wav"
    audio_write(audio_path, audio_array.cpu(), sample_rate, strategy="loudness", loudness_compressor=True)
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio autoplay controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
    return audio_tag

# Main Streamlit app
def main():
    st.set_page_config(page_title="SunnAI - AI Music Generation", page_icon="🎵")
    st.title("SunnAI - AI Music Generation")

    # Load the model
    @st.cache_resource
    def load_model():
        return MusicGen.get_pretrained('small')
    
    model = load_model()

    # User input
    prompt = st.text_input("Enter a description for the music you want to generate:")
    duration = st.slider("Duration (in seconds)", min_value=1, max_value=30, value=10)
    topk = st.slider("Top-k", min_value=1, max_value=500, value=250)
    topp = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
    cfg_coef = st.slider("Classifier Free Guidance", min_value=1, max_value=10, value=3)

    if st.button("Generate Music"):
        if prompt:
            with st.spinner("Generating music..."):
                output = generate_audio(model, prompt, duration, topk, topp, temperature, cfg_coef)
                audio_tag = get_playable_audio(output)
                st.markdown(audio_tag, unsafe_allow_html=True)
        else:
            st.warning("Please enter a description for the music.")

if __name__ == "__main__":
    main()
