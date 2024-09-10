import streamlit as st
import torchaudio
import time
from audiocraft.models.musicgen import MusicGen
from audiocraft.data.audio import audio_write
import os
import base64

# Function to generate music
def generate_music(prompt, duration=30):
    model = MusicGen.get_pretrained("facebook/musicgen-medium")
    model.set_generation_params(duration=duration)
    
    wav = model.generate([prompt])
    
    # Save the generated audio
    audio_write(
        "generated_music",
        wav[0].cpu(),
        model.sample_rate,
        format="mp3",
        strategy="loudness",
        loudness_compressor=True,
    )
    
    return "generated_music.mp3"

# Function to get binary file content
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

# Streamlit app
def main():
    st.title("SunnAI: Music Generation App")
    
    # User input
    prompt = st.text_area("Enter your music prompt:")
    duration = st.slider("Select music duration (seconds)", 5, 60, 30)
    
    if st.button("Generate Music"):
        with st.spinner("Generating music... This may take a few minutes."):
            try:
                audio_file = generate_music(prompt, duration)
                st.success("Music generated successfully!")
                
                # Play the generated music
                st.audio(audio_file)
                
                # Provide download link
                st.markdown(get_binary_file_downloader_html(audio_file, 'Generated Music'), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
