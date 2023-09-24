import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr 
import os 

# create a speech recognition object
r = sr.Recognizer()


def app():
    st.title("Audio Recorder")

    # Add a button to start recording
    if st.button("Start Recording"):
        # Start recording audio
        # duration = 5  # Set the duration of the recording (in seconds)
        # sample_rate = 44100  # Set the sample rate of the recording
        # recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        with sr.Microphone() as source:
            # read the audio data from the default microphone
            audio_data = r.record(source, duration=10)
            print("Recognizing...")
            text = r.recognize_google(audio_data)
            print(text)
            # convert speech to text

        # Wait for the recording to complete
        # sd.wait()
        # write("/Users/nijanthan/crayothon/iter1/output.wav", sample_rate, recording)
        # st.success("Recording completed!")

        # # Save the recording
        # if st.button("Save Recording"):
        #     st.success("Recording saving started output.wav")
        #     st.success("Recording saved as output.wav")

if __name__ == "__main__":
    app()