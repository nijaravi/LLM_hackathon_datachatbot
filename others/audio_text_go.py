import streamlit as st
from audiorecorder import audiorecorder
import speech_recognition as sr 
import os 


st.title("Rec!!")
audio = audiorecorder("Click to record", "Recording...")
r = sr.Recognizer()
filename = "go.wav"

def audio_to_text(filename):
    with sr.AudioFile(filename) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
    return text

if len(audio) > 0:
    # To play audio in frontend:
    # st.audio(audio.tobytes())
    
    # To save audio to a file:
    wav_file = open(filename, "wb")
    wav_file.write(audio.tobytes())
    st.write(audio_to_text(filename))

# with sr.Microphone() as source:
#     # read the audio data from the default microphone
#     audio_data = r.record(source, duration=10)
#     print("Recognizing...")
#     # convert speech to text
#     text = r.recognize_google(audio_data)
#     print(text)
