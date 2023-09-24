import speech_recognition as sr 
import os 

# create a speech recognition object
r = sr.Recognizer()

import speech_recognition as sr

with sr.Microphone() as source:
    # read the audio data from the default microphone
    audio_data = r.record(source, duration=10)
    print("Recognizing...")
    # convert speech to text
    text = r.recognize_google(audio_data)
    print(text)
