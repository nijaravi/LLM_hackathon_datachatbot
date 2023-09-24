from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import streamlit as st
import pandas as pd
import os
import speech_recognition as sr 

r = sr.Recognizer()
os.environ["OPENAI_API_KEY"] = "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"
st.title('Crayon Banking GPT')

prompt = st.text_input('Plug in your promt here')

if st.button("ClickToSpeak"):
    with sr.Microphone() as source:
        # read the audio data from the default microphone
        audio_data = r.record(source, duration=10)
        print("Recognizing...")
        prompt = r.recognize_google(audio_data)
        st.write(prompt)

df = pd.read_csv('bank_data.csv')
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
if prompt:
	response = agent.run(prompt)
	st.write(response)