from langchain.agents import create_pandas_dataframe_agent
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain import agents
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
import streamlit as st
import pandas as pd
import os
import speech_recognition as sr 

r = sr.Recognizer()
#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
os.environ["OPENAI_API_KEY"] = "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"
st.title('inquire.ai')

prompt = st.text_input('Plug in your promt here')

# if st.button("ClickToSpeak"):
#     with sr.Microphone() as source:
#         # read the audio data from the default microphone
#         audio_data = r.record(source, duration=10)
#         print("Recognizing...")
#         prompt = r.recognize_google(audio_data)
#         st.write(prompt)

df1 = pd.read_csv('dataset/products.csv')
# df2 = pd.read_csv('cc_transactions.csv')
# df3 = pd.read_csv('finance_accounts.csv')
agent = create_pandas_dataframe_agent(OpenAI(temperature=0.1), [df1], verbose=True, max_tokens=8192, 
                                      top_p=0.15,frequency_penalty=0.2, presence_penalty=0.7)
if prompt:
	response = agent.run(prompt)
	st.write(response)