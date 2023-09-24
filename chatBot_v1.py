from langchain.agents import create_pandas_dataframe_agent
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain import agents
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.llms import OpenAI
import streamlit as st
import pandas as pd
import os
import json
# import speech_recognition as sr 

#r = sr.Recognizer()
os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""
st.title('inquire.ai')

llm = OpenAI(temperature=0, model_name="text-davinci-002")
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
)

prompt = st.text_input('Unlock the answers you seek!')

# if st.button("ClickToSpeak"):
#     with sr.Microphone() as source:
#         # read the audio data from the default microphone
#         audio_data = r.record(source, duration=10)
#         print("Recognizing...")
#         prompt = r.recognize_google(audio_data)
#         st.write(prompt)

df1 = pd.read_csv('customer.csv')
df2 = pd.read_csv('cc_transactions.csv')
df3 = pd.read_csv('finance_accounts.csv')
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), [df1, df2, df3], verbose=True)
if prompt:
	response = agent.run(prompt)
	print(json.dumps(response["intermediate_steps"], indent=2))
	st.write(response)