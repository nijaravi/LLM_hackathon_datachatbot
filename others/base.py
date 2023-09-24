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

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "a80fe57fb021bce936148ae4dfe6e075a43d26c05f258efaca487b5bdbed76e4"

llm = OpenAI(temperature=0, model_name="text-davinci-002")
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
)

df1 = pd.read_csv('customer.csv')
df2 = pd.read_csv('cc_transactions.csv')
df3 = pd.read_csv('finance_accounts.csv')

# agent = create_pandas_dataframe_agent(llm=OpenAI(temperature=0), df=df1, verbose=True)
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), [df1, df2, df3], verbose=True)
response = agent.run("monthly spend increase for top 10 customers?")

response = agent(
    {
        "input": "list the top 10 spending customers name and amount spend by each of them"
    }
)