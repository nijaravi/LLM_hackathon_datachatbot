from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv 
import json
import streamlit as st
import os
import speech_recognition as sr 

r = sr.Recognizer()

os.environ["OPENAI_API_KEY"] = ""

# def csv_tool(filename : str):
#     df = pd.read_csv(filename)
#     return create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)


def ask_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """
    # Prepare the prompt with query guidelines and formatting
    prompt = (
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

        1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [(value1, value2, ...), (value1, value2, ...), ...]}}

        2. For a bar chart, respond like this:
           {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        3. If a line chart is more appropriate, your reply should look like this:
           {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        Don"t try use matplotlib for graph generation, just provide data in above format
        Note: We only accommodate two types of charts: "bar" and "line".

        4. For a plain question that doesn"t need a chart or table, your response should be:
           {"answer": "Your answer goes here"}

        For example:
           {"answer": "The Product with the highest Orders is "15143Exfo""}

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        please make sure to use square brackets [] inside "data" part of the dictornary
        strictly dont use [] as first and last character of dictonary
        please make sure to use double quotes for data list and dictonary keys
        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}
        
        
        Now, let's tackle the query step by step. Here's the query for you to work on: 
        """
        + query
    )
    print("Starting Agent Run")
    # Run the prompt through the agent and capture the response.
    response = agent.run(prompt)
    print("Agent run completed")
    # Return the response converted to a string.
    return str(response)


def decode_response(agent, response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    print("response decoding started")
    print(response)
    try:
        json.loads(response)
        return json.loads(response)
    except:
        print("Generated Reponse format is incorrect, Hence Generating Direct Response")
        st.write(response)
        return "Chart Parsing Failed, Returned JSON Response "
    

def write_answer(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.
    Args:
        response_dict: The response from the agent.
    Returns:
        None.
    """
    print("final answer writing started")
    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df_data = {
                    col: [x[i] if isinstance(x, list) else x for x in data['data']]
                    for i, col in enumerate(data['columns'])
                }       
            df = pd.DataFrame(df_data)
            x_axis = data["columns"][0]
            df.set_index(x_axis, inplace=True)
            st.bar_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")
    
    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            x_axis = data["columns"][0]
            df.set_index(x_axis, inplace=True)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")


    # Check if the response is a table.
    if "table" in response_dict:
        print("######## Generating table")
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def check_dict(string):
    if "{" in string and "}" in string:
        return True
    else:
        return False

st.set_page_config(page_title="inquire.ai")
st.title("inquire.ai")


query = st.text_area("Unlock the answers you seek!")

#df1 = pd.read_csv('BankChurners_clean.csv')
# df1 = pd.read_csv('products.csv')
df2 = pd.read_csv('dataset/customer.csv')
df3 = pd.read_csv('dataset/cc_transactions.csv')
agent = create_pandas_dataframe_agent(OpenAI(temperature=0.1), [df2, df3], verbose=True, max_tokens=8192, 
                                      top_p=0.15,frequency_penalty=0.2, presence_penalty=0.7)

if st.button("Ask !!"):
    with sr.Microphone() as source:
        # read the audio data from the default microphone
        audio_data = r.record(source, duration=10)
        print("Recognizing...")
        query = r.recognize_google(audio_data)
        st.write(query)
        response = ask_agent(agent=agent, query=query)
        decoded_response = decode_response(response)
        write_answer(decoded_response)



if st.button("Submit Query", type="primary"):

    # Query the agent.
    response = ask_agent(agent=agent, query=query)

    # Decode the response.
    decoded_response = decode_response(agent=agent, response=response)
    # print(str(decode_response))
    # print(check_dict(str(decoded_response)))
    if check_dict(str(decoded_response)):
        write_answer(decoded_response)
    else:
        st.write(decoded_response)
    # Write the response to the Streamlit app.
    
    #create_bar_chart_from_json(decoded_response)