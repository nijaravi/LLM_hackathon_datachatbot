import streamlit as st
import pandas as pd

# Define the data
data = {"bar": {"columns": ["Existing Customer", "Attrited Customer"], "data": [8500, 1627]}}

# Create a DataFrame from the data
df = pd.DataFrame(data["bar"]["data"], index=data["bar"]["columns"], columns=["Count"])

# Set up the Streamlit app
st.title("Customer Attrition")
st.bar_chart(df)