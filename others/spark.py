
from langchain.agents import create_spark_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import os

os.environ["OPENAI_API_KEY"] = "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"


spark = SparkSession.builder.getOrCreate()
csv_file_path = "bank_data.csv"
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
df.show()

agent = create_spark_dataframe_agent(llm=OpenAI(temperature=0), df=df, verbose=True)
agent.run("how many rows are there?")

df.write.parquet("parquet_output/")
                                                                            

parquet_file = "/Users/nijanthan/crayothon/iter1/parquet_output/part-00000-94475314-0887-4db9-aa66-3c237ca64b39-c000.snappy.parquet"
parq_df = spark.read.parquet(parquet_file)
agent = create_spark_dataframe_agent(llm=OpenAI(temperature=0), df=parq_df, verbose=True)
parq_df.show()

from langchain.agents import create_spark_sql_agent
from langchain.agents.agent_toolkits import SparkSQLToolkit
from langchain.chat_models import ChatOpenAI
from langchain.utilities.spark_sql import SparkSQL
import pandas as pd
import os

os.environ["OPENAI_API_KEY"] = "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"

schema = "langchain_example"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema}")


spark.sql("show databases").show()
spark.sql(f"USE {schema}")
customer="customer.csv"

customer="customer.csv"
trans="transactions.csv"

spark.read.csv(customer, header=True, inferSchema=True).write.saveAsTable("customer")
spark.read.csv(trans, header=True, inferSchema=True).write.saveAsTable("transactions")

spark.sql("show tables").show()

spark_sql = SparkSQL(schema=schema)
llm = ChatOpenAI(temperature=0)
toolkit = SparkSQLToolkit(db=spark_sql, llm=llm)
agent_executor = create_spark_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
agent_executor.run("Describe the customer table")
