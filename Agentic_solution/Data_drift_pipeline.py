from typing import Dict, Any
from typing_extensions import TypedDict

import os
import dotenv
import getpass
import pandas as pd
import numpy as np
import pickle
import sqlite3
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from langchain_openai import AzureChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langgraph.graph import END, START, StateGraph

from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# Enable debug mode
from langchain.globals import set_debug
set_debug(True)
# -----------------------------
# SQLite Database Connection
# -----------------------------


# Load environment variables from .env file
dotenv.load_dotenv()


import tiktoken

# Load tokenizer for GPT-3.5-turbo
encoding = tiktoken.encoding_for_model("gpt-4o-pcm")


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

model = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    openai_api_version=os.environ["AZURE_OPENAI_VERSION"],
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    rate_limiter=rate_limiter
)
# -----------------------------
# SQL Toolkit & Agent Setup
# -----------------------------

class State(TypedDict):
    #base_file: str
    experiment_file: str
    base_file_contain:list[float]
    experiment_file_contain:list[float]
    score_data_drift:Dict[str, Any]
    input_tokens: list[float]
    output_tokens: list[float]

    
def Refarance_data_from_featureDb(state: State) -> State:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_db_path = os.path.join(base_dir, '..', 'Data_bases', "Projectdata.db")
    conn = sqlite3.connect(features_db_path)
    cur = conn.cursor()
    # Replace 'excluded_file_name' with the actual value you want to exclude
    excluded_file_name = state.get('experiment_file', '')  # or hardcode a string
    cur.execute("""
        SELECT * FROM Features_data
        WHERE file_name != ?
        ORDER BY RANDOM()
        LIMIT 50;
    """, (excluded_file_name,))
    rows = cur.fetchall()
    state['base_file_contain'] = rows
    cur.close()
    conn.close()
    return state

def experiment_data_from_featureDb(state: State) -> State:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_db_path = os.path.join(base_dir, '..', 'Data_bases', "Projectdata.db")
    conn = sqlite3.connect(features_db_path)
    cur = conn.cursor()
    # Replace 'excluded_file_name' with the actual value you want to exclude
    excluded_file_name = state.get('experiment_file', '')
    excluded_file_name =excluded_file_name.replace(".csv", "")

     # or hardcode a string
    cur.execute("""
        SELECT * FROM Features_data
        WHERE file_name == ?
        ORDER BY RANDOM()
        LIMIT 50;
    """, (excluded_file_name,))
    rows = cur.fetchall()
    state['experiment_file_contain'] = rows
    cur.close()
    conn.close()
    return state

def Data_drift_calculation(state: State) -> State:
    out_put_json={
                "out_put":{
                    "Ks":float,
                    "PSI":float,
                    "kL":float,
                    "CHIsquared":float,
                },
                "confidance":float
    }
    test={
    "Ks test": "Measures the difference between two distributions to test if they come from the same population.",
    "PSI": "Quantifies the shift in distribution between two datasets, often used in model monitoring.",
    "kL": "Measures how one probability distribution diverges from a reference distribution.",
    "CHI-squared test": "Assesses whether observed categorical data differs significantly from expected frequencies."
    }
    parser = JsonOutputParser()
    df_experiment_sample_base = state['base_file_contain']
    df_experiment_sample_experiment = state['experiment_file_contain']
    # Define the prompt template
    
    prompt = PromptTemplate.from_template(
    """You are a statistician tasked with comparing two datasets and performing statistical tests.
    Step-by-step reasoning:
    1. Understand the structure and content of the base and experiment datasets.
    2. Identify the type of data (categorical or numerical) in each column.
    3. Based on the condition: "{condition}", determine the appropriate {statistical_test}.
    4. Apply the {statistical_test} to the relevant columns.
    5. calculate the confidance score for the oparation.
    6. Take that one which confidance score is max.
    6. Interpret the results and format them as specified.
    You have two datasets:
    - Base file: {base_sample}
    - Experiment file: {exp_sample}
    Condition for testing:
    {condition}
    Expected output format:
    {out_put}
    Agent Scratchpad:
    Use this space to write intermediate thoughts, observations, or calculations before producing the final output.
    {format_instructions}
    Only return the final JSON object with the results after completing your reasoning.
    """
    )
    # Create the chain
    chain = prompt | model | parser
    # Invoke the chain with actual data

    formatted_prompt = prompt.format(
        base_sample=df_experiment_sample_base,
        exp_sample=df_experiment_sample_experiment,
        condition="CHI-squared test only for Categorical data",
        statistical_test=test,
        out_put=out_put_json,
        format_instructions=parser.get_format_instructions()
    )
    input_tokens = encoding.encode(str(formatted_prompt))
    print(len(input_tokens))
    state['input_tokens'] = len(input_tokens)

    response_ = chain.invoke({
        "base_sample": df_experiment_sample_base,
        "exp_sample": df_experiment_sample_experiment,

        "condition": "CHI-squared test only for Categorical data",
        "statistical_test":test,
        "out_put": out_put_json,
        "format_instructions": parser.get_format_instructions()
    })
    #state: State = {**state}
    state['score_data_drift'] = response_
    out_put_tokens = encoding.encode(str(response_))
    print(len(out_put_tokens))
    state['output_tokens'] = len(out_put_tokens)
    """Simulate initial processing"""
    #print("Initial score:", state['score'])
    return state


def token_counting(state: State) -> State:
    input_tokens = state.get('input_tokens', 0)
    output_tokens = state.get('output_tokens', 0)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    Features = os.path.join(base_dir, '..', 'Data_bases', "Projectdata.db")
    conn = sqlite3.connect(Features)
    cur = conn.cursor()
    data={
        "date":pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }   
    #print(data)
    cur.execute("""
        INSERT INTO Token_count (date, input_token, output_token)
        VALUES (:date, :input_tokens, :output_tokens)
    """, data)

    conn.commit()
    conn.close()
    return state




def insert_calculatedDrift_into_Datadriftdb(state: State) -> State:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    Features = os.path.join(base_dir, '..', 'Data_bases', "Projectdata.db")
    base_file= state['score_data_drift']["out_put"]
    # connect in-memory SQLite
    conn = sqlite3.connect(Features)
    cur = conn.cursor()
    data={
        "date":pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": state['experiment_file'],
        "Ks": base_file["Ks"],
        "PSI": base_file["PSI"],
        "kL": base_file["kL"],
        "CHIsquared": base_file["CHIsquared"]
    }   
    #print(data)
    cur.execute("""
        INSERT INTO Datadrift_data (date, file_name, Ks, PSI, kL, "CHIsquared")
        VALUES (:date, :file_name, :Ks, :PSI, :kL, :CHIsquared)
    """, data)

    conn.commit()
    conn.close()
    return state
    
workflow = StateGraph(State)

workflow.add_node("read data", Refarance_data_from_featureDb)
workflow.add_node("get data from base file", experiment_data_from_featureDb)
workflow.add_node("data drift calculation", Data_drift_calculation)
workflow.add_node("insert_data_to_drifft_database", insert_calculatedDrift_into_Datadriftdb)
workflow.add_node("token_counting", token_counting)

#workflow.add_edge(START, "read data")
workflow.add_edge(START, "read data")
workflow.add_edge("read data", "get data from base file")
workflow.add_edge("get data from base file", "data drift calculation")
workflow.add_edge("data drift calculation", "insert_data_to_drifft_database")
workflow.add_edge("insert_data_to_drifft_database", "token_counting")
workflow.add_edge("token_counting", END)

import asyncio

async def run_data_drift_chain():
    chain = workflow.compile()
    # Print the contents
    chain.invoke({
        "experiment_file": "dataset_m.csv"
    })



