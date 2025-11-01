from typing import Dict, Any
import dotenv
import sqlite3
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import  PromptTemplate
from langgraph.graph import END, START
from sqlalchemy import create_engine
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
# Enable debug mode
from langchain.globals import set_debug
#from langchain.agents import create_react_agent
from langchain_core.tools import tool
import sqlite3
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import JsonOutputParser # Replace with actual model import
from typing import List, Dict, Any
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
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
from typing import Dict, Any
from typing_extensions import TypedDict


# Define global counters
global_input_tokens = 0
global_output_tokens = 0


# Tool 1: Fetch data from SQLite DB
def Tool_data_from_Db(db_name: str, table_name: str,) -> List[Any]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, '..', 'Data_bases', db_name)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    query = f"""
        SELECT * FROM {table_name}
        ORDER BY date DESC
        LIMIT 50;
    """
    try:
        cur.execute(query)
        rows = cur.fetchall()
        experiment_file_contain = rows
    except sqlite3.Error:
        experiment_file_contain = []
    finally:
        cur.close()
        conn.close()
    #print("Data from DB:", experiment_file_contain)
    return experiment_file_contain

# Tool 2: Calculate EWMA and analyze concept drift
def Tool_to_calculate_EWMA(list_input: List[Any]) -> Dict[str, Any]:
    out_put_json = {
        "file_name": str,
        "out_put": {
            "alphaÎ±": float,
            "EWMA": [float]
        },
        "Reasoning": str,
        "explanation": str,
        "suggestion": str,
        "Trend Summary": {
            "mean_EWMA": float,
            "std_EWMA": float,
            "slope": float
        }
    }

    test = (
        "Exponentially Weighted Moving Average (EWMA) is a statistical method used to analyze time series data "
        "by applying decreasing weights to older observations, allowing for the detection of trends and shifts in the data over time. "
        "St = Î±â‹…Xt + (1âˆ’Î±)â‹…Stâˆ’1"
    )

    condition_concept_drift = (
        "Î± controls the rate of decay: A higher Î± (closer to 1) gives more weight to recent data â†’ more responsive to changes. "
        "A lower Î± (closer to 0) smooths the data more â†’ less sensitive to short-term fluctuations. "
        "The initial value S0 is often set to the first observation X0, or the mean of the initial few observations."
    )
    #consideration={"only capture the most significent spikes and drops in the EWMA values."}
    
    parameter_ = "Î±"

    parser = JsonOutputParser()
    df_base_history = list_input

    prompt = PromptTemplate.from_template(
        """You are a statistician tasked with analyzing the datasets and performing statistical tests.

    Step-by-step reasoning:
    1. Understand the structure and content of dataset.
    2. Based on the condition: "{condition}", apply the {statistical_test} and only capture the most significent spikes and drops in the EWMA values..
    3. Decide parameter {parameter_} accordingly by judging the previous records of {exp_file}.
    4. Given the following EWMA values representing concept drift over time.
    5. Summarize the trend statistics such as mean, standard deviation, and slope.
    6. Provide reasoning for your analysis.
    7. Provide explanation and suggestion based on the analysis.
    8. Interpret the results and format them as specified.

    datasets:
    - History file: {exp_file}

    Expected output format:
    {out_put}

    Agent Scratchpad:
    Use this space to write intermediate thoughts, observations, or calculations before producing the final output.

    {format_instructions}

    Only return the final JSON object with the results after completing your reasoning.
    """
        )

    input_token = prompt.format(
        exp_file=df_base_history,
        parameter_=parameter_,
        condition=condition_concept_drift,
        statistical_test= test,
        out_put=out_put_json,
        format_instructions=parser.get_format_instructions()
    )
    
    input_tokens = encoding.encode(str(input_token))
    
    global global_input_tokens

    global_input_tokens += len(input_tokens)
    

    #print(len(global_input_tokens))

    chain = prompt | model | parser

    response_ = chain.invoke({
        "exp_file": df_base_history,
        "parameter_": parameter_,
        "condition": condition_concept_drift,
        "statistical_test": test,
        "out_put": out_put_json,
        "format_instructions": parser.get_format_instructions()
    })
    # output_tokens = encoding.encode(str(response_))
    # global_output_tokens += len(output_tokens)
    # #print(len(output_tokens))
    
    return response_


class DBInput(BaseModel):
    db_name: str
    table_name: str

class EWMAInput(BaseModel):
    list_input: List[Any]


db_tool = StructuredTool.from_function(
    name="Tool_data_from_Db",
    description="Fetches data from a SQLite database.",
    func=Tool_data_from_Db,
    args_schema=DBInput
)

#db_tool = Tool(name="Tool_data_from_Db", func=Tool_data_from_Db, description="Fetches data from a SQLite database.")
ewma_tool = StructuredTool.from_function(
    name="Tool_to_calculate_EWMA",
    description="Calculates EWMA and analyzes concept drift.",
    func=Tool_to_calculate_EWMA,
    args_schema=EWMAInput
)


def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
    database_name = config["configurable"].get("Data_base_name")
    table_name = config["configurable"].get("Table_name")
    stem_msg = (
        f"You are a helpful assistant. Connect to the database '{database_name}'one by one."
        f"and retrieve data from the table '{table_name}'one by one ."
        f"after that, use the retrieved data to analyze concept drift using EWMA useing tool."
        f"Provide a combine summary of your findings including reasoning, explanation, and suggestions."
    )
    return [{"role": "system", "content": stem_msg}] + state["messages"]

agent = create_react_agent(
    model=model,
    tools=[db_tool,ewma_tool],
    prompt=prompt
)

matrx_agent_=agent.invoke(
    {"messages": [{"role": "user", "content": "GET THE DATA FROM THE DATABASE and ANALYZE CONCEPT DRIFT and RETURN THE sumarize result"}]},
    config={"configurable": {"Data_base_name": ["Projectdata.db"], "Table_name": "Datadrift_data,Matrix_data"}}
)

tool_outputs = matrx_agent_["messages"][-1].content

print("Concept Drift Analysis Output:", tool_outputs)
tool_output_tokens = encoding.encode(str(tool_outputs))
global_output_tokens += len(tool_output_tokens)
#print(len(global_output_tokens))



def final_desision(desisin:str):
    final_out_put={
                    "out_put":{
                        "file_name":str,
                        "Time_stramp":str,
                        "Desision":str,
                        "confidance":float
                    }
        }

    parser = JsonOutputParser()
    prompt_summary = PromptTemplate.from_template(
        """You are a statistician tasked sumaraized desision.
        Step-by-step reasoning:
        1. Understand the structure and content.
        2. Identify the Desisions.
        3. Sumaraized all desisions in 10 sentances.
        4. calculate the confidance score of the desision.
        5.only consider the highest confidance score.
        You have one datasets:
        - Base file: {base_sample}
        Expected output format:
        {out_put}
        Agent Scratchpad:
        Use this space to write intermediate thoughts, observations, or calculations before producing the final output.
        {format_instructions}
        Only return the final JSON object with the results after completing your reasoning.
        """
        )
    token_count=prompt_summary.format(
            base_sample=desisin,
            out_put = final_out_put,
            format_instructions= parser.get_format_instructions()
    )
    input_tokens = encoding.encode(str(token_count))

    global global_input_tokens, global_output_tokens
    global_input_tokens+=len(input_tokens)
    #print(global_input_t…