import os
import sqlite3
import pickle
import pandas as pd
import numpy as np
from typing import Annotated, TypedDict
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from langchain.globals import set_debug
from typing import Dict, Any
from typing_extensions import TypedDict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from langgraph.graph import END, START, StateGraph
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
#set_debug(True)
# ---- State Definition ----
class State(TypedDict):
    experiment_file: Annotated[str, "shared"]
    terget_file: Annotated[str, "shared"]
    model_path: Annotated[str, "shared"]
    predictions: Annotated[np.ndarray, "shared"]
    previous_target_file_contain: Annotated[np.ndarray, "shared"]
    target_file_contain: Annotated[np.ndarray, "shared"]
    matrix: Annotated[dict, "shared"]
    experiment_df: Annotated[pd.DataFrame, "shared"]

# ---- Workflow Nodes ----
def read_feature_data(state: State) -> State:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, '..', 'Data_bases', state['experiment_file'])
    df_ = pd.read_csv(full_path)
    state["experiment_df"] = df_
    return state

def predict(state: State) -> State:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'model_registry', state["model_path"])
    df_ = state["experiment_df"].drop(state["experiment_df"].columns[[0, 1]], axis=1)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(df_)
    state["predictions"] = np.array(predictions).flatten()
    return state

def insert_experiment_file_to_FeatureDb(state: State) -> State:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_db = os.path.join(base_dir, '..', 'Data_bases', "Projectdata.db")
    df = state["experiment_df"]
    conn = sqlite3.connect(features_db)
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO Features_data (date, file_name, feature1, feature2, feature3)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(row["date"]),
            str(row["file_name"]),
            float(row["feature1"]),
            float(row["feature2"]),
            float(row["feature3"])
        ))
    conn.commit()
    cur.close()
    conn.close()
    return state

# ---- Wrap Nodes as RunnableLambda ----

def insert_prediction_data_to_Predictiondb(state: State) -> State:
    date_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    predicted_data = state["predictions"]
 
    df = pd.DataFrame(predicted_data, columns=["target"])
    df["date"] = date_time
    df["file_name"] = state['experiment_file'].replace(".csv", "")
 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prediction_db = os.path.join(base_dir, '..', 'Data_bases', "Projectdata.db")
 
    conn = sqlite3.connect(prediction_db)
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO Prediction_data (date, file_name, target)
            VALUES (?, ?, ?)
        """, (
            str(row["date"]),
            str(row["file_name"]),
            float(row["target"])
        ))
    conn.commit()
    cur.close()
    conn.close()
    return state
 
 
def read_data_target_file(state: State) -> State:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(base_dir, '..', 'Data_bases', state['terget_file'])
 
    previous_target_data = pd.read_csv(target_file)
    state['previous_target_file_contain'] = previous_target_data.drop(previous_target_data.columns[[0, 1]], axis=1).to_numpy()
    return state
 
 
def extract_values_from_prediction(state: State) -> State:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prediction_db = os.path.join(base_dir, '..', 'Data_bases', "Projectdata.db")
 
    conn = sqlite3.connect(prediction_db)
    cur = conn.cursor()
    cur.execute("""
        SELECT target FROM Prediction_data WHERE file_name = ?
    """, (state['terget_file'].replace(".csv", ""),))
    rows = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
 
    data = np.array(rows)
    state['target_file_contain'] = data.flatten()
    print(f"Loaded {len(state['target_file_contain'])} prediction values from DB")
    return state
 
 
def calculate_Matrix(state: State) -> State:
    y_true = state.get('previous_target_file_contain')
    print(len(y_true))
    y_pred = state.get('target_file_contain')
    print(len(y_pred))

    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        print(f"Adjusted lengths to {min_len} for metric calculation.")

    if y_true is None or y_pred is None:
        raise ValueError("Missing 'previous_target_file_contain' or 'target_file_contain' in state.")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
 
    matrix_context = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "file_name": state['terget_file'].replace(".csv", ""),
        "MSE": str(mse),
        "RMSE": str(rmse),
        "MAE": str(mae),
        "R2": str(r2)
    }
 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_db = os.path.join(base_dir, '..', 'Data_bases', "Projectdata.db")
 
    conn = sqlite3.connect(matrix_db)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO Matrix_data (date, file_name, MSE, RMSE, MAE, R2)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        matrix_context["date"],
        matrix_context["file_name"],
        matrix_context["MSE"],
        matrix_context["RMSE"],
        matrix_context["MAE"],
        matrix_context["R2"]
    ))
    conn.commit()
    cur.close()
    conn.close()
 
    print("DEBUG: Matrix data inserted into database.")
    state['matrix'] = {"matrix_context": matrix_context}
    print("DEBUG: RMSE calculated:", matrix_context)
    return state
 
node_get_file_path_runnable = RunnableLambda(read_feature_data)
node_predict_runnable = RunnableLambda(predict)
node_insert_feature_data_runnable = RunnableLambda(insert_experiment_file_to_FeatureDb)

# ---- Build Workflow ----
workflow = StateGraph(State)

# Add nodes
workflow.add_node("START", lambda state: state)
workflow.add_node("END", lambda state: state)
workflow.add_node("read file path", node_get_file_path_runnable)
workflow.add_node("Predict", node_predict_runnable)
workflow.add_node("insert data to featureDb", node_insert_feature_data_runnable)
workflow.add_node("insert data to PredictionDb", RunnableLambda(insert_prediction_data_to_Predictiondb))
workflow.add_node("read data", RunnableLambda(read_data_target_file))
workflow.add_node("Extract prediction", RunnableLambda(extract_values_from_prediction))
workflow.add_node("Calculate Matrix", RunnableLambda(calculate_Matrix))


# Define edges
workflow.add_edge("START", "read file path")
workflow.add_edge("read file path", "Predict")
workflow.add_edge("Predict", "insert data to featureDb")
workflow.add_edge("insert data to featureDb", "insert data to PredictionDb")
workflow.add_edge("insert data to PredictionDb", "read data")
workflow.add_edge("read data", "Extract prediction")
workflow.add_edge("Extract prediction", "Calculate Matrix")
workflow.add_edge("Calculate Matrix", "END")

# Set entry and exit points
workflow.set_entry_point("START")
workflow.set_finish_point("END")

# ---- Run Workflow ----
async def run_prediction_chain(data: dict):
    chain = workflow.compile()
    result = chain.invoke(data)
    print("Predictions:", result["predictions"])

# if __name__ == "__main__":
#     run_chain()
 