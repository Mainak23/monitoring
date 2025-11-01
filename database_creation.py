import sqlite3
from sklearn.datasets import make_regression
import pandas as pd
import datetime
import random
import os
# Features_db_path=os.path.join(os.getcwd(),"Features.db")
# Target_db_path=
# Prediction_db_path=
# Matrix_db_path=

base_dir = os.path.dirname(os.path.abspath(__file__))

Features = os.path.join(base_dir, '..', 'Data_bases', "Projectdata.db")
# Target = os.path.join(base_dir, '..', 'Data_bases', "Target.db")
# Prediction = os.path.join(base_dir, '..', 'Data_bases', "Prediction.db")
# Matrix = os.path.join(base_dir, '..', 'Data_bases', "Matrix.db")
# Datadrifft = os.path.join(base_dir, '..', 'Data_bases', "Datadrift.db")
# Desision = os.path.join(base_dir, '..', 'Data_bases', "Desision.db")

# connect SQLite
Conn_f = sqlite3.connect(Features)
# Conn_t = sqlite3.connect(Target)
# Conn_p = sqlite3.connect(Prediction)
# Conn_m = sqlite3.connect(Matrix)
# Conn_d = sqlite3.connect(Datadrifft)
# Conn_de = sqlite3.connect(Desision)


cur_f = Conn_f.cursor()
# cur_t = Conn_t.cursor()
# cur_p = Conn_p.cursor()
# cur_m = Conn_m.cursor()
# cur_d = Conn_d.cursor()
# cur_de = Conn_de.cursor()


# drop old table


# create table with composite key
cur_f.execute("""
CREATE TABLE IF NOT EXISTS Features_data (
    date DATE NOT NULL,
    file_name TEXT NOT NULL,
    feature1 REAL,
    feature2 REAL,
    feature3 REAL
)
""")

cur_f.execute("""
CREATE TABLE IF NOT EXISTS Target_data (
    date DATE NOT NULL,
    file_name TEXT NOT NULL,
    target REAL
)
""")

cur_f.execute("""
CREATE TABLE IF NOT EXISTS Prediction_data (
    date DATE NOT NULL,
    file_name TEXT NOT NULL,
    target REAL
)
""")

cur_f.execute("""
CREATE TABLE IF NOT EXISTS Matrix_data (
    date DATE NOT NULL,
    file_name TEXT NOT NULL,
    MSE REAL,
    RMSE REAL,
    MAE REAL,
    R2 REAL
)
""")

cur_f.execute("""
CREATE TABLE IF NOT EXISTS Datadrift_data (        
    date DATE NOT NULL,
    file_name TEXT NOT NULL,
    Ks REAL,
    PSI REAL,
    kL REAL,
    CHIsquared REAL
)
""")

cur_f.execute("""
CREATE TABLE IF NOT EXISTS AIDesision (        
    date DATE NOT NULL,
    key_name TEXT NOT NULL,
    Desision string
)
""")

cur_f.execute("""
CREATE TABLE IF NOT EXISTS Token_count (        
    date DATE NOT NULL,
    input_token REAL NOT NULL,
    output_token REAL NOT NULL
)
""")

cur_f.close()
# cur_t.close()
# cur_p.close()
# cur_m.close()
# cur_d.close()
# cur_de.close()

Conn_f.close()
# Conn_t.close()
# Conn_p.close()
# Conn_m.close()
# Conn_d.close()
# Conn_de.close()
