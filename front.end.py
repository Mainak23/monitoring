import os
import sqlite3
import pandas as pd
import altair as alt
import pandas as pd
import panel as pn
import hvplot.pandas

pn.extension("vega", sizing_mode="stretch_width")

base_dir = os.getcwd()
datadrift_db_path = os.path.join(base_dir, 'Data_bases', 'Projectdata.db')

# Connect to SQLite database
conn = sqlite3.connect(datadrift_db_path)
cursor = conn.cursor()

# Load Datadrift_data
datadrift = cursor.execute("""
    SELECT * FROM Datadrift_data
    ORDER BY date DESC
    LIMIT 50;
""")
rows_datadrift = datadrift.fetchall()
df_datadrift = pd.DataFrame(rows_datadrift, columns=["date", "file_name", "Ks", "PSI", "kL", "CHIsquared"])
df_datadrift = df_datadrift.dropna(axis=1, how='any')
df_datadrift.drop(columns=["date","file_name"], inplace=True)
df_scaled_data_drift = df_datadrift.apply(
    lambda row: (row - row.min()) / (row.max() - row.min()) if row.max() != row.min() else row,
    axis=1
)

print(df_scaled_data_drift)



# Load Matrix_data
matrix = cursor.execute("""
    SELECT * FROM Matrix_data
    ORDER BY date DESC
    LIMIT 50;
""")
rows_matrix = matrix.fetchall()
df_matrix = pd.DataFrame(rows_matrix, columns=["date", "file_name", "MAE", "MSE", "RMSE", "R2_score"])
df_matrix = df_matrix.dropna(axis=1, how='any')
df_matrix.drop(columns=["date","file_name"], inplace=True)



print(df_matrix)
# Load token uses
token = cursor.execute("""
    SELECT * FROM Token_count
    ORDER BY date DESC
    LIMIT 50;
""")
rows_token= token.fetchall()
df_token  = pd.DataFrame(rows_token, columns=["date","input_token","output_token"])
df_token.drop(columns=["date"], inplace=True)
print(df_token)

# Load latest AI decision
drift_decision = cursor.execute("""
    SELECT * FROM AIDesision
    ORDER BY date DESC
    LIMIT 1;
""")
df_decision = pd.DataFrame(drift_decision.fetchall(), columns=["date", "file_name", "AIDesision"])
df_decision.drop(columns=["date", "file_name"], inplace=True)
decision_=df_decision.iloc[0,0]
print("Latest AI Decision:", decision_)
# Close database connection
cursor.close()
conn.close()



pn.extension("vega", sizing_mode="stretch_width")

data_collection=[df_scaled_data_drift,df_matrix,df_token,decision_]

period_ = list(range(51))


print(data_collection)

"date","input_token","output_token"



df_token  = pd.DataFrame(rows_token, columns=["date","input_token","output_token"])
period_ = list(range(51))




pn.extension()

# Set full height for the template
template = pn.template.FastListTemplate(
    title="Data Quality and Drift Monitoring Dashboard",
    sidebar=[pn.pane.Markdown(f"## Latest AI Decision: {decision_}")],  # No sidebar for full width
    main=[
        pn.Column(
         pn.pane.Markdown("## Token Consumption Insights Dashboard"),
            pn.pane.Markdown(
                "This graph helps us track the usage of input and output tokens over time, "
                "allowing us to analyze how token values evolve and fluctuate across different time periods."
            ),
        df_token.hvplot().opts(height=400,width=1000, responsive=True)),
        
        pn.Column(
         pn.pane.Markdown("## Model Accuracy & Drift Insights Dashboard"),
            pn.pane.Markdown(
                "This graph provides insights into various concept drift metrics over time, "
                "enabling us to monitor and analyze changes in data distribution and model performance."
            ),
        df_matrix.hvplot().opts(height=400,width=1000, responsive=True)),
        
        pn.Column(
        pn.pane.Markdown(
        "<h2 style='font-size:24px; font-weight:bold;'>Data Drift Insights</h2>"
        "<p2>This graph provides insights into various data drift metrics over time, "
        "enabling us to monitor and analyze changes in data distribution.</p2>",
        sizing_mode="stretch_width"
        ),
        df_datadrift.hvplot().opts(height=400,width=1000, responsive=True)),
        #pn.pane.Markdown(f"## Latest AI Decision: {decision_}")
    ],
   
).servable()
