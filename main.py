import Agentic_solution.concept_drift_pipeline as cd
import Agentic_solution.Data_drift_pipeline as dd
import Agentic_solution.prediction_pipiline as ap

import asyncio

pay_load={
        "experiment_file": "dataset_m.csv",
        "terget_file": "dataset_n.csv",
        "model_path": "linear_regression_model.pkl"
    }


async def scheduler():
    tasks = [
        ap.run_prediction_chain(pay_load),
        dd.run_data_drift_chain(),
        cd.run_concept_drift_chain()]
    results = await asyncio.gather(*tasks)
    print("All tasks completed.")
    return results

with asyncio.Runner() as runner:
    runner.run(scheduler())