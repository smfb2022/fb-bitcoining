import os
from random import random, randint
import mlflow
from mlflow import log_metric, log_param, log_artifacts, set_tracking_uri
from pathlib import Path

if __name__ == "__main__":
   
    # Log a parameter (key-value pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")

    experiment = mlflow.get_experiment_by_name("testbitcoin")
    print(experiment)
    if experiment is None: 
        experiment = mlflow.create_experiment('testbitcoin', artifact_location=Path.cwd().joinpath("mlruns1").as_uri())
        print(experiment)
        experiment = mlflow.get_experiment(experiment)
    print(f'Artifact : {experiment.artifact_location}')
    with mlflow.start_run(experiment_id = experiment.experiment_id, nested=True):
        mlflow.log_metric('Accuracy', 1)
        print('Logged Metric')

