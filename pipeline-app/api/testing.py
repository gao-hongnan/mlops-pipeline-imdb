import mlflow
import os
import numpy as np
from typing import *
import pickle
from rich.pretty import pprint


def process_input_data(
    genres: Union[str, List[str]], titles: Union[str, List[str]]
) -> List[str]:
    genres = validate_input(genres, "genres")
    titles = validate_input(titles, "titles")

    if len(genres) != len(titles):
        raise ValueError("genres and titles must have the same length.")

    return [title + "-" + genre for title, genre in zip(titles, genres)]


def validate_input(input_data: Union[str, List[str]], input_name: str) -> List[str]:
    if isinstance(input_data, str):
        input_data = [input_data]
    elif isinstance(input_data, list):
        if not all(isinstance(item, str) for item in input_data):
            raise ValueError(f"All elements in the {input_name} must be strings.")
    else:
        raise TypeError(f"{input_name} must be a string or a list of strings.")
    return input_data


tracking_uri = "http://34.143.176.217:5001/"
client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


def get_latest_production_model(client, model_name):
    model = client.get_registered_model(model_name)
    for mv in model.latest_versions:
        if mv.current_stage == "Production":
            return mv
    return None


# Usage
model_name = "imdb_sgd"
latest_production_model = get_latest_production_model(client, model_name)
latest_production_version = latest_production_model.version
latest_production_run_id = latest_production_model.run_id
print(
    f"Latest production version of {model_name}: {latest_production_version} and run_id: {latest_production_run_id}"
)

latest_production_run = client.get_run(latest_production_run_id)
latest_production_experiment_id = latest_production_run.info.experiment_id
logged_model = f"gs://gaohn/imdb/artifacts/{latest_production_experiment_id}/{latest_production_run_id}/artifacts/registry"
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/Users/reighns/gaohn/pipeline/mlops-pipeline-imdb/pipeline-training/gcp-storage-service-account.json"
# model = mlflow.pyfunc.load_model(logged_model)
# print(model)
# TODO: super bad if i load from here i cannot use predict proba so i use below
model = mlflow.sklearn.load_model(logged_model)

artifacts = client.list_artifacts(latest_production_run_id)
pprint(artifacts)

stores_path = "stores"

stores_local_path = mlflow.artifacts.download_artifacts(
    run_id=latest_production_run_id,
    artifact_path=stores_path,
    tracking_uri=tracking_uri,
)
pprint(stores_local_path)
metadata = pickle.load(open(stores_local_path + "/artifacts/metadata.pkl", "rb"))
pprint(metadata)


vectorizer = metadata.model_artifacts["vectorizer"]


genres = "Action_Adventure_Sci-Fi"
titles = "The Matrix"
text = process_input_data(genres, titles)

text = vectorizer.transform(text)
pprint(model.predict_proba(text))
pprint(model.predict(text))
