# app/api.py
from http import HTTPStatus
from typing import Any, Dict, Optional
import pickle
from rich.pretty import pprint
import os
from common_utils.core.decorators import construct_response
from fastapi import FastAPI, Request
from api import schemas
from api.config import get_settings
from typing import Any, List
import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas.predictions import RatingInput, RatingOutput, Rating

# TODO: DESERIALIZE CFG.PKL AND METADATA.PKL

from typing import List, Dict, Any, Union

# TODO: Check paul's usage of api_router decorator
api_router = APIRouter()

# disabling unused-argument and redefined-builtin because of FastAPI
# pylint: disable=unused-argument,redefined-builtin
# Define application
app = FastAPI(
    title="IMDb Rating Predictor",
    description="Predict user ratings for movies based on title and genre.",
    version="0.1",
)
tracking_uri = "http://34.143.176.217:5001/"
client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


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


def predict(texts: List[str], vectorizer, model) -> List[Dict[str, Any]]:
    """Predict on data."""
    X = vectorizer.transform(texts)
    y_preds = model.predict(X)
    y_probs = model.predict_proba(X)

    predictions = []
    for text, y_pred, y_prob in zip(texts, y_preds, y_probs):
        predictions.append(
            {
                "text": text,
                "rating": y_pred,
                "probability": y_prob.tolist(),  # display all probabilities
            }
        )
    return predictions


def get_latest_production_model(client, model_name):
    model = client.get_registered_model(model_name)
    for mv in model.latest_versions:
        if mv.current_stage == "Production":
            return mv
    return None


# pylint: disable=global-variable-undefined
@api_router.on_event("startup")
async def load_best_artifacts_from_mlflow():
    """
    Asynchronously load the best MLflow artifacts on startup.

    This function uses the `on_event("startup")` decorator, meaning it runs
    once when the application starts. It employs MLflow's tracking
    functionality to load artifacts associated with a previous machine
    learning experiment, determined as 'best' by the
    `main.load_best_artifacts_from_mlflow` function. The `run_id` argument
    defaults to None, indicating the function selects the best run.

    The `artifacts` are stored in a global variable to avoid reloading
    them with every prediction request, which can be performance
    critical, especially for larger models or artifacts. Without
    declaring `artifacts` as `global`, the artifacts would need to be
    reloaded each time a prediction is made, significantly slowing
    response times.

    Globals:
    --------
    artifacts : Dict[str, Any]
        Contains the loaded MLflow artifacts. The structure and
        contents depend on the implementation of
        `main.load_best_artifacts_from_mlflow()`.
    """
    global artifacts
    global vectorizer
    global model
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

    artifacts = metadata.model_artifacts

    vectorizer = artifacts["vectorizer"]


@api_router.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: Optional[str] = None) -> Dict:
    """Get the performance metrics."""
    print(artifacts)
    performance = artifacts["overall_performance"]
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@api_router.get("/args", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """Get a specific parameter's value used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"args": artifacts["params"]},
    }
    return response


@api_router.get("/args/{arg}", tags=["Arguments"])
@construct_response
def _arg(request: Request, arg: str) -> Dict:
    """Get a specific parameter's value used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            arg: artifacts["params"].get(arg, ""),
        },
    }
    return response


@api_router.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, data: RatingInput) -> Dict[str, Any]:
    """Predict the user rating for a movie."""
    texts = process_input_data(genres=data.genres, titles=data.titles)

    # Here `predict` is the function you use to generate predictions
    predictions = predict(texts=texts, vectorizer=vectorizer, model=model)
    ratings = [Rating(**prediction) for prediction in predictions]
    ratings_output = RatingOutput(ratings=ratings)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": ratings_output,
    }
    return response


@api_router.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict[str, Any]:
    """
    Perform a health check on the server.

    This function is a simple health check endpoint that can be used to
    verify if the server is running correctly. It returns a dictionary
    with a message indicating the status of the server, the HTTP status
    code, and an empty data dictionary.

    Parameters
    ----------
    request : Request
        The request object that contains all the HTTP request
        information.

    Returns
    -------
    response : Dict[str, Any]
        A dictionary containing:
        - message: A string indicating the status of the server. If the
          server is running correctly, this will be "OK".
        - status-code: An integer representing the HTTP status code. If
          the server is running correctly, this will be 200.
        - data: An empty dictionary. This can be used to include any
          additional data if needed in the future.
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response
