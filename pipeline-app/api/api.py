# app/api.py
from http import HTTPStatus
from typing import Any, Dict, Optional

from common_utils.core.decorators import construct_response
from fastapi import FastAPI, Request

from src.trainer.inference import predict
from app.schemas import RatingInput, RatingOutput, Rating
import main

DESERIALIZE CFG.PKL AND METADATA.PKL

# disabling unused-argument and redefined-builtin because of FastAPI
# pylint: disable=unused-argument,redefined-builtin
# Define application
app = FastAPI(
    title="IMDb Rating Predictor",
    description="Predict user ratings for movies based on title and genre.",
    version="0.1",
)

# pylint: disable=global-variable-undefined
@app.on_event("startup")
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
    artifacts = main.load_artifacts_from_mlflow(run_id=None)


@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: Optional[str] = None) -> Dict:
    """Get the performance metrics."""
    print(artifacts)
    performance = artifacts["performance"]
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@app.get("/args", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """Get a specific parameter's value used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"args": artifacts["params"]},
    }
    return response


@app.get("/args/{arg}", tags=["Arguments"])
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


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, data: RatingInput) -> Dict[str, Any]:
    """Predict the user rating for a movie."""
    texts = main.process_input_data(genres=data.genres, titles=data.titles)

    # Here `predict` is the function you use to generate predictions
    predictions = predict(texts=texts, artifacts=artifacts)
    ratings = [Rating(**prediction) for prediction in predictions]
    ratings_output = RatingOutput(ratings=ratings)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": ratings_output,
    }
    return response


@app.get("/", tags=["General"])
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
