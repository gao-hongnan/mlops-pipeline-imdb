import mlflow

import warnings
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from rich.pretty import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, precision_recall_fscore_support
from pipeline_training.data_preparation.resampling import get_data_splits

# from cfg.base import Config
# from cfg.model import ScikitLearnModel
# from src.datamodule.preprocess import get_data_splits
# from src.utils.common import log_data_splits_summary

tracking_uri = "http://34.94.42.137:5001"
experiment_name = "imdb_revamp"
run_name = "untuned_imdb_sgd_dummy"
nested = False

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name=experiment_name)


warnings.filterwarnings("ignore")


def predict_on_holdout_set(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Predict on holdout set."""
    y_pred = model.predict(X_test)
    metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    y_prob = model.predict_proba(X_test)
    test_loss = log_loss(y_test, y_prob)
    performance = {
        "test_loss": test_loss,
        "test_precision": metrics[0],
        "test_recall": metrics[1],
        "test_f1": metrics[2],
    }
    return performance


from sklearn.linear_model import SGDClassifier

model = SGDClassifier(
    loss="log",
    penalty="l2",
    alpha=0.0001,
    max_iter=1,
    learning_rate="optimal",
    eta0=0.1,
    power_t=0.1,
    warm_start=True,
)


def train_model(cfg, logger, metadata) -> Dict[str, Any]:
    """Train model."""
    logger = cfg.logger.logger
    logger.info("Training model...")

    # Tf-idf
    vectorizer = TfidfVectorizer(
        analyzer=cfg.misc.analyzer, ngram_range=cfg.misc.ngram_range
    )

    df = pd.read_csv(input_filepath)

    X = df["concat_title_genres"].to_numpy()
    y = df["rounded_averageRating"].to_numpy()
    # binary > 5
    y = np.where(y > 5, 1, 0)

    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(cfg=cfg, X=X, y=y)
    pprint(X_train)

    # row is your data point, column is your feature
    # which is the tf-idf score of the word in that data point.
    # pprint(X_train.shape)

    # log_data_splits_summary(
    #     logger,
    #     splits={
    #         "X_train": X_train,
    #         "X_val": X_val,
    #         "X_test": X_test,
    #     },
    #     total_size=len(df),
    # )

    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # NOTE: purposely put max_iter = 1 to illustrate the concept of
    # gradient descent. This will raise convergence warning.
    # Model initialization

    model = cfg.model.model
    pprint(model.get_params())

    # Training

    for epoch in range(cfg.misc.num_epochs):
        model.fit(X_train, y_train)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        # evaluate on validation data
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        # Log performance metrics for the current epoch
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if not epoch % 10:
            logger.info(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}"
            )

    # Evaluate on test data
    performance = predict_on_holdout_set(model, X_test, y_test)

    # Log the model with a signature that defines the schema of the model's inputs and outputs.
    # When the model is deployed, this signature will be used to validate inputs.
    signature = mlflow.models.infer_signature(X_test, model.predict(X_test))

    artifacts = {
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
        "signature": signature,
        "model_config": model.get_params(),
    }
    return artifacts
