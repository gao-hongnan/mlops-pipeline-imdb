import warnings
from typing import Any, Dict
from common_utils.core.common import seed_all
import mlflow
import numpy as np
from rich.pretty import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, precision_recall_fscore_support

import pickle
from pipeline_training.data_preparation.resampling import get_data_splits


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


def train_model(cfg, logger, metadata, model) -> Dict[str, Any]:
    """Train model."""
    logger.info("Training model...")

    # Tf-idf
    vectorizer = TfidfVectorizer(**cfg.train.vectorizer)

    df = metadata.processed_df

    X = df["concat_title_genres"].to_numpy()
    y = df["rounded_averageRating"].to_numpy()
    # binary > 5
    y = np.where(y > 5, 1, 0)

    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(cfg=cfg, X=X, y=y)

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

    pprint(model.get_params())

    # Training

    for epoch in range(cfg.train.num_epochs):
        model.fit(X_train, y_train)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        # evaluate on validation data
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        # Log performance metrics for the current epoch
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if not epoch % cfg.train.log_every_n_epoch:
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

    model_artifacts = {
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
        "signature": signature,
        "model_config": model.get_params(),
    }
    metadata.model_artifacts = model_artifacts
    return metadata


def train(cfg, logger, metadata, model):
    seed_all(cfg.general.seed, seed_torch=False)
    mlflow.set_experiment(experiment_name=cfg.exp.experiment_name)

    # nested=True because this is nested under a parent train func in main.py.
    with mlflow.start_run(**cfg.exp.start_run):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        metadata = train_model(cfg, logger, metadata, model)
        pprint(metadata.model_artifacts)

        mlflow.sklearn.log_model(
            sk_model=metadata.model_artifacts["model"],
            artifact_path="model",
            signature=metadata.model_artifacts["signature"],
        )

        with open(f"{cfg.general.dirs.stores.artifacts}/cfg.pkl", "wb") as file:
            pickle.dump(cfg, file)

        with open(f"{cfg.general.dirs.stores.artifacts}/metadata.pkl", "wb") as file:
            pickle.dump(metadata, file)

        stores_path = cfg.general.dirs.stores.base
        pprint(stores_path)

        mlflow.log_artifacts(local_dir=stores_path, artifact_path="stores")

    return metadata
