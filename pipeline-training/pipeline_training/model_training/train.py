import pickle
import warnings
from typing import Any, Dict

import mlflow
import numpy as np
from common_utils.core.common import seed_all
from pipeline_training.data_preparation.resampling import get_data_splits
from pipeline_training.utils.common import log_data_splits_summary
from rich.pretty import pprint
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier  # pylint: disable=unused-import
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# FIXME: model is predicting all 1s
# FIXME: write a class to encapsulate the model and its methods for training
# NOTE: note very carefully that if you don't add the if not trial flag, the code will run into error
# since mlflow will try to create a new experiment with the same name. There's other workarounds but for now we'll
# just use this.


# TODO: dump confusion matrix and classification report to image/media.
def predict_on_holdout_set(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Predict on holdout set."""
    # TODO: make metrics an abstract object instead of dict
    performance = {"overall": {}, "report": {}, "per_class": {}}

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    classes = np.unique(y_test)
    num_classes = len(classes)

    prf_metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    test_loss = log_loss(y_test, y_prob)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Brier score
    if num_classes == 2:
        test_brier_score = brier_score_loss(y_test, y_prob[:, 1])
        test_roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        test_brier_score = np.mean(
            [brier_score_loss(y_test == i, y_prob[:, i]) for i in range(num_classes)]
        )
        test_roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

    overall_performance = {
        "test_loss": test_loss,
        "test_precision": prf_metrics[0],
        "test_recall": prf_metrics[1],
        "test_f1": prf_metrics[2],
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced_accuracy,
        "test_roc_auc": test_roc_auc,
        "test_brier_score": test_brier_score,
    }
    performance["overall"] = overall_performance

    test_confusion_matrix = confusion_matrix(y_test, y_pred)
    test_classification_report = classification_report(
        y_test, y_pred, output_dict=True
    )  # output_dict=True to get result as dictionary

    performance["report"] = {
        "test_confusion_matrix": test_confusion_matrix,
        "test_classification_report": test_classification_report,
    }

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )  # None to get per-class metrics

    for i, _class in enumerate(classes):
        performance["per_class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }
    return performance


def create_model(model_config: Dict[str, Any]) -> BaseEstimator:
    model_class = eval(model_config.pop("model_name"))
    model = model_class(**model_config)
    return model


def create_vectorizer(vectorizer_config: Dict[str, Any]) -> TfidfVectorizer:
    vectorizer_class = eval(vectorizer_config.pop("vectorizer_name"))
    vectorizer = vectorizer_class(**vectorizer_config)
    return vectorizer


def train_model(cfg, logger, metadata, trial=None) -> Dict[str, Any]:
    """Train model."""
    seed_all(cfg.general.seed, seed_torch=False)

    logger.info("Training model...")

    # Tf-idf
    vectorizer = create_vectorizer(cfg.train.vectorizer)

    df = metadata.processed_df
    pprint(df.head())
    pprint(df["rounded_averageRating"])

    X = df["concat_title_genres"].to_numpy()
    y = df["rounded_averageRating"].to_numpy()
    # binary > 5
    y = np.where(y > 5, 1, 0)

    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(cfg=cfg, X=X, y=y)

    # row is your data point, column is your feature
    # which is the tf-idf score of the word in that data point.
    # pprint(X_train.shape)

    log_data_splits_summary(
        logger,
        splits={
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
        },
        total_size=len(df),
    )

    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # create model
    model = create_model(cfg.train.model)

    # Training
    for epoch in range(cfg.train.num_epochs):
        model.fit(X_train, y_train)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        # evaluate on validation data
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        val_accuracy = accuracy_score(y_val, model.predict(X_val))

        # Log performance metrics for the current epoch
        if not trial:  # if not hyperparameter tuning then we log to mlflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )

        if not epoch % cfg.train.log_every_n_epoch:
            logger.info(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}, "
                f"val_accuracy: {val_accuracy:.5f}"
            )

    # Evaluate on test data
    performance = predict_on_holdout_set(model, X_test, y_test)

    # Log the model with a signature that defines the schema of the model's inputs and outputs.
    # When the model is deployed, this signature will be used to validate inputs.
    if not trial:
        logger.info(
            "This is not in a trial, it is likely training a final model with the best hyperparameters"
        )
        signature = mlflow.models.infer_signature(X_test, model.predict(X_test))

    model_artifacts = {
        "vectorizer": vectorizer,
        "model": model,
        "overall_performance": performance["overall"],
        "report_performance": performance["report"],
        "per_class_performance": performance["per_class"],
        "signature": signature if not trial else None,
        "model_config": model.get_params(),
    }
    metadata.model_artifacts = model_artifacts
    return metadata


def log_all_metrics_to_mlflow(metrics: Dict[str, Any]) -> None:
    """Log all metrics to MLFlow."""
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def dump_cfg_and_metadata(cfg, metadata):
    with open(f"{cfg.general.dirs.stores.artifacts}/cfg.pkl", "wb") as file:
        pickle.dump(cfg, file)

    with open(f"{cfg.general.dirs.stores.artifacts}/metadata.pkl", "wb") as file:
        pickle.dump(metadata, file)


def get_experiment_id_via_experiment_name(experiment_name: str) -> int:
    """Get experiment ID via experiment name."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    return experiment_id


def train(cfg, logger, metadata, trial=None):
    mlflow.set_experiment(experiment_name=cfg.exp.experiment_name)

    # nested=True because this is nested under a parent train func in main.py.
    with mlflow.start_run(**cfg.exp.start_run):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        metadata = train_model(cfg, logger, metadata, trial=trial)

        mlflow.sklearn.log_model(
            sk_model=metadata.model_artifacts["model"],
            artifact_path="registry",
            signature=metadata.model_artifacts["signature"],
        )

        logger.info("✅ Logged the model to MLflow.")

        overall_performance = metadata.model_artifacts["overall_performance"]
        logger.info(
            f"✅ Training completed. The model overall_performance is {overall_performance}."
        )
        logger.info("✅ Logged the model's overall performance to MLflow.")
        log_all_metrics_to_mlflow(overall_performance)

        logger.info("✅ Dumping cfg and metadata to artifacts.")
        dump_cfg_and_metadata(cfg, metadata)

        stores_path = cfg.general.dirs.stores.base

        mlflow.log_artifacts(
            local_dir=stores_path, artifact_path=cfg.exp.log_artifacts["artifact_path"]
        )

        # log to model registry
        # log to model registry
        experiment_id = get_experiment_id_via_experiment_name(
            experiment_name=cfg.exp.experiment_name
        )

        # FIXME: tidy up all hardcoded mlflow paths

        signature = metadata.model_artifacts["signature"]
        mlflow.models.signature.set_signature(
            model_uri=cfg.exp.set_signature["model_uri"].format(
                experiment_id=experiment_id, run_id=run_id
            ),
            signature=signature,
        )

        model_version = mlflow.register_model(
            model_uri=cfg.exp.register_model["model_uri"].format(
                experiment_id=experiment_id, run_id=run_id
            ),  # this is relative to the run_id! rename to registry to be in sync with local stores
            name=cfg.exp.register_model["name"],
            tags={
                "dev-exp-id": experiment_id,
                "dev-exp-name": cfg.exp.experiment_name,
                "dev-exp-run-id": run_id,
                "dev-exp-run-name": cfg.exp.start_run["run_name"],
            },
        )
        mlflow.log_param("model_version", model_version.version)
        logger.info(
            f"✅ Logged the model to the model registry with version {model_version.version}."
        )

    return metadata
