import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import requests
from common_utils.core.common import YamlAdapter
from common_utils.core.file_system_utils import list_files_recursively
from common_utils.versioning.git.core import get_git_commit_hash
from prettytable import PrettyTable


def get_experiment_id_via_experiment_name(experiment_name: str) -> int:
    """Get experiment ID via experiment name."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    return experiment_id


def log_all_pydantic_configs_to_mlflow(config) -> None:
    """Log all Pydantic config classes to MLFlow."""
    config_outer_dict = vars(config)
    for key, value in config_outer_dict.items():
        if key == "secret":
            # skip logging the secret
            continue
        config_inner_dict = vars(value)
        mlflow.log_params(config_inner_dict)


def log_all_metrics_to_mlflow(metrics: Dict[str, Any]) -> None:
    """Log all metrics to MLFlow."""
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


# TODO: loading and dumping artifacts are not very well implemented, to review.
def load_artifacts_and_params_from_mlflow(
    run_id: str, artifact_dir: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load model artifacts from a given MLFlow run ID."""
    artifacts = {}

    yaml_adapter = YamlAdapter()
    tmp_dir_stem = mlflow.get_run(run_id).data.params["tmp_dir_stem"]

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="",
        dst_path=artifact_dir,
    )

    local_path = Path(local_path) / tmp_dir_stem
    # Print the contents of the temporary directory
    list_files_recursively(local_path)

    vectorizer = joblib.load(local_path / "vectorizer.joblib")
    model = joblib.load(local_path / "model.joblib")
    performance = joblib.load(local_path / "performance.joblib")
    model_config = joblib.load(local_path / "model_config.joblib")

    # model_config = yaml_adapter.load_to_dict(local_path / "model.yaml")

    artifacts["vectorizer"] = vectorizer
    artifacts["model"] = model
    artifacts["performance"] = performance
    artifacts["model_config"] = model_config

    run = mlflow.get_run(run_id=run_id)
    params = run.data.params
    # TODO: note here params encompasses all params, including the model params.
    # May not be clean since we are "mutating" artifacts to include what was
    # not originally there.
    artifacts["params"] = params

    return artifacts, params


def dump_artifacts_and_params_to_mlflow(artifacts: Dict[str, Any], config) -> None:
    """Log artifacts and params to MLFlow.

    NOTE:
        1. Artifacts are the output of the training process:
            - model
            - vectorizer
            - performance
            - signature
    """
    yaml_adapter = YamlAdapter()
    # log artifacts
    with tempfile.TemporaryDirectory() as tmp_dir:
        # save tmp_dir_stem as an artifact to retrieve the artifacts later.
        mlflow.log_param("tmp_dir_stem", tmp_dir.split("/")[-1])

        # Save all items in artifacts as artifacts
        for key, value in artifacts.items():
            joblib.dump(value, Path(tmp_dir) / f"{key}.joblib")

        # Save the log_output_dir as an artifact
        log_dir_path = Path(tmp_dir) / "log_output_dir"
        log_dir_path.mkdir(parents=True, exist_ok=True)
        for log_file in config.logger.log_output_dir.glob("*"):
            shutil.copy(log_file, log_dir_path)

        # Log all items in config as parameters
        for key, value in vars(config).items():
            if key == "secret":
                # skip logging the secret
                continue
            config_inner_dict = vars(value)
            yaml_adapter.save_as_dict(
                data=config_inner_dict,
                filepath=Path(tmp_dir) / f"{key}.yaml",
                sort_keys=False,
            )

        additional_metadata = log_additional_metadata()
        mlflow.log_params(additional_metadata)
        yaml_adapter.save_as_dict(
            data=additional_metadata,
            filepath=Path(tmp_dir) / "additional_metadata.yaml",
            sort_keys=False,
        )

        mlflow.log_artifact(tmp_dir, artifact_path=None)


def log_additional_metadata():
    current_commit_hash = get_git_commit_hash()
    additional_metadata_dict = {"git_commit_hash": current_commit_hash}
    return additional_metadata_dict


def log_data_splits_summary(logger, splits, total_size=None):
    # Create a pretty table
    table = PrettyTable()
    table.field_names = ["Data Split", "Size", "Percentage"]
    table.align = "l"

    for split_name, split_data in splits.items():
        percentage = (len(split_data) / total_size) * 100
        table.add_row([split_name, len(split_data), f"{percentage:.2f}%"])

    logger.info(f"Data splits summary:\n{table}")


def get_last_modified(url: str, utc8: bool = True) -> datetime:
    response = requests.head(url, timeout=30)
    last_modified = response.headers.get("Last-Modified")

    if last_modified:
        last_modified_datetime = datetime.strptime(
            last_modified, "%a, %d %b %Y %H:%M:%S %Z"
        )
        if utc8:
            last_modified_datetime = last_modified_datetime.replace(
                tzinfo=timezone.utc
            ).astimezone(timezone(timedelta(hours=8)))
        return last_modified_datetime

    print("Last-Modified header not found.")
    return None


if __name__ == "__main__":
    url = "https://datasets.imdbws.com/title.basics.tsv.gz"  # Update this with the URL of the file you want to check
    last_modified_datetime = get_last_modified(url)

    if last_modified_datetime:
        print(f"The dataset was last modified on: {last_modified_datetime}")
