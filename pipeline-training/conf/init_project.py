"""To refactor."""
import os
from types import SimpleNamespace
from typing import Literal, Union

import rich
from common_utils.core.common import get_root_dir, load_env_vars
from common_utils.versioning.git.core import get_git_commit_hash
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords


from conf.init_dirs import create_new_dirs


STOPWORDS = set(stopwords.words("english"))

# Prepare your schema outside, in production this should be in a constant file or database
RAW_SCHEMA = {
    "tconst": "object",
    "primaryTitle": "object",
    "originalTitle": "object",
    "isAdult": "float64",
    "startYear": "float64",
    "endYear": "float64",
    "runtimeMinutes": "object",
    "genres": "object",
    "averageRating": "float64",
    "numVotes": "float64",
    "last_modified": "object",
}


def is_docker():
    path = "/.dockerenv"
    return os.path.exists(path)


# pylint: disable=invalid-name
def initialize_project(root_dir: str) -> SimpleNamespace:
    DIRS = create_new_dirs()

    ROOT_DIR = get_root_dir(env_var="ROOT_DIR", root_dir=root_dir)
    os.environ["ROOT_DIR"] = str(ROOT_DIR)

    if not is_docker():
        print("Not running inside docker")
        load_env_vars(root_dir=ROOT_DIR)

    PROJECT_ID = os.getenv("PROJECT_ID")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    GCS_BUCKET_PROJECT_NAME = os.getenv("GCS_BUCKET_PROJECT_NAME")
    BIGQUERY_RAW_DATASET = os.getenv("BIGQUERY_RAW_DATASET")
    BIGQUERY_RAW_TABLE_NAME = os.getenv("BIGQUERY_RAW_TABLE_NAME")

    rich.print("ENV", PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS, GCS_BUCKET_NAME)

    git_commit_hash: Union[str, Literal["N/A"]] = get_git_commit_hash(
        check_git_status=False
    )
    if git_commit_hash == "N/A":
        git_commit_hash = os.getenv("GIT_COMMIT_HASH", None)
        print(f"git_commit_hash: {git_commit_hash} this is inside docker likely.")
        if git_commit_hash is None:
            raise ValueError(
                """Git commit hash is 'N/A' and environment variable 'GIT_COMMIT_HASH'
                is not set. Please set the 'GIT_COMMIT_HASH' environment variable
                when running inside a Docker container."""
            )

    cfg = SimpleNamespace(**{})
    general_config = {
        "seed": 42,
        "git_commit_hash": git_commit_hash,
        "dirs": DIRS,
        "root_dir": ROOT_DIR,
        "raw_schema": RAW_SCHEMA,
    }
    cfg.general = SimpleNamespace(**general_config)

    env_config = {
        "project_id": PROJECT_ID,
        "google_application_credentials": GOOGLE_APPLICATION_CREDENTIALS,
        "gcs_bucket_name": GCS_BUCKET_NAME,
        "gcs_bucket_project_name": GCS_BUCKET_PROJECT_NAME,
        "bigquery_raw_dataset": BIGQUERY_RAW_DATASET,
        "bigquery_raw_table_name": BIGQUERY_RAW_TABLE_NAME,
    }
    cfg.env = SimpleNamespace(**env_config)

    transform_config = {"stem": False, "lower": True, "stopwords": STOPWORDS}
    cfg.transform = SimpleNamespace(**transform_config)

    resampling_config = {
        "strategy": {
            "train_test_split": {"train_size": 0.7, "random_state": 42, "shuffle": True}
        },
    }
    cfg.resampling = SimpleNamespace(**resampling_config)

    train_config = {
        "vectorizer": {"analyzer": "char_wb", "ngram_range": (2, 5)},
        "num_epochs": 5,
        "log_every_n_epoch": 1,
    }
    cfg.train = SimpleNamespace(**train_config)

    exp_config = {
        "experiment_name": "imdb_mlops_pipeline",
        "start_run": {
            "run_name": "untuned_imdb_sgd_5_epochs",
            "nested": True,
            "description": "Imdb sentiment analysis with sklearn SGDClassifier",
            "tags": {"framework": "sklearn", "type": "classification"},
        },
    }
    cfg.exp = SimpleNamespace(**exp_config)

    return cfg
