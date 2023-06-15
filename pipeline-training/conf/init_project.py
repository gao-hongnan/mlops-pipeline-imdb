"""To refactor."""
import os
from types import SimpleNamespace

import rich
from common_utils.core.common import get_root_dir, load_env_vars
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


# pylint: disable=invalid-name
def initialize_project(root_dir: str) -> SimpleNamespace:
    DIRS = create_new_dirs()

    ROOT_DIR = get_root_dir(env_var="ROOT_DIR", root_dir=root_dir)
    os.environ["ROOT_DIR"] = str(ROOT_DIR)

    load_env_vars(root_dir=ROOT_DIR)

    PROJECT_ID = os.getenv("PROJECT_ID")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    GCS_BUCKET_PROJECT_NAME = os.getenv("GCS_BUCKET_PROJECT_NAME")
    BIGQUERY_RAW_DATASET = os.getenv("BIGQUERY_RAW_DATASET")
    BIGQUERY_RAW_TABLE_NAME = os.getenv("BIGQUERY_RAW_TABLE_NAME")

    rich.print("ENV", PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS, GCS_BUCKET_NAME)

    cfg = SimpleNamespace(**{})
    dict_cfg = {
        "dirs": DIRS,
        "root_dir": ROOT_DIR,
        "project_id": PROJECT_ID,
        "raw_schema": RAW_SCHEMA,
        "google_application_credentials": GOOGLE_APPLICATION_CREDENTIALS,
        "gcs_bucket_name": GCS_BUCKET_NAME,
        "gcs_bucket_project_name": GCS_BUCKET_PROJECT_NAME,
        "bigquery_raw_dataset": BIGQUERY_RAW_DATASET,
        "bigquery_raw_table_name": BIGQUERY_RAW_TABLE_NAME,
    }

    cfg.__dict__.update(dict_cfg)

    transform_config = {"stem": False, "lower": True, "stopwords": STOPWORDS}
    cfg.__dict__.update(transform_config)

    resampling_config = {
        "strategy": {
            "train_test_split": {"train_size": 0.7, "random_state": 42, "shuffle": True}
        },
    }
    cfg.__dict__.update(resampling_config)

    train_config = {"analyzer": "char_wb", "ngram_range": (2, 5)}
    return cfg
