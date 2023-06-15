"""To refactor."""
import os
from types import SimpleNamespace

import rich
from common_utils.core.common import get_root_dir, load_env_vars

from conf.init_dirs import create_new_dirs


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
        "google_application_credentials": GOOGLE_APPLICATION_CREDENTIALS,
        "gcs_bucket_name": GCS_BUCKET_NAME,
        "gcs_bucket_project_name": GCS_BUCKET_PROJECT_NAME,
        "bigquery_raw_dataset": BIGQUERY_RAW_DATASET,
        "bigquery_raw_table_name": BIGQUERY_RAW_TABLE_NAME,
    }

    cfg.__dict__.update(dict_cfg)
    return cfg
