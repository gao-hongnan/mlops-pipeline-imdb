from pathlib import Path
from typing import Optional

from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC
from rich.pretty import pprint

from conf.init_dirs import ROOT_DIR, Directories
from conf.init_project import initialize_project
from conf.metadata import Metadata
from pipeline_training.data_extraction.extract import test_extract_from_data_warehouse
from pipeline_training.data_loading.load import load
from pipeline_training.model_training.train import train_model
from pipeline_training.data_validation.validate import test_validate_raw
from pipeline_training.data_preparation.transform import preprocess_data

cfg = initialize_project(ROOT_DIR)

logger = Logger(
    log_file="pipeline_training.log",
    log_root_dir=cfg.dirs.logs,
    module_name=__name__,
    propagate=False,
).logger

gcs = GCS(
    project_id=cfg.project_id,
    google_application_credentials=cfg.google_application_credentials,
    bucket_name=cfg.gcs_bucket_name,
)
dvc = SimpleDVC(
    storage=gcs,
    remote_bucket_project_name=cfg.gcs_bucket_project_name,
    data_dir=cfg.dirs.raw,
)

### extract.py
metadata = test_extract_from_data_warehouse(logger=logger, cfg=cfg)

### validate.py
test_validate_raw(cfg=cfg, logger=logger, metadata=metadata)

### load.py
metadata = load(metadata=metadata, logger=logger, dirs=cfg.dirs, dvc=dvc)
pprint(metadata)

dvc = SimpleDVC(
    storage=gcs,
    remote_bucket_project_name=cfg.gcs_bucket_project_name,
    data_dir=cfg.dirs.processed,
)

### transform.py
metadata = preprocess_data(
    cfg=cfg, metadata=metadata, logger=logger, dirs=cfg.dirs, dvc=dvc
)
pprint(metadata)

# train()
