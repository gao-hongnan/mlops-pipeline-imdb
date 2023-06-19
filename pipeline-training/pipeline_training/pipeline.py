import mlflow
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC
from rich.pretty import pprint

from common_utils.core.common import seed_all

from conf.init_dirs import ROOT_DIR
from conf.init_project import initialize_project
from pipeline_training.data_extraction.extract import test_extract_from_data_warehouse
from pipeline_training.data_loading.load import load
from pipeline_training.data_preparation.transform import preprocess_data
from pipeline_training.data_validation.validate import test_validate_raw
from pipeline_training.model_training.train import train
from pipeline_training.model_evaluation.hyperparameter_tuning import optimize

cfg = initialize_project(ROOT_DIR)
seed_all(cfg.general.seed, seed_torch=False)

logger = Logger(
    log_file="pipeline_training.log",
    log_root_dir=cfg.general.dirs.stores.logs,
    module_name=__name__,
    propagate=False,
).logger

gcs = GCS(
    project_id=cfg.env.project_id,
    google_application_credentials=cfg.env.google_application_credentials,
    bucket_name=cfg.env.gcs_bucket_name,
)
dvc = SimpleDVC(
    storage=gcs,
    remote_bucket_project_name=cfg.env.gcs_bucket_project_name,
    data_dir=cfg.general.dirs.data.raw,
    metadata_dir=cfg.general.dirs.stores.blob.raw,
)

### extract.py
metadata = test_extract_from_data_warehouse(logger=logger, cfg=cfg)

### validate.py
test_validate_raw(cfg=cfg, logger=logger, metadata=metadata)

### load.py
metadata = load(metadata=metadata, logger=logger, dirs=cfg.general.dirs, dvc=dvc)
pprint(metadata)

dvc = SimpleDVC(
    storage=gcs,
    remote_bucket_project_name=cfg.env.gcs_bucket_project_name,
    data_dir=cfg.general.dirs.data.processed,
    metadata_dir=cfg.general.dirs.stores.blob.processed,
)

### transform.py
metadata = preprocess_data(
    cfg=cfg, metadata=metadata, logger=logger, dirs=cfg.general.dirs, dvc=dvc
)
pprint(metadata)

### validate.py # TODO
# test_validate_processed(cfg=cfg, logger=logger, metadata=metadata)
# train()

### train.py
# NOTE: purposely put max_iter = 1 to illustrate the concept of
# gradient descent. This will raise convergence warning.
# Model initialization

mlflow.set_tracking_uri(cfg.exp.tracking_uri)
# evaluate
metadata, cfg = optimize(cfg=cfg, metadata=metadata, logger=logger)
pprint(metadata)
pprint(cfg)

metadata = train(cfg=cfg, metadata=metadata, logger=logger, trial=None)
pprint(metadata)
