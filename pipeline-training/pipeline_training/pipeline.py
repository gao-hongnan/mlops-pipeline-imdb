import mlflow
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC
from rich.pretty import pprint
from sklearn.linear_model import SGDClassifier

from conf.init_dirs import ROOT_DIR
from conf.init_project import initialize_project
from pipeline_training.data_extraction.extract import test_extract_from_data_warehouse
from pipeline_training.data_loading.load import load
from pipeline_training.data_preparation.transform import preprocess_data
from pipeline_training.data_validation.validate import test_validate_raw
from pipeline_training.model_training.train import train, train_model

cfg = initialize_project(ROOT_DIR)

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

tracking_uri = "http://34.94.42.137:5001"
mlflow.set_tracking_uri(tracking_uri)
metadata = train(cfg=cfg, metadata=metadata, logger=logger, model=model)
