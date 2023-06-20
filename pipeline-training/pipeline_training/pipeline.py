import mlflow
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.common import seed_all
from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC
from rich.pretty import pprint

from conf.init_dirs import ROOT_DIR
from conf.init_project import initialize_project
from pipeline_training.data_extraction.extract import test_extract_from_data_warehouse
from pipeline_training.data_loading.load import load
from pipeline_training.data_preparation.transform import preprocess_data
from pipeline_training.data_validation.validate import test_validate_raw
from pipeline_training.model_evaluation.hyperparameter_tuning import optimize
from pipeline_training.model_training.train import train
from mlflow.tracking import MlflowClient
from common_utils.experiment_tracking.promoter.core import MLFlowPromotionManager


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
# FIXME: properly implement preprocess_data so that the model has predictive power and not predict all 1s.
metadata = preprocess_data(
    cfg=cfg, metadata=metadata, logger=logger, dirs=cfg.general.dirs, dvc=dvc
)
pprint(metadata)

### validate.py # TODO
# test_validate_processed(cfg=cfg, logger=logger, metadata=metadata)
# train()


mlflow.set_tracking_uri(cfg.exp.tracking_uri)

# evaluate
metadata, cfg = optimize(cfg=cfg, metadata=metadata, logger=logger)
pprint(metadata)
pprint(cfg)

# NOTE: at this junction cfg is updated with the best hyperparameters
# FIXME: technically once params are tuned, you train on the full dataset and not split.
# NOTE: purposely put max_iter = 1 to illustrate the concept of
# gradient descent. This will raise convergence warning.
# Model initialization
### train.py
logger.info("Training model with best hyperparameters...")
metadata = train(cfg=cfg, metadata=metadata, logger=logger, trial=None)
pprint(metadata)

# promote.py
client = MlflowClient(tracking_uri=cfg.exp.tracking_uri)
promoter = MLFlowPromotionManager(
    client=client, model_name=cfg.exp.register_model["name"], logger=logger
)

promoter.promote_to_production(metric_name="test_accuracy")
