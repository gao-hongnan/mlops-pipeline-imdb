import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from common_utils.core.base import Connection
from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC
from rich.pretty import pprint

from conf.metadata import Metadata


def fetch_dvc_metadata(dvc: SimpleDVC, filename: str) -> dict:
    """
    Fetches the metadata for a file from its metadata JSON file.

    Args:
        dvc (SimpleDVC): Instance of SimpleDVC.
        filename (str): Name of the file.

    Returns:
        dict: Metadata dictionary for the file.
    """
    return dvc._load_metadata(filename)


def extract_from_dvc(
    cfg: SimpleNamespace,
    dvc: SimpleDVC,
    logger: Logger,
    metadata: Metadata,
) -> Metadata:
    logger.info("Production Environment: starting data extraction...")

    try:
        filename = cfg.bigquery_raw_table_name
        filename = f"{filename}.csv"
        dvc_metadata = fetch_dvc_metadata(filename=filename, dvc=dvc)

        dvc.pull(
            filename=dvc_metadata["filename"],
            remote_bucket_project_name=dvc_metadata["remote_bucket_project_name"],
        )

        raw_df: pd.DataFrame = pd.read_csv(Path(cfg.dirs.raw) / filename)
        pprint(raw_df.head())

        logger.info("✅ Data extraction from DVC completed. Updating metadata...")

        num_rows, num_cols = raw_df.shape

        attr_dict = {
            "raw_df": raw_df,
            "raw_num_rows": num_rows,
            "raw_num_cols": num_cols,
        }

        metadata.set_attrs(attr_dict)

        return metadata
    except Exception as error:
        logger.error(f"❌ Data extraction from DVC failed. Error: {error}")
        raise error


if __name__ == "__main__":
    from common_utils.cloud.gcp.storage.gcs import GCS
    from common_utils.core.logger import Logger

    from conf.init_dirs import ROOT_DIR
    from conf.init_project import initialize_project

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
    metadata = Metadata()
    extract_from_dvc(cfg=cfg, dvc=dvc, logger=logger, metadata=metadata)
