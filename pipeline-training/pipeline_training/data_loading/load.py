from pathlib import Path
from typing import Optional

from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC
from rich.pretty import pprint

from conf.init_dirs import Directories
from conf.metadata import Metadata


def load_to_feature_store():
    pass


def load(
    metadata: Metadata,
    logger: Logger,
    dirs: Directories,
    dvc: Optional[SimpleDVC] = None,
) -> Metadata:
    """
    This function loads data from the metadata into a CSV file, and
    optionally tracks that file using Data Version Control (DVC).

    Design Principles:
    ------------------
    1. Single Responsibility Principle (SRP): This function solely
       handles loading data and potentially tracking it with DVC.

    2. Dependency Inversion Principle (DIP): Dependencies (logger,
       directories, metadata, and optionally DVC) are passed as
       arguments for better testability and flexibility.

    3. Open-Closed Principle (OCP): The function supports optional DVC
       tracking, showing it can be extended without modifying the
       existing code. In other words, the function is open for
       extension because it can be extended to support DVC tracking,
       and closed for modification because the existing code does not
       need to be modified to support DVC tracking.

    4. Use of Type Hints: Type hints are used consistently for clarity
       and to catch type-related errors early.

    5. Logging: Effective use of logging provides transparency and aids
       in troubleshooting.


    Areas of Improvement:
    ---------------------
    1. Exception Handling: More specific exception handling could
       improve error management.

    2. Liskov Substitution Principle (LSP): Using interfaces or base
       classes for the inputs could enhance flexibility and adherence
       to LSP.

    3. Encapsulation: Consider if direct manipulation of `metadata`
       attributes should be encapsulated within `metadata` methods.

    Parameters
    ----------
    metadata: Metadata
        The Metadata object containing the data to be loaded.
    logger: Logger
        The Logger object for logging information and errors.
    dirs: Directories
        The Directories object with the directories for data loading.
    dvc: Optional[SimpleDVC]
        The optional DVC object for data file tracking.

    Returns
    -------
    Metadata
        The Metadata object with updated information.
    """
    logger.info("Reading data from metadata computed in the previous step...")
    raw_df = metadata.raw_df
    table_name = metadata.raw_table_name
    filepath: Path = dirs.raw / f"{table_name}.csv"

    raw_df.to_csv(filepath, index=False)
    # Calculate the size of the file
    metadata.raw_file_size = filepath.stat().st_size

    # Determine the file format
    metadata.raw_file_format = filepath.suffix[1:]  # remove the leading dot

    metadata.release("raw_df")

    if dvc is not None:
        # add local file to dvc
        dvc_metadata = dvc.add(dirs.raw / f"{table_name}.csv")
        try:
            dvc.push(dirs.raw / f"{table_name}.csv")
        except Exception as error:
            logger.error(f"File is already tracked by DVC. Error: {error}")

        metadata.dvc_metadata = dvc_metadata
        pprint(metadata.dvc_metadata)

    return metadata


if __name__ == "__main__":
    # these can be tests, note create_new_dirs in production or a new run should
    # only be called once! Same for the logger.

    from common_utils.cloud.gcp.storage.gcs import GCS
    from common_utils.core.logger import Logger
    from rich.pretty import pprint

    from conf.init_dirs import ROOT_DIR
    from conf.init_project import initialize_project
    from pipeline_training.data_extraction.extract import (
        test_extract_from_data_warehouse,
    )

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

    ### load.py
    metadata = load(metadata=metadata, logger=logger, dirs=cfg.dirs, dvc=dvc)
    pprint(metadata)

    # reinitialize to pull
    # cfg = initialize_project(ROOT_DIR)
    # gcs = GCS(
    #     project_id=cfg.project_id,
    #     google_application_credentials=cfg.google_application_credentials,
    #     bucket_name=cfg.gcs_bucket_name,
    # )
    # dvc = SimpleDVC(data_dir=cfg.dirs.raw, storage=gcs)
    # filename = "filtered_movies_incremental.csv"
    # remote_project_name = "imdb"
    # dvc.pull(filename=filename, remote_project_name=remote_project_name)
