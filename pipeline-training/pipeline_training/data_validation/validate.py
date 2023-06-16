"""In production, one should use great_expectations to validate data.
This is just a simple example of how to validate data using pandas without
any external dependencies.

Also, best is put schema in a immutable place so that it can be used by
any downstream processes.

TODO:
    - Refactor the function such that if there exists errors, it will
        update the metadata with the error and status. So the next step
        can halt upon seeing the error.
"""
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from common_utils.core.logger import Logger
from common_utils.data_validator.core import DataFrameValidator

from conf.metadata import Metadata


# pre-load-post-extract
def validate_raw(
    df: pd.DataFrame, schema: Dict[str, Any], metadata: Metadata
) -> Metadata:
    """
    Validates a raw DataFrame against a schema and updates metadata.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be validated.
    schema : Dict[str, Any]
        Schema to validate the DataFrame against.
    metadata : Metadata
        Metadata object where validation results and relevant attributes will be stored.

    Returns
    -------
    Metadata
        Updated metadata with attributes set from the validation results.

    Raises
    ------
    ValidationError
        If the DataFrame does not conform to the schema.
    """
    validator = DataFrameValidator(df=df, schema=schema)
    validator.validate_schema().validate_data_types().validate_missing()

    attr_dict = {
        "validation_status": True,
        "validation_time": datetime.now(),
        "validation_errors": None,
    }
    metadata.raw_validation_dict = attr_dict
    return metadata


def validate_processed(df: pd.DataFrame, schema: Dict[str, Any]) -> None:
    ...


from types import SimpleNamespace


def test_validate_raw(cfg: SimpleNamespace, metadata: Metadata, logger: Logger):
    raw_df = metadata.raw_df

    validate_raw(df=raw_df, schema=cfg.general.raw_schema, metadata=metadata)


if __name__ == "__main__":
    # these can be tests, note create_new_dirs in production or a new run should
    # only be called once! Same for the logger.

    from conf.init_dirs import ROOT_DIR
    from conf.init_project import initialize_project
    from pipeline_training.data_extraction.extract import (
        test_extract_from_data_warehouse,
    )

    cfg = initialize_project(ROOT_DIR)
    logger = Logger(
        log_file="pipeline_training.log",
        log_root_dir=cfg.general.dirs.stores.logs,
        module_name=__name__,
        propagate=False,
    ).logger

    metadata: Metadata = test_extract_from_data_warehouse(cfg, logger)

    test_validate_raw(cfg=cfg, logger=logger, metadata=metadata)
