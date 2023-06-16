"""
This module is used to create the directory structure for the project and
demonstrates the application of several design patterns and good software
engineering practices:

1. Singleton Pattern: The Enums (`BaseDirType`, `DataDirType`, `StoresDirType`)
   act as a form of the Singleton pattern, defining a set of **unique** and
   **constant** values.

2. Factory Pattern: The `create_new_dirs` function acts as a factory, producing
   a complex `Directories` object based on an optional `run_id` parameter.
   It is not the in the strict sense of the Factory pattern, but it is a
   "factor" that encapsulates the logic of creating the `Directories` object.

3. Dependency Injection: `create_new_dirs` function demonstrates Dependency
   Injection by accepting an optional `run_id`, increasing its flexibility.

4. Use of Model Classes: The `Directories` class is a Pydantic model, which
   allows for easy data validation and settings management.

5. Law of Demeter: By providing the individual directories as properties of the
   `Directories` model, the code adheres to the Law of Demeter (also known as
   the "principle of least knowledge").

6. Immutability: The Paths are all built from immutable `Path` objects, aiding
   in program state management.

7. Separation of Concerns: The code cleanly separates the concerns of defining
   the directory structure (`BaseDirType`, `DataDirType`, `StoresDirType`,
   `Directories`) and creating the directories (`create_new_dirs`).
"""


from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from common_utils.core.common import generate_uuid
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module
from rich.pretty import pprint

# Note, using enum means that the values are unique and constant.
# Using pydanitc models means that the values may be unique, but they are not
# constant.


class BaseDirType(Enum):
    DATA = "data"
    STORES = "stores"


class DataDirType(Enum):
    RAW = "raw"
    PROCESSED = "processed"


class StoresDirType(Enum):
    LOGS = "logs"
    REGISTRY = "registry"
    ARTIFACTS = "artifacts"
    BLOB = "blob"


class BlobDirType(Enum):
    RAW = "raw"
    PROCESSED = "processed"


class BlobDirectories(BaseModel):
    raw: Path = Field(..., description="This is the directory for raw blob data.")
    processed: Path = Field(
        ..., description="This is the directory for processed blob data."
    )


class DataDirectories(BaseModel):
    raw: Path = Field(..., description="This is the directory for raw data.")
    processed: Path = Field(
        ..., description="This is the directory for processed data."
    )


class StoresDirectories(BaseModel):
    logs: Path = Field(..., description="This is the directory for logs.")
    registry: Path = Field(..., description="This is the directory for registry.")
    artifacts: Path = Field(..., description="This is the directory for artifacts.")
    blob: BlobDirectories = Field(..., description="This is the blob directories.")


class Directories(BaseModel):
    root: Path = Field(..., description="This is the root directory.")
    data: DataDirectories = Field(..., description="This is the data directories.")
    stores: StoresDirectories = Field(
        ..., description="This is the stores directories."
    )


ROOT_DIR = Path(__file__).parent.parent.absolute()
pprint(f"ROOT_DIR: {ROOT_DIR}")


def create_new_dirs(run_id: Optional[str] = None) -> Directories:
    if not run_id:
        run_id = generate_uuid()

    data_dir = Path(ROOT_DIR, BaseDirType.DATA.value)
    stores_dir = Path(ROOT_DIR, BaseDirType.STORES.value, run_id)

    data_dirs_dict: Dict[str, Path] = {}
    stores_dirs_dict: Dict[str, Path] = {}
    blob_dirs_dict: Dict[str, Path] = {}

    # Creating directories under data
    for data_dir_type in DataDirType:
        data_dir_path = Path(data_dir, data_dir_type.value)
        data_dir_path.mkdir(parents=True, exist_ok=True)
        data_dirs_dict[data_dir_type.value] = data_dir_path

    # Creating directories under stores
    for store_dir_type in StoresDirType:
        store_dir_path = Path(stores_dir, store_dir_type.value)
        store_dir_path.mkdir(parents=True, exist_ok=True)
        if store_dir_type.value == "blob":
            for blob_dir_type in BlobDirType:
                blob_dir_path = Path(store_dir_path, blob_dir_type.value)
                blob_dir_path.mkdir(parents=True, exist_ok=True)
                blob_dirs_dict[blob_dir_type.value] = blob_dir_path
            stores_dirs_dict[store_dir_type.value] = BlobDirectories(**blob_dirs_dict)
        else:
            stores_dirs_dict[store_dir_type.value] = store_dir_path

    dirs_dict = {
        "root": ROOT_DIR,
        "data": DataDirectories(**data_dirs_dict),
        "stores": StoresDirectories(**stores_dirs_dict),
    }

    return Directories(**dirs_dict)


if __name__ == "__main__":
    # TODO: can go into tests
    dirs = create_new_dirs()
    pprint(dirs)
