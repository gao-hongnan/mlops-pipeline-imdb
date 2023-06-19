import re
from pathlib import Path
from typing import Any, Dict, List, Union

from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC
from nltk.stem import PorterStemmer
from rich.pretty import pprint

from conf.init_dirs import Directories
from conf.metadata import Metadata


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_stopwords(text: str, stopwords: List[str]) -> str:
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    return pattern.sub("", text)


def add_spacing_between_objects(text: str) -> str:
    return re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)


def remove_non_alphanumeric_chars(text: str) -> str:
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    return text.strip()


def remove_links(text: str) -> str:
    return re.sub(r"http\S+", "", text)


def stem_text(text: str) -> str:
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split(" ")])


def clean_text(text: str, lower: bool, stem: bool, stopwords: List[str]) -> str:
    text = to_lowercase(text) if lower else text
    text = remove_stopwords(text, stopwords)
    text = add_spacing_between_objects(text)
    text = remove_non_alphanumeric_chars(text)
    text = remove_links(text)
    text = stem_text(text) if stem else text
    return text


def select_columns(df):
    return df[["originalTitle", "genres", "averageRating"]]


def round_average_rating(df) -> None:
    df["rounded_averageRating"] = df["averageRating"].round().astype(int)


def replace_space_with_underscore(df, column_name: str) -> None:
    df[column_name] = df[column_name].apply(lambda x: x.replace(" ", "_"))


def drop_na(df, subset: List[str]) -> None:
    df.dropna(subset=subset, inplace=True)


def concat_title_genres(df) -> None:
    df["concat_title_genres"] = df["cleaned_originalTitle"] + "-" + df["cleaned_genres"]


def create_cleaned_column(df, column_name: str, cleaning_args: Dict[str, Any]) -> None:
    df[f"cleaned_{column_name}"] = df[column_name].apply(clean_text, **cleaning_args)


def preprocess_data(
    cfg, metadata: Metadata, logger: Logger, dirs: Directories, dvc: SimpleDVC
) -> Metadata:
    """Preprocess the data."""
    logger.info("Preprocessing the data...")

    df = metadata.raw_df.copy()

    df = select_columns(df)

    create_cleaned_column(df, "originalTitle", cfg.transform.__dict__)

    round_average_rating(df)

    create_cleaned_column(df, "genres", cfg.transform.__dict__)

    replace_space_with_underscore(df, "cleaned_genres")

    # hardcode here because one title is ¬ø and somehow it become nan after cleaning.
    drop_na(df, ["cleaned_originalTitle", "cleaned_genres"])
    concat_title_genres(df)

    pprint(df.rounded_averageRating.value_counts())
    pprint(df.head())

    filepath: Path = dirs.data.processed / f"{metadata.raw_table_name}.csv"
    df.to_csv(filepath, index=False)

    if dvc is not None:
        # add local file to dvc
        processed_dvc_metadata = dvc.add(filepath)
        try:
            dvc.push(filepath)
        except Exception as error:  # pylint: disable=broad-except
            logger.error(f"File is already tracked by DVC. Error: {error}")

    attr_dict = {
        "processed_df": df,
        "processed_num_rows": df.shape[0],
        "processed_num_cols": df.shape[1],
        "processed_file_size": filepath.stat().st_size,
        "processed_dvc_metadata": processed_dvc_metadata if dvc is not None else None,
    }
    metadata.set_attrs(attr_dict)

    logger.info("Releasing raw_df...")
    metadata.release("raw_df")
    assert metadata.raw_df is None

    logger.info("Preprocessing complete.")
    return metadata


def validate_input(input_data: Union[str, List[str]], input_name: str) -> List[str]:
    if isinstance(input_data, str):
        input_data = [input_data]
    elif isinstance(input_data, list):
        if not all(isinstance(item, str) for item in input_data):
            raise ValueError(f"All elements in the {input_name} must be strings.")
    else:
        raise TypeError(f"{input_name} must be a string or a list of strings.")
    return input_data


def process_input_data(
    genres: Union[str, List[str]], titles: Union[str, List[str]]
) -> List[str]:
    genres = validate_input(genres, "genres")
    titles = validate_input(titles, "titles")

    if len(genres) != len(titles):
        raise ValueError("genres and titles must have the same length.")

    return [title + "-" + genre for title, genre in zip(titles, genres)]
