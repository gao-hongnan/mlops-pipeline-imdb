import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC
from nltk.stem import PorterStemmer
from rich.pretty import pprint
from sklearn.model_selection import train_test_split

from conf.init_dirs import Directories
from conf.metadata import Metadata


def clean_text(text: str, lower: bool, stem: bool, stopwords: List[str]) -> str:
    """Clean raw text.
    Args:
        text (str): raw text to be cleaned.
        lower (bool): whether to lowercase the text.
        stem (bool): whether to stem the text.
    Returns:
        str: cleaned text.
    """
    # Lower
    if lower:
        text = text.lower()

    # Remove stopwords
    if stopwords is not None or len(stopwords) > 0:
        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)

    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        stemmer = PorterStemmer()
        text = " ".join(
            [stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")]
        )

    return text


def preprocess_data(
    cfg, metadata: Metadata, logger: Logger, dirs: Directories, dvc: SimpleDVC
) -> Metadata:
    """Preprocess the data."""
    logger.info("Preprocessing the data...")

    df = metadata.raw_df.copy()

    df = df[["originalTitle", "genres", "averageRating"]]
    df["cleaned_originalTitle"] = df["originalTitle"].apply(
        clean_text,
        lower=cfg.lower,
        stem=cfg.stem,
        stopwords=cfg.stopwords,
    )
    df["rounded_averageRating"] = df["averageRating"].round().astype(int)
    df["cleaned_genres"] = df["genres"].apply(
        clean_text,
        lower=cfg.lower,
        stem=cfg.stem,
        stopwords=cfg.stopwords,
    )
    df["cleaned_genres"] = df["cleaned_genres"].apply(
        lambda x: x.replace(" ", "_")
    )  # concat space

    # hardcode here because one title is ¬ø and somehow it become nan after cleaning.
    df = df.dropna(subset=["cleaned_genres", "cleaned_originalTitle"])
    df["concat_title_genres"] = df["cleaned_originalTitle"] + "-" + df["cleaned_genres"]

    pprint(df.rounded_averageRating.value_counts())
    pprint(df.head())

    filepath: Path = dirs.processed / f"{metadata.raw_table_name}.csv"
    df.to_csv(filepath, index=False)

    if dvc is not None:
        # add local file to dvc
        processed_dvc_metadata = dvc.add(filepath, save_metadata=True)
        try:
            dvc.push(filepath)
        except Exception as error:  # pylint: disable=broad-except
            logger.error(f"File is already tracked by DVC. Error: {error}")

        metadata.processed_dvc_metadata = processed_dvc_metadata
        pprint(metadata.processed_dvc_metadata)

    attr_dict = {
        "processed_df": df,
        "processed_num_rows": df.shape[0],
        "processed_num_cols": df.shape[1],
        "processed_file_size": filepath.stat().st_size,
    }
    metadata.set_attrs(attr_dict)

    logger.info("Releasing raw_df...")
    metadata.release("raw_df")
    assert metadata.raw_df is None

    logger.info("Preprocessing complete.")

    return metadata


def process_input_data(genres: str, titles: str) -> List[str]:
    if isinstance(genres, str) and isinstance(titles, str):
        return [titles + "-" + genres]
    # both are lists
    return [title + "-" + genre for title, genre in zip(titles, genres)]
