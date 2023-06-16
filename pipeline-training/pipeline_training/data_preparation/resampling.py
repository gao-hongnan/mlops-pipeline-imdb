"""
Simple train-val-test split instead of K-Folds and its variants.
As mentioned in GCP blog, the output of data prepartion is
the data *splits* in the prepared format.
"""
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def get_data_splits(cfg, X: np.ndarray, y: np.ndarray) -> Tuple:
    """Generate balanced data splits.

    NOTE: As emphasized this is not a modelling project. This is a project
    to illustrate end-to-end MLOps. Therefore, we will not be using
    a lot of "SOTA" methods.

    Args:
        X (pd.Series): input features.
        y (np.ndarray): encoded labels.
        train_size (float, optional): proportion of data to use for training. Defaults to 0.7.
    Returns:
        Tuple: data splits as Numpy arrays.
    """
    # 70-15-15 split
    X_train, X_, y_train, y_ = train_test_split(
        X, y, stratify=y, **cfg.resampling.strategy["train_test_split"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_, random_state=42, shuffle=True
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test)
