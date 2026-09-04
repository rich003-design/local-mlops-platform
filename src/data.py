"""
Load and prepare the training dataset.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset():
    """
    Load the breast cancer dataset and create train/test splits.
    """

    dataset = load_breast_cancer(
        as_frame=True
    )

    dataframe = dataset.frame.copy()

    X = dataframe.drop(
        columns=["target"]
    )

    y = dataframe["target"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        list(X.columns),
    )
