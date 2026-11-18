import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def sample_data():
    """A small, deterministic sample of census-style rows."""
    rows = [
        [39, "State-gov", 77516, "Bachelors", 13, "Never-married",
         "Adm-clerical", "Not-in-family", "White", "Male", 2174, 0, 40,
         "United-States", "<=50K"],
        [50, "Self-emp-not-inc", 83311, "Bachelors", 13,
         "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male",
         0, 0, 13, "United-States", "<=50K"],
        [38, "Private", 215646, "HS-grad", 9, "Divorced",
         "Handlers-cleaners", "Not-in-family", "White", "Male", 0, 0, 40,
         "United-States", "<=50K"],
        [52, "Self-emp-inc", 287927, "HS-grad", 9, "Married-civ-spouse",
         "Exec-managerial", "Wife", "White", "Female", 15024, 0, 40,
         "United-States", ">50K"],
        [31, "Private", 45781, "Masters", 14, "Never-married",
         "Prof-specialty", "Not-in-family", "White", "Female", 14084, 0, 50,
         "United-States", ">50K"],
        [42, "Private", 159449, "Bachelors", 13, "Married-civ-spouse",
         "Exec-managerial", "Husband", "White", "Male", 5178, 0, 40,
         "United-States", ">50K"],
    ]
    columns = [
        "age", "workclass", "fnlgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "salary",
    ]
    return pd.DataFrame(rows, columns=columns)


def test_process_data_shapes(sample_data):
    """ Process data should return feature and label arrays of the correct shape, 
    and the label array should be 0 or 1."""
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary",
        training=True,
    )
    assert X.shape[0] == len(sample_data)
    assert y.shape[0] == len(sample_data)
    assert set(np.unique(y)).issubset({0, 1})
    assert encoder is not None and lb is not None


def test_compute_model_metrics_known_values():
    """Compute model metrics should return the correct precision, recall, and F1 values."""
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 1, 0])
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_train_model_returns_random_forest(sample_data):
    """Train_model should return a fitted RandomForestClassifier, inference should return binary predictions."""
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary",
        training=True,
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})