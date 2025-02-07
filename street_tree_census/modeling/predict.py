from pathlib import Path

import typer
import joblib

import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

from street_tree_census.modeling.model import *
from street_tree_census.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    FEATURE_COLUMNS,
    TARGET,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = typer.Typer()


def predict(model, preprocessor, target_le, X):
    model.eval()

    X = preprocessor.transform(X)
    X = torch.FloatTensor(X)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    labels = []

    torch.manual_seed(SEED)
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs[0].to(DEVICE)
            outputs = model(inputs).cpu()
            labels.extend(torch.argmax(outputs, 1).numpy())

    labels = target_le.inverse_transform(labels)
    labels = pd.DataFrame({TARGET[0]: labels})

    return labels


@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    preprocessor_path: Path = MODELS_DIR / "preprocessor.pkl",
    target_le_path: Path = MODELS_DIR / "target_le.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
):
    X = pd.read_csv(data_path)
    X = X[FEATURE_COLUMNS]

    preprocessor = joblib.load(preprocessor_path)
    target_le = joblib.load(target_le_path)
    model = torch.load(model_path, weights_only=False)

    y = predict(model, preprocessor, target_le, X)

    y.to_csv(predictions_path)


if __name__ == "__main__":
    app()
