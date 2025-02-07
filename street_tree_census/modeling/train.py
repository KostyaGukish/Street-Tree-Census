from pathlib import Path

import typer
from tqdm import tqdm

import pandas as pd
import numpy as np

import joblib

from sklearn.compose import ColumnTransformer

import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from street_tree_census.modeling.model import TreeCensusModel
from street_tree_census.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET,
    OPTIMIZER_LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    DROPOUT,
    SCHEDULER_STEP_SIZE,
    SCHEDULER_GAMMA,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = typer.Typer()


def fit_epoch(model, train_loader, criterion, optimizer):
    torch.manual_seed(SEED)
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc


def train(train_dataset, model, epochs, opt, criterion, scheduler, batch_size):
    torch.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} train_acc {t_acc:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt)

            pbar_outer.update(1)
            tqdm.write(
                log_template.format(ep=epoch + 1, t_loss=train_loss, t_acc=train_acc)
            )

            scheduler.step()

    return


@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    preprocessor_path: Path = MODELS_DIR / "preprocessor.pkl",
    target_le_path: Path = MODELS_DIR / "target_le.pkl",
):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    data = pd.read_csv(data_path)

    target_le = LabelEncoder()
    y = target_le.fit_transform(data[TARGET].squeeze()).astype(int)
    joblib.dump(target_le, target_le_path)

    X = data[FEATURE_COLUMNS]
    categories_list = [X[col].unique().tolist() for col in CATEGORICAL_COLUMNS]
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), NUMERICAL_COLUMNS),
            (
                "OHE",
                OneHotEncoder(
                    sparse_output=False, drop="first", categories=categories_list
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
    )
    X = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, preprocessor_path)

    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    dataset = TensorDataset(X, y)

    model = TreeCensusModel(
        input_dim=X.shape[1], output_dim=len(y.unique()), dropout_prob=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=OPTIMIZER_LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
    )

    model.to(DEVICE)
    train(
        dataset,
        model=model,
        epochs=EPOCHS,
        opt=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        batch_size=BATCH_SIZE,
    )

    torch.save(model, model_path)


if __name__ == "__main__":
    app()
