import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib

import torch
from torch.utils.data import TensorDataset, DataLoader

from modeling.model import *
from config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET, SEED

data_path = (PROCESSED_DATA_DIR / "dataset.csv",)
model_path = (MODELS_DIR / "model.pkl",)
preprocessor_path = (MODELS_DIR / "preprocessor.pkl",)
target_le_path = (MODELS_DIR / "target_le.pkl",)
predictions_path = (PROCESSED_DATA_DIR / "predictions.csv",)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()


class Tree(BaseModel):
    curb_loc: str
    spc_common: str
    guards: str
    sidewalk: str
    user_type: str
    borough: str
    block_id: float
    tree_dbh: float
    steward: float
    postcode: float
    cncldist: float
    st_senate: float
    nta: float
    ct: float
    x_sp: float
    y_sp: float
    problems: float
    root_problems: float
    trunk_problems: float
    brch_problems: float


def predict_data(model, preprocessor, target_le, X):
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


@app.post("/predict")
def predict(tree: Tree):
    features = {
        "block_id": [tree.block_id],
        "steward": [tree.steward],
        "tree_dbh": [tree.tree_dbh],
        "postcode": [tree.postcode],
        "cncldist": [tree.cncldist],
        "st_senate": [tree.st_senate],
        "nta": [tree.nta],
        "ct": [tree.ct],
        "x_sp": [tree.x_sp],
        "y_sp": [tree.y_sp],
        "problems": [tree.problems],
        "root_problems": [tree.root_problems],
        "trunk_problems": [tree.trunk_problems],
        "brch_problems": [tree.brch_problems],
        "curb_loc": [tree.curb_loc],
        "spc_common": [tree.spc_common],
        "guards": [tree.guards],
        "sidewalk": [tree.sidewalk],
        "user_type": [tree.user_type],
        "borough": [tree.borough],
    }

    X = pd.DataFrame.from_dict(features)

    preprocessor = joblib.load(preprocessor_path[0])
    target_le = joblib.load(target_le_path[0])
    model = torch.load(model_path[0], weights_only=False)

    return predict_data(model, preprocessor, target_le, X)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=80)
