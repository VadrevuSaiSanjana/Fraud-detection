"""
predict.py – Inference engine
=============================

• Loads every *.pkl in backend/models/models/
• Runs predictions on the merged DataFrame
• Saves a predictions CSV with individual model flags, ensemble vote & metadata
"""

import glob
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS_DIR = Path(__file__).parent / "models"


# ────────────────────────── Utilities ─────────────────────────────
def _load_models() -> Dict[str, any]:
    models = {}
    for pkl in glob.glob(str(MODELS_DIR / "*.pkl")):
        name = Path(pkl).stem
        try:
            models[name] = joblib.load(pkl)
            logger.info(f"Loaded model ▶ {name}")
        except Exception as e:
            logger.error(f"Failed loading {name}: {e}")
    if not models:
        raise RuntimeError(f"No .pkl models found in {MODELS_DIR}")
    return models


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Provider" not in df.columns:
        raise ValueError("Merged data lacks required 'Provider' column.")
    X = df.drop(columns=["Provider"]).copy()
    # Coerce any residual objects to numeric
    for c in X.select_dtypes(include="object").columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X.fillna(0)


# ────────────────────────── Main entry ────────────────────────────
def run_models(merged_csv: Union[str, Path], out_dir: Union[str, Path]) -> str:
    df = pd.read_csv(merged_csv)
    X = _prepare_features(df)
    models = _load_models()

    preds = pd.DataFrame({"Provider": df["Provider"]})
    votes: List[np.ndarray] = []

    for name, model in models.items():
        y_hat = model.predict(X)
        preds[f"{name}_flag"] = y_hat
        votes.append(y_hat)

    # Majority vote
    vote_matrix = np.column_stack(votes)
    ensemble = (vote_matrix.mean(axis=1) >= 0.5).astype(int)
    preds["Fraud_Flag"] = ensemble
    preds["Model_Count"] = len(models)
    preds["Run_Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.csv"
    preds.to_csv(out_file, index=False)
    logger.info(f"Predictions saved → {out_file}")

    return str(out_file)
