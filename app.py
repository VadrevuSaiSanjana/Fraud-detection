"""
app.py – FastAPI server for Insurance-Fraud Detector
===================================================

Endpoints
---------
POST /upload-csvs   : Upload four CSVs, run preprocessing + models, return predictions.
POST /predict-db    : Provide DB credentials, pull four tables, run full pipeline.
GET  /download/{fn} : (Optional) download any file cached during a session.
GET  /health        : Health-check.

All heavy lifting is delegated to:
    preprocessing.merge_and_clean.merge_csvs
    models.predict.run_models
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import create_engine

from preprocessing.merge_and_clean import merge_csvs
from models.predict import run_models

# ────────────────────────── FastAPI setup ──────────────────────────
app = FastAPI(title="Insurance Fraud Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional cache of generated artefacts this run
generated_files: Dict[str, str] = {}

# ────────────────────────── Helpers ────────────────────────────────
def _save_upload(upload: UploadFile, dst: str) -> str:
    with open(dst, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dst


def _temp_dir() -> str:
    return tempfile.mkdtemp(prefix="frauddet_")

# ────────────────────────── Routes ─────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Insurance-Fraud Detector backend is live."}


@app.post("/upload-csvs")
async def upload_csvs(
    bene: UploadFile = File(...),
    ip:   UploadFile = File(...),
    op:   UploadFile = File(...),
    tgt:  UploadFile = File(...),
):
    try:
        tmp = _temp_dir()
        paths = {
            "bene": _save_upload(bene, os.path.join(tmp, "bene.csv")),
            "ip":   _save_upload(ip,   os.path.join(tmp, "ip.csv")),
            "op":   _save_upload(op,   os.path.join(tmp, "op.csv")),
            "tgt":  _save_upload(tgt,  os.path.join(tmp, "tgt.csv")),
        }

        merged_csv = merge_csvs(
            paths["bene"], paths["ip"], paths["op"], paths["tgt"], out_dir=tmp
        )
        preds_csv  = run_models(merged_csv, tmp)

        generated_files[Path(preds_csv).name] = preds_csv
        return FileResponse(
            preds_csv,
            filename="predictions.csv",
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict-db")
async def predict_db(
    hostname: str = Form(...),
    port:     int = Form(...),
    user:     str = Form(...),
    password: str = Form(...),
    dbname:   str = Form(...),
    bene_tbl: str = Form(default="beneficiary"),
    ip_tbl:   str = Form(default="inpatient"),
    op_tbl:   str = Form(default="outpatient"),
    tgt_tbl:  str = Form(default="target"),
):
    try:
        tmp = _temp_dir()
        eng = create_engine(f"postgresql://{user}:{password}@{hostname}:{port}/{dbname}")

        tbl_map = {"bene": bene_tbl, "ip": ip_tbl, "op": op_tbl, "tgt": tgt_tbl}
        csv_paths = {}
        for key, tbl in tbl_map.items():
            df = pd.read_sql_table(tbl, eng)
            pth = os.path.join(tmp, f"{key}.csv")
            df.to_csv(pth, index=False)
            csv_paths[key] = pth

        merged_csv = merge_csvs(
            csv_paths["bene"], csv_paths["ip"], csv_paths["op"], csv_paths["tgt"], tmp
        )
        preds_csv = run_models(merged_csv, tmp)

        generated_files[Path(preds_csv).name] = preds_csv
        return FileResponse(
            preds_csv,
            filename="predictions.csv",
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/download/{filename}")
async def download(filename: str):
    if filename in generated_files and os.path.exists(generated_files[filename]):
        return FileResponse(generated_files[filename], filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
