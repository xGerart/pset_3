"""Registro de corridas en MLflow con el mismo contrato para todas las fases.

Cada corrida guarda: parámetros (hiperparámetros + semilla), datos (tabla de origen,
fecha de corte y hash del snapshot), métricas (entrenamiento, CV media y desviación,
costo de negocio, segundos de ajuste), tags (fase, familia, commit de Git) y el
modelo entrenado con su firma y un ejemplo de entrada.
"""
from __future__ import annotations

import os
import subprocess

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature

from src.datos import RAIZ, Snapshot
from src.pipeline import ENTRADA, INICIO_PRUEBA, SEMILLA, Resultado

EXPERIMENTO = "ventas-hora-matriz"
MODELO_REGISTRADO = f"{EXPERIMENTO}-modelo"


def configurar() -> str:
    load_dotenv(RAIZ / ".env")
    uri = os.environ["MLFLOW_TRACKING_URI"]   # definido en .env (ver .env.example)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENTO)
    return uri


def commit_git() -> str:
    """Commit del código que generó la corrida; '-dirty' si src/ tenía cambios sin
    commitear. Los notebooks no cuentan: al ejecutarlos cambian sus salidas, no su código."""
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
                         text=True, check=True).stdout.strip()
    sucio = subprocess.run(["git", "status", "--porcelain", "--", "src"],
                           cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()
    return sha + ("-dirty" if sucio else "")


def _params_modelo(modelo) -> dict:
    """Hiperparámetros del estimador final (no los del preprocesamiento)."""
    est = modelo.named_steps["modelo"] if hasattr(modelo, "named_steps") else modelo
    return {k: v for k, v in est.get_params(deep=False).items()
            if isinstance(v, (int, float, str, bool, type(None)))}


def registrar(nombre: str, modelo, resultado: Resultado, *, fase: str, familia: str,
              snapshot: Snapshot, train: pd.DataFrame, params_extra: dict | None = None,
              nested: bool = False, guardar_modelo: bool = True) -> str:
    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        mlflow.log_params({**_params_modelo(modelo), "semilla": SEMILLA, **(params_extra or {})})
        mlflow.log_params({
            "datos_tabla_origen": snapshot.tabla_origen,
            "datos_fecha_corte": snapshot.fecha_corte,
            "datos_hash": snapshot.hash,
            "datos_inicio_prueba": INICIO_PRUEBA.date().isoformat(),
            "datos_filas_train": len(train),
        })
        mlflow.log_input(mlflow.data.from_pandas(
            train, source=snapshot.tabla_origen, name=f"obt_matriz@{snapshot.fecha_corte}",
            digest=snapshot.hash[:20], targets="sales_count"), context="training")
        mlflow.log_metrics({k: v for k, v in resultado.metricas.items() if np.isfinite(v)})
        for i, w in enumerate(resultado.wape_pliegues):
            mlflow.log_metric("cv_wape_pliegue", w, step=i)
        mlflow.set_tags({"fase": fase, "familia": familia, "git_commit": commit_git(),
                         "modelo_clase": type(modelo.named_steps["modelo"] if hasattr(modelo, "named_steps") else modelo).__name__})
        if guardar_modelo:
            ejemplo = train[ENTRADA].head(5)
            # La firma solo necesita el tipo de la salida: se toma de las predicciones fuera de
            # pliegue. Predecir sobre el entrenamiento no tiene sentido para un modelo zero-shot
            # (Chronos o el LLM pronostican lo que viene DESPUÉS de su historia).
            salida = resultado.oof[~np.isnan(resultado.oof)][:5].astype(float)
            firma = infer_signature(ejemplo, salida)
            # cloudpickle + src/ empaquetado: las clases propias (regla, Chronos, LLM) se
            # cargan desde models:/ en un entorno limpio. skops no admite clases propias.
            mlflow.sklearn.log_model(resultado.modelo, name="modelo", signature=firma,
                                     input_example=ejemplo, serialization_format="cloudpickle",
                                     code_paths=[str(RAIZ / "src")])
        return run.info.run_id
