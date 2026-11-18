"""Preparación, validación y métricas compartidas por todos los notebooks.

Todo lo que se *ajusta* a los datos (imputación, codificación, escalado) vive dentro
de un Pipeline de scikit-learn, así que en la validación cruzada se ajusta solo con
el pliegue de entrenamiento. Las features derivadas de `agregar_features` no se
ajustan: son transformaciones deterministas que solo miran fechas >= 7 días atrás,
por eso se calculan una vez sobre toda la tabla sin fuga.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEMILLA = 42
TARGET = "sales_count"

# Columnas de la OBT que NO pueden ser predictores: se observan en la misma hora que
# se quiere predecir (o se calculan a partir del target).
FUGA = ["net_sales_amount", "units_sold", "admin_sales_count", "load_per_staff",
        "is_overloaded", "obt_built_at"]

# Inicio del conjunto de prueba (hold-out): las dos últimas semanas completas.
INICIO_PRUEBA = pd.Timestamp("2026-10-26")
N_PLIEGUES = 5

# Regla de negocio del PSet 1: una hora necesita refuerzo si la carga es >= 8 ventas
# por vendedor presente.
UMBRAL_CARGA = 8.0
# Costos del error (PSet 1): una hora sobrecargada que no se refuerza pierde, como
# mínimo, una venta atendida (margen bruto medio $7,19); reforzar una hora que no lo
# necesitaba cuesta una hora de medio tiempo ($2,65).
COSTO_FN = 7.19
COSTO_FP = 2.65

CATEGORICAS = ["hour_of_day", "iso_day_of_week"]
BINARIAS = ["is_holiday", "is_last_open_hour", "ventana_pago"]
NUMERICAS = ["sales_lag_7d", "sales_lag_14d", "sales_lag_21d", "sales_lag_28d",
             "sales_avg_4w_same_hour", "ventas_dia_lag_7d", "staff_present"]
FEATURES = CATEGORICAS + BINARIAS + NUMERICAS
# Entrada de todos los modelos: las features + la fecha (los modelos de series de
# tiempo la necesitan para ordenar la historia; el ColumnTransformer la descarta).
ENTRADA = ["hour_date"] + FEATURES


# ----------------------------------------------------------------------- features
def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas (ver notebook 02). Ambas existen al momento de planificar
    la semana siguiente: una es calendario y la otra mira 7 días atrás."""
    out = df.copy()
    dia = out["hour_date"].dt.day
    # Hallazgo EDA: alrededor del fin de mes (pago de sueldos) la venta sube.
    out["ventana_pago"] = ((dia >= 28) | (dia <= 3)).astype(int)
    # Venta total del mismo día de la semana anterior: nivel del día, menos ruidoso
    # que el rezago de una sola hora.
    diaria = out.groupby("hour_date", as_index=False)[TARGET].sum()
    diaria["hour_date"] = diaria["hour_date"] + pd.Timedelta(days=7)
    diaria = diaria.rename(columns={TARGET: "ventas_dia_lag_7d"})
    out = out.merge(diaria, on="hour_date", how="left")
    for c in ["is_holiday", "is_last_open_hour"]:
        out[c] = out[c].astype(int)
    return out


# ------------------------------------------------------------------ preparación
def construir_preprocesador(escalar: bool = True) -> ColumnTransformer:
    numericas = [("imputar", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        numericas.append(("escalar", StandardScaler()))
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
            ("bin", "passthrough", BINARIAS),
            ("num", Pipeline(numericas), NUMERICAS),
        ],
        verbose_feature_names_out=False,
    )


def construir_pipeline(modelo, escalar: bool = True) -> Pipeline:
    return Pipeline([("prep", construir_preprocesador(escalar)), ("modelo", modelo)])


# ---------------------------------------------------------------- división y CV
@dataclass(frozen=True)
class Division:
    train: pd.DataFrame
    test: pd.DataFrame


def dividir(df: pd.DataFrame) -> Division:
    """Divide una sola vez: todo lo anterior a INICIO_PRUEBA es entrenamiento."""
    df = df.sort_values(["hour_date", "hour_of_day"]).reset_index(drop=True)
    es_test = df["hour_date"] >= INICIO_PRUEBA
    return Division(df[~es_test].reset_index(drop=True), df[es_test].reset_index(drop=True))


def pliegues_temporales(train: pd.DataFrame, n: int = N_PLIEGUES):
    """TimeSeriesSplit por semanas: cada pliegue valida una semana completa y entrena
    con todas las semanas anteriores (ventana expansiva). Se corta por semana y no por
    fila porque la decisión real es planificar la semana siguiente entera."""
    semanas = train["week_start_date"].drop_duplicates().sort_values().to_list()
    pliegues = []
    for semana in semanas[-n:]:
        idx_train = np.flatnonzero(train["week_start_date"] < semana)
        idx_val = np.flatnonzero(train["week_start_date"] == semana)
        pliegues.append((idx_train, idx_val))
    return pliegues


# --------------------------------------------------------------------- métricas
def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / y.sum())


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def decision_refuerzo(yhat, staff, umbral: float = UMBRAL_CARGA) -> np.ndarray:
    return np.asarray(yhat, float) / np.asarray(staff, float) >= umbral


def costo_negocio(y, yhat, staff, umbral: float = UMBRAL_CARGA) -> dict:
    """Costo de decidir el refuerzo con la predicción, frente a la carga real."""
    real = decision_refuerzo(y, staff)
    pred = decision_refuerzo(yhat, staff, umbral)
    fn = int((real & ~pred).sum())
    fp = int((~real & pred).sum())
    tp = int((real & pred).sum())
    return {
        "costo": fn * COSTO_FN + fp * COSTO_FP,
        "fn": fn, "fp": fp, "tp": tp,
        "recall_sobrecarga": tp / real.sum() if real.sum() else float("nan"),
    }


def umbral_optimo(y, yhat, staff, grilla=np.arange(5.0, 10.01, 0.25)) -> float:
    """Umbral de carga PREDICHA que minimiza el costo (se elige con predicciones fuera
    de pliegue del entrenamiento, nunca con la prueba)."""
    costos = [costo_negocio(y, yhat, staff, u)["costo"] for u in grilla]
    return float(grilla[int(np.argmin(costos))])


# ------------------------------------------------------------------ líneas base
class ReglaPromedio4Semanas(BaseEstimator, RegressorMixin):
    """Línea base de negocio del PSet 1: la venta de la misma hora y día de la semana,
    promediada en las 4 semanas anteriores. Si no hay historia, usa el promedio de esa
    hora en el entrenamiento."""

    def fit(self, X, y):
        self.media_hora_ = pd.Series(np.asarray(y, float)).groupby(
            X["hour_of_day"].to_numpy()).mean()
        self.media_global_ = float(np.mean(y))
        return self

    def predict(self, X):
        regla = X["sales_avg_4w_same_hour"].to_numpy(float)
        respaldo = X["hour_of_day"].map(self.media_hora_).fillna(self.media_global_).to_numpy(float)
        return np.where(np.isnan(regla), respaldo, regla)


# ------------------------------------------------------------------- evaluación
@dataclass
class Resultado:
    metricas: dict
    oof: np.ndarray            # predicciones fuera de pliegue (NaN fuera de la CV)
    wape_pliegues: list
    modelo: object             # ajustado en todo el entrenamiento


def evaluar(modelo, train: pd.DataFrame, pliegues=None) -> Resultado:
    """Misma evaluación para todos: CV temporal por semanas + ajuste final en todo el
    entrenamiento (para la métrica de entrenamiento y el tiempo)."""
    pliegues = pliegues or pliegues_temporales(train)
    X, y = train[ENTRADA], train[TARGET].to_numpy()
    oof = np.full(len(train), np.nan)
    wapes, maes = [], []
    for idx_tr, idx_va in pliegues:
        m = clone(modelo).fit(X.iloc[idx_tr], y[idx_tr])
        oof[idx_va] = m.predict(X.iloc[idx_va])
        wapes.append(wape(y[idx_va], oof[idx_va]))
        maes.append(mae(y[idx_va], oof[idx_va]))

    final = clone(modelo)
    t0 = time.perf_counter()
    final.fit(X, y)
    seg_fit = time.perf_counter() - t0
    # Un modelo zero-shot no se ajusta a los datos: no tiene métrica de entrenamiento.
    pred_train = None if getattr(final, "zero_shot", False) else final.predict(X)

    en_cv = ~np.isnan(oof)
    negocio = costo_negocio(y[en_cv], oof[en_cv], X["staff_present"].to_numpy()[en_cv])
    metricas = {
        "train_wape": wape(y, pred_train) if pred_train is not None else float("nan"),
        "train_mae": mae(y, pred_train) if pred_train is not None else float("nan"),
        "cv_wape_mean": float(np.mean(wapes)),
        "cv_wape_std": float(np.std(wapes, ddof=1)),
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes, ddof=1)),
        "cv_costo_negocio": negocio["costo"],
        "cv_recall_sobrecarga": negocio["recall_sobrecarga"],
        "fit_seconds": seg_fit,
    }
    metricas["brecha_wape"] = metricas["cv_wape_mean"] - metricas["train_wape"]
    return Resultado(metricas, oof, wapes, final)
