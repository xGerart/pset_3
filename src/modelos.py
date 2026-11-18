"""Modelos que no son estimadores tabulares: Chronos-2 (fundacional de series de
tiempo, zero-shot) y un LLM por prompting (Claude Haiku 5.5).

Ambos envuelven la interfaz fit/predict de scikit-learn para pasar por la MISMA
validación (pliegues por semana) y la misma métrica que los demás. `fit` solo guarda
la historia; `predict` pronostica las filas pedidas, que deben venir justo después de
esa historia (así ocurre en cada pliegue y en la prueba).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.datos import RAIZ

COVARIABLES = ["hour_of_day", "iso_day_of_week", "is_holiday", "is_last_open_hour", "ventana_pago"]


def _historia(X, y) -> pd.DataFrame:
    h = X[["hour_date"] + COVARIABLES].copy()
    h["target"] = np.asarray(y, float)
    return h.sort_values(["hour_date", "hour_of_day"]).reset_index(drop=True)


class Chronos2Pronostico(BaseEstimator, RegressorMixin):
    """Chronos-2 (Amazon, 2025) sin entrenamiento. La serie son las horas abiertas
    concatenadas; como cierra de noche, el tiempo se indexa por posición (paso horario
    sintético). Calendario y feriados entran como covariables futuras conocidas."""

    zero_shot = True

    def __init__(self, modelo_id: str = "amazon/chronos-2", usar_covariables: bool = True):
        self.modelo_id = modelo_id
        self.usar_covariables = usar_covariables

    def fit(self, X, y):
        self.historia_ = _historia(X, y)
        return self

    def predict(self, X):
        futuro = X[["hour_date"] + COVARIABLES].copy()
        orden = futuro.sort_values(["hour_date", "hour_of_day"]).index
        futuro = futuro.loc[orden].reset_index(drop=True)
        n_h, n_f = len(self.historia_), len(futuro)
        reloj = pd.date_range("2000-01-01", periods=n_h + n_f, freq="h")

        contexto = self.historia_.drop(columns="hour_date").assign(
            item_id="matriz", timestamp=reloj[:n_h])
        futuro = futuro.drop(columns="hour_date").assign(item_id="matriz", timestamp=reloj[n_h:])
        if not self.usar_covariables:
            contexto = contexto[["item_id", "timestamp", "target"]]
            futuro = None
        # torch corre en otro proceso: ver src/chronos_worker.py.
        with tempfile.TemporaryDirectory() as tmp:
            ctx, fut, out = (os.path.join(tmp, f) for f in ("ctx.parquet", "fut.parquet", "out.parquet"))
            contexto.to_parquet(ctx, index=False)
            if futuro is not None:
                futuro.to_parquet(fut, index=False)
            subprocess.run([sys.executable, "-m", "src.chronos_worker", ctx,
                            fut if futuro is not None else "-", out, self.modelo_id, str(n_f)],
                           cwd=RAIZ, check=True, capture_output=True)
            pred = pd.read_parquet(out)
        valores = np.clip(pred["predictions"].to_numpy(float), 0, None)
        out = pd.Series(valores, index=orden)
        return out.loc[X.index].to_numpy()


# --------------------------------------------------------------------------- LLM
DIAS = {1: "lun", 2: "mar", 3: "mié", 4: "jue", 5: "vie", 6: "sáb", 7: "dom"}

INSTRUCCIONES = """Eres analista de demanda de una tienda minorista en Ecuador (abre de 8:00 a 19:00; \
los feriados cierra a las 18:00). Recibes la cantidad de ventas (comprobantes) por hora de las \
semanas anteriores y debes pronosticar las ventas de cada hora de la semana siguiente.
Reglas:
- Usa solo la historia que se te entrega.
- Devuelve exactamente una predicción por cada fila pedida, con la misma fecha y hora.
- Las ventas son conteos no negativos; puedes usar decimales (es un valor esperado)."""


class LLMPronostico(BaseEstimator, RegressorMixin):
    """Pronóstico por prompting: la historia reciente va en el prompt como tabla CSV y
    el modelo devuelve JSON validado (structured outputs). Cada respuesta se guarda en
    docs/llm/ con su prompt: re-ejecutar el notebook reproduce las mismas predicciones
    sin volver a llamar a la API (las respuestas de un LLM no son deterministas)."""

    zero_shot = True

    def __init__(self, modelo: str = "claude-haiku-5-5", semanas_contexto: int = 4,
                 dir_cache: str = "docs/llm"):
        self.modelo = modelo
        self.semanas_contexto = semanas_contexto
        self.dir_cache = dir_cache

    def fit(self, X, y):
        self.historia_ = _historia(X, y)
        return self

    def _prompt(self, X) -> str:
        h = self.historia_
        desde = h["hour_date"].max() - pd.Timedelta(weeks=self.semanas_contexto) + pd.Timedelta(days=1)
        h = h[h["hour_date"] >= desde]
        filas_h = "\n".join(
            f"{r.hour_date:%Y-%m-%d},{DIAS[r.iso_day_of_week]},{r.hour_of_day},"
            f"{'feriado' if r.is_holiday else ''},{int(r.target)}"
            for r in h.itertuples())
        f = X.sort_values(["hour_date", "hour_of_day"])
        filas_f = "\n".join(
            f"{r.hour_date:%Y-%m-%d},{DIAS[r.iso_day_of_week]},{r.hour_of_day},"
            f"{'feriado' if r.is_holiday else ''}"
            for r in f.itertuples())
        return (f"Historia (fecha,día,hora,feriado,ventas):\n{filas_h}\n\n"
                f"Pronostica estas {len(f)} filas (fecha,día,hora,feriado):\n{filas_f}")

    def _llamar(self, prompt: str) -> dict:
        clave = hashlib.sha256(f"{self.modelo}\n{INSTRUCCIONES}\n{prompt}".encode()).hexdigest()[:16]
        ruta = RAIZ / self.dir_cache / f"{clave}.json"
        if ruta.exists():
            return json.loads(ruta.read_text())

        import anthropic
        from dotenv import load_dotenv
        from pydantic import BaseModel

        class Prediccion(BaseModel):
            fecha: str
            hora: int
            ventas: float

        class Respuesta(BaseModel):
            predicciones: list[Prediccion]

        load_dotenv(RAIZ / ".env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Falta ANTHROPIC_API_KEY en .env y no hay respuesta guardada en "
                               f"{ruta.relative_to(RAIZ)}")
        cliente = anthropic.Anthropic()
        resp = cliente.messages.parse(
            model=self.modelo,
            max_tokens=16000,
            system=INSTRUCCIONES,
            messages=[{"role": "user", "content": prompt}],
            output_format=Respuesta,
        )
        if resp.stop_reason != "end_turn":
            raise RuntimeError(f"Respuesta incompleta del LLM: stop_reason={resp.stop_reason}")
        registro = {
            "modelo": self.modelo,
            "instrucciones": INSTRUCCIONES,
            "prompt": prompt,
            "predicciones": [p.model_dump() for p in resp.parsed_output.predicciones],
            "uso": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        }
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(registro, ensure_ascii=False, indent=1))
        return registro

    def predict(self, X):
        registro = self._llamar(self._prompt(X))
        pred = {(p["fecha"], int(p["hora"])): max(float(p["ventas"]), 0.0)
                for p in registro["predicciones"]}
        claves = [(f"{d:%Y-%m-%d}", int(h)) for d, h in zip(X["hour_date"], X["hour_of_day"])]
        faltan = [k for k in claves if k not in pred]
        if faltan:
            raise ValueError(f"El LLM omitió {len(faltan)} filas, p. ej. {faltan[:3]}")
        return np.array([pred[k] for k in claves])
