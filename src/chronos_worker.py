"""Inferencia de Chronos-2 en un proceso aparte.

torch y LightGBM/scikit-learn traen cada uno su propio libomp; cargados en el mismo
proceso (el kernel del notebook) se cuelgan o terminan en segfault. El envoltorio
`Chronos2Pronostico` escribe contexto y futuro en parquet y llama a este script.

Uso: python -m src.chronos_worker <contexto.parquet> <futuro.parquet|-> <salida.parquet> <modelo_id> <n>
"""
import sys

import pandas as pd
from chronos import Chronos2Pipeline


def main(ruta_ctx, ruta_fut, ruta_out, modelo_id, n):
    contexto = pd.read_parquet(ruta_ctx)
    futuro = None if ruta_fut == "-" else pd.read_parquet(ruta_fut)
    pipeline = Chronos2Pipeline.from_pretrained(modelo_id, device_map="cpu")
    pred = pipeline.predict_df(contexto, future_df=futuro, prediction_length=int(n),
                               quantile_levels=[0.5])
    pred[["timestamp", "predictions"]].to_parquet(ruta_out, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
