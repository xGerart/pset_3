"""Acceso a la OBT del PSet 2 y snapshot congelado para los experimentos.

Todos los notebooks leen los datos con `cargar_obt()`. La primera llamada consulta
Snowflake y guarda un snapshot en data/ (no versionado); las siguientes leen ese
snapshot, así cada corrida usa exactamente los mismos datos. El hash SHA-256 del
snapshot se registra en MLflow como evidencia de qué datos entrenaron cada modelo.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
TABLA_ORIGEN = "PSET2.OBT.OBT_SALES_HOURLY"
SNAPSHOT = RAIZ / "data" / "obt_matriz.parquet"
MATRIZ_BRANCH_ID = 1

# Solo la matriz: el refuerzo de personal se decide ahí, y la sucursal 2 queda
# reservada como grupo de control del diseño causal (PSet 1/2).
CONSULTA = f"""
select *
from {TABLA_ORIGEN}
where branch_id = {MATRIZ_BRANCH_ID}
order by hour_date, hour_of_day
"""


@dataclass(frozen=True)
class Snapshot:
    datos: pd.DataFrame
    tabla_origen: str
    fecha_corte: str   # última fecha con datos (fecha de corte del snapshot)
    hash: str          # SHA-256 del contenido, ordenado por el grano


def _conectar():
    import snowflake.connector
    from cryptography.hazmat.primitives import serialization

    load_dotenv(RAIZ / ".env")
    ruta_llave = RAIZ / os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"]
    llave = serialization.load_pem_private_key(ruta_llave.read_bytes(), password=None)
    der = llave.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        role=os.environ["SNOWFLAKE_ROLE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        private_key=der,
    )


def descargar_obt() -> pd.DataFrame:
    """Consulta la OBT en Snowflake (solo la matriz)."""
    with _conectar() as con:
        df = con.cursor().execute(CONSULTA).fetch_pandas_all()
    df.columns = [c.lower() for c in df.columns]
    df["hour_date"] = pd.to_datetime(df["hour_date"])
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    # Los enteros con nulos llegan como float/object según el conector: se fijan los tipos.
    for c in ["sales_lag_7d", "sales_lag_14d", "sales_lag_21d", "sales_lag_28d"]:
        df[c] = df[c].astype("Float64").astype(float)
    for c in ["sales_count", "units_sold", "admin_sales_count", "hour_of_day",
              "iso_day_of_week", "iso_week", "staff_present"]:
        df[c] = df[c].astype("int64")
    for c in ["net_sales_amount", "sales_avg_4w_same_hour", "load_per_staff"]:
        df[c] = df[c].astype(float)
    return df.sort_values(["hour_date", "hour_of_day"]).reset_index(drop=True)


def hash_datos(df: pd.DataFrame) -> str:
    contenido = df.sort_values(["hour_date", "hour_of_day"]).to_csv(index=False).encode()
    return hashlib.sha256(contenido).hexdigest()


def cargar_obt(refrescar: bool = False) -> Snapshot:
    """Devuelve el snapshot congelado; con refrescar=True lo vuelve a bajar de Snowflake."""
    if refrescar or not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(exist_ok=True)
        descargar_obt().to_parquet(SNAPSHOT, index=False)
    df = pd.read_parquet(SNAPSHOT)
    return Snapshot(
        datos=df,
        tabla_origen=TABLA_ORIGEN,
        fecha_corte=df["hour_date"].max().date().isoformat(),
        hash=hash_datos(df),
    )
