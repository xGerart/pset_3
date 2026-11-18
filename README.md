# PSet 3 · Experimentación y selección del modelo

Pronóstico de las **ventas por hora** de la matriz para la semana siguiente. El pronóstico decide qué horas necesitan un vendedor de refuerzo: aquellas con una carga de 8 o más ventas por vendedor presente (PSet 1). Los datos son la OBT que construye el pipeline del [PSet 2](https://github.com/xGerart/pset_2) en Snowflake.

```
OBT (Snowflake) → EDA → Features → Pipeline → Líneas base → Candidatos → Tuning → Selección → Model Registry (MLflow)
```

## Estructura

| Ruta | Contenido |
|---|---|
| `docker-compose.yml`, `mlflow/` | Servidor de tracking de MLflow. Backend store: Postgres. Artifact store: un volumen que sirve el propio servidor |
| `src/datos.py` | Lee la OBT de Snowflake y congela un snapshot con su hash SHA-256 |
| `src/pipeline.py` | Features derivadas, `ColumnTransformer` + `Pipeline`, división train/prueba, CV temporal por semanas, WAPE y costo de negocio |
| `src/modelos.py`, `src/chronos_worker.py` | Chronos-2 (*zero-shot*) y el LLM por prompting, envueltos con la interfaz de scikit-learn |
| `src/tracking.py` | Contrato de registro de cada corrida en MLflow |
| `notebooks/01…06` | EDA, features, líneas base y candidatos, tuning, selección y registro, reproducción |
| `docs/corridas.csv` | Todas las corridas del experimento (`mlflow.search_runs`) |
| `docs/llm/` | Prompts y respuestas del LLM: re-ejecutar el notebook 03 no vuelve a llamar a la API |
| `docs/figuras/`, `docs/resultados_*.csv` | Gráficos y tablas que usa el documento técnico |
| `docs/PSet3_memo_Torres_Chicaiza.pdf` | Documento técnico |

## 1. Entorno de Python

```bash
uv venv --python 3.12 .venv          # o: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## 2. Credenciales

```bash
cp .env.example .env
```

- **Snowflake.** Se usa el mismo usuario de servicio del PSet 2, que se autentica con par de llaves y no tiene contraseña. Completa `SNOWFLAKE_ACCOUNT` y copia la llave privada a `secrets/snowflake_rsa_key.p8` (en el PSet 2 está en `pset_2/secrets/`). El rol `PSET2_ROLE` ya tiene lectura sobre `PSET2.OBT`.
- **MLflow.** Pon cualquier valor aleatorio en `MLFLOW_DB_PASSWORD`; es la contraseña del Postgres interno.
- **LLM (opcional).** `ANTHROPIC_API_KEY` solo hace falta si se borran las respuestas guardadas en `docs/llm/`.

`.env` y `secrets/` están en `.gitignore`: el repositorio no contiene credenciales.

## 3. Levantar MLflow

```bash
docker compose up -d --build --wait
```

La interfaz queda en <http://localhost:5050>. El puerto 5000 lo ocupa AirPlay en macOS, y el servicio solo escucha en `127.0.0.1` porque no tiene autenticación. Los datos persisten en los volúmenes `pset3_mlflow-postgres` y `pset3_mlflow-artifacts`.

## 4. Orden de ejecución de los notebooks

Ejecuta cada uno con **Restart & Run All**, en este orden. Cada notebook lee de MLflow lo que registró el anterior.

| # | Notebook | Qué hace | Corridas |
|---|---|---|---|
| 1 | `01_eda.ipynb` | Primera consulta a Snowflake y snapshot en `data/`; perfil, atípicos, correlaciones, hallazgos | — |
| 2 | `02_features.ipynb` | Features derivadas, ablación y prueba de no-fuga | — |
| 3 | `03_baseline_candidatos.ipynb` | 2 líneas base y 7 candidatos (5 familias + Chronos-2 + LLM) | 9 |
| 4 | `04_tuning.ipynb` | Optuna (TPE), 30 configuraciones × 3 modelos, como corridas hijas | 93 |
| 5 | `05_seleccion.ipynb` | Ganador frente al segundo, prueba única, umbral por costo, registro `champion`/`challenger`, exporta `docs/corridas.csv` | 2 |
| 6 | `06_reproduccion.ipynb` | Carga `models:/ventas-hora-matriz-modelo@champion` y reproduce la métrica registrada | — |

Desde la terminal:

```bash
for nb in notebooks/0*.ipynb; do .venv/bin/jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

`src/datos.py` guarda el snapshot en `data/obt_matriz.parquet`. Las corridas siguientes leen ese archivo, de modo que todas usan los mismos datos (el hash queda en cada corrida). Para volver a bajar la OBT: `cargar_obt(refrescar=True)`.

## Notas de reproducibilidad

- Semilla fija (`SEMILLA = 42`) en los modelos y en el muestreador de Optuna.
- Cada corrida registra el commit de Git del código en `src/`. Si había cambios sin commitear, el valor lleva `-dirty`.
- Chronos-2 se descarga de Hugging Face (`amazon/chronos-2`) la primera vez y corre en un proceso aparte, porque el OpenMP de torch choca con el de LightGBM dentro del mismo kernel.
- Las respuestas de un LLM no son deterministas. Por eso se guarda cada prompt con su respuesta, y la corrida se reproduce desde ese archivo.
