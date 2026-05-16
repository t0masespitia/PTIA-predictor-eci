# PTIA-RUL-Predictor

Backend FastAPI + PyTorch para prediccion de Remaining Useful Life (RUL) sobre C-MAPSS FD001.

## Resumen

- Modelo principal: `CNN-BiLSTM` en PyTorch.
- Features de entrada: 17 por ciclo.
  - 3 operational settings: `op1`, `op2`, `op3`
  - 14 sensores: `s2`, `s3`, `s4`, `s7`, `s8`, `s9`, `s11`, `s12`, `s13`, `s14`, `s15`, `s17`, `s20`, `s21`
- Ventana temporal: 30 ciclos.
- Persistencia: SQLite + SQLAlchemy 2.x para historico de predicciones.

## Estructura

```text
├── app/
│   ├── api/
│   │   ├── dtos/
│   │   └── routes/
│   │       ├── history.py
│   │       ├── metrics.py
│   │       └── prediction.py
│   ├── core/
│   ├── data/
│   ├── models/
│   │   ├── cnn_bilstm.py
│   │   ├── entities/
│   │   └── trainer.py
│   ├── repository/
│   └── services/
├── artifacts/
│   ├── models/
│   └── plots/
├── data/
├── scripts/
│   ├── baseline_rf.py
│   ├── demo_prediccion.py
│   ├── generate_plots.py
│   └── train_model.py
├── tests/
├── main.py
├── Dockerfile
└── requirements.txt
```

## Resultados

Metricas actuales del modelo CNN-BiLSTM reentrenado con 17 features:

- RMSE: `22.34`
- MAE: `16.88`
- Unidades evaluadas: `100`

## Como ejecutar

### Reentrenar (offline, ~25 min CPU)

```bash
python scripts/train_model.py --epochs 100
```

### Levantar la API

```bash
uvicorn main:app --reload
```

Swagger en `http://127.0.0.1:8000/docs`.

### Endpoints

- `POST /predict` -> `{engine_id, sequence: [30 readings de 17 campos]}`
- `GET /predictions` -> historico persistido en SQLite
- `GET /metrics` -> RMSE/MAE sobre `test_FD001`
- `GET /health`

### Tests

```bash
pytest tests/ -v
```

### Instalacion

```bash
git clone https://github.com/t0masespitia/PTIA-predictor-eci.git
cd PTIA-predictor-eci
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Stack tecnologico

- PyTorch - modelo CNN-BiLSTM y entrenamiento
- FastAPI - API REST y Swagger
- Pydantic - validacion de DTOs
- scikit-learn - `MinMaxScaler`
- SQLite + SQLAlchemy 2.x - persistencia de predicciones
- pytest - pruebas automatizadas
- Docker - empaquetado

## Scripts utiles

- `python scripts/train_model.py --epochs 100`
- `python scripts/baseline_rf.py`
- `python scripts/demo_prediccion.py`
- `python scripts/generate_plots.py`
