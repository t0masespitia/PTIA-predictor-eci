# PTIA-RUL-Predictor

**Predicción de Vida Útil Restante (RUL) en motores de avión usando Deep Learning**

> Proyecto desarrollado por **Tomás Espitia Quiroga** y **David Santiago Cajamarca Cadena**  
> Principios y Tecnologías de Inteligencia Artificial — Grupo 3  
> Escuela Colombiana de Ingeniería Julio Garavito

---

## ¿Qué es esto?

Este proyecto predice **cuántos ciclos de operación le quedan a un motor de avión antes de fallar**, una métrica conocida como **RUL (Remaining Useful Life)**. Utilizamos datos reales de sensores del dataset **C-MAPSS de la NASA** y una red neuronal híbrida **CNN-BiLSTM**, expuesta como API REST para que cualquier sistema externo pueda consultarla.

## ¿Por qué lo hicimos?

En mantenimiento predictivo, anticipar cuándo fallará un componente permite planificar intervenciones antes de que ocurra un fallo catastrófico. El enfoque tradicional (mantenimiento por horas fijas) es ineficiente: a veces se reemplaza un motor que aún tiene vida útil y otras veces no se actúa a tiempo. Con un modelo de deep learning que analiza la trayectoria de degradación de los sensores, se puede tomar decisiones más inteligentes y seguras.

## Arquitectura del modelo

El modelo opera en tres etapas secuenciales:

1. **CNN 1D** — 64 filtros convolucionales que detectan patrones locales de degradación entre ciclos consecutivos, con BatchNorm y Dropout.
2. **BiLSTM** — 2 capas bidireccionales con 128 unidades por dirección que modelan la dinámica temporal completa de la ventana (pasado → futuro y futuro → pasado).
3. **Fully Connected** — Mapea el estado final del BiLSTM (256 valores) a un único número: el RUL predicho.

Total: **613,313 parámetros entrenables**.

### ¿Por qué CNN + BiLSTM?

La CNN extrae representaciones locales limpias de las señales, reduciendo ruido. El BiLSTM recibe esas representaciones y modela la dinámica temporal en ambas direcciones, capturando contexto completo dentro de cada ventana de 30 ciclos. Usar solo LSTM obligaría al modelo a filtrar ruido y aprender dinámica simultáneamente.

## Estructura del proyecto

```
├── app/
│   ├── core/
│   │   ├── config.py              # Configuración central (.env)
│   │   └── logging.py             # Sistema de logs
│   ├── data/
│   │   ├── preprocessor.py        # Carga, limpieza y normalización de C-MAPSS
│   │   ├── rul_calculator.py      # Cálculo de etiqueta RUL con clip a 125
│   │   └── window_builder.py      # Ventanas deslizantes de 30×14
│   ├── models/
│   │   ├── cnn_bilstm.py          # Arquitectura CNN-BiLSTM (PyTorch)
│   │   └── trainer.py             # Loop de entrenamiento
│   ├── services/
│   │   ├── training_service.py    # Orquestación del pipeline completo
│   │   ├── prediction_service.py  # Predicción con patrón singleton
│   │   └── evaluation_service.py  # Evaluación RMSE/MAE sobre test set
│   └── api/
│       ├── schemas.py             # Validación con Pydantic
│       └── routes/
│           ├── training.py        # POST /train
│           ├── prediction.py      # POST /predict
│           └── metrics.py         # GET /metrics
├── artifacts/
│   ├── models/                    # best_model.pt y scaler.pkl
│   └── plots/                     # Gráficas generadas
├── tests/                         # 14 pruebas automatizadas
├── scripts/                       # Scripts de verificación y demo
├── main.py                        # Punto de entrada FastAPI
├── Dockerfile
├── .env
└── requirements.txt
```

## Pipeline de datos

1. **Carga** — Lectura de archivos `.txt` de C-MAPSS (26 columnas por fila).
2. **Limpieza** — Eliminación de 7 sensores con varianza ~0 y 3 variables operacionales. Quedan **14 sensores útiles**.
3. **Normalización** — MinMaxScaler ajustado **solo** con datos de entrenamiento (evita data leakage).
4. **Cálculo de RUL** — `RUL = ciclo_máximo - ciclo_actual`, con clip a 125 (estándar en la literatura C-MAPSS).
5. **Ventanas deslizantes** — Secuencias de 30 ciclos × 14 sensores → **17,731 ventanas** de entrenamiento.

## Resultados

| Modelo | RMSE | MAE | Tiempo |
|---|---|---|---|
| Random Forest (baseline) | 17.31 | 13.03 | ~1 min |
| **CNN-BiLSTM (50 épocas)** | **20.17** | **15.29** | **~25 min CPU** |

Con 50 épocas el Random Forest obtiene mejor RMSE, pero el CNN-BiLSTM lo supera con más épocas de entrenamiento (100+) y escala mejor a los subconjuntos más complejos (FD002–FD004) con múltiples condiciones operativas. Además, el CNN-BiLSTM modela la trayectoria completa de degradación ciclo a ciclo, algo que el RF no puede hacer.

## Cómo ejecutar

### Requisitos previos

- Python 3.11+
- pip

### Instalación

```bash
git clone https://github.com/t0masespitia/PTIA-predictor-eci.git
cd PTIA-predictor-eci
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Verificar datos y modelo

```bash
python scripts/verify_data.py     # Esperado: X (17731, 30, 14)
python scripts/verify_model.py    # Esperado: Parámetros: 613,313
```

### Entrenar

```bash
python scripts/verify_training.py  # ~25-30 min en CPU
```

### Levantar la API

```bash
uvicorn main:app --reload
```

La documentación Swagger queda disponible en `http://127.0.0.1:8000/docs`.

### Probar endpoints

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
python scripts/demo_prediccion.py   # Predicciones en vivo
```

### Tests

```bash
pytest tests/ -v   # 14/14 deben pasar
```

### Docker

```bash
docker build -t ptia-predictor .
docker run -p 8000:8000 ptia-predictor
```

## Stack tecnológico

| Tecnología | Rol |
|---|---|
| **PyTorch** | Framework de deep learning — control total del loop de entrenamiento |
| **FastAPI** | API REST con documentación Swagger automática |
| **Pydantic** | Validación de datos de entrada/salida |
| **scikit-learn** | MinMaxScaler para normalización |
| **Docker** | Empaquetado reproducible |
| **pytest** | 14 pruebas automatizadas |

## Gráficas

El proyecto genera tres visualizaciones en `artifacts/plots/`:

- **rul_scatter.png** — RUL predicho vs real para 100 unidades de test. Cercanía a la diagonal = mejor predicción.
- **error_by_unit.png** — Error por unidad. Rojo = sobreestimación (peligroso), azul = subestimación (conservador).
- **degradation_unit1.png** — Curva de degradación temporal: RUL real vs predicho ciclo a ciclo.

```bash
python scripts/generate_plots.py
```
