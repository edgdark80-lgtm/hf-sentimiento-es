# Selección de modelos preentrenados de Hugging Face — sentimiento en español

Prototipo reproducible para **identificar e implementar el modelo preentrenado más adecuado** para clasificar menciones en redes sociales en español como *negativo*, *neutral* o *positivo*. Se comparan cuatro modelos del Hub (más una variante con ajuste mínimo) en Google Colab con GPU, con métricas estándar, latencia, recursos y una regla de selección fijada antes de ver los resultados.

- **Rama de la IA:** Procesamiento de Lenguaje Natural (clasificación de texto)
- **Caso de uso:** monitoreo de la voz del cliente de un fabricante industrial (escenario simulado, sin datos internos)
- **Dataset:** InterTASS 2017 (UMSAB / XLM-T), 3 033 tuits balanceados — ver [`data/README.md`](data/README.md)

## Modelos evaluados

| Clave | Modelo | Tarea | Parámetros | Licencia |
|---|---|---|---|---|
| M1 | [`lxyuan/distilbert-base-multilingual-cased-sentiments-student`](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student) | text-classification | ~135 M | Apache-2.0 |
| M2 | [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) | text-classification | ~278 M | No declarada en la ficha |
| M3 | [`pysentimiento/robertuito-sentiment-analysis`](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis) | text-classification | ~109 M | No declarada; datos TASS no comerciales |
| M4 | [`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) | zero-shot-classification | ~279 M | MIT |
| M1 +FT | M1 con 3 épocas sobre el split train | text-classification | ~135 M | Apache-2.0 |

## Estructura del repositorio

```
hf-sentimiento-es/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md                     # origen, estructura y procedimiento de obtención
│   ├── raw/                          # archivos originales de XLM-T (sin modificar)
│   └── processed/                    # sentimiento_es_intertass.csv / .xlsx
├── notebooks/
│   └── hf_seleccion_modelos_sentimiento_es.ipynb   # cuaderno de Colab
├── src/
│   ├── config.py                     # semilla, modelos, umbrales y supuestos
│   ├── data_prep.py                  # descarga y preparación del dataset
│   ├── models.py                     # carga de modelos y salida homogénea [neg, neu, pos]
│   ├── evaluate.py                   # métricas, bootstrap, latencia, costo (CLI)
│   ├── finetune.py                   # ajuste mínimo de M1
│   ├── selection.py                  # regla de selección del modelo
│   ├── plots.py                      # figuras
│   └── build_report.py               # genera el reporte Word desde results/
├── results/                          # métricas, predicciones y figuras (los genera el notebook)
└── docs/
    └── reporte_analisis_comparativo.docx
```

## Entorno de ejecución

| Elemento | Valor de referencia |
|---|---|
| Plataforma | Google Colab, entorno con GPU **NVIDIA T4** (16 GB) |
| Python | 3.10 o superior (el que provea Colab) |
| Librerías | `requirements.txt` (torch, transformers, datasets, huggingface_hub, scikit-learn…) |
| Semilla | 42 (`src/config.py`) |
| Autenticación | Token de lectura de Hugging Face en *Colab → Secrets* como `HF_TOKEN` (opcional: los modelos son públicos) |

Las versiones exactas usadas en cada ejecución quedan registradas en `results/selection.json`.

## Pasos de ejecución

### Opción A — Google Colab (recomendada)

1. Abre `notebooks/hf_seleccion_modelos_sentimiento_es.ipynb` en Colab (*Archivo → Abrir cuaderno → GitHub*).
2. *Entorno de ejecución → Cambiar tipo de entorno → GPU T4*.
3. (Opcional) Agrega tu token como secreto `HF_TOKEN`.
4. En la sección 2, cambia `REPO_URL` por la URL de este repositorio (o sube el `.zip` cuando se solicite).
5. *Entorno de ejecución → Ejecutar todo* (≈ 10–15 min). Al final se generan `results/` y `docs/reporte_analisis_comparativo.docx` con los números reales.
6. Descarga `results/` y el reporte, y súbelos al repositorio.

### Opción B — Línea de comandos (máquina con GPU)

```bash
git clone https://github.com/TU_USUARIO/hf-sentimiento-es.git && cd hf-sentimiento-es
pip install -r requirements.txt && pip install --no-deps pysentimiento
python -m src.data_prep            # (re)genera el dataset; verifica el SHA-256
python -m src.evaluate             # evalúa M1–M4 -> results/metrics.csv, predictions_test.csv, selection.json
python -m src.finetune             # ajuste mínimo de M1 (opcional)
python -m src.build_report --author "Tu nombre"   # docs/reporte_analisis_comparativo.docx
```

Prueba rápida sin GPU: `python -m src.evaluate --limit 60`.

## Metodología en breve

- **Mismo protocolo para todos:** split test (870 tuits), truncado a 128 tokens, salida reordenada a `[negativo, neutral, positivo]` leyendo `id2label` de cada modelo.
- **Métricas:** accuracy, precision, recall y F1 (macro, ponderadas y por clase); IC 95 % del F1-macro por bootstrap y bootstrap pareado entre los dos mejores.
- **Eficiencia:** latencia por petición (batch = 1; media, p50, p95), throughput con lotes de 32, memoria GPU pico, parámetros y costo estimado por millón de predicciones (supuesto: 0.35 USD/h de T4, editable).
- **Regla de selección (previa):** cumplir F1-macro ≥ 0.60, recall negativo ≥ 0.70 y p95 ≤ 100 ms; entre los que cumplen, mayor F1-macro; si otro está a ≤ 0.02 y es más eficiente, se prefiere el eficiente.

## Resultados y conclusiones

Los resultados se generan al ejecutar el notebook y quedan en `results/metrics.csv` y en el reporte Word. Tras la ejecución, completa esta tabla con `results/metrics.csv`:

| Modelo | Accuracy | F1-macro | Recall neg | p95 (ms) | Textos/s |
|---|---|---|---|---|---|
| M1 | | | | | |
| M2 | | | | | |
| M3 | | | | | |
| M4 | | | | | |
| M1 +FT | | | | | |

**Modelo recomendado:** *(ver `results/selection.json`)*

**Conclusiones de diseño (independientes de los números):**

1. Una regla de decisión fijada de antemano y un protocolo idéntico evitan elegir por intuición o sólo por el F1 más alto.
2. La integridad de la evaluación importa: M2 entrenó con el split train de este corpus y M3 con TASS 2020 (que reutiliza InterTASS); por eso se incluye M1 + FT como comparación justa en datos.
3. La licencia es un criterio de selección, no un detalle: dos candidatos no la declaran y uno hereda restricciones no comerciales.
4. El umbral de confianza (0.60) convierte el clasificador en un sistema de triaje: automatiza la mayoría de los casos y envía los dudosos a revisión humana.
5. Siguiente paso: validar con menciones reales del dominio, fijar la revisión del modelo en el Hub, publicar una demo en un Space y desplegar como API con monitoreo de deriva.

## Referencias

- Barbieri, F., Espinosa Anke, L., & Camacho-Collados, J. (2022). *XLM-T: Multilingual language models in Twitter for sentiment analysis and beyond*. LREC 2022.
- Díaz-Galiano, M. C., et al. (2018). The democratization of deep learning in TASS 2017. *Procesamiento del Lenguaje Natural, 60*.
- Pérez, J. M., et al. (2022). *RoBERTuito: A pre-trained language model for social media text in Spanish*. LREC 2022.
- Laurer, M., et al. (2024). Less annotating, more classifying. *Political Analysis, 32*(1).

## Licencia

Código bajo licencia MIT (`LICENSE`). El dataset conserva su licencia original (CC BY 3.0, términos de Twitter/X) y cada modelo, la indicada en su ficha del Hub.
