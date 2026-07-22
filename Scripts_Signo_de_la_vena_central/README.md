#Clasificación automática del Central Vein Sign en lesiones de esclerosis múltiple mediante Deep Learning

Sistema de clasificación binaria basado en una red neuronal convolucional 3D multimodal (FLAIR + SWI) para la detección automática del **signo de la vena central (Central Vein Sign, CVS)** en lesiones de esclerosis múltiple, a partir de parches volumétricos extraídos de resonancias magnéticas.

Proyecto desarrollado por **Paula Santalla Estévez** durante sus prácticas en **Qubiotech**.

---

## Descripción

El signo de la vena central es un biomarcador radiológico emergente que ayuda a diferenciar las lesiones desmielinizantes propias de la esclerosis múltiple de otras lesiones de sustancia blanca de origen vascular, lo que tiene relevancia directa en el diagnóstico diferencial de la enfermedad. Su identificación manual sobre secuencias SWI (*Susceptibility Weighted Imaging*) es un proceso costoso y dependiente del observador, lo que motiva el desarrollo de un sistema automático de apoyo a la decisión clínica.

Este proyecto implementa un pipeline completo, de extremo a extremo, para entrenar y evaluar un clasificador que decide, para cada lesión individual, si presenta o no el signo de la vena central (CVS+ / CVS−). El sistema parte de imágenes FLAIR y SWI ya adquiridas y segmentadas, extrae parches volumétricos centrados en cada lesión, construye un conjunto de datos balanceable, entrena una red convolucional 3D de dos ramas (una por modalidad) y evalúa su rendimiento sobre un conjunto de test independiente, incluyendo un análisis posterior de optimización del umbral de decisión.

El código de este directorio (`Experimento1`) corresponde a la configuración **de referencia**: el modelo se entrena únicamente con lesiones reales, sin ningún tipo de aumento de datos mediante imágenes sintéticas generadas por redes generativas.

---

## Características

A partir del análisis del código se han identificado las siguientes funcionalidades implementadas:

- **Extracción de parches volumétricos** (28×28×28 vóxeles) centrados en el centroide de cada lesión, tanto para lesiones positivas como negativas, a partir de imágenes FLAIR, SWI y máscara de lesión.
- **Aumento de datos** mediante rotaciones de 90° en torno al eje axial (0°, 90°, 180°, 270°) aplicadas a las lesiones positivas.
- **Construcción automática del dataset** en formato JSON compatible con MONAI, con extracción del identificador de paciente y etiquetado binario de cada muestra.
- **Entrenamiento de un clasificador 3D multimodal** (CVSNet) con parada temprana, reducción adaptativa de la tasa de aprendizaje y guardado del mejor modelo según la pérdida de validación.
- **Evaluación cuantitativa** sobre el conjunto de test independiente (exactitud, precisión, sensibilidad, especificidad, F1-score, ROC-AUC), con generación de matriz de confusión y curva ROC.
- **Optimización posterior del umbral de decisión**, mediante barrido de 0 a 1 y selección según F1-score e índice de Youden.
- **Seguimiento del aprendizaje por lesión** durante el entrenamiento: en cada época se registra qué lesiones del conjunto de entrenamiento se clasifican correctamente, generando un histórico por lesión.

---

## Estructura del proyecto

```
Experimento1/
│
├── extract_one_patch.py     # Extrae parches 28×28×28 (FLAIR, SWI, máscara) centrados en el
│                             # centroide de cada lesión, a partir de lesiones_positivas.csv o
│                             # lesiones_negativas.csv . Comprueba que las tres modalidades
│                             # tengan idénticas dimensiones antes de recortar.
│
├── data_augmentation.py     # Genera 4 variantes (original, rot90, rot180, rot270) de cada
│                             # parche de lesión positiva mediante rotación en torno al eje Z.
│
├── build_dataset.py         # Recorre las carpetas de parches (CVS_pos y CVS_neg) y construye
│                             # dataset.json: una lista de muestras con rutas de FLAIR/SWI,
│                             # etiqueta, paciente y variante de aumento.
│
├── model.py                 # Arquitectura CVSNet: CNN 3D de dos ramas (FLAIR y SWI) con
│                             # fusión intermedia y clasificador binario final.
│
├── test_monai_dataset.py    # Carga dataset.json, realiza el split train/val/test estratificado
│                             # por lesión y define las transformaciones MONAI de entrada al modelo.
│
├── train.py                 # Script principal de entrenamiento: bucle de entrenamiento y
│                             # validación, scheduler, early stopping, guardado del mejor modelo
│                             # y registro del histórico de métricas por época.
│
├── evaluate.py               # Carga el mejor modelo guardado y lo evalúa sobre el conjunto de
│                             # test: métricas agregadas, matriz de confusión y curva ROC.
│
└── optimize_threshold.py    # Barrido del umbral de clasificación (0.00–1.00) a partir de las
                              # probabilidades ya calculadas por evaluate.py, sin reentrenar el modelo.
```

> **Nota:** los archivos `dataset.json`, `best_model.pth` y `history.csv` se generan en el directorio padre (`segmentaciones/`), ya que `BASE_DIR` se define como `Path(__file__).parent.parent` en varios scripts. Los resultados de evaluación (`resultados_evaluacion/`) se generan igualmente en ese directorio padre.

---

## Pipeline

El flujo completo, tal y como está implementado en el código, es el siguiente:

1. **Punto de partida.** El proceso parte de dos archivos CSV (`lesiones_positivas.csv` y `lesiones_negativas.csv`) que contienen, para cada lesión, el dataset de origen, el paciente, la visita, un identificador de lesión, su volumen en vóxeles, las coordenadas del centroide y la etiqueta CVS. 
2. **Extracción de parches** (`extract_one_patch.py`): para cada fila del CSV correspondiente, se localizan las imágenes FLAIR, SWI (priorizando versiones registradas al espacio FLAIR) y la máscara de lesión del paciente/visita indicados, se comprueba que las tres compartan dimensiones y se recorta un cubo de 28×28×28 vóxeles centrado en el centroide, guardándose en `patches/CVS_pos/` o `patches/CVS_neg/` según corresponda.
3. **Aumento de datos** (`data_augmentation.py`): sobre las lesiones positivas ya extraídas, se generan las cuatro rotaciones (0°, 90°, 180°, 270°) de cada modalidad (FLAIR, SWI y máscara).
4. **Construcción del dataset** (`build_dataset.py`): se recorren las carpetas de parches de ambas clases, se valida la existencia de los archivos de cada modalidad, se extrae el identificador de paciente mediante expresiones regulares y se genera `dataset.json`, con una entrada por cada combinación de lesión y variante de aumento.
5. **Partición y carga** (`test_monai_dataset.py`): `dataset.json` se divide en entrenamiento (70 %), validación (15 %) y test (15 %) de forma estratificada por etiqueta y agrupada por lesión base (para que las distintas rotaciones de una misma lesión no queden repartidas entre particiones), con semilla fija. Se definen a continuación las transformaciones MONAI que cargan FLAIR y SWI, las normalizan por intensidad y las concatenan en un único tensor de dos canales.
6. **Entrenamiento** (`train.py`): se instancia `CVSNet`, se entrena durante un máximo de 100 épocas con parada temprana y reducción adaptativa de la tasa de aprendizaje, y se conserva el modelo con menor pérdida de validación en `best_model.pth`.
7. **Evaluación** (`evaluate.py`): se carga `best_model.pth` y se evalúa sobre el conjunto de test, generando métricas agregadas, la matriz de confusión y la curva ROC.
8. **Optimización del umbral** (`optimize_threshold.py`): a partir de las probabilidades guardadas por `evaluate.py`, se analiza el efecto de mover el punto de corte de clasificación entre 0 y 1, identificando el umbral óptimo según F1-score y según el índice de Youden.

---

## Arquitectura del modelo

La arquitectura, definida en `model.py` bajo la clase `CVSNet`, es una red convolucional 3D de **doble rama con fusión intermedia**:

- La entrada es un tensor de 2 canales y tamaño 28×28×28 vóxeles, donde el canal 0 corresponde a FLAIR y el canal 1 a SWI.
- Cada modalidad se procesa de forma **independiente** a través de una rama convolucional idéntica en estructura (`make_branch`), compuesta por dos bloques `Conv3d → BatchNorm3d → ReLU`, un `MaxPool3d` tras el primer bloque y un `Dropout3d(p=0.2)` tras el segundo. La rama de FLAIR pasa de 1 a 8 y después a 16 canales; la rama de SWI, de forma independiente, hace lo mismo.
- Las características de ambas ramas (16 + 16 canales) se **concatenan** por canal y se fusionan mediante una capa `Conv3d → BatchNorm3d → ReLU` que las combina en 32 canales.
- A continuación se aplica un `AdaptiveAvgPool3d(1)` (Global Average Pooling), reduciendo cada mapa de características a un único valor por canal.
- Tras un `Dropout` (`p=0.5` por defecto) se aplica una capa totalmente conectada (`nn.Linear(32, 2)`) que produce los logits de las dos clases (CVS− / CVS+).

No se ha encontrado en el código ninguna inicialización explícita de pesos, por lo que se asume el comportamiento por defecto de PyTorch para `Conv3d`, `BatchNorm3d` y `Linear`.

---

## Ejecución

Los scripts deben ejecutarse en el siguiente orden desde este directorio, ya que cada uno depende de los artefactos generados por el anterior:

```bash
# 1. Extracción de parches (lesiones positivas y negativas)
python extract_one_patch.py --all
python extract_one_patch.py --all --neg

# 2. Aumento de datos sobre las lesiones positivas
python data_augmentation.py

# 3. Construcción del dataset.json
python build_dataset.py

# 4. (Opcional) Comprobación del split y de la carga MONAI
python test_monai_dataset.py

# 5. Entrenamiento del modelo
python train.py

# 6. Evaluación sobre el conjunto de test
python evaluate.py

# 7. Optimización del umbral de decisión
python optimize_threshold.py
```

---

## Resultados

*(Sección a completar con los resultados obtenidos tras la ejecución del pipeline.)*

- Métricas de test (`evaluation_metrics.txt`):
- Matriz de confusión: `confusion_matrix.png`
- Curva ROC: `roc_curve.png`
- Umbral óptimo (F1 / Youden): `threshold_optimization_summary.txt`

