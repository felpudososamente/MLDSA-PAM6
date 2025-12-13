# Modelo Transfer leasrning para detección de Mascotas
Juan Felipe Reyes Botero
## Entendimiento del Negocio
El siguiente proyecto consiste en un modelo de redes neuronales convolucionales con la finalidad de final de detectar si una imagen presentada consiste en un perro o un gato.
Principalmente se trabajará con la  librería de imágenes del dataset Oxford‑IIIT Pet Dataset : https://www.robots.ox.ac.uk/~vgg/data/pets/.

<img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg" width="250">


## Metodología
Se trabajará con el conjunto que proporcionará imágenes de múltiples razas de gatos y perros, a partir de las cuales se construirá una tarea de clasificación de gato o perro. En primera instancia, se tratará de definir si es o no un gato, dando una clasificación binaria de Gato o No Gato. Posteriormente, si el modelo da los resultados esperados, se trabajará en la diferenciación de gato o perro. Se planea utilizar un modelo CNN con optimizador Adam, función de pérdida binary cross-entropy y métrica principal accuracy.

```
data/
├── train/
│   ├── cat/
│   └── not_cat/
└── val/
    ├── cat/
    └── not_cat/

```

## Carga de los Datos
 Los datos se cargarán de la siguiente manera
 ```
!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz -O images.tar.gz
!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz -O annotations.tar.gz

!mkdir -p data/raw
!tar -xzf images.tar.gz -C data/raw
!tar -xzf annotations.tar.gz -C data/raw

print("Dataset descargado y descomprimido.")
 ```
## Cronograma
| Etapa                                        | Duración Estimada | Fechas                              |
|----------------------------------------------|-------------------|-------------------------------------|
| Entendimiento del negocio y carga de datos   | 1 semana          | 21 nov 2025 al 27 nov 2025          |
| Preprocesamiento y análisis exploratorio      | 1 semana          | 28 nov 2025 al 4 dic 2025           |
| Modelamiento y extracción de características  | 1 semana          | 5 dic 2025 al 11 dic 2025           |
| Despliegue                                   | 2 días            | 12 dic 2025 al 13 dic 2025          |
| Evaluación y entrega final                    | 2 días            | 12 dic 2025 al 13 dic 2025          |
## Carga e importacion de las librerías
```
import sys
import os
import shutil
from pathlib import Path
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gradio as gr



print("Python:", sys.version)
print("scikit-learn:", sklearn.__version__)
print("TensorFlow:", tf.__version__)
print("Gradio:", gr.__version__)
print("NumPy:", np.__version__)
print("Seaborn:", sns.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pandas:", pd.__version__)
print("Pillow:", Image.__version__)
## Análisis Exploratorio

```
Python: 3.11.14 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 18:30:03) [MSC v.1929 64 bit (AMD64)]
scikit-learn: 1.8.0
TensorFlow: 2.20.0
Gradio: 6.1.0
NumPy: 2.3.5
Seaborn: 0.13.2
Matplotlib: 3.10.8
Pandas: 2.3.3
Pillow: 12.0.0
```


```
## Equipo del Proyecto.
Juan Felipe Reyes Botero
jfreyesb@unal.edu.co
cc 1020818791
 
