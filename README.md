# Modelo CNN para detección de Mascotas
Juan Felipe Reyes Botero
## Entendimiento del Negocio
El siguiente proyecto consiste en un modelo de redes neuronales convolucionales con la finalidad de final de detectar si una imagen presentada consiste en un perro o un gato.
Principalmente se trabajará con la  librería de imágenes del dataset Oxford‑IIIT Pet Dataset : https://www.robots.ox.ac.uk/~vgg/data/pets/.

## Metodología
Se trabajará con el conjunto que proporcionará imágenes de múltiples razas de gatos y perros, a partir de las cuales se construirá una tarea de clasificación de gato o perro. En primera instancia, se tratará de definir si es o no un gato, dando una clasificación binaria de Gato o No Gato. Posteriormente, si el modelo da los resultados esperados, se trabajará en la diferenciación de gato o perro. Se planea utilizar un modelo CNN con optimizador Adam, función de pérdida binary cross-entropy y métrica principal accuracy.

```
data/
└── train/
    ├── cat/ 
    └── not_cat/
```

## Equipo del Proyecto.
Juan Felipe Reyes Botero
jfreyesb@unal.edu.co
cc 1020818791
 
