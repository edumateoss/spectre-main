# Carpeta de datos (no incluida en el repositorio)

Los *datasets* de espectrogramas y los artefactos de entrenamiento no se suben a
GitHub por su tamaño (varios GB). Esta carpeta documenta la estructura esperada
para reproducir los experimentos; el contenido se descarga aparte.

## Estructura esperada

```
data/
├── mini3_detector_python_v3/            # Dataset v3 (Stage 1): 3150 espectrogramas DJI Mini 3
├── stage2_classifier_v1/                # Dataset de campo: F450, Mavic Pro, Hunter (zero-shot)
└── stage2_classifier_v2_mavic_split/    # Dataset RFUAV (Stage 2): 6 clases
```

## Descarga

Los *datasets* y los pesos del clasificador ResNet18 de la segunda etapa están
disponibles en el enlace facilitado: https://atcdatos.ugr.es/index.php/s/xcOo6TT3ErYyTu3?path=%2F
