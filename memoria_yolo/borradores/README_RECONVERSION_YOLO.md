# Reconversión del TFG hacia detección YOLO Stage 1

Esta versión del proyecto Overleaf cambia el foco de la memoria:

- Antes: clasificación global `drone` vs `interference` con ResNet18 sobre espectrogramas completos.
- Ahora: detección/localización de ráfagas con YOLOv8 como Stage 1.
- Se mantiene: clasificación entre modelos de drones como Stage 2 posterior.
- Se conserva: distancia como variable experimental complementaria, no como objetivo principal de localización.

## Archivos modificados

- `main.tex`: subtítulo y keywords.
- `preliminares/resumen_es.tex`: resumen actualizado.
- `preliminares/resumen_en.tex`: abstract actualizado.
- `capitulos/01_introduccion.tex`: introducción reconvertida a YOLO/two-stage.
- `capitulos/03_requisitos_y_diseno.tex`: requisitos y diseño centrados en YOLO.
- `capitulos/04_implementacion.tex`: adquisición, espectrogramas, auto-etiquetado y entrenamiento YOLO.
- `capitulos/05_pruebas_y_resultados.tex`: resultados run01/run02, métricas y discusión.
- `capitulos/06_conclusiones.tex`: conclusiones acordes al nuevo enfoque.
- `capitulos/07_anexos.tex`: pseudocódigo actualizado.
- `bibliografia.bib`: añadido un registro básico para Ultralytics YOLOv8.
- `figuras/resultados/`: copiadas las figuras de métricas YOLO disponibles.

## Idea central nueva

El TFG ya no debe defender que una CNN clasifica bien una imagen completa. Debe defender que una red de detección de objetos puede localizar ráfagas FHSS en espectrogramas y separar morfológicamente OcuSync del DJI Mini 3 frente a interferencias Bluetooth/WiFi residuales.

## Resultado central incorporado

YOLOv8n run02_60ep:

- mAP@50 global: 0.388
- mAP@50-95 global: 0.203
- Precision global: 0.760
- Recall global: 0.391
- mAP@50 drone: 0.575
- mAP@50 interference: 0.201

Lectura clave: la matriz de confusión normalizada muestra 0.00 de confusión cruzada entre `drone` e `interference`. Los errores dominantes son falsos negativos hacia `background`, no confusión entre clases activas.
