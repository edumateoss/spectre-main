
# Datos crudos (IQ / SigMF)

Esta carpeta debe contener las capturas de señal en bruto (muestras IQ en formato SigMF).

> **IMPORTANTE:** estos datos son **OPCIONALES**. No hacen falta para entrenar ni evaluar el sistema sobre los espectrogramas ya generados (esos van en `data.zip`). Solo se necesitan si se quiere:
> - regenerar los espectrogramas desde cero a partir del IQ, o
> - ejecutar el barrido de SNR / el zero-shot directamente desde la señal SigMF.

## Para obtener los datos

1. Descarga el fichero `data_raw.zip` desde el enlace facilitado al tribunal:


2. Descomprímelo en la **raíz del proyecto**, de modo que quede:

   ```
   spectre-main/data_raw/...
   ```
