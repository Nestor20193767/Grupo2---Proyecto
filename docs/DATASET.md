# Dataset y partición experimental

## Fuente

El proyecto final usa **PhysioNet ECG-Arrhythmia, versión 1.0.0**, con ECG de 12 derivaciones a 500 Hz y etiquetas SNOMED-CT. El dataset no se versiona en GitHub. Debe descargarse desde su fuente oficial y respetar sus condiciones de uso.

## Preprocesamiento reportado

1. Filtro paso alto de 0.5 Hz.
2. Filtro paso bajo de 40 Hz.
3. Detección de pico R mediante Pan-Tompkins.
4. Extracción de ventanas de 650 ms: 125 muestras antes y 200 después del pico R.
5. Cada latido contiene 325 muestras a 500 Hz.
6. División por registros: 60% `GEN_SPLIT`, 20% `CLF_REAL` y 20% `CLF_HELD`.

## Clases

El generador usa seis categorías: `AF`, `AFL`, `NSR`, `Others`, `SB` y `ST`. El manuscrito indica que `Others` se excluyó del clasificador por heterogeneidad diagnóstica; por ello, el código separa explícitamente las clases del generador de las clases del clasificador.

## Datos locales

Coloque datos sin procesar en `data/raw/` y derivados en `data/processed/`. Ambos directorios se ignoran en Git para evitar publicar información pesada o sensible.
