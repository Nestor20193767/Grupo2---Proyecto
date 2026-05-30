# -*- coding: utf-8 -*-
"""ECG_Avance2_Pipeline.ipynb

# Avance 2 – Dataset, Preprocesamiento y Pipeline de Datos
## Proyecto: Generación Sintética de Señales ECG con TCN-CVAE
### Base de Datos: MIT-BIH Arrhythmia Database

---
**Flujo del notebook:**
1. Instalación de dependencias
2. Carga del dataset desde Kaggle
3. Descripción detallada del dataset
4. Pipeline de preprocesamiento (filtrado, segmentación, normalización)
5. Visualización antes/después del preprocesamiento
6. Mapeo de etiquetas AAMI
7. División train/val/test (sin contaminación por paciente)
8. Manejo del desbalance de clases
9. Resumen final del pipeline

## 1. Instalación de Dependencias
"""

# Instalación de todas las librerías necesarias

print(" Dependencias instaladas correctamente")

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wfdb
import neurokit2 as nk
import kagglehub
from kagglehub import KaggleDatasetAdapter
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, butter as butter_sos
from scipy.signal import butter as _butter
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import glob

warnings.filterwarnings('ignore')
np.random.seed(42)

# Estilo de gráficos
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['font.size'] = 11

print(" Librerías importadas correctamente")

"""## 2. Carga del Dataset desde Kaggle

**Dataset:** MIT-BIH Arrhythmia Database  
**Fuente:** PhysioNet (Kaggle mirror)  
**Enlace:** https://www.kaggle.com/datasets/mondejar/mitbih-database  
**Justificación:** Es el benchmark gold-standard para clasificación y generación de arritmias. Contiene registros con anotaciones de expertos cardiólogos, frecuencia de muestreo de 360 Hz y la diversidad de clases necesaria para entrenar un modelo generativo condicional (CVAE).
"""

# ─────────────────────────────────────────────────────────
#  DESCARGA DEL DATASET (Directamente de PhysioNet)
# ─────────────────────────────────────────────────────────
import wfdb
import os
import glob

# Creamos una carpeta local para guardar los datos
wfdb_dir = os.path.join(os.getcwd(), 'mitdb_data')
os.makedirs(wfdb_dir, exist_ok=True)

print(" Descargando MIT-BIH directamente desde PhysioNet (puede tardar un momento)...")

# wfdb descarga la base de datos oficial ('mitdb') con los formatos correctos
wfdb.dl_database('mitdb', dl_dir=wfdb_dir)
print(f" Dataset descargado exitosamente en: {wfdb_dir}")

# Buscar los archivos .dat que necesita tu código
dat_files = glob.glob(os.path.join(wfdb_dir, '*.dat'))
print(f"\n Archivos .dat encontrados: {len(dat_files)}")

# Detectar los IDs de los pacientes
if dat_files:
    record_ids = sorted(set([os.path.basename(f).replace('.dat', '') for f in dat_files]))
    print(f"\n Registros de pacientes disponibles ({len(record_ids)} total):")
    print(record_ids)
else:
    print(" ERROR: No se encontraron archivos .dat.")
    record_ids = []

"""## 3. Descripción Detallada del Dataset"""

# ─────────────────────────────────────────────────────────
#  EXPLORACIÓN DE UN REGISTRO REPRESENTATIVO
# ─────────────────────────────────────────────────────────

# Validación de seguridad (evita errores si se saltó la celda anterior)
if not record_ids:
    raise ValueError("La lista 'record_ids' está vacía. Ejecuta primero la celda de descarga.")

sample_record_id = record_ids[0]  # Tomamos el primer registro
record_path = os.path.join(wfdb_dir, sample_record_id)

# Leer el registro con WFDB
record   = wfdb.rdrecord(record_path)
ann      = wfdb.rdann(record_path, 'atr')

print("=" * 60)
print(f"  DESCRIPCIÓN DEL DATASET  MIT-BIH ARRHYTHMIA DATABASE")
print("=" * 60)
print(f"\n INFORMACIÓN GENERAL")
print(f"  Nombre            : MIT-BIH Arrhythmia Database")
print(f"  Fuente            : PhysioNet (Descarga directa vía wfdb)") # Texto actualizado
print(f"  Número de pacientes: {len(record_ids)}")
print(f"  Canales ECG       : {record.n_sig} (MLII y V5)")
print(f"  Frecuencia muest. : {record.fs} Hz")
print(f"  Duración/registro : ~30 minutos")
print(f"  Resolución        : 11 bits, ganancia 200 ADC/mV")
print(f"  Formato archivos  : .dat (señal binaria), .hea (cabecera), .atr (anotaciones)")
print(f"  Tipo de dato      : Señal biomédica 1D (ECG)")
print(f"\n DETALLE DEL REGISTRO '{sample_record_id}'")
print(f"  Muestras totales  : {record.sig_len:,}")
print(f"  Anotaciones       : {len(ann.sample)} latidos")
print(f"  Nombres de señal  : {record.sig_name}")
print(f"  Unidades          : {record.units}")

# Mostrar las primeras anotaciones
print(f"\n Primeras 10 anotaciones del registro {sample_record_id}:")
ann_df = pd.DataFrame({'sample': ann.sample[:10], 'symbol': ann.symbol[:10]})
print(ann_df.to_string(index=False))

# ─────────────────────────────────────────────────────────
#  CONTEO GLOBAL DE LATIDOS EN TODOS LOS REGISTROS
# ─────────────────────────────────────────────────────────

# Mapeo de etiquetas MIT-BIH → 5 clases AAMI
AAMI_MAP = {
    # N – Normal / latidos supraventriculares de escape
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # S – Supraventricular ectópico
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V – Ventricular ectópico
    'V': 'V', 'E': 'V',
    # F – Fusión
    'F': 'F',
    # Q – Desconocido / artifacto
    '/': 'Q', 'f': 'Q', 'Q': 'Q', '?': 'Q'
}

global_counts_raw  = Counter()
global_counts_aami = Counter()
total_beats = 0

print(" Contando latidos en todos los registros...")
for rid in record_ids:
    try:
        a = wfdb.rdann(os.path.join(wfdb_dir, rid), 'atr')
        for sym in a.symbol:
            global_counts_raw[sym] += 1
            if sym in AAMI_MAP:
                global_counts_aami[AAMI_MAP[sym]] += 1
                total_beats += 1
    except Exception:
        pass

print(f"\n Latidos totales (válidos AAMI): {total_beats:,}")
print(f"\n DISTRIBUCIÓN POR CLASE AAMI:")
print(f"{'Clase':<8} {'Descripción':<28} {'Latidos':>8} {'%':>7}")
print("-" * 55)
class_desc = {'N': 'Normal', 'S': 'Supraventricular', 'V': 'Ventricular', 'F': 'Fusión', 'Q': 'Desconocido'}
for cls in ['N', 'S', 'V', 'F', 'Q']:
    cnt  = global_counts_aami[cls]
    pct  = cnt / total_beats * 100
    print(f"  {cls:<6} {class_desc[cls]:<28} {cnt:>8,} {pct:>6.1f}%")
print("-" * 55)
print(f"  {'TOTAL':<34} {total_beats:>8,} {'100.0%':>7}")

# ─────────────────────────────────────────────────────────
#  VISUALIZACIÓN DE LA DISTRIBUCIÓN DE CLASES
# ─────────────────────────────────────────────────────────
classes  = ['N', 'S', 'V', 'F', 'Q']
counts   = [global_counts_aami[c] for c in classes]
labels   = [f"{c}\n{class_desc[c]}" for c in classes]
colors   = ['#4CAF50', '#2196F3', '#F44336', '#FF9800', '#9E9E9E']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribución de Clases AAMI – MIT-BIH Arrhythmia Database', fontsize=13, fontweight='bold')

# Barras
bars = ax1.bar(labels, counts, color=colors, edgecolor='white', linewidth=1.5)
ax1.set_ylabel('Número de latidos')
ax1.set_title('Distribución absoluta')
ax1.set_yscale('log')  # Escala log para visualizar mejor el desbalance
for bar, cnt in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
             f'{cnt:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Pastel
wedges, texts, autotexts = ax2.pie(counts, labels=labels, colors=colors,
                                    autopct='%1.1f%%', startangle=90,
                                    pctdistance=0.75, wedgeprops=dict(edgecolor='white', linewidth=2))
ax2.set_title('Distribución porcentual')

plt.tight_layout()
plt.savefig('distribucion_clases.png', dpi=150, bbox_inches='tight')
plt.show()
print("  DESBALANCE SEVERO: La clase N domina (~90%). Se requiere estrategia de balanceo.")

"""## 4. Pipeline de Preprocesamiento
### Paso 4.1 – Filtrado de la señal ECG cruda
"""

# ─────────────────────────────────────────────────────────
#  FUNCIONES DE FILTRADO  (Procesamiento de Señales)
# ─────────────────────────────────────────────────────────
FS = 360  # Frecuencia de muestreo MIT-BIH

def highpass_filter(signal, cutoff=0.5, fs=FS, order=4):
    """Elimina la deriva de la línea base (wandering baseline)."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, signal)

def notch_filter(signal, freq=60.0, fs=FS, quality=30):
    """Elimina el ruido de la red eléctrica (60 Hz para MIT-BIH)."""
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal)

def lowpass_filter(signal, cutoff=45.0, fs=FS, order=4):
    """Elimina ruido muscular (EMG) conservando el pico R."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

def full_ecg_pipeline(raw_signal, fs=FS):
    """Pipeline completo: highpass → notch → lowpass."""
    step1 = highpass_filter(raw_signal, cutoff=0.5, fs=fs)
    step2 = notch_filter(step1, freq=60.0, fs=fs)
    step3 = lowpass_filter(step2, cutoff=45.0, fs=fs)
    return step3

print(" Funciones de filtrado definidas:")
print("  → highpass_filter  : Butterworth 4° orden, corte 0.5 Hz (elimina baseline wander)")
print("  → notch_filter     : IIR Notch, 60 Hz, Q=30 (elimina ruido red eléctrica)")
print("  → lowpass_filter   : Butterworth 4° orden, corte 45 Hz (elimina EMG muscular)")
print("  → full_ecg_pipeline: Encadena los 3 filtros en secuencia")

# ─────────────────────────────────────────────────────────
#  VISUALIZACIÓN ANTES Y DESPUÉS DEL FILTRADO
# ─────────────────────────────────────────────────────────
# Cargamos 10 segundos del primer registro
record_sample = wfdb.rdrecord(os.path.join(wfdb_dir, record_ids[0]), sampto=10*FS)
raw_ecg = record_sample.p_signal[:, 0].copy()  # Canal MLII

# Aplicar pipeline paso a paso para visualizar cada etapa
ecg_hp  = highpass_filter(raw_ecg)
ecg_hp_notch = notch_filter(ecg_hp)
ecg_clean = lowpass_filter(ecg_hp_notch)

time_axis = np.arange(len(raw_ecg)) / FS

fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
fig.suptitle(f'Pipeline de Filtrado – Registro {record_ids[0]} (primeros 10 s)',
             fontsize=13, fontweight='bold')

signals  = [raw_ecg,  ecg_hp,  ecg_hp_notch, ecg_clean]
titles   = [' Señal CRUDA (sin filtrar)',
             ' Paso 1: Tras filtro pasa-alto 0.5 Hz (elimina baseline wander)',
             ' Paso 2: Tras filtro Notch 60 Hz (elimina ruido red eléctrica)',
             ' Paso 3: SEÑAL LIMPIA – tras filtro pasa-bajo 45 Hz (elimina EMG)']
colors_sig = ['#e74c3c', '#f39c12', '#e67e22', '#27ae60']

for ax, sig, title, col in zip(axes, signals, titles, colors_sig):
    ax.plot(time_axis, sig, color=col, linewidth=0.8, alpha=0.9)
    ax.set_ylabel('Amplitud (mV)', fontsize=9)
    ax.set_title(title, fontsize=10, pad=3)
    ax.set_ylim([sig.mean() - 3*sig.std(), sig.mean() + 3*sig.std()])

axes[-1].set_xlabel('Tiempo (segundos)')
plt.tight_layout()
plt.savefig('filtrado_ecg.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Filtrado completo. La señal verde es la que se usará para segmentación.")

# ─────────────────────────────────────────────────────────
#  ANÁLISIS DE DENSIDAD ESPECTRAL (PSD)
#  Para verificar que los filtros operaron correctamente
# ─────────────────────────────────────────────────────────
from scipy.signal import welch

fig, ax = plt.subplots(figsize=(12, 4))
fig.suptitle('Densidad Espectral de Potencia (PSD) – Antes vs Después del Filtrado', fontweight='bold')

freqs_raw,   psd_raw   = welch(raw_ecg,   fs=FS, nperseg=512)
freqs_clean, psd_clean = welch(ecg_clean, fs=FS, nperseg=512)

ax.semilogy(freqs_raw,   psd_raw,   color='#e74c3c', linewidth=1.5, alpha=0.8, label='Señal cruda')
ax.semilogy(freqs_clean, psd_clean, color='#27ae60', linewidth=1.5, alpha=0.8, label='Señal filtrada')

# Marcar las frecuencias de corte
ax.axvline(0.5,  color='#f39c12', linestyle='--', alpha=0.7, label='Corte HP: 0.5 Hz')
ax.axvline(60,   color='#9b59b6', linestyle='--', alpha=0.7, label='Notch: 60 Hz')
ax.axvline(45,   color='#3498db', linestyle='--', alpha=0.7, label='Corte LP: 45 Hz')

ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('PSD (mV²/Hz)')
ax.legend()
ax.set_xlim([0, 100])
plt.tight_layout()
plt.savefig('psd_analisis.png', dpi=150, bbox_inches='tight')
plt.show()
print(" PSD confirma: ruido de red eléctrica (60 Hz) y artefactos de alta frecuencia eliminados.")

"""### Paso 4.2 – Detección de Picos R y Segmentación"""

# ─────────────────────────────────────────────────────────
#  SEGMENTACIÓN DE LATIDOS
#  Usamos las anotaciones MIT-BIH (precisas) como posiciones R
# ─────────────────────────────────────────────────────────

WINDOW_SIZE  = 256    # Longitud fija del latido (potencia de 2)
HALF_WINDOW  = WINDOW_SIZE // 2  # 128 muestras antes y después del pico R

def segment_beats(record_id, wfdb_dir, aami_map, window=WINDOW_SIZE):
    """
    Extrae latidos segmentados de un registro MIT-BIH.

    Returns:
        beats  : np.ndarray de shape (N_beats, window)
        labels : list de etiquetas AAMI
        patient_ids: list con el ID del paciente
    """
    half = window // 2
    rpath = os.path.join(wfdb_dir, record_id)

    rec = wfdb.rdrecord(rpath)
    ann = wfdb.rdann(rpath, 'atr')

    # Señal cruda (canal MLII)
    raw = rec.p_signal[:, 0]

    # Señal filtrada
    clean = full_ecg_pipeline(raw, fs=rec.fs)

    beats, labels, pids = [], [], []

    for sample, sym in zip(ann.sample, ann.symbol):
        if sym not in aami_map:
            continue  # Ignorar anotaciones que no son latidos

        start = sample - half
        end   = sample + half

        # Verificar límites: descartar latidos en los bordes del registro
        if start < 0 or end > len(clean):
            continue

        beat = clean[start:end]

        # Normalización Min-Max individual (0 a 1)
        b_min, b_max = beat.min(), beat.max()
        if b_max - b_min < 1e-6:  # Evitar división por cero
            continue
        beat_norm = (beat - b_min) / (b_max - b_min)

        beats.append(beat_norm)
        labels.append(aami_map[sym])
        pids.append(record_id)

    return np.array(beats, dtype=np.float32), labels, pids

print(" Función de segmentación definida.")
print(f"   Ventana por latido : {WINDOW_SIZE} muestras = {WINDOW_SIZE/FS*1000:.1f} ms")
print(f"   Centro en pico R   : ±{HALF_WINDOW} muestras ({HALF_WINDOW/FS*1000:.1f} ms)")
print(f"   Normalización      : Min-Max [0, 1] individual por latido")

# ─────────────────────────────────────────────────────────
#  EXTRACCIÓN DE TODOS LOS LATIDOS (todos los registros)
# ─────────────────────────────────────────────────────────
all_beats  = []
all_labels = []
all_pids   = []

print(f" Segmentando latidos de {len(record_ids)} registros...")
for i, rid in enumerate(record_ids):
    try:
        beats, labels, pids = segment_beats(rid, wfdb_dir, AAMI_MAP)
        all_beats.append(beats)
        all_labels.extend(labels)
        all_pids.extend(pids)
        print(f"  [{i+1:02d}/{len(record_ids)}] Registro {rid}: {len(labels):>5,} latidos extraídos")
    except Exception as e:
        print(f"    Error en registro {rid}: {e}")

# Concatenar todo
X = np.concatenate(all_beats, axis=0)    # Shape: (N_total, 256)
y = np.array(all_labels)                 # Shape: (N_total,)
patients = np.array(all_pids)            # Shape: (N_total,)

print(f"\n Segmentación completa.")
print(f"   X shape  : {X.shape}  → ({X.shape[0]:,} latidos × {X.shape[1]} muestras)")
print(f"   y shape  : {y.shape}")
print(f"   dtype    : {X.dtype}")
print(f"   Memoria  : {X.nbytes / 1e6:.1f} MB")

# ─────────────────────────────────────────────────────────
#  VISUALIZACIÓN DE LATIDOS SEGMENTADOS POR CLASE
# ─────────────────────────────────────────────────────────
classes_present = [c for c in ['N', 'S', 'V', 'F', 'Q'] if c in y]
colors_class = {'N': '#27ae60', 'S': '#2980b9', 'V': '#e74c3c', 'F': '#e67e22', 'Q': '#7f8c8d'}
n_samples_per_class = 5

fig, axes = plt.subplots(len(classes_present), n_samples_per_class,
                          figsize=(15, 2.5 * len(classes_present)))
fig.suptitle('Latidos Segmentados y Normalizados por Clase AAMI (5 ejemplos c/u)',
             fontsize=12, fontweight='bold')

t = np.arange(WINDOW_SIZE) / FS * 1000  # tiempo en ms

for row, cls in enumerate(classes_present):
    idx = np.where(y == cls)[0]
    sample_idx = np.random.choice(idx, size=min(n_samples_per_class, len(idx)), replace=False)

    for col, sidx in enumerate(sample_idx):
        ax = axes[row, col] if len(classes_present) > 1 else axes[col]
        ax.plot(t, X[sidx], color=colors_class[cls], linewidth=1.2)
        ax.axvline(x=HALF_WINDOW/FS*1000, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_ylim([-0.1, 1.1])
        if col == 0:
            ax.set_ylabel(f'Clase {cls}\n{class_desc[cls]}', fontsize=9, fontweight='bold')
        if row == 0:
            ax.set_title(f'Muestra {col+1}', fontsize=9)
        if row == len(classes_present) - 1:
            ax.set_xlabel('ms')
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('latidos_segmentados.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Latidos centrados en el pico R y normalizados [0,1].")

"""## 5. Indexación con Polars y Mapeo AAMI"""

# ─────────────────────────────────────────────────────────
#  GESTIÓN DE METADATA CON POLARS
# ─────────────────────────────────────────────────────────
# Crear índice para cada latido (el array X se referencia por posición)
beat_ids = np.arange(len(y))

meta_df = pl.DataFrame({
    'beat_id'  : beat_ids.tolist(),
    'patient'  : all_pids,
    'aami_class': all_labels,
})

# Añadir one-hot encoding en el DataFrame
for cls in ['N', 'S', 'V', 'F', 'Q']:
    meta_df = meta_df.with_columns(
        pl.when(pl.col('aami_class') == cls).then(1).otherwise(0).alias(f'oh_{cls}')
    )

print(" DataFrame de metadata (Polars):")
print(meta_df.head(10))
print(f"\n📐 Shape: {meta_df.shape[0]:,} filas × {meta_df.shape[1]} columnas")
print(f"\n📈 Distribución por clase:")
print(meta_df.group_by('aami_class').len().sort('len', descending=True))

"""## 6. División Train / Validation / Test (sin contaminación)"""

# ─────────────────────────────────────────────────────────
#  SPLIT BASADO EN PACIENTES (Patient-Level Split)
#  CRITERIO ANTI-CONTAMINACIÓN: un paciente NUNCA aparece
#  a la vez en entrenamiento y en evaluación.
# ─────────────────────────────────────────────────────────

unique_patients = list(set(all_pids))
np.random.seed(42)
np.random.shuffle(unique_patients)

n_total = len(unique_patients)
n_train = int(0.70 * n_total)  # 70%
n_val   = int(0.15 * n_total)  # 15%
# n_test  = remaining          # 15%

train_patients = set(unique_patients[:n_train])
val_patients   = set(unique_patients[n_train:n_train + n_val])
test_patients  = set(unique_patients[n_train + n_val:])

print(" CRITERIO DE DIVISIÓN: Patient-Level Split")
print("   → Un mismo paciente NO puede aparecer en train Y en val/test simultáneamente.")
print("   → Esto previene el data leakage: el modelo no ve morfologías del paciente de prueba.")
print()

# Crear máscaras booleanas
patient_arr  = np.array(all_pids)
train_mask   = np.array([p in train_patients for p in patient_arr])
val_mask     = np.array([p in val_patients   for p in patient_arr])
test_mask    = np.array([p in test_patients  for p in patient_arr])

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"{'Conjunto':<12} {'Pacientes':>10} {'Latidos':>12} {'%':>7}")
print("-" * 45)
for name, patients, arr in [
    ('Train',      train_patients, X_train),
    ('Validation', val_patients,   X_val),
    ('Test',       test_patients,  X_test)
]:
    pct = len(arr) / len(X) * 100
    print(f"  {name:<10} {len(patients):>10}   {len(arr):>10,}   {pct:>6.1f}%")
print("-" * 45)
print(f"  {'TOTAL':<10} {n_total:>10}   {len(X):>10,}   {'100.0%':>7}")

# Verificación de no contaminación
assert len(train_patients & val_patients)  == 0, " Contaminación train-val!"
assert len(train_patients & test_patients) == 0, " Contaminación train-test!"
assert len(val_patients   & test_patients) == 0, " Contaminación val-test!"
print("\n Verificación de no-contaminación: PASSED (0 pacientes solapados)")

# ─────────────────────────────────────────────────────────
#  DISTRIBUCIÓN DE CLASES EN CADA SPLIT
# ─────────────────────────────────────────────────────────
def class_distribution(labels, title):
    cnts = Counter(labels)
    total = sum(cnts.values())
    print(f"\n  {title}")
    for cls in ['N', 'S', 'V', 'F', 'Q']:
        c = cnts.get(cls, 0)
        print(f"    {cls}: {c:>7,}  ({c/total*100:.1f}%)")

print(" Distribución de clases por split:")
class_distribution(y_train, f"TRAIN  ({len(y_train):,} latidos)")
class_distribution(y_val,   f"VAL    ({len(y_val):,} latidos)")
class_distribution(y_test,  f"TEST   ({len(y_test):,} latidos)")

"""## 7. Estrategia de Manejo del Desbalance de Clases"""

# ─────────────────────────────────────────────────────────
#  ESTRATEGIA DE BALANCEO (solo en entrenamiento)
#
#  Estrategia elegida: OVERSAMPLING con duplicación aleatoria
#  (compatible con señales 1D y con el CVAE)
#
#  Para el proyecto generativo (CVAE), también se usará
#  generación sintética como balanceo adicional.
# ─────────────────────────────────────────────────────────

def oversample_training_set(X_tr, y_tr, target_ratio=0.15):
    """
    Oversampling de clases minoritarias.
    target_ratio: fracción mínima que debe tener cada clase respecto al total.
    """
    class_counts = Counter(y_tr)
    n_majority   = max(class_counts.values())

    new_X, new_y = [X_tr.copy()], list(y_tr.copy())

    for cls, cnt in class_counts.items():
        target = int(n_majority * target_ratio)
        if cnt < target:
            deficit = target - cnt
            idx = np.where(y_tr == cls)[0]
            chosen = np.random.choice(idx, size=deficit, replace=True)
            new_X.append(X_tr[chosen])
            new_y.extend([cls] * deficit)

    X_bal = np.concatenate(new_X, axis=0)
    y_bal = np.array(new_y)

    # Shuffle
    shuffle_idx = np.random.permutation(len(X_bal))
    return X_bal[shuffle_idx], y_bal[shuffle_idx]

X_train_bal, y_train_bal = oversample_training_set(X_train, y_train, target_ratio=0.10)

print(" Comparación ANTES vs DESPUÉS del balanceo (solo train):")
print(f"\n{'Clase':<8} {'Antes':>10} {'%':>7}   {'Después':>10} {'%':>7}")
print("-" * 50)
before = Counter(y_train)
after  = Counter(y_train_bal)
for cls in ['N', 'S', 'V', 'F', 'Q']:
    b = before.get(cls, 0)
    a = after.get(cls, 0)
    bp = b / len(y_train) * 100
    ap = a / len(y_train_bal) * 100
    print(f"  {cls:<6} {b:>10,} {bp:>6.1f}%   {a:>10,} {ap:>6.1f}%")
print("-" * 50)
print(f"  {'TOTAL':<6} {len(y_train):>10,}         {len(y_train_bal):>10,}")

print(f"\n Dataset de entrenamiento balanceado: {len(X_train_bal):,} latidos")
print("ℹ️  Val y Test NO se balancean (deben reflejar la distribución real del mundo)")

# ─────────────────────────────────────────────────────────
#  VISUALIZACIÓN DEL BALANCEO
# ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Distribución de Clases en Entrenamiento – Antes vs Después del Balanceo',
             fontweight='bold')

x_pos = np.arange(len(classes_present))
width = 0.35
before_vals = [before.get(c, 0) for c in classes_present]
after_vals  = [after.get(c, 0)  for c in classes_present]

ax1.bar(x_pos - width/2, before_vals, width, label='Antes', color='#e74c3c', alpha=0.8)
ax1.bar(x_pos + width/2, after_vals,  width, label='Después', color='#27ae60', alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"{c}\n{class_desc[c]}" for c in classes_present])
ax1.set_ylabel('Número de latidos')
ax1.set_title('Escala lineal')
ax1.legend()

ax2.bar(x_pos - width/2, before_vals, width, label='Antes', color='#e74c3c', alpha=0.8)
ax2.bar(x_pos + width/2, after_vals,  width, label='Después', color='#27ae60', alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"{c}\n{class_desc[c]}" for c in classes_present])
ax2.set_ylabel('Número de latidos (log)')
ax2.set_title('Escala logarítmica')
ax2.set_yscale('log')
ax2.legend()

plt.tight_layout()
plt.savefig('balanceo_clases.png', dpi=150, bbox_inches='tight')
plt.show()

"""## 8. Resumen Final del Pipeline"""

# ─────────────────────────────────────────────────────────
#  RESUMEN EJECUTIVO DEL PIPELINE
# ─────────────────────────────────────────────────────────
print("=" * 65)
print("           RESUMEN FINAL DEL PIPELINE DE DATOS")
print("           Avance 2 – Proyecto TCN-CVAE ECG")
print("=" * 65)

print("\n DATASET")
print(f"   Nombre   : MIT-BIH Arrhythmia Database")
print(f"   Fuente   : PhysioNet (Kaggle: mondejar/mitbih-database)")
print(f"   Registros: {len(record_ids)} pacientes")
print(f"   Señal    : ECG 1D (Lead MLII), 360 Hz, ~30 min/paciente")
print(f"   Formato  : .dat (señal), .hea (header), .atr (anotaciones)")

print("\n🔧 PREPROCESAMIENTO")
print(f"   1. Filtro pasa-alto  : Butterworth 4°, 0.5 Hz  (elimina baseline wander)")
print(f"   2. Filtro Notch      : IIR 60 Hz, Q=30         (elimina ruido red eléctrica)")
print(f"   3. Filtro pasa-bajo  : Butterworth 4°, 45 Hz   (elimina EMG muscular)")
print(f"   4. Segmentación      : Ventana de {WINDOW_SIZE} muestras centrada en pico R")
print(f"                         ({WINDOW_SIZE/FS*1000:.0f} ms de duración)")
print(f"   5. Normalización     : Min-Max [0,1] por latido individual")
print(f"   6. Mapeo AAMI        : {len(AAMI_MAP)} símbolos → 5 clases")

print("\n ESTADÍSTICAS DE LOS DATOS")
print(f"   Total latidos válidos: {len(X):,}")
print(f"   Shape tensor         : {X.shape} (float32)")
print(f"   Memoria              : {X.nbytes/1e6:.1f} MB")

print("\n🔀 DIVISIÓN DEL DATASET")
print(f"   Estrategia: Patient-Level Split (anti-contaminación)")
print(f"   Train      : {len(X_train):>8,} latidos  ({len(X_train)/len(X)*100:.1f}%) – {len(train_patients)} pacientes")
print(f"   Validation : {len(X_val):>8,} latidos  ({len(X_val)/len(X)*100:.1f}%) – {len(val_patients)} pacientes")
print(f"   Test       : {len(X_test):>8,} latidos  ({len(X_test)/len(X)*100:.1f}%) – {len(test_patients)} pacientes")

print("\n  MANEJO DE DESBALANCE")
print(f"   Técnica: Oversampling aleatorio de clases minoritarias")
print(f"   Aplicado: Solo en entrenamiento")
print(f"   Train balanceado: {len(X_train_bal):,} latidos")

print("\n LISTO PARA FASE 3: Arquitectura TCN-CVAE")
print("   X_train_bal → Encoder TCN → z (latente) → Decoder TCN → señal sintética")
print("=" * 65)

# ─────────────────────────────────────────────────────────
#  GUARDAR LOS ARRAYS PROCESADOS EN DISCO  (opcional)
#  Útil para no tener que volver a correr el preprocesamiento
# ─────────────────────────────────────────────────────────
import os
save_dir = './ecg_processed'
os.makedirs(save_dir, exist_ok=True)

np.save(f'{save_dir}/X_train.npy',     X_train_bal)
np.save(f'{save_dir}/y_train.npy',     y_train_bal)
np.save(f'{save_dir}/X_val.npy',       X_val)
np.save(f'{save_dir}/y_val.npy',       y_val)
np.save(f'{save_dir}/X_test.npy',      X_test)
np.save(f'{save_dir}/y_test.npy',      y_test)

# También guardar el metadata con Polars
meta_df.write_parquet(f'{save_dir}/metadata.parquet')

print(" Arrays guardados en", save_dir)
for f in os.listdir(save_dir):
    fpath = os.path.join(save_dir, f)
    print(f"   {f:<30} {os.path.getsize(fpath)/1e6:.1f} MB")

print("\n Para cargarlos en el siguiente notebook:")
print("   X_train = np.load('/content/ecg_processed/X_train.npy')")
print("   y_train = np.load('/content/ecg_processed/y_train.npy')")

"""---
##  Fin del Avance 2

**Lo que queda para el Avance 3:** Implementación de la arquitectura TCN-CVAE (Fases 3 y 4 del plan de implementación):
- Módulo Encoder TCN
- Módulo de Reparametrización
- Módulo Decoder TCN con upsampling
- Función de pérdida MSE + KL con β-scheduling
- Generación de latidos sintéticos condicionales
"""
# -*- coding: utf-8 -*-
"""ECG_TCNCVAE_Entrenamiento.ipynb

#  TCN-CVAE – Entrenamiento y Generación Sintética de ECG
## Proyecto: Generación Sintética de Señales ECG · MIT-BIH Arrhythmia Database

---
**Prerequisito:** haber ejecutado `ECG_Avance2_Pipeline.ipynb` y tener los arrays en `/content/ecg_processed/`.

**Flujo de este notebook:**
1. Instalación y configuración GPU
2. Carga de los datos preprocesados (Avance 2)
3. `ECGDataset` + `DataLoader` PyTorch
4. Bloques constructores: `CausalDilatedBlock` + `TCNStack`
5. Módulos `TCNEncoder`, `Reparametrize`, `TCNDecoder`
6. Modelo raíz `TCNCVAE` + conteo de parámetros
7. Función de pérdida + β-scheduler
8. Bucle de entrenamiento con curvas de loss
9. Generación de latidos sintéticos condicionales
10. Validación: visualización, análisis espectral y t-SNE

## 1. Instalación y Configuración
"""

# PyTorch instalado localmente.
# Solo necesitamos verificar que la GPU está disponible.

print(' torchinfo instalado')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import os, time, copy, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torchinfo import summary

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# ── Reproducibilidad ──────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Dispositivo ───────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  Dispositivo: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ── Estilo matplotlib ─────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'font.size': 11
})
print(' Imports completados')

"""## 2. Carga de Datos Preprocesados (desde Avance 2)"""

# ─────────────────────────────────────────────────────────
#  CARGA DE ARRAYS GUARDADOS EN EL AVANCE 2
# ─────────────────────────────────────────────────────────
DATA_DIR = './ecg_processed'

# Si no existe el directorio, recordar al usuario que primero ejecute el Avance 2
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        ' No se encontró /content/ecg_processed/.\n'
        '   Por favor ejecuta primero el pipeline para generar los arrays.'
    )

X_train = np.load(f'{DATA_DIR}/X_train.npy')   # shape: (N_train, 256)
y_train = np.load(f'{DATA_DIR}/y_train.npy')   # shape: (N_train,)  dtype: str
X_val   = np.load(f'{DATA_DIR}/X_val.npy')
y_val   = np.load(f'{DATA_DIR}/y_val.npy')
X_test  = np.load(f'{DATA_DIR}/X_test.npy')
y_test  = np.load(f'{DATA_DIR}/y_test.npy')

# Mapeo de clase string → índice entero (N=0, S=1, V=2, F=3, Q=4)
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
CLASS_IDX   = {c: i for i, c in enumerate(CLASS_NAMES)}
N_CLASSES   = len(CLASS_NAMES)

def labels_to_idx(y_str):
    return np.array([CLASS_IDX[c] for c in y_str], dtype=np.int64)

y_train_idx = labels_to_idx(y_train)
y_val_idx   = labels_to_idx(y_val)
y_test_idx  = labels_to_idx(y_test)

print(' Arrays cargados correctamente')
print(f'   X_train : {X_train.shape}  |  y_train: {y_train.shape}')
print(f'   X_val   : {X_val.shape}  |  y_val  : {y_val.shape}')
print(f'   X_test  : {X_test.shape}  |  y_test : {y_test.shape}')
print(f'\n   Distribución train: {dict(Counter(y_train))}')

"""## 3. Dataset y DataLoader PyTorch"""

# ─────────────────────────────────────────────────────────
#  CLASE ECGDataset
#  Envuelve los arrays numpy en la interfaz Dataset de PyTorch.
#  Cada __getitem__ devuelve:
#    x      : tensor float32 shape (1, 256)  → señal con dim de canal
#    c      : tensor float32 shape (5,)      → one-hot de la clase AAMI
#    label  : tensor int64                   → índice numérico de clase
# ─────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, X, y_idx, n_classes=5):
        # X: numpy (N, 256)  y_idx: numpy (N,) int
        self.X        = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,256)
        self.y        = torch.tensor(y_idx, dtype=torch.long)
        self.n_classes = n_classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x     = self.X[idx]          # (1, 256)
        label = self.y[idx]          # escalar int64
        # One-hot vector de condición
        c = F.one_hot(label, num_classes=self.n_classes).float()  # (5,)
        return x, c, label

# ── Hiperparámetros del DataLoader ────────────────────────
BATCH_SIZE  = 128
NUM_WORKERS = 2

train_ds = ECGDataset(X_train, y_train_idx)
val_ds   = ECGDataset(X_val,   y_val_idx)
test_ds  = ECGDataset(X_test,  y_test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))

print(' DataLoaders creados')
print(f'   Train  : {len(train_ds):,} muestras  →  {len(train_loader)} batches de {BATCH_SIZE}')
print(f'   Val    : {len(val_ds):,} muestras  →  {len(val_loader)} batches')
print(f'   Test   : {len(test_ds):,} muestras  →  {len(test_loader)} batches')

# Verificar shapes de un batch
x_b, c_b, lbl_b = next(iter(train_loader))
print(f'\n   Shapes de un batch de ejemplo:')
print(f'   x shape : {x_b.shape}   → (Batch, Canal=1, L=256)')
print(f'   c shape : {c_b.shape}  → (Batch, N_clases=5)')
print(f'   lbl shape: {lbl_b.shape} → (Batch,)')

"""## 4. Bloques Constructores: CausalDilatedBlock y TCNStack"""

# ─────────────────────────────────────────────────────────
#  CausalDilatedBlock
#  Bloque residual TCN con:
#   - Convolución dilatada con padding SIMÉTRICO (no causal)
#     porque el ECG no es tiempo real; aprovechamos contexto
#     pasado Y futuro alrededor del pico R.
#   - WeightNorm para estabilizar el gradiente en capas profundas.
#   - Activación GELU (suaviza el gradiente vs ReLU).
#   - Skip connection con Conv1x1 si los canales cambian.
# ─────────────────────────────────────────────────────────
class CausalDilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        # Padding simétrico = dilation * (kernel_size - 1) // 2
        pad = dilation * (kernel_size - 1) // 2

        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                  dilation=dilation, padding=pad)),
            nn.GELU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                  dilation=dilation, padding=pad)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Skip connection: 1x1 conv si los canales difieren
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x) + self.skip(x)

# ─────────────────────────────────────────────────────────
#  TCNStack
#  Apila N bloques con dilataciones crecientes [1, 2, 4, 8].
#  El campo receptivo efectivo con kernel=3, 4 bloques:
#  RF = 1 + sum(2*(k-1)*d) = 1 + 2*2*(1+2+4+8) = 1 + 60 = 61 muestras
#  Con 256 muestras por latido, cubre todo el complejo QRS.
# ─────────────────────────────────────────────────────────
class TCNStack(nn.Module):
    def __init__(self, in_ch, hidden_ch=64, kernel_size=3,
                 dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        layers = []
        ch = in_ch
        for d in dilations:
            layers.append(CausalDilatedBlock(ch, hidden_ch, kernel_size, d, dropout))
            ch = hidden_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)    # shape preservada: (B, hidden_ch, L)

print(' CausalDilatedBlock y TCNStack definidos')
# Verificación rápida de shapes
_x = torch.randn(4, 6, 256)    # (Batch=4, Canales=6, L=256)
_out = TCNStack(in_ch=6, hidden_ch=64)(_x)
print(f'   Input shape : {_x.shape}')
print(f'   Output shape: {_out.shape}  (debe ser [4, 64, 256])')

"""## 5. Módulos TCNEncoder, Reparametrize y TCNDecoder"""

# ─────────────────────────────────────────────────────────
#  TCNEncoder
#  Toma la señal x (B,1,256) + condición c (B,5) y produce
#  los parámetros del espacio latente: mu y log_var.
#
#  Pasos:
#    1. Expandir c a (B,5,256) y concatenar con x → (B,6,256)
#    2. Pasar por TCNStack → (B,64,256)
#    3. GlobalAveragePooling → (B,64)
#    4. Dos capas Linear independientes → mu (B,Z) y log_var (B,Z)
# ─────────────────────────────────────────────────────────
class TCNEncoder(nn.Module):
    def __init__(self, signal_len=256, n_classes=5, hidden_ch=64,
                 latent_dim=32, dropout=0.1):
        super().__init__()
        self.signal_len = signal_len
        # in_ch = 1 (señal) + 5 (one-hot expandido) = 6
        self.tcn = TCNStack(in_ch=1 + n_classes, hidden_ch=hidden_ch,
                            dropout=dropout)
        # GlobalAvgPool y proyección a espacio latente
        self.fc_mu      = nn.Linear(hidden_ch, latent_dim)
        self.fc_log_var = nn.Linear(hidden_ch, latent_dim)

    def forward(self, x, c):
        # x: (B,1,256)  c: (B,5)
        # Expandir c para concatenar en la dimensión temporal
        c_exp = c.unsqueeze(-1).expand(-1, -1, self.signal_len)  # (B,5,256)
        xc    = torch.cat([x, c_exp], dim=1)                     # (B,6,256)

        h   = self.tcn(xc)                   # (B,64,256)
        h   = h.mean(dim=-1)                 # GlobalAvgPool → (B,64)

        mu      = self.fc_mu(h)              # (B, latent_dim)
        log_var = self.fc_log_var(h)         # (B, latent_dim)
        return mu, log_var

print(' TCNEncoder definido')
_enc = TCNEncoder()
_x, _c = torch.randn(4,1,256), torch.randn(4,5)
_mu, _lv = _enc(_x, _c)
print(f'   mu shape     : {_mu.shape}   (B=4, latent_dim=32)')
print(f'   log_var shape: {_lv.shape}')

# ─────────────────────────────────────────────────────────
#  Reparametrize
#  Truco de la reparametrización: z = mu + sigma * epsilon
#  donde epsilon ~ N(0, I) es ruido muestreado.
#
#  ¿Por qué? Si z = mu + sigma*eps, el gradiente puede fluir
#  a través de mu y sigma (deterministas), mientras que la
#  aleatoriedad queda en eps (no diferenciable, pero no necesita
#  gradiente). Sin este truco, el backpropagation no funcionaría
#  a través de una operación de muestreo.
# ─────────────────────────────────────────────────────────
class Reparametrize(nn.Module):
    def forward(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)           # sigma = exp(0.5 * log_var)
            eps = torch.randn_like(std)               # eps ~ N(0, I)
            return mu + std * eps
        else:
            # En evaluación usamos el valor esperado (determinista)
            return mu

print(' Reparametrize definido')
_reparam = Reparametrize()
_reparam.train()
_z = _reparam(_mu, _lv)
print(f'   z shape (train): {_z.shape}   ← muestreado con ruido')
_reparam.eval()
_z_eval = _reparam(_mu, _lv)
print(f'   z shape (eval) : {_z_eval.shape}  ← solo mu (determinista)')

# ─────────────────────────────────────────────────────────
#  TCNDecoder
#  Genera la señal sintética a partir de z + c.
#
#  Pasos:
#    1. Concat z (B,32) y c (B,5) → (B,37)
#    2. Linear → (B, 64*32) → Reshape → (B, 64, 32)
#    3. Upsampling ×3 con F.interpolate + CausalDilatedBlock:
#       (B,64,32) → (B,64,64) → (B,64,128) → (B,64,256)
#    4. Conv1d(64→1) + Sigmoid → señal en [0,1]
# ─────────────────────────────────────────────────────────
class TCNDecoder(nn.Module):
    def __init__(self, signal_len=256, n_classes=5, hidden_ch=64,
                 latent_dim=32, dropout=0.1):
        super().__init__()
        self.signal_len = signal_len
        self.hidden_ch  = hidden_ch
        self.init_len   = signal_len // 8   # 256//8 = 32

        # 1. Proyección z+c → secuencia inicial
        self.fc_in = nn.Sequential(
            nn.Linear(latent_dim + n_classes, hidden_ch * self.init_len),
            nn.GELU()
        )

        # 2. Tres bloques de upsampling: cada uno dobla la longitud
        #    ×2: 32→64  ×2: 64→128  ×2: 128→256
        self.up1 = CausalDilatedBlock(hidden_ch, hidden_ch, dilation=1, dropout=dropout)
        self.up2 = CausalDilatedBlock(hidden_ch, hidden_ch, dilation=2, dropout=dropout)
        self.up3 = CausalDilatedBlock(hidden_ch, hidden_ch, dilation=4, dropout=dropout)

        # 3. Capa de salida: proyectar a 1 canal
        self.out_conv = nn.Sequential(
            nn.Conv1d(hidden_ch, 1, kernel_size=1),
            nn.Sigmoid()   # señal normalizada [0, 1]
        )

    def forward(self, z, c):
        # z: (B, latent_dim)   c: (B, 5)
        zc = torch.cat([z, c], dim=-1)              # (B, latent_dim+5)
        h  = self.fc_in(zc)                         # (B, 64*32)
        h  = h.view(h.size(0), self.hidden_ch,
                    self.init_len)                   # (B, 64, 32)

        # Upsampling progresivo: interpolate lineal + refinamiento TCN
        h = F.interpolate(h, scale_factor=2, mode='linear', align_corners=False)  # (B,64,64)
        h = self.up1(h)

        h = F.interpolate(h, scale_factor=2, mode='linear', align_corners=False)  # (B,64,128)
        h = self.up2(h)

        h = F.interpolate(h, scale_factor=2, mode='linear', align_corners=False)  # (B,64,256)
        h = self.up3(h)

        x_hat = self.out_conv(h)                    # (B, 1, 256)
        return x_hat

print(' TCNDecoder definido')
_dec = TCNDecoder()
_z_test = torch.randn(4, 32)
_c_test = torch.zeros(4, 5); _c_test[:, 2] = 1.0   # clase V
_out = _dec(_z_test, _c_test)
print(f'   Input z: {_z_test.shape}  c: {_c_test.shape}')
print(f'   Output : {_out.shape}   (debe ser [4, 1, 256])')
print(f'   Rango salida: [{_out.min().item():.3f}, {_out.max().item():.3f}]  (debe estar en [0,1])')

"""## 6. Modelo Completo TCNCVAE + Resumen de Parámetros"""

# ─────────────────────────────────────────────────────────
#  TCNCVAE – Modelo raíz
#  Ensambla Encoder + Reparametrize + Decoder.
#
#  forward() → (x_hat, mu, log_var)   para el bucle de entrenamiento
#  generate() → (B, 1, 256)            para inferencia sin encoder
# ─────────────────────────────────────────────────────────
class TCNCVAE(nn.Module):
    def __init__(self, signal_len=256, n_classes=5,
                 hidden_ch=64, latent_dim=32, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = TCNEncoder(signal_len, n_classes, hidden_ch, latent_dim, dropout)
        self.reparam    = Reparametrize()
        self.decoder    = TCNDecoder(signal_len, n_classes, hidden_ch, latent_dim, dropout)

    def forward(self, x, c):
        """Paso completo (usado durante entrenamiento)."""
        mu, log_var = self.encoder(x, c)
        z           = self.reparam(mu, log_var)
        x_hat       = self.decoder(z, c)
        return x_hat, mu, log_var

    @torch.no_grad()
    def generate(self, c, n_samples=1, device=None):
        """
        Generación pura: samplea z ~ N(0,I) y decodifica.
        c: tensor (n_samples, 5) one-hot  O  índice int escalar.
        """
        self.eval()
        dev = device or next(self.parameters()).device

        # Si se pasa un entero, construir el one-hot
        if isinstance(c, int):
            c_oh = F.one_hot(torch.tensor([c] * n_samples), num_classes=5).float().to(dev)
        else:
            c_oh = c.to(dev)

        z     = torch.randn(n_samples, self.latent_dim).to(dev)
        x_gen = self.decoder(z, c_oh)    # (n_samples, 1, 256)
        return x_gen

# ── Instanciación y resumen ───────────────────────────────
model = TCNCVAE(
    signal_len  = 256,
    n_classes   = N_CLASSES,
    hidden_ch   = 64,
    latent_dim  = 32,
    dropout     = 0.1
).to(DEVICE)

print(' Modelo TCNCVAE creado y enviado a', DEVICE)

# Conteo de parámetros
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\n   Parámetros totales      : {total_params:,}')
print(f'   Parámetros entrenables  : {trainable_params:,}')
print(f'   Memoria aprox. (float32): {total_params * 4 / 1e6:.1f} MB')

# Resumen detallado con torchinfo
summary(model,
        input_data=[torch.randn(BATCH_SIZE, 1, 256).to(DEVICE),
                    torch.randn(BATCH_SIZE, 5).to(DEVICE)],
        col_names=['input_size', 'output_size', 'num_params'],
        depth=3, verbose=1)

"""## 7. Función de Pérdida y β-Scheduler"""

# ─────────────────────────────────────────────────────────
#  FUNCIÓN DE PÉRDIDA DEL CVAE
#
#  Loss = Reconstrucción + β · KL
#
#  Reconstrucción (MSE):
#    Mide qué tan fielmente el decoder reprodujo la señal original.
#    Usamos reducción 'mean' para que la escala sea independiente
#    del batch_size y la longitud de señal.
#
#  KL Divergence:
#    Regulariza el espacio latente para que z siga una N(0,I).
#    Fórmula analítica: -0.5 * sum(1 + log_var - mu² - exp(log_var))
#    con media por latido para escala consistente.
#
#  β-Scheduler (KL Annealing):
#    Empieza en β=0 para que el encoder aprenda a reconstruir
#    sin restricción del espacio latente (evita el posterior collapse).
#    Luego sube linealmente hasta β_max.
# ─────────────────────────────────────────────────────────
def cvae_loss(x_hat, x, mu, log_var, beta=1.0):
    """
    Calcula la pérdida del CVAE.
    Retorna: loss_total, recon_loss, kl_loss (todos escalares)
    """
    # Pérdida de reconstrucción
    recon = F.mse_loss(x_hat, x, reduction='mean')

    # Divergencia KL (forma analítica)
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Pérdida combinada
    total = recon + beta * kl
    return total, recon, kl

class BetaScheduler:
    """
    Scheduler lineal para el peso β de la KL divergence.
    Durante las primeras `warmup_epochs` épocas, β=0.
    Luego sube linealmente hasta β_max en `anneal_epochs` épocas.
    """
    def __init__(self, beta_max=1.0, warmup_epochs=5, anneal_epochs=20):
        self.beta_max      = beta_max
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs

    def get_beta(self, epoch):
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.warmup_epochs + self.anneal_epochs:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return self.beta_max * progress
        else:
            return self.beta_max

# ── Visualización del β-schedule ─────────────────────────
scheduler_preview = BetaScheduler(beta_max=1.0, warmup_epochs=5, anneal_epochs=20)
epochs_preview = range(50)
betas_preview  = [scheduler_preview.get_beta(e) for e in epochs_preview]

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(epochs_preview, betas_preview, color='#9b59b6', linewidth=2)
ax.axvspan(0,  5, alpha=0.08, color='#e74c3c', label='Warmup (β=0): solo reconstrucción')
ax.axvspan(5, 25, alpha=0.08, color='#f39c12', label='Annealing lineal (β: 0→1)')
ax.axvspan(25,50, alpha=0.08, color='#27ae60', label='Entrenamiento estable (β=1)')
ax.set_xlabel('Época')
ax.set_ylabel('β (peso KL)')
ax.set_title('β-Scheduler: KL Annealing', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('beta_schedule.png', dpi=130, bbox_inches='tight')
plt.show()
print(' Función de pérdida y β-Scheduler definidos')

"""## 8. Bucle de Entrenamiento"""

# ─────────────────────────────────────────────────────────
#  HIPERPARÁMETROS DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────
N_EPOCHS      = 50       # Ajustar según tiempo disponible
LR            = 1e-3     # Learning rate inicial para AdamW
WEIGHT_DECAY  = 1e-4     # Regularización L2
GRAD_CLIP     = 1.0      # Gradient clipping (previene explosión en TCN profundo)
BETA_MAX      = 1.0
WARMUP_EPOCHS = 5
ANNEAL_EPOCHS = 20

# Directorio para checkpoints
CKPT_DIR = './tcncvae_checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Optimizador y LR Scheduler ───────────────────────────
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=LR, weight_decay=WEIGHT_DECAY)

# CosineAnnealingLR: decae el LR suavemente, útil para VAEs
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_EPOCHS, eta_min=1e-5
)

beta_scheduler = BetaScheduler(BETA_MAX, WARMUP_EPOCHS, ANNEAL_EPOCHS)

print(' Configuración de entrenamiento:')
print(f'   Épocas        : {N_EPOCHS}')
print(f'   Batch size    : {BATCH_SIZE}')
print(f'   Optimizador   : AdamW  lr={LR}  wd={WEIGHT_DECAY}')
print(f'   LR Scheduler  : CosineAnnealingLR  T_max={N_EPOCHS}')
print(f'   Grad clip     : {GRAD_CLIP}')
print(f'   β-schedule    : warmup={WARMUP_EPOCHS}  anneal={ANNEAL_EPOCHS}  β_max={BETA_MAX}')
print(f'   Checkpoints   : {CKPT_DIR}')

# ─────────────────────────────────────────────────────────
#  FUNCIONES DE TRAIN / VALIDATE POR ÉPOCA
# ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, beta, device):
    model.train()
    total_loss = recon_loss_sum = kl_loss_sum = 0.0

    for x, c, _ in loader:
        x, c = x.to(device), c.to(device)

        optimizer.zero_grad()
        x_hat, mu, log_var = model(x, c)
        loss, recon, kl    = cvae_loss(x_hat, x, mu, log_var, beta)

        loss.backward()
        # Gradient clipping: evita explosión en TCN profundo
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss    += loss.item()
        recon_loss_sum += recon.item()
        kl_loss_sum    += kl.item()

    n = len(loader)
    return total_loss/n, recon_loss_sum/n, kl_loss_sum/n

@torch.no_grad()
def validate(model, loader, beta, device):
    model.eval()
    total_loss = recon_loss_sum = kl_loss_sum = 0.0

    for x, c, _ in loader:
        x, c = x.to(device), c.to(device)
        x_hat, mu, log_var = model(x, c)
        loss, recon, kl    = cvae_loss(x_hat, x, mu, log_var, beta)
        total_loss     += loss.item()
        recon_loss_sum += recon.item()
        kl_loss_sum    += kl.item()

    n = len(loader)
    return total_loss/n, recon_loss_sum/n, kl_loss_sum/n

# ─────────────────────────────────────────────────────────
#  BUCLE PRINCIPAL DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────
history = {
    'train_loss': [], 'val_loss': [],
    'train_recon': [], 'val_recon': [],
    'train_kl': [],   'val_kl': [],
    'beta': [], 'lr': []
}

best_val_loss  = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

print(' Iniciando entrenamiento...\n')
print(f'{"Epoch":>6} {"β":>5} {"Train Loss":>12} {"Recon":>9} {"KL":>7}  '
      f'{"Val Loss":>10} {"Recon":>9} {"KL":>7}  {"LR":>8}  {"t(s)":>5}')
print('─' * 100)

for epoch in range(N_EPOCHS):
    t0   = time.time()
    beta = beta_scheduler.get_beta(epoch)
    current_lr = optimizer.param_groups[0]['lr']

    tr_loss, tr_recon, tr_kl = train_one_epoch(model, train_loader, optimizer, beta, DEVICE)
    vl_loss, vl_recon, vl_kl = validate(model, val_loader, beta, DEVICE)

    lr_scheduler.step()
    elapsed = time.time() - t0

    # Guardar historial
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_recon'].append(tr_recon)
    history['val_recon'].append(vl_recon)
    history['train_kl'].append(tr_kl)
    history['val_kl'].append(vl_kl)
    history['beta'].append(beta)
    history['lr'].append(current_lr)

    # Guardar mejor modelo
    if vl_loss < best_val_loss:
        best_val_loss  = vl_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'history': history},
                   f'{CKPT_DIR}/best_model.pt')
        flag = '  mejor'
    else:
        flag = ''

    print(f'{epoch+1:>6} {beta:>5.2f} {tr_loss:>12.5f} {tr_recon:>9.5f} {tr_kl:>7.4f}  '
          f'{vl_loss:>10.5f} {vl_recon:>9.5f} {vl_kl:>7.4f}  '
          f'{current_lr:>8.2e}  {elapsed:>4.1f}s{flag}')

# Restaurar mejor modelo
model.load_state_dict(best_model_wts)
print(f'\n Entrenamiento completado. Mejor val_loss = {best_val_loss:.5f}')
print(f'   Checkpoint guardado en {CKPT_DIR}/best_model.pt')

# ─────────────────────────────────────────────────────────
#  CURVAS DE PÉRDIDA
# ─────────────────────────────────────────────────────────
epochs_range = range(1, N_EPOCHS + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Curvas de Entrenamiento – TCN-CVAE', fontsize=13, fontweight='bold')

# Loss total
ax = axes[0, 0]
ax.plot(epochs_range, history['train_loss'], label='Train', color='#e74c3c', linewidth=1.8)
ax.plot(epochs_range, history['val_loss'],   label='Val',   color='#3498db', linewidth=1.8)
ax.set_title('Loss Total (MSE + β·KL)')
ax.set_xlabel('Época'); ax.set_ylabel('Loss'); ax.legend()

# Reconstrucción
ax = axes[0, 1]
ax.plot(epochs_range, history['train_recon'], label='Train', color='#e74c3c', linewidth=1.8)
ax.plot(epochs_range, history['val_recon'],   label='Val',   color='#3498db', linewidth=1.8)
ax.set_title('Loss de Reconstrucción (MSE)')
ax.set_xlabel('Época'); ax.set_ylabel('MSE'); ax.legend()

# KL divergence
ax = axes[1, 0]
ax.plot(epochs_range, history['train_kl'], label='Train', color='#9b59b6', linewidth=1.8)
ax.plot(epochs_range, history['val_kl'],   label='Val',   color='#1abc9c', linewidth=1.8)
ax.set_title('KL Divergence')
ax.set_xlabel('Época'); ax.set_ylabel('KL'); ax.legend()

# β y LR
ax  = axes[1, 1]
ax2 = ax.twinx()
ax.plot(epochs_range, history['beta'], color='#e67e22', linewidth=2, label='β')
ax2.plot(epochs_range, history['lr'], color='#2c3e50', linewidth=1.5,
         linestyle='--', label='LR')
ax.set_title('β-schedule y Learning Rate')
ax.set_xlabel('Época')
ax.set_ylabel('β', color='#e67e22')
ax2.set_ylabel('Learning Rate', color='#2c3e50')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=9)

plt.tight_layout()
plt.savefig('curvas_entrenamiento.png', dpi=150, bbox_inches='tight')
plt.show()

"""## 9. Generación de Latidos Sintéticos Condicionales"""

# ─────────────────────────────────────────────────────────
#  GENERACIÓN SINTÉTICA POR CLASE AAMI
#  Para cada una de las 5 clases, se genera a partir de z ~ N(0,I)
#  y el vector condicional c correspondiente.
#  El encoder NO interviene en inferencia.
# ─────────────────────────────────────────────────────────
N_GEN = 6  # Latidos sintéticos por clase

class_colors = {
    'N': '#27ae60', 'S': '#2980b9', 'V': '#e74c3c',
    'F': '#e67e22', 'Q': '#7f8c8d'
}
class_desc = {
    'N': 'Normal', 'S': 'Supraventricular',
    'V': 'Ventricular', 'F': 'Fusión', 'Q': 'Desconocido'
}

model.eval()
t_ms = np.arange(256) / 360 * 1000  # eje temporal en ms

fig, axes = plt.subplots(N_CLASSES, N_GEN, figsize=(15, 2.4 * N_CLASSES))
fig.suptitle('Latidos Sintéticos Generados por Clase AAMI (TCN-CVAE)',
             fontsize=12, fontweight='bold')

for row, (cls, cls_idx) in enumerate(CLASS_IDX.items()):
    # Generar N_GEN latidos sintéticos de esta clase
    x_syn = model.generate(c=cls_idx, n_samples=N_GEN, device=DEVICE)
    x_syn = x_syn.squeeze(1).cpu().numpy()   # (N_GEN, 256)

    for col in range(N_GEN):
        ax = axes[row, col]
        ax.plot(t_ms, x_syn[col], color=class_colors[cls], linewidth=1.2)
        ax.axvline(x=256/2/360*1000, color='gray', linestyle='--',
                   alpha=0.4, linewidth=0.8)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel(f'{cls}\n{class_desc[cls]}',
                          fontsize=9, fontweight='bold')
        if row == 0:
            ax.set_title(f'Sintético {col+1}', fontsize=9)
        if row == N_CLASSES - 1:
            ax.set_xlabel('ms')

plt.tight_layout()
plt.savefig('latidos_sinteticos.png', dpi=150, bbox_inches='tight')
plt.show()
print(' Latidos sintéticos generados por clase AAMI.')

# ─────────────────────────────────────────────────────────
#  COMPARACIÓN: REAL vs SINTÉTICO
#  Para las 3 clases más importantes (N, V, S),
#  se comparan 3 latidos reales contra 3 sintéticos.
# ─────────────────────────────────────────────────────────
COMPARE_CLASSES = ['N', 'V', 'S']
N_COMP = 3

fig, axes = plt.subplots(len(COMPARE_CLASSES), N_COMP * 2,
                          figsize=(15, 2.8 * len(COMPARE_CLASSES)))
fig.suptitle('Comparación: Latidos Reales (izq.) vs Sintéticos (der.) por clase',
             fontsize=12, fontweight='bold')

for row, cls in enumerate(COMPARE_CLASSES):
    cls_idx = CLASS_IDX[cls]
    col_color = class_colors[cls]

    # Latidos reales del set de test
    real_idx = np.where(y_test == cls)[0]
    chosen   = np.random.choice(real_idx, size=N_COMP, replace=False)
    X_real   = X_test[chosen]     # (N_COMP, 256)

    # Latidos sintéticos
    x_syn = model.generate(c=cls_idx, n_samples=N_COMP, device=DEVICE)
    X_syn = x_syn.squeeze(1).cpu().numpy()   # (N_COMP, 256)

    for col in range(N_COMP):
        # Real
        ax = axes[row, col]
        ax.plot(t_ms, X_real[col], color=col_color, linewidth=1.2)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel(f'{cls} – {class_desc[cls]}\n(REAL)',
                          fontsize=8, fontweight='bold')
        if row == 0: ax.set_title(f'Real {col+1}', fontsize=9, color='#444')
        if row == len(COMPARE_CLASSES)-1: ax.set_xlabel('ms')

        # Sintético
        ax = axes[row, col + N_COMP]
        ax.plot(t_ms, X_syn[col], color=col_color, linewidth=1.2,
                linestyle='--', alpha=0.85)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel(f'{cls} – {class_desc[cls]}\n(SINTÉTICO)',
                          fontsize=8, fontweight='bold', color='#555')
        if row == 0: ax.set_title(f'Sintético {col+1}', fontsize=9, color='#888')
        if row == len(COMPARE_CLASSES)-1: ax.set_xlabel('ms')

plt.tight_layout()
plt.savefig('comparacion_real_sintetico.png', dpi=150, bbox_inches='tight')
plt.show()

"""## 10. Validación de Calidad Generativa"""

# ─────────────────────────────────────────────────────────
#  ANÁLISIS ESPECTRAL: REAL vs SINTÉTICO
#  Compara la densidad espectral de potencia (PSD) media
#  para las 5 clases. Si el modelo aprendió bien la morfología,
#  los espectros reales y sintéticos deben solaparse.
# ─────────────────────────────────────────────────────────
from scipy.signal import welch

FS = 360
N_SPECTRAL = 200  # Latidos por clase para estimar el PSD medio

fig, axes = plt.subplots(1, N_CLASSES, figsize=(16, 4), sharey=True)
fig.suptitle('Análisis Espectral PSD: Reales vs Sintéticos por clase AAMI',
             fontsize=11, fontweight='bold')

model.eval()
for col, cls in enumerate(CLASS_NAMES):
    cls_idx = CLASS_IDX[cls]
    ax = axes[col]

    # PSD de latidos reales del test set
    real_idx  = np.where(y_test == cls)[0]
    n_samples = min(N_SPECTRAL, len(real_idx))
    chosen    = np.random.choice(real_idx, size=n_samples, replace=False)
    psds_real = []
    for i in chosen:
        f, p = welch(X_test[i], fs=FS, nperseg=64)
        psds_real.append(p)
    psd_real_mean = np.mean(psds_real, axis=0)
    psd_real_std  = np.std(psds_real,  axis=0)

    # PSD de latidos sintéticos
    x_syn = model.generate(c=cls_idx, n_samples=n_samples, device=DEVICE)
    X_syn = x_syn.squeeze(1).cpu().numpy()
    psds_syn = []
    for s in X_syn:
        f, p = welch(s, fs=FS, nperseg=64)
        psds_syn.append(p)
    psd_syn_mean = np.mean(psds_syn, axis=0)
    psd_syn_std  = np.std(psds_syn,  axis=0)

    # Plot
    ax.semilogy(f, psd_real_mean, color=class_colors[cls],
                linewidth=2, label='Real')
    ax.fill_between(f, psd_real_mean-psd_real_std,
                    psd_real_mean+psd_real_std,
                    alpha=0.2, color=class_colors[cls])
    ax.semilogy(f, psd_syn_mean, color=class_colors[cls],
                linewidth=1.5, linestyle='--', alpha=0.7, label='Sintético')
    ax.fill_between(f, psd_syn_mean-psd_syn_std,
                    psd_syn_mean+psd_syn_std,
                    alpha=0.1, color=class_colors[cls], hatch='//')

    ax.set_title(f'Clase {cls}\n{class_desc[cls]}', fontsize=9)
    ax.set_xlabel('Hz')
    if col == 0: ax.set_ylabel('PSD')
    ax.legend(fontsize=8)
    ax.set_xlim([0, FS/2])

plt.tight_layout()
plt.savefig('analisis_espectral.png', dpi=150, bbox_inches='tight')
plt.show()
print(' Análisis espectral completado. Los espectros solapados indican buena calidad generativa.')

# ─────────────────────────────────────────────────────────
#  t-SNE DEL ESPACIO LATENTE
#  Visualiza cómo se organizan las clases en el espacio z.
#  Si el CVAE aprendió bien:
#    - Los latidos reales de cada clase formarán clusters separados.
#    - Los z sintéticos sampleados de N(0,I) se mezclarán
#      con los reales de su misma clase.
# ─────────────────────────────────────────────────────────
N_TSNE = 300  # Latidos por clase (para que t-SNE sea manejable)

zs_real  = []
zs_syn   = []
labs_real = []
labs_syn  = []

model.eval()
with torch.no_grad():
    for cls in CLASS_NAMES:
        cls_idx = CLASS_IDX[cls]

        # Latidos reales → pasar por encoder → obtener mu como representación z
        real_idx = np.where(y_test == cls)[0]
        n = min(N_TSNE, len(real_idx))
        chosen = np.random.choice(real_idx, size=n, replace=False)

        x_r  = torch.tensor(X_test[chosen], dtype=torch.float32).unsqueeze(1).to(DEVICE)
        c_r  = F.one_hot(torch.tensor([cls_idx]*n), num_classes=5).float().to(DEVICE)
        mu_r, _ = model.encoder(x_r, c_r)
        zs_real.append(mu_r.cpu().numpy())
        labs_real.extend([cls] * n)

        # Latidos sintéticos → samplear z ~ N(0,I) directamente
        z_s = torch.randn(n, model.latent_dim).to(DEVICE)
        zs_syn.append(z_s.cpu().numpy())
        labs_syn.extend([cls] * n)

Z_real = np.concatenate(zs_real, axis=0)  # (N_total, latent_dim)
Z_syn  = np.concatenate(zs_syn,  axis=0)
Z_all  = np.concatenate([Z_real, Z_syn], axis=0)
origin = ['Real'] * len(Z_real) + ['Sintético'] * len(Z_syn)
labels_all = labs_real + labs_syn

# Reducción t-SNE a 2D
print(' Calculando t-SNE (puede tardar ~30s)...')
tsne  = TSNE(n_components=2, perplexity=40, n_iter=800,
             random_state=SEED, verbose=0)
Z_2d  = tsne.fit_transform(Z_all)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('t-SNE del Espacio Latente z – TCN-CVAE', fontsize=12, fontweight='bold')

# Por clase AAMI
for cls in CLASS_NAMES:
    mask = [l == cls for l in labels_all]
    ax1.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
               c=class_colors[cls], s=8, alpha=0.5, label=f'{cls} ({class_desc[cls]})')
ax1.set_title('Coloreado por clase AAMI')
ax1.legend(fontsize=8, markerscale=2)
ax1.set_xlabel('t-SNE dim 1'); ax1.set_ylabel('t-SNE dim 2')

# Real vs Sintético
for orig, col, mk in [('Real', '#2c3e50', 'o'), ('Sintético', '#e74c3c', '^')]:
    mask = [o == orig for o in origin]
    ax2.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
               c=col, s=8, alpha=0.4, marker=mk, label=orig)
ax2.set_title('Coloreado por origen (Real vs Sintético)')
ax2.legend(fontsize=9, markerscale=2)
ax2.set_xlabel('t-SNE dim 1'); ax2.set_ylabel('t-SNE dim 2')

plt.tight_layout()
plt.savefig('tsne_latente.png', dpi=150, bbox_inches='tight')
plt.show()
print(' t-SNE completado.')
print('   Si los clusters de clases están separados → el CVAE aprendió representaciones discriminativas.')
print('   Si Real y Sintético se mezclan → el decoder genera distribuciones realistas.')

# ─────────────────────────────────────────────────────────
#  EXPORTAR LATIDOS SINTÉTICOS PARA USO EN AUGMENTATION
#  Genera N latidos por clase y los guarda en .npy
# ─────────────────────────────────────────────────────────
N_EXPORT_PER_CLASS = 1000  # Latidos sintéticos por clase a exportar

syn_dir = './ecg_synthetic'
os.makedirs(syn_dir, exist_ok=True)

X_synthetic = []
y_synthetic = []

model.eval()
print(f' Generando {N_EXPORT_PER_CLASS} latidos sintéticos por clase...')
for cls in CLASS_NAMES:
    cls_idx = CLASS_IDX[cls]
    # Generar en batches de 256 para no saturar la VRAM
    batches  = []
    remaining = N_EXPORT_PER_CLASS
    while remaining > 0:
        n_batch = min(256, remaining)
        x_b = model.generate(c=cls_idx, n_samples=n_batch, device=DEVICE)
        batches.append(x_b.squeeze(1).cpu().numpy())
        remaining -= n_batch
    X_cls = np.concatenate(batches, axis=0)  # (N_EXPORT, 256)
    X_synthetic.append(X_cls)
    y_synthetic.extend([cls] * N_EXPORT_PER_CLASS)
    print(f'   Clase {cls} ({class_desc[cls]}): {len(X_cls)} latidos generados')

X_synthetic = np.concatenate(X_synthetic, axis=0)  # (5*N_EXPORT, 256)
y_synthetic = np.array(y_synthetic)

np.save(f'{syn_dir}/X_synthetic.npy', X_synthetic)
np.save(f'{syn_dir}/y_synthetic.npy', y_synthetic)

print(f'\n Dataset sintético guardado en {syn_dir}/')
print(f'   X_synthetic shape: {X_synthetic.shape}')
print(f'   y_synthetic shape: {y_synthetic.shape}')
print(f'   Distribución: {dict(Counter(y_synthetic))}')
print(f'   Memoria: {X_synthetic.nbytes / 1e6:.1f} MB')

# ─────────────────────────────────────────────────────────
#  RESUMEN FINAL DEL ENTRENAMIENTO
# ─────────────────────────────────────────────────────────
print('=' * 65)
print('          RESUMEN – TCN-CVAE ENTRENAMIENTO COMPLETO')
print('=' * 65)

print('\n  ARQUITECTURA')
print(f'   Señal de entrada  : (1, 256) – ECG normalizado [0,1]')
print(f'   Condición c       : one-hot (5 clases AAMI)')
print(f'   Dimensión latente : {model.latent_dim}')
print(f'   Canales ocultos   : 64')
print(f'   Dilataciones TCN  : [1, 2, 4, 8]')
print(f'   Parámetros        : {sum(p.numel() for p in model.parameters()):,}')

print('\n  ENTRENAMIENTO')
print(f'   Épocas            : {N_EPOCHS}')
print(f'   Mejor val_loss    : {best_val_loss:.5f}')
best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
print(f'   Mejor época       : {best_epoch}')
print(f'   β final           : {history["beta"][-1]:.2f}')
final_lr = history['lr'][-1]
print(f'   LR final          : {final_lr:.2e}')

print('\n  GENERACIÓN SINTÉTICA')
print(f'   Latidos exportados: {len(X_synthetic):,}')
print(f'   Por clase         : {N_EXPORT_PER_CLASS} por clase × 5 clases')
print(f'   Guardado en       : /content/ecg_synthetic/')

print('\n  ARCHIVOS GENERADOS')
for fname in ['beta_schedule.png', 'curvas_entrenamiento.png',
              'latidos_sinteticos.png', 'comparacion_real_sintetico.png',
              'analisis_espectral.png', 'tsne_latente.png']:
    exists = '' if os.path.exists(fname) else ''
    print(f'   {exists}  {fname}')

print('=' * 65)

"""---
##  Fin del Notebook de Entrenamiento

**Próximos pasos sugeridos:**
- Combinar `X_train_bal` con `X_synthetic` y evaluar si el clasificador downstream mejora.
- Experimentar con `latent_dim=64` o más bloques TCN si el MSE de reconstrucción no converge.
- Ajustar `beta_max` a valores más pequeños (0.1–0.5) si las señales generadas pierden detalle morfológico.
"""
