# -*- coding: utf-8 -*-
"""
TCN-cVAE + Clasificador TSTR – PhysioNet ECG-Arrhythmia (12 Derivaciones, 500 Hz)
===================================================================================
CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR:
  1. DILATIONS corregido a (1,2,4,8,16,32) → campo receptivo > 325 muestras
  2. NOISE_STD y LABEL_SMOOTH conectados al loop de entrenamiento
  3. BETA_MAX subido a 0.035 (rango 0.02-0.05) con warmup de 30 épocas
  4. latent_frechet_distance usa PCA a 10 componentes si N < 2*LATENT_DIM
  5. noise del banco latente subido a 1.0 para mayor coverage
  6. El dataset se divide en 3 grupos:
       - GEN_SPLIT  (60%) → entrenar el TCN-cVAE
       - CLF_REAL   (20%) → pool de datos reales del clasificador
       - CLF_HELD   (20%) → test final del clasificador (nunca visto)
  7. Clasificador TCN 1D independiente con 3 evaluaciones:
       a) Solo datos REALES   (CLF_REAL)
       b) Solo datos SINTÉTICOS generados con el cVAE
       c) REAL + SINTÉTICO combinados (aug.)
  8. Métricas del clasificador: Accuracy, Recall, F1-score (macro)
     por modo de entrenamiento + matriz de confusión
  9. Exportación del generador a ONNX (encoder + decoder por separado)
 10. Resumen final coherente (se eliminan variables huérfanas del resumen)
"""

# ══════════════════════════════════════════════════════════════════════════════
#  1. DEPENDENCIAS
# ══════════════════════════════════════════════════════════════════════════════
!pip install -q wfdb neurokit2 polars imbalanced-learn tqdm scikit-learn onnx onnxruntime
print("✅ Dependencias instaladas")

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import os, glob, warnings
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
import wfdb
import neurokit2 as nk
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = '#f8f9fa'
plt.rcParams['axes.grid']        = True
plt.rcParams['grid.alpha']       = 0.4
plt.rcParams['font.size']        = 11

print(f"✅ Librerías importadas | Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
#  2. DESCARGA DEL DATASET (PhysioNet 12-lead, 500 Hz)
# ══════════════════════════════════════════════════════════════════════════════
PHYSIONET_DIR = '/content/physionet.org/files/ecg-arrhythmia/1.0.0'

if not os.path.exists(PHYSIONET_DIR):
    print('Descargando dataset desde PhysioNet (puede tardar 30-60 min)...')
    os.makedirs(PHYSIONET_DIR, exist_ok=True)
    !wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/
    print('Descarga completada.')
else:
    print(f'Dataset ya descargado en: {PHYSIONET_DIR}')

hea_files    = glob.glob(os.path.join(PHYSIONET_DIR, 'WFDBRecords', '**', '*.hea'), recursive=True)
record_paths = [p.replace('.hea', '') for p in hea_files]
print(f'Total de registros encontrados: {len(record_paths):,}')

# ══════════════════════════════════════════════════════════════════════════════
#  3. MAPEO SNOMED CT Y EXTRACCIÓN DE ETIQUETAS
# ══════════════════════════════════════════════════════════════════════════════
SNOMED_MAP = {
    '270492004': 'I-AVB',  '164889003': 'AF',    '164890007': 'AFL',
    '426627000': 'Brady',  '713427006': 'RBBB',   '713426002': 'IRBBB',
    '39732003':  'LBBB',   '445118002': 'LQRSV',  '164947007': 'LQRS',
    '164917005': 'LSAD',   '251146004': 'LVQRS',  '698252002': 'NSIVCB',
    '10370003':  'PR',     '365413008': 'PAC',    '427172004': 'PVC',
    '164934002': 'QTIE',   '59931005':  'QWAVE',  '426177001': 'SB',
    '426783006': 'NSR',    '427393009': 'SA',     '427084000': 'ST',
    '429622005': 'STD',    '164931005': 'STE',    '251200008': 'TAb',
}

def extract_snomed_labels(hea_path):
    try:
        with open(hea_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#Dx:'):
                    codes = line.replace('#Dx:', '').strip().split(',')
                    return [c.strip() for c in codes]
    except:
        pass
    return []

N_META = min(500, len(record_paths))
print(f'Extrayendo etiquetas SNOMED CT de {N_META} registros...')
all_records_meta = []
for rp in tqdm(record_paths[:N_META]):
    hea   = rp + '.hea'
    codes = extract_snomed_labels(hea)
    prim  = codes[0] if codes else 'UNK'
    name  = SNOMED_MAP.get(prim, 'Other')
    all_records_meta.append({'record': rp, 'snomed': prim, 'label_name': name})

df_meta = pl.DataFrame(all_records_meta)
print(df_meta.head(5))

# ══════════════════════════════════════════════════════════════════════════════
#  4. FILTRADO Y DETECCIÓN DE PICOS R (PRUEBA DE CONCEPTO)
# ══════════════════════════════════════════════════════════════════════════════
FS         = 500
WIN_BEFORE = int(0.250 * FS)   # 125 muestras
WIN_AFTER  = int(0.400 * FS)   # 200 muestras
SEQ_LEN    = WIN_BEFORE + WIN_AFTER  # 325

rp = record_paths[0]
record   = wfdb.rdrecord(rp)
lead_idx = record.sig_name.index('II') if 'II' in record.sig_name else 1
ecg_raw  = record.p_signal[:, lead_idx]

ecg_filt = nk.ecg_clean(ecg_raw, sampling_rate=FS, method='neurokit')
_, rpeaks_dict = nk.ecg_peaks(ecg_filt, sampling_rate=FS, method='pantompkins1985')
peaks    = rpeaks_dict['ECG_R_Peaks']
windows  = np.array([ecg_filt[p-WIN_BEFORE:p+WIN_AFTER]
                     for p in peaks if p-WIN_BEFORE >= 0 and p+WIN_AFTER <= len(ecg_filt)])
print(f'Latidos extraidos en prueba: {windows.shape}')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
t = np.arange(len(ecg_raw)) / FS
seg = int(5 * FS)
axes[0].plot(t[:seg], ecg_raw[:seg], alpha=0.45, label='Cruda', color='#90A4AE')
axes[0].plot(t[:seg], ecg_filt[:seg], label='Filtrada', color='#1565C0', lw=1.2)
seg_peaks = [p for p in peaks if p < seg]
axes[0].scatter(np.array(seg_peaks)/FS, ecg_filt[seg_peaks], color='#E53935', marker='x', s=80, zorder=5)
axes[0].set_title('ECG Derivacion II (5 s)'); axes[0].legend()
t_beat = np.arange(SEQ_LEN) / FS * 1000
for i in range(min(20, len(windows))):
    axes[1].plot(t_beat, windows[i], alpha=0.3, lw=0.9, color='#1565C0')
axes[1].plot(t_beat, windows.mean(0), color='#E53935', lw=2.5, label='Media')
axes[1].set_title(f'Latidos – {SEQ_LEN} muestras (650 ms)'); axes[1].legend()
plt.tight_layout(); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  5. DISTRIBUCIÓN DE CLASES
# ══════════════════════════════════════════════════════════════════════════════
class_counts = (df_meta.group_by('label_name')
                .agg(pl.len().alias('count'))
                .sort('count', descending=True))
labels_plot  = class_counts['label_name'].to_list()
counts_plot  = class_counts['count'].to_list()
total        = sum(counts_plot)

print('=' * 55)
print('  DISTRIBUCIÓN – PhysioNet ECG-Arrhythmia')
print('=' * 55)
for l, c in zip(labels_plot, counts_plot):
    print(f'  {l:<10} {c:>5,}  ({100*c/total:.1f}%)')

palette = plt.cm.get_cmap('tab20', len(labels_plot))
colors  = [palette(i) for i in range(len(labels_plot))]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(labels_plot[::-1], counts_plot[::-1], color=colors[::-1])
axes[0].set_title('Distribución por Diagnóstico SNOMED CT'); axes[0].set_xlabel('Registros')
axes[1].pie(counts_plot, labels=labels_plot, colors=colors, autopct='%1.1f%%',
            wedgeprops=dict(width=0.55), startangle=90, textprops={'fontsize': 8})
axes[1].set_title('Proporción de clases')
plt.tight_layout(); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  6. PREPROCESAMIENTO MASIVO Y GUARDADO EN DISCO
# ══════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR      = 'processed_tensors'
BATCH_SIZE_REC  = 500
N_RECORDS_LIMIT = len(record_paths)  # ← reducir p.ej. a 2000 para pruebas rápidas
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_label_names = df_meta['label_name'].to_list()
le = LabelEncoder()
le.fit(all_label_names)
CLASS_NAMES = list(le.classes_)
N_CLASSES   = len(CLASS_NAMES)
print(f'Clases ({N_CLASSES}): {CLASS_NAMES}')

batch_idx, all_windows, all_labels, n_errors = 0, [], [], 0

for i, rp in enumerate(tqdm(record_paths[:N_RECORDS_LIMIT])):
    try:
        hea_path    = rp + '.hea'
        snomed_codes = extract_snomed_labels(hea_path)
        primary_code = snomed_codes[0] if snomed_codes else 'UNK'
        label_name   = SNOMED_MAP.get(primary_code, 'Other')
        if label_name not in CLASS_NAMES:
            label_name = CLASS_NAMES[0]
        label_int = le.transform([label_name])[0]

        rec     = wfdb.rdrecord(rp)
        li      = rec.sig_name.index('II') if 'II' in rec.sig_name else 0
        ecg_raw = rec.p_signal[:, li]
        ecg_f   = nk.ecg_clean(ecg_raw, sampling_rate=FS, method='neurokit')
        _, rpd  = nk.ecg_peaks(ecg_f, sampling_rate=FS, method='pantompkins1985')
        pks     = rpd['ECG_R_Peaks']

        for p in pks:
            if p - WIN_BEFORE >= 0 and p + WIN_AFTER <= len(ecg_f):
                w = ecg_f[p-WIN_BEFORE:p+WIN_AFTER]
                all_windows.append(w.astype(np.float32))
                all_labels.append(label_int)
    except Exception:
        n_errors += 1

    if (i + 1) % BATCH_SIZE_REC == 0 or (i + 1) == N_RECORDS_LIMIT:
        if all_windows:
            X_t = torch.tensor(np.array(all_windows), dtype=torch.float32)
            y_t = torch.tensor(np.array(all_labels),  dtype=torch.long)
            torch.save((X_t, y_t), os.path.join(OUTPUT_DIR, f'batch_{batch_idx}.pt'))
            batch_idx += 1; all_windows = []; all_labels = []

print(f'Pre-procesamiento: {batch_idx} lotes | {n_errors} errores')

# ══════════════════════════════════════════════════════════════════════════════
#  7. CARGA COMPLETA Y DIVISIÓN EN 3 GRUPOS
#     GEN_SPLIT  60% → entrenar el TCN-cVAE
#     CLF_REAL   20% → pool de datos REALES para entrenar el clasificador
#     CLF_HELD   20% → test HELD-OUT del clasificador (nunca visto)
# ══════════════════════════════════════════════════════════════════════════════

class PhysionetDiskDataset(Dataset):
    """Carga todos los lotes .pt del disco, normaliza por latido y añade canal."""
    def __init__(self, data_dir, le):
        super().__init__()
        files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        X_list, y_list = [], []
        for f in files:
            X, y = torch.load(f, weights_only=True)
            X_list.append(X); y_list.append(y)
        self.X = torch.cat(X_list, dim=0)
        self.y = torch.cat(y_list, dim=0)
        # Z-score por latido
        mu  = self.X.mean(dim=1, keepdim=True)
        std = self.X.std(dim=1, keepdim=True) + 1e-8
        self.X = ((self.X - mu) / std).unsqueeze(1)   # (N, 1, 325)
        self.le = le
        print(f'Dataset: {self.X.shape[0]:,} latidos | seq_len={self.X.shape[2]}')

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


full_dataset = PhysionetDiskDataset(OUTPUT_DIR, le)
n_total      = len(full_dataset)

# Índices aleatorios mezclados con semilla fija
rng_split = np.random.default_rng(SEED)
all_idx   = rng_split.permutation(n_total)

n_gen  = int(0.60 * n_total)
n_clf  = int(0.20 * n_total)
n_held = n_total - n_gen - n_clf

idx_gen  = all_idx[:n_gen].tolist()
idx_clf  = all_idx[n_gen:n_gen + n_clf].tolist()
idx_held = all_idx[n_gen + n_clf:].tolist()

# Sub-datasets
from torch.utils.data import Subset
gen_ds  = Subset(full_dataset, idx_gen)
clf_ds  = Subset(full_dataset, idx_clf)
held_ds = Subset(full_dataset, idx_held)

# ── DataLoaders del generador (gen_ds → 70/15/15) ────────────────────────────
n_g    = len(gen_ds)
n_tr   = int(0.70 * n_g); n_va = int(0.15 * n_g); n_te = n_g - n_tr - n_va
gen_train_ds, gen_val_ds, gen_test_ds = torch.utils.data.random_split(
    gen_ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(SEED))

train_labels_arr = full_dataset.y[gen_train_ds.dataset.indices[gen_train_ds.indices]
                                  if hasattr(gen_train_ds.dataset, 'indices')
                                  else gen_train_ds.indices].numpy()
# Forma segura de obtener etiquetas del subset anidado
_tr_idx = [idx_gen[i] for i in gen_train_ds.indices]
train_labels_arr = full_dataset.y[_tr_idx].numpy()

class_counts_arr = np.bincount(train_labels_arr, minlength=N_CLASSES)
class_weights    = 1.0 / (class_counts_arr + 1e-6)
sample_weights   = class_weights[train_labels_arr]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(gen_train_ds), replacement=True)

BATCH_SIZE   = 256
train_loader = DataLoader(gen_train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(gen_val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(gen_test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f'GEN  → Train:{n_tr:,} | Val:{n_va:,} | Test:{n_te:,}')
print(f'CLF_REAL  → {len(clf_ds):,} latidos')
print(f'CLF_HELD  → {len(held_ds):,} latidos')

CLASS_IDX  = {name: i for i, name in enumerate(CLASS_NAMES)}
class_desc = {name: name for name in CLASS_NAMES}   # desc = propio nombre en PhysioNet
colors_map = {name: plt.cm.get_cmap('tab10', N_CLASSES)(i)
              for i, name in enumerate(CLASS_NAMES)}

# ══════════════════════════════════════════════════════════════════════════════
#  8. ARQUITECTURA TCN-cVAE  (DILATIONS CORREGIDO → campo receptivo ≥ SEQ_LEN)
# ══════════════════════════════════════════════════════════════════════════════
INPUT_LENGTH = SEQ_LEN        # 325
NUM_CLASSES  = N_CLASSES

LATENT_DIM   = 32
HIDDEN_CH    = 64
# FIX 1 ── Dilataciones ampliadas para campo receptivo ≥ 325 muestras
# campo = sum(2*(kernel-1)*d for d in dilations) = 2*4*(1+2+4+8+16+32) = 504 > 325
DILATIONS    = (1, 2, 4, 8, 16, 32)
KERNEL_SIZE  = 5
DROPOUT_RATE = 0.1

print(f'INPUT_LENGTH={INPUT_LENGTH} | NUM_CLASSES={NUM_CLASSES}')
print(f'DILATIONS={DILATIONS}  → campo receptivo ≥ {2*(KERNEL_SIZE-1)*sum(DILATIONS)} muestras')


class TCNResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.norm1    = nn.BatchNorm1d(out_ch)
        self.norm2    = nn.BatchNorm1d(out_ch)
        self.dropout  = nn.Dropout(dropout)
        self.skip     = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self._pad     = pad

    def _causal_trim(self, x, pad):
        return x[:, :, :-pad] if pad > 0 else x

    def forward(self, x):
        out = self._causal_trim(self.conv1(x), self._pad)
        out = F.relu(self.norm1(out))
        out = self.dropout(out)
        out = self._causal_trim(self.conv2(out), self._pad)
        out = F.relu(self.norm2(out))
        out = self.dropout(out)
        return out + self.skip(x)


class TCNStack(nn.Module):
    def __init__(self, in_ch, hidden_ch, dilations=(1,2,4,8,16,32), dropout=0.1):
        super().__init__()
        layers, ch = [], in_ch
        for d in dilations:
            layers.append(TCNResBlock(ch, hidden_ch, dilation=d, dropout=dropout))
            ch = hidden_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class TCNEncoder(nn.Module):
    def __init__(self, input_len, n_classes, latent_dim,
                 hidden_ch=64, dilations=(1,2,4,8,16,32), dropout=0.1):
        super().__init__()
        self.cond_proj = nn.Linear(n_classes, hidden_ch)
        self.tcn       = TCNStack(1, hidden_ch, dilations, dropout)
        self.pool      = nn.AdaptiveAvgPool1d(1)
        self.fc_mu     = nn.Linear(hidden_ch * 2, latent_dim)
        self.fc_lv     = nn.Linear(hidden_ch * 2, latent_dim)

    def forward(self, x, c):
        h     = self.pool(self.tcn(x)).squeeze(-1)
        c_emb = F.relu(self.cond_proj(c))
        h_cat = torch.cat([h, c_emb], dim=-1)
        return self.fc_mu(h_cat), self.fc_lv(h_cat)


class TCNDecoder(nn.Module):
    def __init__(self, input_len, n_classes, latent_dim,
                 hidden_ch=64, dilations=(1,2,4,8,16,32), dropout=0.1):
        super().__init__()
        self.input_len = input_len
        self.hidden_ch = hidden_ch
        self.proj = nn.Linear(latent_dim + n_classes, hidden_ch * input_len)
        self.tcn  = TCNStack(hidden_ch, hidden_ch, dilations, dropout)
        self.out  = nn.Conv1d(hidden_ch, 1, kernel_size=1)

    def forward(self, z, c):
        zc  = torch.cat([z, c], dim=-1)
        h   = self.proj(zc).view(-1, self.hidden_ch, self.input_len)
        return self.out(self.tcn(h))


class TCNCVAE(nn.Module):
    def __init__(self, input_len, n_classes, latent_dim=32,
                 hidden_ch=64, dilations=(1,2,4,8,16,32), dropout=0.1):
        super().__init__()
        self.input_len  = input_len
        self.n_classes  = n_classes
        self.latent_dim = latent_dim
        self.encoder = TCNEncoder(input_len, n_classes, latent_dim, hidden_ch, dilations, dropout)
        self.decoder = TCNDecoder(input_len, n_classes, latent_dim, hidden_ch, dilations, dropout)

    def reparametrize(self, mu, log_var):
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)

    def forward(self, x, c):
        mu, lv   = self.encoder(x, c)
        x_recon  = self.decoder(self.reparametrize(mu, lv), c)
        return x_recon, mu, lv

    @torch.no_grad()
    def generate(self, cls_idx, n_samples, device):
        z = torch.randn(n_samples, self.latent_dim, device=device)
        c = F.one_hot(torch.tensor([cls_idx]*n_samples), self.n_classes).float().to(device)
        return self.decoder(z, c)


model = TCNCVAE(
    input_len  = INPUT_LENGTH,
    n_classes  = NUM_CLASSES,
    latent_dim = LATENT_DIM,
    hidden_ch  = HIDDEN_CH,
    dilations  = DILATIONS,
    dropout    = DROPOUT_RATE,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'✅ TCN-cVAE instanciado | {n_params:,} parámetros')

# Prueba de dimensiones
model.eval()
with torch.no_grad():
    _xb = torch.randn(4, 1, INPUT_LENGTH).to(DEVICE)
    _cb = F.one_hot(torch.zeros(4, dtype=torch.long), NUM_CLASSES).float().to(DEVICE)
    _r, _mu, _lv = model(_xb, _cb)
assert _r.shape == (4, 1, INPUT_LENGTH), f"Shape error: {_r.shape}"
print(f'   Prueba dimensiones OK: {_r.shape}')
model.train()

# ══════════════════════════════════════════════════════════════════════════════
#  9. ENTRENAMIENTO  (NOISE + LABEL_SMOOTH + BETA_MAX CORREGIDOS)
# ══════════════════════════════════════════════════════════════════════════════

# FIX 2 ── Beta aumentado para mejor geometría latente
BETA_MAX      = 0.035       # anterior: 0.008
WARMUP_EPOCHS = 30          # anterior: 20
MAX_EPOCHS    = 140
LR            = 3e-4
WEIGHT_DECAY  = 1e-3
GRAD_CLIP     = 1.0
PATIENCE      = 20

# FIX 2 ── Regularizadores ahora activos
NOISE_STD    = 0.03         # ruido gaussiano sobre la señal de entrada
LABEL_SMOOTH = 0.05         # label smoothing sobre el one-hot condicional
WARMUP_FRAC  = WARMUP_EPOCHS / MAX_EPOCHS

CKPT_DIR = '/content/tcncvae_checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CKPT_DIR, 'best_model.pt')


def beta_schedule(epoch, max_epochs, beta_max=0.035, warmup_frac=0.3):
    warmup_epochs = int(max_epochs * warmup_frac)
    return beta_max * min(epoch / (warmup_epochs + 1e-8), 1.0)


def vae_loss(x, x_recon, mu, log_var, beta=1.0):
    recon = F.mse_loss(x_recon, x, reduction='mean')
    kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + beta * kl, recon, kl


def smooth_onehot(c_int, n_classes, eps=0.05):
    """Label smoothing sobre one-hot: c*(1-eps) + eps/n_classes."""
    c_oh = F.one_hot(c_int, num_classes=n_classes).float()
    return c_oh * (1 - eps) + eps / n_classes


optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

history = {k: [] for k in ['train_loss','val_loss','train_recon','val_recon',
                             'train_kl','val_kl','beta','lr','generalization_gap']}
best_val_recon  = float('inf')
patience_counter = 0

print(f'🚀 Entrenamiento | max_épocas={MAX_EPOCHS} | BETA_MAX={BETA_MAX} | patience={PATIENCE}')

for epoch in range(1, MAX_EPOCHS + 1):
    beta = beta_schedule(epoch - 1, MAX_EPOCHS, BETA_MAX, WARMUP_FRAC)

    # ── TRAIN ────────────────────────────────────────────────────────────────
    model.train()
    tr_losses, tr_recons, tr_kls = [], [], []
    for x, c_int in train_loader:
        x, c_int = x.to(DEVICE), c_int.to(DEVICE)

        # FIX 2a ── Ruido gaussiano sobre la señal de entrada
        x_noisy = x + NOISE_STD * torch.randn_like(x)

        # FIX 2b ── Label smoothing en la condición condicional
        c_smooth = smooth_onehot(c_int, NUM_CLASSES, LABEL_SMOOTH)

        optimizer.zero_grad()
        x_recon, mu, lv = model(x_noisy, c_smooth)
        loss, recon, kl = vae_loss(x, x_recon, mu, lv, beta)   # target = x limpio
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        tr_losses.append(loss.item()); tr_recons.append(recon.item()); tr_kls.append(kl.item())

    # ── VALIDACIÓN ───────────────────────────────────────────────────────────
    model.eval()
    vl_losses, vl_recons, vl_kls = [], [], []
    with torch.no_grad():
        for x, c_int in val_loader:
            x, c_int = x.to(DEVICE), c_int.to(DEVICE)
            c_oh = F.one_hot(c_int, num_classes=NUM_CLASSES).float()
            x_recon, mu, lv = model(x, c_oh)
            loss, recon, kl = vae_loss(x, x_recon, mu, lv, beta)
            vl_losses.append(loss.item()); vl_recons.append(recon.item()); vl_kls.append(kl.item())

    tr_r = np.mean(tr_recons); vl_r = np.mean(vl_recons)
    gap  = vl_r - tr_r

    history['train_loss'].append(np.mean(tr_losses))
    history['val_loss'].append(np.mean(vl_losses))
    history['train_recon'].append(tr_r); history['val_recon'].append(vl_r)
    history['train_kl'].append(np.mean(tr_kls)); history['val_kl'].append(np.mean(vl_kls))
    history['beta'].append(beta); history['lr'].append(optimizer.param_groups[0]['lr'])
    history['generalization_gap'].append(gap)
    scheduler.step()

    if vl_r < best_val_recon:
        best_val_recon   = vl_r
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        marker = ' ← MEJOR'
    else:
        patience_counter += 1
        marker = f' (p={patience_counter}/{PATIENCE})'

    if epoch % 10 == 0 or epoch == 1 or patience_counter == 0:
        print(f'Ep {epoch:4d} β={beta:.4f}  tr_recon={tr_r:.5f}  vl_recon={vl_r:.5f}  '
              f'gap={gap:+.5f}{marker}')

    if patience_counter >= PATIENCE:
        print(f'\n⏹ Early stopping en época {epoch}'); break

if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    print(f'\n✅ Mejor modelo cargado: val_recon={best_val_recon:.5f}')

# ══════════════════════════════════════════════════════════════════════════════
#  10. CURVAS DE APRENDIZAJE
# ══════════════════════════════════════════════════════════════════════════════
epochs_ran = list(range(1, len(history['val_loss']) + 1))
best_ep    = int(np.argmin(history['val_recon'])) + 1
gap_fin    = history['generalization_gap'][-1]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Curvas de Aprendizaje TCN-cVAE – PhysioNet', fontsize=13, fontweight='bold')
axes[0].plot(epochs_ran, history['train_loss'], label='Train', color='#2196F3')
axes[0].plot(epochs_ran, history['val_loss'],   label='Val',   color='#F44336')
axes[0].axvline(best_ep, color='gray', ls='--', alpha=0.6); axes[0].set_title('Loss Total')
axes[0].legend()
axes[1].plot(epochs_ran, history['train_recon'], label='Train', color='#4CAF50')
axes[1].plot(epochs_ran, history['val_recon'],   label='Val',   color='#FF9800')
axes[1].axvline(best_ep, color='gray', ls='--', alpha=0.6,
                label=f'Mejor (ep.{best_ep})')
axes[1].set_title('Reconstrucción MSE'); axes[1].legend()
axes[2].plot(epochs_ran, history['train_kl'], label='KL Train', color='#9C27B0')
axes[2].plot(epochs_ran, history['val_kl'],   label='KL Val',   color='#00BCD4')
axes[2].set_title('KL Divergence'); axes[2].legend()
plt.tight_layout()
plt.savefig('curvas_aprendizaje_physionet.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Gap final: {gap_fin:+.5f} → {"OK" if gap_fin < 0.01 else "Overfitting leve"}')

# ══════════════════════════════════════════════════════════════════════════════
#  11. MÉTRICAS DE RECONSTRUCCIÓN POR CLASE
# ══════════════════════════════════════════════════════════════════════════════

def calc_recon_metrics(loader, model, device, class_names, n_classes):
    model.eval()
    pc = {c: {'mse': [], 'mae': [], 'psnr': [], 'r2': []} for c in class_names}
    with torch.no_grad():
        for x, labels in loader:
            x    = x.to(device)
            c_oh = F.one_hot(labels.to(device), num_classes=n_classes).float()
            x_r, _, _ = model(x, c_oh)
            for i in range(len(labels)):
                cn  = class_names[labels[i].item()]
                xo  = x[i, 0].cpu().numpy()
                xrr = x_r[i, 0].cpu().numpy()
                mse = float(np.mean((xo - xrr)**2))
                mae = float(np.mean(np.abs(xo - xrr)))
                rng = xo.max() - xo.min()
                psnr = 20 * np.log10(rng / (np.sqrt(mse) + 1e-8)) if mse > 0 else 100.0
                ss_res = np.sum((xo - xrr)**2); ss_tot = np.sum((xo - xo.mean())**2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                pc[cn]['mse'].append(mse); pc[cn]['mae'].append(mae)
                pc[cn]['psnr'].append(psnr); pc[cn]['r2'].append(r2)
    return {c: {k: np.mean(v) for k, v in m.items()} for c, m in pc.items()}


print('⏳ Métricas de reconstrucción...')
val_metrics  = calc_recon_metrics(val_loader,  model, DEVICE, CLASS_NAMES, NUM_CLASSES)
test_metrics = calc_recon_metrics(test_loader, model, DEVICE, CLASS_NAMES, NUM_CLASSES)

for split_name, metrics in [('VALIDACIÓN', val_metrics), ('TEST', test_metrics)]:
    print(f'\n📊 RECONSTRUCCIÓN – {split_name}')
    print(f"   {'Clase':<10} {'MSE':>8} {'MAE':>8} {'PSNR':>10} {'R²':>7}")
    for c in CLASS_NAMES:
        m = metrics[c]
        print(f"   {c:<10} {m['mse']:>8.5f} {m['mae']:>8.5f} {m['psnr']:>10.2f} {m['r2']:>7.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  12. BANCO LATENTE Y GENERACIÓN SINTÉTICA
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_latent_bank(model, loader, device, n_classes):
    model.eval()
    bank = {i: {'mu': [], 'log_var': []} for i in range(n_classes)}
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        c  = F.one_hot(labels, num_classes=n_classes).float()
        mu, lv = model.encoder(x, c)
        for ci in range(n_classes):
            mask = (labels == ci)
            if mask.any():
                bank[ci]['mu'].append(mu[mask].detach().cpu())
                bank[ci]['log_var'].append(lv[mask].detach().cpu())
    for ci in range(n_classes):
        if bank[ci]['mu']:
            bank[ci]['mu']      = torch.cat(bank[ci]['mu'],      dim=0)
            bank[ci]['log_var'] = torch.cat(bank[ci]['log_var'], dim=0)
        else:
            bank[ci]['mu']      = torch.zeros(1, LATENT_DIM)
            bank[ci]['log_var'] = torch.zeros(1, LATENT_DIM)
    return bank


# FIX 3 ── noise subido a 1.0 para mayor coverage
def sample_z_from_bank(bank, cls_idx, n, device, noise=1.0):
    mu  = bank[cls_idx]['mu']
    lv  = bank[cls_idx]['log_var']
    idx = torch.randint(0, len(mu), (n,))
    mu_s  = mu[idx].to(device)
    std_s = torch.exp(0.5 * lv[idx].to(device))
    return mu_s + noise * std_s * torch.randn_like(std_s)


@torch.no_grad()
def generate_from_bank(model, bank, cls_idx, n, device, noise=1.0):
    model.eval()
    z = sample_z_from_bank(bank, cls_idx, n, device, noise)
    c = F.one_hot(torch.tensor([cls_idx]*n), num_classes=NUM_CLASSES).float().to(device)
    return model.decoder(z, c)


@torch.no_grad()
def generate_class_samples(model, bank, cls_idx, n_samples, device, batch_size=256):
    generated = []
    remaining = n_samples
    while remaining > 0:
        nb = min(batch_size, remaining)
        x_syn = generate_from_bank(model, bank, cls_idx, nb, device)
        generated.append(x_syn.squeeze(1).cpu().numpy())
        remaining -= nb
    return np.concatenate(generated, axis=0).astype(np.float32)


latent_bank = build_latent_bank(model, train_loader, DEVICE, NUM_CLASSES)
print('✅ Banco latente construido:')
for cls in CLASS_NAMES:
    ci = CLASS_IDX[cls]
    print(f'   {cls}: {len(latent_bank[ci]["mu"]):,} vectores z')

# Visualización de señales sintéticas
N_GEN     = 6
t_idx_plt = np.arange(INPUT_LENGTH)
model.eval()
fig, axes = plt.subplots(N_CLASSES, N_GEN, figsize=(15, 2.4 * N_CLASSES))
if N_CLASSES == 1: axes = axes.reshape(1, -1)
fig.suptitle('Señales Sintéticas – TCN-cVAE / PhysioNet', fontsize=12, fontweight='bold')
for row, (cls, cls_idx) in enumerate(CLASS_IDX.items()):
    x_syn = generate_from_bank(model, latent_bank, cls_idx, N_GEN, DEVICE).squeeze(1).cpu().numpy()
    for col in range(N_GEN):
        ax = axes[row, col]
        ax.plot(t_idx_plt, x_syn[col], color=colors_map[cls], linewidth=1.2)
        ax.tick_params(labelsize=7)
        if col == 0: ax.set_ylabel(cls, fontsize=9, fontweight='bold')
        if row == 0: ax.set_title(f'Sintético {col+1}', fontsize=9)
plt.tight_layout()
plt.savefig('senales_sinteticas_physionet.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  13. EVALUACIÓN CUANTITATIVA DEL GENERADOR  (con PCA en Fréchet si N pequeño)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_signals(model, signals, cls_idx, n_classes, device, batch_size=256):
    model.eval()
    out = []
    for start in range(0, len(signals), batch_size):
        batch = signals[start:start+batch_size]
        x     = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)
        lbl   = torch.full((len(batch),), cls_idx, dtype=torch.long, device=device)
        c     = F.one_hot(lbl, num_classes=n_classes).float()
        mu, _ = model.encoder(x, c)
        out.append(mu.cpu().numpy())
    return np.concatenate(out, axis=0)


def mmd_rbf_squared(real, synthetic):
    n, m = len(real), len(synthetic)
    if n < 2 or m < 2: return np.nan
    rng_e = np.random.default_rng(SEED)
    combined = np.concatenate([real, synthetic], axis=0)
    idx      = rng_e.choice(len(combined), min(400, len(combined)), replace=False)
    sample   = combined[idx]
    dist_sq  = cdist(sample, sample, metric='sqeuclidean')
    pos      = dist_sq[dist_sq > 0]
    if len(pos) == 0: return 0.0
    gamma = 1.0 / (2.0 * np.median(pos) + 1e-12)
    k_xx = np.exp(-gamma * cdist(real, real, metric='sqeuclidean'))
    k_yy = np.exp(-gamma * cdist(synthetic, synthetic, metric='sqeuclidean'))
    k_xy = np.exp(-gamma * cdist(real, synthetic, metric='sqeuclidean'))
    np.fill_diagonal(k_xx, 0); np.fill_diagonal(k_yy, 0)
    return float(max(k_xx.sum()/(n*(n-1)) + k_yy.sum()/(m*(m-1)) - 2*k_xy.mean(), 0.0))


def sliced_wasserstein_distance(real, synthetic, n_proj=64):
    rng_e   = np.random.default_rng(SEED)
    dirs    = rng_e.normal(size=(real.shape[1], n_proj))
    dirs   /= np.linalg.norm(dirs, axis=0, keepdims=True) + 1e-12
    r_proj  = real @ dirs
    s_proj  = synthetic @ dirs
    return float(np.mean([wasserstein_distance(r_proj[:,i], s_proj[:,i]) for i in range(n_proj)]))


# FIX 4 ── PCA antes de Fréchet si N < 2*LATENT_DIM
def latent_frechet_distance(real_lat, syn_lat, pca_dim=10):
    if len(real_lat) < 2 or len(syn_lat) < 2: return np.nan
    min_n = min(len(real_lat), len(syn_lat))
    if min_n < 2 * LATENT_DIM:
        pca = PCA(n_components=min(pca_dim, min_n - 1))
        combined = np.concatenate([real_lat, syn_lat], axis=0)
        pca.fit(combined)
        real_lat = pca.transform(real_lat)
        syn_lat  = pca.transform(syn_lat)
    mu_r, mu_s   = real_lat.mean(0), syn_lat.mean(0)
    cov_r = np.atleast_2d(np.cov(real_lat, rowvar=False)) + 1e-6 * np.eye(real_lat.shape[1])
    cov_s = np.atleast_2d(np.cov(syn_lat,  rowvar=False)) + 1e-6 * np.eye(syn_lat.shape[1])
    sqrt_cov = sqrtm(cov_r @ cov_s)
    if np.iscomplexobj(sqrt_cov): sqrt_cov = sqrt_cov.real
    diff = mu_r - mu_s
    return float(max(diff @ diff + np.trace(cov_r + cov_s - 2*sqrt_cov), 0.0))


def mean_pairwise_distance(signals):
    return float(np.mean(pdist(signals, metric='euclidean'))) if len(signals) >= 2 else np.nan


def coverage_and_memorization(real, synthetic):
    if len(real) < 2 or len(synthetic) < 1: return np.nan, np.nan
    rr = cdist(real, real, metric='euclidean'); np.fill_diagonal(rr, np.inf)
    ref_dist  = np.median(np.min(rr, axis=1))
    rs = cdist(real, synthetic, metric='euclidean')
    cov  = float(np.mean(np.min(rs, axis=1) <= ref_dist))
    mem  = float(np.mean(np.min(rs, axis=0) <= 0.05 * ref_dist))
    return cov, mem


GENERATOR_EVAL_MAX = 300
N_PROJECTIONS      = 64
GEN_BATCH_SIZE     = 256

# Extraer datos de test del generador
X_test_eval = full_dataset.X[[idx_gen[i] for i in gen_test_ds.indices]].squeeze(1).cpu().numpy()
y_test_eval = full_dataset.y[[idx_gen[i] for i in gen_test_ds.indices]].cpu().numpy()

generator_results = []
generator_samples = {}
model.eval()
print('⏳ Evaluando generador...')

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    real_class  = X_test_eval[y_test_eval == cls_idx]
    n_eval      = min(GENERATOR_EVAL_MAX, len(real_class))
    if n_eval < 2: continue

    rng_ev   = np.random.default_rng(SEED)
    real_sel = real_class[rng_ev.choice(len(real_class), n_eval, replace=False)].astype(np.float32)
    syn_sel  = generate_class_samples(model, latent_bank, cls_idx, n_eval, DEVICE, GEN_BATCH_SIZE)

    generator_samples[cls_name] = {'real': real_sel, 'synthetic': syn_sel}

    rl = encode_signals(model, real_sel, cls_idx, NUM_CLASSES, DEVICE)
    sl = encode_signals(model, syn_sel,  cls_idx, NUM_CLASSES, DEVICE)

    mmd  = mmd_rbf_squared(real_sel, syn_sel)
    swd  = sliced_wasserstein_distance(real_sel, syn_sel, N_PROJECTIONS)
    fre  = latent_frechet_distance(rl, sl)
    d_r  = mean_pairwise_distance(real_sel)
    d_s  = mean_pairwise_distance(syn_sel)
    cov, mem = coverage_and_memorization(real_sel, syn_sel)

    generator_results.append({
        'Clase': cls_name, 'N': n_eval,
        'MMD² RBF': mmd, 'SWD': swd, 'Fréchet latente': fre,
        'MAE media': float(np.mean(np.abs(real_sel.mean(0)-syn_sel.mean(0)))),
        'MAE std':   float(np.mean(np.abs(real_sel.std(0) -syn_sel.std(0)))),
        'Ratio diversidad': d_s/(d_r+1e-12), 'Coverage': cov, 'Memorization': mem,
    })
    print(f'   {cls_name}: MMD²={mmd:.5f} SWD={swd:.5f} Fréchet={fre:.4f} Coverage={cov:.3f}')

gen_df = pd.DataFrame(generator_results)
num_cols = ['MMD² RBF','SWD','Fréchet latente','MAE media','Ratio diversidad','Coverage','Memorization']
print('\n📊 MÉTRICAS GENERADOR POR CLASE')
print(gen_df[['Clase','N']+num_cols].round(5).to_string(index=False))
print('\nMACRO:', gen_df[num_cols].mean().round(5).to_dict())
gen_df.to_csv('metricas_generador_physionet.csv', index=False)

# Visualización métricas generador
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('Evaluación cuantitativa del generador TCN-cVAE – PhysioNet', fontsize=13, fontweight='bold')
for ax, (col, title) in zip(axes.flatten(), [
    ('MMD² RBF',       'MMD² RBF (↓)'),
    ('SWD',            'Sliced Wasserstein (↓)'),
    ('Fréchet latente','Fréchet latente (↓)'),
    ('Ratio diversidad','Ratio diversidad (ideal≈1)'),
    ('Coverage',        'Coverage (↑)'),
    ('Memorization',    'Memorización (↓)'),
]):
    ax.bar(gen_df['Clase'], gen_df[col])
    ax.set_title(title); ax.set_xlabel('Clase')
    if col == 'Ratio diversidad':
        ax.axhline(1.0, ls='--', lw=1.2, label='ideal'); ax.legend()
plt.tight_layout()
plt.savefig('metricas_generador_physionet.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  14. t-SNE DEL ESPACIO LATENTE
# ══════════════════════════════════════════════════════════════════════════════
X_test_tsne = full_dataset.X[[idx_gen[i] for i in gen_test_ds.indices]].cpu().numpy()
y_test_tsne = full_dataset.y[[idx_gen[i] for i in gen_test_ds.indices]].cpu().numpy()

N_TSNE = min(200, len(y_test_tsne) // max(N_CLASSES, 1))
zs_real, zs_syn, labs_real, labs_syn = [], [], [], []

model.eval()
with torch.no_grad():
    for cls in CLASS_NAMES:
        ci   = CLASS_IDX[cls]
        ridx = np.where(y_test_tsne == ci)[0]
        n    = min(N_TSNE, len(ridx))
        if n == 0: continue
        chosen = np.random.choice(ridx, n, replace=False)
        x_r = torch.tensor(X_test_tsne[chosen], dtype=torch.float32).to(DEVICE)
        c_r = F.one_hot(torch.tensor([ci]*n), num_classes=NUM_CLASSES).float().to(DEVICE)
        mu_r, _ = model.encoder(x_r, c_r)
        zs_real.append(mu_r.cpu().numpy()); labs_real.extend([cls]*n)
        zs_syn.append(torch.randn(n, LATENT_DIM).numpy()); labs_syn.extend([cls]*n)

Z_all    = np.concatenate(zs_real + zs_syn, axis=0)
origin   = ['Real']*len(np.concatenate(zs_real,axis=0)) + ['Sintético']*len(np.concatenate(zs_syn,axis=0))
labs_all = labs_real + labs_syn

print(f'⏳ t-SNE sobre {len(Z_all)} puntos...')
Z_2d = TSNE(n_components=2, perplexity=min(40, len(Z_all)//5),
            n_iter=800, random_state=SEED).fit_transform(Z_all)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('t-SNE Espacio Latente – TCN-cVAE / PhysioNet', fontsize=12, fontweight='bold')
for cls in CLASS_NAMES:
    mask = np.array([l == cls for l in labs_all])
    ax1.scatter(Z_2d[mask,0], Z_2d[mask,1], c=[colors_map[cls]], s=8, alpha=0.6, label=cls)
ax1.set_title('Por clase'); ax1.legend(fontsize=7, markerscale=2)
for orig, col, mk in [('Real','#2c3e50','o'),('Sintético','#e74c3c','^')]:
    mask = np.array([o == orig for o in origin])
    ax2.scatter(Z_2d[mask,0], Z_2d[mask,1], c=col, s=8, alpha=0.5, marker=mk, label=orig)
ax2.set_title('Real vs Sintético'); ax2.legend(fontsize=9, markerscale=2)
plt.tight_layout()
plt.savefig('tsne_latente_physionet.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  15. CLASIFICADOR TCN 1D INDEPENDIENTE
# ══════════════════════════════════════════════════════════════════════════════
# El clasificador es una TCN 1D con cabezal softmax. No comparte pesos con el cVAE.
# Se evalúa en 3 modos:
#   A) Entrenado con datos REALES (clf_ds → CLF_REAL)
#   B) Entrenado con datos SINTÉTICOS (generados con el cVAE)
#   C) Entrenado con REAL + SINTÉTICO
# En los 3 casos el test se hace con held_ds (CLF_HELD → 20% no visto por nadie)
# ─────────────────────────────────────────────────────────────────────────────

class TCNClassifier(nn.Module):
    """TCN 1D con cabezal softmax para clasificación de arritmias."""
    def __init__(self, input_len, n_classes, hidden_ch=64,
                 dilations=(1,2,4,8,16,32), dropout=0.1):
        super().__init__()
        self.tcn  = TCNStack(1, hidden_ch, dilations, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ch, n_classes),
        )

    def forward(self, x):
        h = self.pool(self.tcn(x)).squeeze(-1)
        return self.head(h)


def train_classifier(train_loader_clf, val_loader_clf, n_classes,
                     max_epochs=60, lr=1e-3, patience=10, tag=''):
    """Entrena el TCNClassifier y devuelve el mejor modelo."""
    clf = TCNClassifier(INPUT_LENGTH, n_classes, hidden_ch=64,
                        dilations=DILATIONS, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None
    pat          = 0

    for epoch in range(1, max_epochs+1):
        clf.train()
        for x, y in train_loader_clf:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(clf(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()
        sch.step()

        clf.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in val_loader_clf:
                pred = clf(x.to(DEVICE)).argmax(1).cpu().numpy()
                all_pred.extend(pred); all_true.extend(y.numpy())
        acc = accuracy_score(all_true, all_pred)

        if acc > best_val_acc:
            best_val_acc = acc; pat = 0
            best_state   = {k: v.clone() for k, v in clf.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                print(f'  [{tag}] Early stop ep.{epoch} | best_val_acc={best_val_acc:.4f}')
                break

        if epoch % 10 == 0:
            print(f'  [{tag}] Ep {epoch:3d} | val_acc={acc:.4f}')

    clf.load_state_dict(best_state)
    return clf


def evaluate_classifier(clf, loader, class_names):
    """Devuelve acc, recall_macro, f1_macro y el reporte completo."""
    clf.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = clf(x.to(DEVICE)).argmax(1).cpu().numpy()
            all_pred.extend(pred); all_true.extend(y.numpy())
    acc    = accuracy_score(all_true, all_pred)
    rec    = recall_score(all_true, all_pred, average='macro', zero_division=0)
    f1     = f1_score(all_true, all_pred, average='macro', zero_division=0)
    report = classification_report(all_true, all_pred, target_names=class_names, zero_division=0)
    cm     = confusion_matrix(all_true, all_pred)
    return {'accuracy': acc, 'recall_macro': rec, 'f1_macro': f1,
            'report': report, 'confusion_matrix': cm,
            'y_true': all_true, 'y_pred': all_pred}


# ── Preparar datos CLF_REAL y CLF_HELD ─────────────────────────────────────
X_clf_real = full_dataset.X[idx_clf].squeeze(1).cpu().numpy()
y_clf_real = full_dataset.y[idx_clf].cpu().numpy()
X_clf_held = full_dataset.X[idx_held].squeeze(1).cpu().numpy()
y_clf_held = full_dataset.y[idx_held].cpu().numpy()

# ── Generar datos SINTÉTICOS por clase para igualar el pool real ─────────────
print('\n⏳ Generando datos sintéticos para el clasificador...')
X_clf_syn_list, y_clf_syn_list = [], []
for ci, cls in enumerate(CLASS_NAMES):
    n_real_cls = int((y_clf_real == ci).sum())
    if n_real_cls == 0: continue
    x_syn = generate_class_samples(model, latent_bank, ci, n_real_cls, DEVICE)
    X_clf_syn_list.append(x_syn)
    y_clf_syn_list.append(np.full(n_real_cls, ci, dtype=np.int64))

X_clf_syn = np.concatenate(X_clf_syn_list, axis=0)
y_clf_syn = np.concatenate(y_clf_syn_list, axis=0)
print(f'   Datos sintéticos: {X_clf_syn.shape[0]:,} latidos')

# ── Datos aumentados (REAL + SINTÉTICO) ──────────────────────────────────────
X_clf_aug = np.concatenate([X_clf_real, X_clf_syn], axis=0)
y_clf_aug = np.concatenate([y_clf_real, y_clf_syn], axis=0)

CLF_BATCH = 256
CLF_EPOCHS = 60
CLF_LR     = 1e-3
CLF_PATIENCE = 10


def make_loaders(X_tr, y_tr, X_va, y_va):
    ds_tr = TensorDataset(torch.tensor(X_tr).unsqueeze(1), torch.tensor(y_tr))
    ds_va = TensorDataset(torch.tensor(X_va).unsqueeze(1), torch.tensor(y_va))
    # Sampler balanceado para entrenamiento
    cc = np.bincount(y_tr, minlength=NUM_CLASSES)
    sw = (1.0 / (cc + 1e-6))[y_tr]
    smp = WeightedRandomSampler(sw, num_samples=len(ds_tr), replacement=True)
    return (DataLoader(ds_tr, batch_size=CLF_BATCH, sampler=smp),
            DataLoader(ds_va, batch_size=CLF_BATCH, shuffle=False))


# Separar una porción de validación del pool real para hiperparámetros
X_tr_r, X_va_r, y_tr_r, y_va_r = train_test_split(
    X_clf_real, y_clf_real, test_size=0.15, random_state=SEED, stratify=y_clf_real)

X_tr_s, X_va_s, y_tr_s, y_va_s = train_test_split(
    X_clf_syn, y_clf_syn, test_size=0.15, random_state=SEED, stratify=y_clf_syn)

X_tr_a, X_va_a, y_tr_a, y_va_a = train_test_split(
    X_clf_aug, y_clf_aug, test_size=0.15, random_state=SEED, stratify=y_clf_aug)

loader_held = DataLoader(
    TensorDataset(torch.tensor(X_clf_held).unsqueeze(1), torch.tensor(y_clf_held)),
    batch_size=CLF_BATCH, shuffle=False)

# ── A) Clasificador entrenado con datos REALES ───────────────────────────────
print('\n' + '─'*55)
print('🔵  MODO A – Solo datos REALES')
print('─'*55)
ld_tr_r, ld_va_r = make_loaders(X_tr_r, y_tr_r, X_va_r, y_va_r)
clf_real = train_classifier(ld_tr_r, ld_va_r, NUM_CLASSES,
                             max_epochs=CLF_EPOCHS, lr=CLF_LR,
                             patience=CLF_PATIENCE, tag='REAL')
res_real = evaluate_classifier(clf_real, loader_held, CLASS_NAMES)

# ── B) Clasificador entrenado con datos SINTÉTICOS ───────────────────────────
print('\n' + '─'*55)
print('🟡  MODO B – Solo datos SINTÉTICOS')
print('─'*55)
ld_tr_s, ld_va_s = make_loaders(X_tr_s, y_tr_s, X_va_s, y_va_s)
clf_syn = train_classifier(ld_tr_s, ld_va_s, NUM_CLASSES,
                            max_epochs=CLF_EPOCHS, lr=CLF_LR,
                            patience=CLF_PATIENCE, tag='SINT')
res_syn = evaluate_classifier(clf_syn, loader_held, CLASS_NAMES)

# ── C) Clasificador entrenado con REAL + SINTÉTICO ───────────────────────────
print('\n' + '─'*55)
print('🟢  MODO C – REAL + SINTÉTICO (Augmentado)')
print('─'*55)
ld_tr_a, ld_va_a = make_loaders(X_tr_a, y_tr_a, X_va_a, y_va_a)
clf_aug = train_classifier(ld_tr_a, ld_va_a, NUM_CLASSES,
                            max_epochs=CLF_EPOCHS, lr=CLF_LR,
                            patience=CLF_PATIENCE, tag='AUG')
res_aug = evaluate_classifier(clf_aug, loader_held, CLASS_NAMES)

# ── Tabla resumen de métricas del clasificador ───────────────────────────────
print('\n' + '='*60)
print('  RESULTADOS CLASIFICADOR TCN – TEST HELD-OUT')
print('='*60)
header = f"{'Modo':<20} {'Accuracy':>10} {'Recall':>10} {'F1-macro':>10}"
print(header); print('─'*60)
for tag, res in [('A – Solo REAL', res_real),
                 ('B – Solo SINT', res_syn),
                 ('C – REAL+SINT', res_aug)]:
    print(f"  {tag:<18} {res['accuracy']*100:>9.2f}%  "
          f"{res['recall_macro']*100:>9.2f}%  "
          f"{res['f1_macro']*100:>9.2f}%")

print('\n─── MODO A: Reporte completo ───')
print(res_real['report'])
print('─── MODO B: Reporte completo ───')
print(res_syn['report'])
print('─── MODO C: Reporte completo ───')
print(res_aug['report'])

# ── Matrices de confusión ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle('Matrices de Confusión – TCN Clasificador / PhysioNet', fontsize=13, fontweight='bold')

for ax, (res, title) in zip(axes, [
    (res_real, 'A – Solo REAL'),
    (res_syn,  'B – Solo SINTÉTICO'),
    (res_aug,  'C – REAL + SINTÉTICO'),
]):
    cm = res['confusion_matrix']
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
    ax.set_title(f'{title}\nAcc={res["accuracy"]*100:.1f}% | F1={res["f1_macro"]*100:.1f}%')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=7,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.04)

plt.tight_layout()
plt.savefig('matriz_confusion_clasificador_physionet.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Gráfico comparativo de métricas ─────────────────────────────────────────
modos   = ['A – REAL', 'B – SINT', 'C – AUG']
accs    = [r['accuracy']*100    for r in [res_real, res_syn, res_aug]]
recalls = [r['recall_macro']*100 for r in [res_real, res_syn, res_aug]]
f1s     = [r['f1_macro']*100    for r in [res_real, res_syn, res_aug]]

x    = np.arange(3)
width = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, accs,    width, label='Accuracy',      color='#2196F3')
ax.bar(x,         recalls, width, label='Recall Macro',  color='#4CAF50')
ax.bar(x + width, f1s,     width, label='F1 Macro',      color='#FF9800')
ax.set_xticks(x); ax.set_xticklabels(modos, fontsize=12)
ax.set_ylabel('Métrica (%)'); ax.set_ylim(0, 110)
ax.set_title('Comparación de métricas del clasificador por modo de entrenamiento',
             fontweight='bold')
ax.legend()
for bars in ax.containers:
    ax.bar_label(bars, fmt='%.1f%%', fontsize=9, padding=2)
plt.tight_layout()
plt.savefig('metricas_clasificador_physionet.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Guardar modelos del clasificador ─────────────────────────────────────────
torch.save(clf_real.state_dict(), 'clf_real_physionet.pt')
torch.save(clf_syn.state_dict(),  'clf_sint_physionet.pt')
torch.save(clf_aug.state_dict(),  'clf_aug_physionet.pt')
print('✅ Modelos del clasificador guardados')

# ══════════════════════════════════════════════════════════════════════════════
#  16. EXPORTACIÓN DEL GENERADOR A ONNX
# ══════════════════════════════════════════════════════════════════════════════
# Se exportan encoder y decoder por separado para mayor flexibilidad.
# También se exporta el mejor clasificador (modo C) a ONNX.

import onnx
import onnxruntime as ort

model.eval()
clf_aug.eval()

# ── Encoder ONNX ─────────────────────────────────────────────────────────────
class EncoderWrapper(nn.Module):
    """Wrapper para exportar solo el encoder (devuelve mu)."""
    def __init__(self, m): super().__init__(); self.enc = m.encoder

    def forward(self, x, c):
        mu, _ = self.enc(x, c)
        return mu


_enc_w   = EncoderWrapper(model).to('cpu').eval()
_dummy_x = torch.randn(1, 1, INPUT_LENGTH)
_dummy_c = torch.zeros(1, NUM_CLASSES)

torch.onnx.export(
    _enc_w, (_dummy_x, _dummy_c),
    'tcncvae_encoder_physionet.onnx',
    input_names  = ['signal', 'condition'],
    output_names = ['mu'],
    dynamic_axes = {'signal':    {0: 'batch'}, 'condition': {0: 'batch'}, 'mu': {0: 'batch'}},
    opset_version = 17,
    verbose       = False,
)
print('✅ Encoder exportado: tcncvae_encoder_physionet.onnx')

# ── Decoder ONNX ─────────────────────────────────────────────────────────────
class DecoderWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.dec = m.decoder

    def forward(self, z, c): return self.dec(z, c)


_dec_w    = DecoderWrapper(model).to('cpu').eval()
_dummy_z  = torch.randn(1, LATENT_DIM)

torch.onnx.export(
    _dec_w, (_dummy_z, _dummy_c),
    'tcncvae_decoder_physionet.onnx',
    input_names  = ['z', 'condition'],
    output_names = ['signal_synth'],
    dynamic_axes = {'z': {0: 'batch'}, 'condition': {0: 'batch'}, 'signal_synth': {0: 'batch'}},
    opset_version = 17,
    verbose       = False,
)
print('✅ Decoder exportado: tcncvae_decoder_physionet.onnx')

# ── Clasificador ONNX (modo C: aug) ──────────────────────────────────────────
_clf_w = clf_aug.to('cpu').eval()
torch.onnx.export(
    _clf_w, _dummy_x,
    'clf_aug_physionet.onnx',
    input_names  = ['signal'],
    output_names = ['logits'],
    dynamic_axes = {'signal': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version = 17,
    verbose       = False,
)
print('✅ Clasificador (AUG) exportado: clf_aug_physionet.onnx')

# ── Verificación con ONNXRuntime ──────────────────────────────────────────────
def verify_onnx(path, input_dict, label):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    out  = sess.run(None, {k: v.numpy() for k, v in input_dict.items()})
    print(f'   ✅ {label} | output shape: {out[0].shape}')

verify_onnx('tcncvae_encoder_physionet.onnx',
            {'signal': _dummy_x, 'condition': _dummy_c}, 'Encoder ONNX')
verify_onnx('tcncvae_decoder_physionet.onnx',
            {'z': _dummy_z, 'condition': _dummy_c}, 'Decoder ONNX')
verify_onnx('clf_aug_physionet.onnx',
            {'signal': _dummy_x}, 'Clasificador AUG ONNX')

# Guardar también los pesos PyTorch del generador
torch.save(model.state_dict(), 'tcncvae_generator_physionet.pt')
print('✅ Pesos generador guardados: tcncvae_generator_physionet.pt')

# ══════════════════════════════════════════════════════════════════════════════
#  17. RESUMEN FINAL COMPLETO
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*68)
print('  RESUMEN FINAL – TCN-cVAE + Clasificador TCN / PhysioNet')
print('='*68)

print('\n📊 DATASET')
print(f'   Total latidos    : {n_total:,}')
print(f'   Longitud señal   : {INPUT_LENGTH} muestras @ {FS} Hz')
print(f'   Clases           : {N_CLASSES} → {CLASS_NAMES}')
print(f'   Split generador  : {n_gen:,}  (60%)')
print(f'   Pool CLF real    : {n_clf:,}  (20%)')
print(f'   Held-out CLF     : {n_held:,} (20%)')

print('\n🏗️  ARQUITECTURA TCN-cVAE')
n_p = sum(p.numel() for p in model.parameters())
print(f'   Parámetros       : {n_p:,}')
print(f'   Latent dim       : {LATENT_DIM}')
print(f'   Dilataciones     : {DILATIONS}  (campo rec. ≥ {2*(KERNEL_SIZE-1)*sum(DILATIONS)} muestras)')
print(f'   BETA_MAX         : {BETA_MAX}  (corregido de 0.008)')
print(f'   Noise/LabelSmooth: activados')

print('\n📈 ENTRENAMIENTO GENERADOR')
print(f'   Épocas ejecutadas: {len(history["val_loss"])}')
print(f'   Mejor val_recon  : {best_val_recon:.5f}')
print(f'   Gap final        : {gap_fin:+.5f}')

print('\n📐 MÉTRICAS DEL GENERADOR (promedio macro)')
for k, v in gen_df[num_cols].mean().items():
    print(f'   {k:<22}: {v:.5f}')

print('\n🎯 CLASIFICADOR TCN – TEST HELD-OUT')
print(f"   {'Modo':<20} {'Accuracy':>10} {'Recall':>10} {'F1-macro':>10}")
for tag, res in [('A – Solo REAL', res_real), ('B – Solo SINT', res_syn), ('C – REAL+SINT', res_aug)]:
    print(f"   {tag:<20} {res['accuracy']*100:>9.2f}%  "
          f"{res['recall_macro']*100:>9.2f}%  "
          f"{res['f1_macro']*100:>9.2f}%")

delta_acc = (res_aug['accuracy'] - res_real['accuracy']) * 100
delta_f1  = (res_aug['f1_macro'] - res_real['f1_macro']) * 100
print(f'\n   Δ Accuracy (AUG vs REAL) : {delta_acc:+.2f}pp  ({"↑ mejora" if delta_acc > 0 else "↓ no mejora"})')
print(f'   Δ F1-macro  (AUG vs REAL): {delta_f1:+.2f}pp  ({"↑ mejora" if delta_f1  > 0 else "↓ no mejora"})')

print('\n📁 ARCHIVOS EXPORTADOS')
for fname in [
    'tcncvae_generator_physionet.pt',
    'tcncvae_encoder_physionet.onnx',
    'tcncvae_decoder_physionet.onnx',
    'clf_real_physionet.pt',
    'clf_sint_physionet.pt',
    'clf_aug_physionet.pt',
    'clf_aug_physionet.onnx',
    'curvas_aprendizaje_physionet.png',
    'senales_sinteticas_physionet.png',
    'metricas_generador_physionet.csv',
    'metricas_generador_physionet.png',
    'matriz_confusion_clasificador_physionet.png',
    'metricas_clasificador_physionet.png',
    'tsne_latente_physionet.png',
]:
    estado = '✅' if os.path.exists(fname) else '⏳'
    print(f'   {estado}  {fname}')

print('='*68)
