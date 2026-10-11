"""Run XAI evaluation on trained model."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from app.ml.model import PhysicsInformedBiLSTM, get_device
from app.ml.xai import compute_xai_attribution
from app.ml.xai_evaluation import run_full_xai_evaluation
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load test data
csv_path = Path(__file__).parent / 'data' / 'processed' / 'argo_floats_canonical.csv'
df = pd.read_csv(csv_path)

STANDARD_DEPTHS = [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
TEMP_COLS = [f'temp_{d}' for d in STANDARD_DEPTHS]
SAL_COLS = [f'sal_{d}' for d in STANDARD_DEPTHS]

ARCTIC_WMOS = {6903058, 6903059, 6903060, 6903062, 6903063}
wmos = sorted(df['wmo'].unique().tolist())
arabian = sorted(set(wmos) - ARCTIC_WMOS)
_, test_ar = train_test_split(arabian, test_size=5, random_state=42)
test_df = df[df['wmo'].isin(test_ar)].reset_index(drop=True)
train_df = df[~df['wmo'].isin(test_ar)].reset_index(drop=True)

temp_scaler = StandardScaler().fit(train_df[TEMP_COLS].to_numpy(dtype=np.float64))
sal_scaler = StandardScaler().fit(train_df[SAL_COLS].to_numpy(dtype=np.float64))

def scale_frame(frame):
    t = temp_scaler.transform(frame[TEMP_COLS].to_numpy(dtype=np.float64)).astype(np.float32)
    s = sal_scaler.transform(frame[SAL_COLS].to_numpy(dtype=np.float64)).astype(np.float32)
    out = frame.copy()
    out[TEMP_COLS] = t
    out[SAL_COLS] = s
    return out

def build_windows(df):
    Xs, yTs, ySs, metas = [], [], [], []
    for wmo, g in df.sort_values(['wmo', 'cycle']).groupby('wmo'):
        g = g.dropna(subset=TEMP_COLS + SAL_COLS + ['cycle']).reset_index(drop=True)
        if len(g) < 4: continue
        t = g[TEMP_COLS].to_numpy(dtype=np.float32)
        s = g[SAL_COLS].to_numpy(dtype=np.float32)
        cyc = g['cycle'].to_numpy()
        for i in range(len(g) - 3):
            seq_t = t[i:i+3]; seq_s = s[i:i+3]
            x = np.concatenate([seq_t, seq_s], axis=1)
            Xs.append(x); yTs.append(t[i+3]); ySs.append(s[i+3])
    return np.stack(Xs), np.stack(yTs), np.stack(ySs), metas

Xte, yTte, ySte, _ = build_windows(scale_frame(test_df))
print('Test windows:', len(Xte))

# Compute climatology baseline (mean profile across training set)
climatology_temp = train_df[TEMP_COLS].mean().values.astype(np.float32)  # (16,)
climatology_sal = train_df[SAL_COLS].mean().values.astype(np.float32)    # (16,)
# Scale climatology
climatology_temp_scaled = temp_scaler.transform(climatology_temp.reshape(1, -1)).flatten()
climatology_sal_scaled = sal_scaler.transform(climatology_sal.reshape(1, -1)).flatten()
# Build full 3-cycle climatology input (repeat for 3 cycles)
climatology_full = np.tile(np.concatenate([climatology_temp_scaled, climatology_sal_scaled]), 3).astype(np.float32)
climatology_tensor = torch.from_numpy(climatology_full).reshape(1, 3, 32)
print('Climatology tensor shape:', climatology_tensor.shape)

# Load model
device = get_device()
model = PhysicsInformedBiLSTM().to(device)
artifacts_dir = Path(__file__).parent / 'app' / 'ml' / 'artifacts'
model.load_state_dict(torch.load(artifacts_dir / 'model_weights.pt', map_location=device))
model.eval()

# Test on first sample
input_tensor = torch.from_numpy(Xte[:1].astype(np.float32)).to(device)
print('Input shape:', input_tensor.shape)

# Run XAI evaluation
print()
print('=== XAI EVALUATION ===')
results = run_full_xai_evaluation(
    model=model,
    input_tensor=input_tensor,
    compute_attributions_fn=compute_xai_attribution,
    target_variable='temp',
    target_depth_idx=4,  # 100 dbar
    variable='temp',
    climatology=climatology_tensor,  # Pass climatology for insertion/deletion
)
print()
print('=== ATTRIBUTION METHOD COMPARISON ===')
from app.ml.xai_evaluation import compare_attribution_methods
comparison = compare_attribution_methods(
    model=model,
    input_tensor=input_tensor,
    target_variable='temp',
    target_depth_idx=4,
    n_smoothing_samples=10,  # Reduced for speed
)
print()
print('=== SUMMARY ===')
for k, v in results.items():
    if k not in ['insertion_curve', 'deletion_curve', 'physical_plausibility']:
        print(f'  {k}: {v}')
print()
print('Physical plausibility:')
for pk, pv in results['physical_plausibility'].items():
    print(f'  {pk}: {pv}')