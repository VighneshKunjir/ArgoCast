"""Test SmoothGrad and Expected Gradients."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from app.ml.model import PhysicsInformedBiLSTM, get_device
from app.ml.xai import compute_integrated_gradients_with_smoothing, compute_smoothgrad, compute_expected_gradients
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

device = get_device()
model = PhysicsInformedBiLSTM().to(device)
artifacts_dir = Path(__file__).parent / 'app' / 'ml' / 'artifacts'
model.load_state_dict(torch.load(artifacts_dir / 'model_weights.pt', map_location=device))
model.eval()

input_tensor = torch.from_numpy(Xte[:1].astype(np.float32)).to(device)

print('Testing SmoothGrad (5 samples)...')
attr_sg, delta_sg = compute_smoothgrad(model, input_tensor, 'temp', 4, n_samples=5)
print(f'  SmoothGrad shape: {attr_sg.shape}, delta: {delta_sg}')

print('Testing Expected Gradients (5 samples)...')
attr_eg, delta_eg = compute_expected_gradients(model, input_tensor, 'temp', 4, n_samples=5)
print(f'  Expected Gradients shape: {attr_eg.shape}, delta: {delta_eg}')

print('Testing unified interface...')
attr_none, delta_none = compute_integrated_gradients_with_smoothing(model, input_tensor, 'temp', 4, smoothing='none')
print(f'  None: shape={attr_none.shape}, delta={delta_none}')

attr_sg2, delta_sg2 = compute_integrated_gradients_with_smoothing(model, input_tensor, 'temp', 4, smoothing='smoothgrad', n_smoothing_samples=5)
print(f'  SmoothGrad: shape={attr_sg2.shape}, delta={delta_sg2}')

attr_eg2, delta_eg2 = compute_integrated_gradients_with_smoothing(model, input_tensor, 'temp', 4, smoothing='expected_gradients', n_smoothing_samples=5)
print(f'  Expected Gradients: shape={attr_eg2.shape}, delta={delta_eg2}')

print('All tests passed!')