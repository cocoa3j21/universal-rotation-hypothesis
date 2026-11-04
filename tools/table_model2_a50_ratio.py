import os, sys, csv
import numpy as np

project_dir = 'Labs/universal-rotation-hypothesis'
v2_pred_path = os.path.join(project_dir, 'output', 'v2_predictions_fixed.csv')

with open(v2_pred_path, 'r', encoding='utf-8') as f:
    v2_preds = list(csv.DictReader(f))

m2_preds = [r for r in v2_preds if r['model_id'] == 'model_2']

print(f"=== Model 2 (4.89 km/s) Recovery Status vs A50 Table (N={len(m2_preds)}) ===")
header = f"{'FWHM':>5} | {'Sigma_v':>7} | {'Inc':>4} | {'A50_pred':>8} | {'A_phys/A50':>10} | {'Pred Class':>12}"
print(header)
print("-" * len(header))

for r in m2_preds:
    print(f"{float(r['fwhm_kpc']):5.1f} | {float(r['sigma_v_kms']):7.1f} | {float(r['inc_deg']):4.0f} | {float(r['a50_pred_kms']):8.2f} | {float(r['ratio_phys_to_a50']):10.3f} | {r['pred_class']:>12}")

print("\nModel 2 Pred Class Distribution in v2:")
pclasses = {}
for r in m2_preds:
    pclasses[r['pred_class']] = pclasses.get(r['pred_class'], 0) + 1
for k, v in pclasses.items():
    print(f"  {k}: {v} / {len(m2_preds)} ({v/len(m2_preds)*100:.1f}%)")
