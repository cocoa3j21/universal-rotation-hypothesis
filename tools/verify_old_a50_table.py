import os, sys, csv
import numpy as np

project_dir = 'Labs/universal-rotation-hypothesis'
old_csv_path = os.path.join(project_dir, 'output', 'a50_grid_table.csv')

with open(old_csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Total rows in old A50 table: {len(rows)}")

# Check each row against the formula:
# A_50_pred = (snr_th * sigma_v) / (eta_beam * eta_fit * sin(i))
discrepancies = []
for idx, r in enumerate(rows):
    snr_th = float(r['snr_th_ref'])
    sigma_v = float(r['sigma_v'])
    eta_beam = float(r['eta_beam'])
    eta_fit = float(r['eta_fit'])
    inc = float(r['inc_deg'])
    a50_old = float(r['a50_pred_kms'])

    sin_i = np.sin(np.radians(inc))
    denom = eta_beam * eta_fit * sin_i
    a50_calc = (snr_th * sigma_v) / denom

    diff = abs(a50_calc - a50_old)
    if diff > 0.01:
        discrepancies.append((idx, r, a50_calc, a50_old))

print(f"Discrepancies against formula using recorded eta_beam: {len(discrepancies)}")
if len(discrepancies) > 0:
    for idx, r, calc, old in discrepancies[:5]:
        print(f"  Row {idx}: calc={calc:.3f}, old={old:.3f}")
else:
    print("ALL 54 rows match the formula A_50 = (snr_th * sigma_v) / (eta_beam * eta_fit * sin i) with recorded eta_beam!")

# Now check how eta_beam was computed
print("\nRecorded eta_beam values by model and FWHM:")
eta_dict = {}
for r in rows:
    key = (r['model_id'], float(r['fwhm_kpc']))
    eta_dict[key] = float(r['eta_beam'])
for k, v in sorted(eta_dict.items()):
    print(f"  {k}: eta_beam = {v}")
