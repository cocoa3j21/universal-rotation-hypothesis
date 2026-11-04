import os
import json
import csv
import numpy as np

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(project_dir, 'output', '.stratum_snr_thresholds.json')
csv_path = os.path.join(project_dir, 'output', 'h0_stratum_calibration.csv')

with open(json_path, 'r', encoding='utf-8') as f:
    snr_dict = json.load(f)

rows = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        key = f"{r['model_id']}_{r['inc_deg']}_{r['fwhm_kpc']}_{r['sigma_v']}_{r['geom_error_flag']}_{r['a_los3']}_{r['arm_method']}"
        val = snr_dict.get(key, None)
        if val is not None:
            r['snr_th'] = float(val)
            r['inc_deg'] = float(r['inc_deg'])
            r['sigma_v'] = float(r['sigma_v'])
            r['fwhm_kpc'] = float(r['fwhm_kpc'])
            r['geom_error_flag'] = (r['geom_error_flag'].lower() == 'true')
            rows.append(r)

snrs = [r['snr_th'] for r in rows]
print(f"Total strata: {len(rows)}")
print(f"Overall SNR_th: min={np.min(snrs):.2f}, med={np.median(snrs):.2f}, mean={np.mean(snrs):.2f}, max={np.max(snrs):.2f}")

# By geom_error_flag
for g in [False, True]:
    sub = [r['snr_th'] for r in rows if r['geom_error_flag'] == g]
    print(f"geom_error_flag={g}: n={len(sub)}, min={np.min(sub):.2f}, med={np.median(sub):.2f}, mean={np.mean(sub):.2f}, max={np.max(sub):.2f}")

# By (geom_error_flag, inc_deg)
for g in [False, True]:
    for inc in [30.0, 50.0, 70.0]:
        sub = [r['snr_th'] for r in rows if r['geom_error_flag'] == g and r['inc_deg'] == inc]
        print(f"geom={g}, inc={inc:02.0f}: n={len(sub)}, min={np.min(sub):.2f}, med={np.median(sub):.2f}, max={np.max(sub):.2f}")

# Strata with SNR_th >= 10.0
high_strata = [r for r in rows if r['snr_th'] >= 10.0]
print(f"\nStrata with SNR_th >= 10.0: {len(high_strata)} / {len(rows)}")

# Strata with SNR_th >= 30.0
extreme = [r for r in rows if r['snr_th'] >= 30.0]
print(f"Strata with SNR_th >= 30.0: {len(extreme)}")
for r in extreme:
    print(f"  Stratum {r['stratum_id']}: model={r['model_id']}, inc={r['inc_deg']}, fwhm={r['fwhm_kpc']}, sig={r['sigma_v']}, arm_meth={r['arm_method']}, snr_th={r['snr_th']:.2f}")
