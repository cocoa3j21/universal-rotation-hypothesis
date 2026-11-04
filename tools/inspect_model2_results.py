import os
import csv
import numpy as np

csv_path = 'Labs/universal-rotation-hypothesis/output/injection_recovery_model2_summary.csv'
rows = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

vortex_rows = [r for r in rows if int(r['n_vortex_inj']) > 0]
recalls = [float(r['recall_pos']) for r in vortex_rows]

print(f"Total Model 2 vortex cells: {len(vortex_rows)}")
print(f"Recall pos: min={np.min(recalls):.4f}, med={np.median(recalls):.4f}, mean={np.mean(recalls):.4f}, max={np.max(recalls):.4f}")

# Top 10 cells by recall_pos
sorted_vortex = sorted(vortex_rows, key=lambda x: float(x['recall_pos']), reverse=True)
print("\nTop 10 Model 2 cells by recall_pos:")
for r in sorted_vortex[:10]:
    print(f"Cell: Nv={r['n_vortex_inj']}, inc={r['inc_deg']}, pa={r['pa_deg']}, fwhm={r['fwhm_kpc']}, sig={r['sigma_v_kms']}, geom={r['geom_error_flag']}, arm={r['arm_method']}, a3={r['a_los3_kms']} -> recall_pos={r['recall_pos']}, verdict={r['cell_verdict']}")
