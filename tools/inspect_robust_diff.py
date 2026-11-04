# -*- coding: utf-8 -*-
import csv

with open('output/injection_recovery_summary.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

robust_cells = []
rec_high_not_robust = []

for idx, r in enumerate(rows):
    v = r['cell_verdict']
    rec = float(r['recall_pos']) if (r['recall_pos'] != '' and r['recall_pos'] != 'nan') else 0.0
    msel = float(r['rate_model_select']) if (r['rate_model_select'] != '' and r['rate_model_select'] != 'nan') else 0.0
    
    if v == 'CELL_ROBUST_RECOVERED':
        robust_cells.append((idx, r))
    elif rec >= 0.50:
        rec_high_not_robust.append((idx, r, rec, msel, v))

print(f"Current CELL_ROBUST_RECOVERED count: {len(robust_cells)}")
print("\nCurrent Robust Recovered cells:")
for idx, r in robust_cells:
    print(f"  [{idx}] Nv={r['n_vortex_inj']}, i={r['inc_deg']}, PA={r['pa_deg']}, FWHM={r['fwhm_kpc']}, s={r['sigma_v_kms']}, geom={r['geom_error_flag']}, a3={r['a_los3_kms']}, meth={r['arm_method']}, rec={float(r['recall_pos']):.2f}, msel={float(r['rate_model_select']):.2f}")

print(f"\nCells with recall_pos >= 0.50 but NOT robust recovered ({len(rec_high_not_robust)} cells):")
for idx, r, rec, msel, v in rec_high_not_robust:
    print(f"  [{idx}] Nv={r['n_vortex_inj']}, i={r['inc_deg']}, PA={r['pa_deg']}, FWHM={r['fwhm_kpc']}, s={r['sigma_v_kms']}, geom={r['geom_error_flag']}, a3={r['a_los3_kms']}, meth={r['arm_method']} -> verdict={v}, rec={rec:.2f}, msel={msel:.2f}, th={float(r['delta_bic_th_stratum']):.3f}")
