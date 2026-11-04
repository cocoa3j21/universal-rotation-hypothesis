import os, csv
import numpy as np

cal_path = 'Labs/universal-rotation-hypothesis/output/h0_model2_calibration.csv'
res_path = 'Labs/universal-rotation-hypothesis/output/injection_recovery_model2_summary.csv'

with open(cal_path, 'r', encoding='utf-8') as f:
    cal_rows = list(csv.DictReader(f))

with open(res_path, 'r', encoding='utf-8') as f:
    res_rows = list(csv.DictReader(f))

# Filter control cells (n_vortex_inj == 0) in Stage 4
control_cells = [r for r in res_rows if int(r['n_vortex_inj']) == 0]
print(f"Total Model 2 control cells: {len(control_cells)}")

deg_arm_cells = [r for r in control_cells if r['cell_verdict'] == 'CELL_DEGENERATE_ARM']
print(f"CELL_DEGENERATE_ARM count: {len(deg_arm_cells)} ({len(deg_arm_cells)/len(control_cells)*100:.2f}%)")

# Group by PA
for pa in [0.0, 45.0, 90.0]:
    pa_cells = [r for r in control_cells if float(r['pa_deg']) == pa]
    pa_deg = [r for r in pa_cells if r['cell_verdict'] == 'CELL_DEGENERATE_ARM']
    fpr_arms = [float(r['fpr_arm']) for r in pa_cells if float(r['a_los3_kms']) > 0]
    fpr_h0s = [float(r['fpr_h0_val']) for r in pa_cells if float(r['a_los3_kms']) > 0]
    delta_fprs = [float(r['delta_fpr_arm_lower95']) for r in pa_cells if float(r['a_los3_kms']) > 0]
    print(f"\nPA = {pa} deg (N={len(pa_cells)}, Arm-injected N={len(fpr_arms)}):")
    print(f"  CELL_DEGENERATE_ARM count: {len(pa_deg)} / {len(pa_cells)} ({len(pa_deg)/len(pa_cells)*100:.2f}%)")
    print(f"  Stage 4 fpr_arm: mean={np.mean(fpr_arms):.4f}, median={np.median(fpr_arms):.4f}, max={np.max(fpr_arms):.4f}")
    print(f"  Stage 3 fpr_h0_val: mean={np.mean(fpr_h0s):.4f}, median={np.median(fpr_h0s):.4f}, max={np.max(fpr_h0s):.4f}")
    print(f"  delta_fpr_arm_lower95: mean={np.mean(delta_fprs):.4f}, max={np.max(delta_fprs):.4f}")

# Group by arm_method and a_los3
print("\n--- Breakdown of CELL_DEGENERATE_ARM by arm_method and a_los3 ---")
arm_breakdown = {}
for r in deg_arm_cells:
    key = (r['arm_method'], float(r['a_los3_kms']), float(r['inc_deg']))
    arm_breakdown[key] = arm_breakdown.get(key, 0) + 1

for k, v in sorted(arm_breakdown.items()):
    print(f"  method={k[0]}, a_los3={k[1]}, inc={k[2]}: {v} cells")
