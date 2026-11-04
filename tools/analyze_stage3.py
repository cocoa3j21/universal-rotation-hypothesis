# -*- coding: utf-8 -*-
import csv
from collections import defaultdict

with open("output/h0_stratum_calibration.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total strata: {len(rows)}")

delta_bics = [float(r["delta_bic_th_stratum"]) for r in rows]
fpr_uppers = [float(r["fpr_h0_upper95"]) for r in rows]
fpr_h0s = [float(r["fpr_h0"]) for r in rows]

print(f"Delta_BIC min={min(delta_bics):.3f}, median={sorted(delta_bics)[len(delta_bics)//2]:.3f}, max={max(delta_bics):.3f}")
print(f"FPR_H0 min={min(fpr_h0s):.4f}, median={sorted(fpr_h0s)[len(fpr_h0s)//2]:.4f}, max={max(fpr_h0s):.4f}")
print(f"FPR_upper95 min={min(fpr_uppers):.4f}, median={sorted(fpr_uppers)[len(fpr_uppers)//2]:.4f}, max={max(fpr_uppers):.4f}")

n_pass = sum(1 for u in fpr_uppers if u <= 0.070)
print(f"Pass (upper95 <= 0.070): {n_pass} / {len(rows)} ({n_pass/len(rows)*100:.1f}%)")

by_model = defaultdict(list)
by_inc = defaultdict(list)
by_arm = defaultdict(list)
by_fwhm = defaultdict(list)
by_noise = defaultdict(list)

for r in rows:
    u = float(r["fpr_h0_upper95"])
    by_model[r["model_id"]].append(u <= 0.070)
    by_inc[r["inc_deg"]].append(u <= 0.070)
    by_arm[f"{r['a_los3']}_{r['arm_method']}"].append(u <= 0.070)
    by_fwhm[r["fwhm_kpc"]].append(u <= 0.070)
    by_noise[r["sigma_v"]].append(u <= 0.070)

print("\nPass rate by Model:")
for k, v in by_model.items():
    print(f"  {k}: {sum(v)}/{len(v)} ({sum(v)/len(v)*100:.1f}%)")

print("\nPass rate by Inc:")
for k, v in sorted(by_inc.items()):
    print(f"  inc={k}: {sum(v)}/{len(v)} ({sum(v)/len(v)*100:.1f}%)")

print("\nPass rate by FWHM:")
for k, v in sorted(by_fwhm.items()):
    print(f"  FWHM={k}: {sum(v)}/{len(v)} ({sum(v)/len(v)*100:.1f}%)")

print("\nPass rate by Noise sigma_v:")
for k, v in sorted(by_noise.items()):
    print(f"  sigma_v={k}: {sum(v)}/{len(v)} ({sum(v)/len(v)*100:.1f}%)")

print("\nPass rate by Arm Level:")
for k, v in sorted(by_arm.items()):
    print(f"  arm={k}: {sum(v)}/{len(v)} ({sum(v)/len(v)*100:.1f}%)")
