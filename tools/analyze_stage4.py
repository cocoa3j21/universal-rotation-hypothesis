# -*- coding: utf-8 -*-
import csv
from collections import defaultdict, Counter

with open("output/injection_recovery_summary.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows: {len(rows)}")

verdict_counts = Counter(r["cell_verdict"] for r in rows)
print("\n=== CELL VERDICT BREAKDOWN (ALL 9,072 CELLS) ===")
for k, v in verdict_counts.most_common():
    print(f"  {k}: {v} ({v/len(rows)*100:.2f}%)")

print("\n=== CELL VERDICT BY MODEL ===")
for mid in ["model_1", "model_3"]:
    sub = [r for r in rows if r["model_id"] == mid]
    cnt = Counter(r["cell_verdict"] for r in sub)
    print(f"\nModel: {mid} (Total {len(sub)})")
    for k, v in cnt.most_common():
        print(f"  {k}: {v} ({v/len(sub)*100:.2f}%)")

print("\n=== CELL_ROBUST_RECOVERED BREAKDOWN ===")
robust_cells = [r for r in rows if r["cell_verdict"] == "CELL_ROBUST_RECOVERED"]
print(f"Total CELL_ROBUST_RECOVERED: {len(robust_cells)}")
for r in robust_cells:
    print(f"  Model={r['model_id']}, Nv={r['n_vortex_inj']}, inc={r['inc_deg']}, PA={r['pa_deg']}, FWHM={r['fwhm_kpc']}, s={r['sigma_v_kms']}, geom={r['geom_error_flag']}, a3={r['a_los3_kms']}, meth={r['arm_method']}, recall_pos={float(r['recall_pos']):.2f}, model_sel={float(r['rate_model_select']):.2f}, xi_vortex={r['xi_vortex']}")

print("\n=== RECALL_POS STATS BY MODEL (Nv >= 1) ===")
for mid in ["model_1", "model_3"]:
    sub_inj = [float(r["recall_pos"]) for r in rows if r["model_id"] == mid and int(r["n_vortex_inj"]) >= 1]
    print(f"{mid} (Nv>=1, {len(sub_inj)} cells): min={min(sub_inj):.3f}, mean={sum(sub_inj)/len(sub_inj):.3f}, max={max(sub_inj):.3f}")

print("\n=== RECALL_POS_VIS STATS BY MODEL (Nv >= 1) ===")
for mid in ["model_1", "model_3"]:
    sub_vis = [float(r["recall_pos_vis"]) for r in rows if r["model_id"] == mid and int(r["n_vortex_inj"]) >= 1 and r["recall_pos_vis"] != ""]
    if len(sub_vis) > 0:
        print(f"{mid} (vis, {len(sub_vis)} cells): min={min(sub_vis):.3f}, mean={sum(sub_vis)/len(sub_vis):.3f}, max={max(sub_vis):.3f}")
