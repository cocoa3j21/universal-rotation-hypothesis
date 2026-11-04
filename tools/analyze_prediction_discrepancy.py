import os
import csv
import numpy as np

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pred_path = os.path.join(project_dir, 'output', 'cell_predictions_preregistered.csv')
res_path = os.path.join(project_dir, 'output', 'injection_recovery_summary.csv')

pred_rows = []
with open(pred_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        pred_rows.append(r)

res_rows = []
with open(res_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        res_rows.append(r)

assert len(pred_rows) == len(res_rows), f"Length mismatch: {len(pred_rows)} vs {len(res_rows)}"

pairs = []
for p, r in zip(pred_rows, res_rows):
    # Verify alignment
    assert p['model_id'] == r['model_id']
    assert p['n_vortex_inj'] == r['n_vortex_inj']
    assert p['inc_deg'] == r['inc_deg']
    assert p['pa_deg'] == r['pa_deg']
    pairs.append((p, r))

print(f"Total cells: {len(pairs)}")

# Filter cells with vortex injection (n_vortex_inj > 0)
vortex_pairs = [(p, r) for (p, r) in pairs if int(r['n_vortex_inj']) > 0]
print(f"Total cells with vortex injection: {len(vortex_pairs)}")

# Model 1 vortex cells:
m1_vortex = [(p, r) for (p, r) in vortex_pairs if r['model_id'] == 'model_1']
m3_vortex = [(p, r) for (p, r) in vortex_pairs if r['model_id'] == 'model_3']
print(f"Model 1 vortex cells: {len(m1_vortex)} (Total M1 cells: {len([p for p in pairs if p[0]['model_id'] == 'model_1'])})")
print(f"Model 3 vortex cells: {len(m3_vortex)}")

# Evaluation excluding BOUNDARY
eval_pairs = [(p, r) for (p, r) in vortex_pairs if p['pred_class'] in ['RECOVERED', 'UNDETECTED']]
print(f"Vortex evaluation cells (excluding BOUNDARY): {len(eval_pairs)}")

tp = fp = fn = tn = 0
fp_list = []
for p, r in eval_pairs:
    p_rec = (p['pred_class'] == 'RECOVERED')
    a_rec = (float(r['recall_pos']) >= 0.5)
    if p_rec and a_rec: tp += 1
    elif p_rec and not a_rec:
        fp += 1
        fp_list.append((p, r))
    elif not p_rec and a_rec: fn += 1
    else: tn += 1

print(f"\nStrict recall_pos >= 0.5 Confusion Matrix:")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
acc = (tp + tn) / len(eval_pairs)
all_undet_acc = (fn + tn) / len(eval_pairs)
p0 = acc
pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (len(eval_pairs)**2)
kappa = (p0 - pe) / (1 - pe) if (1 - pe) > 0 else 0
print(f"Accuracy: {acc:.4f} (All-UNDETECTED: {all_undet_acc:.4f})")
print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, Cohen's kappa: {kappa:.4f}")

# Subsets analysis
def eval_subset(name, cond_func):
    sub = [(p, r) for (p, r) in eval_pairs if cond_func(p, r)]
    s_tp = s_fp = s_fn = s_tn = 0
    for p, r in sub:
        p_rec = (p['pred_class'] == 'RECOVERED')
        a_rec = (float(r['recall_pos']) >= 0.5)
        if p_rec and a_rec: s_tp += 1
        elif p_rec and not a_rec: s_fp += 1
        elif not p_rec and a_rec: s_fn += 1
        else: s_tn += 1
    s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0
    s_rec = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0
    print(f"\n--- {name} (N={len(sub)}) ---")
    print(f"TP={s_tp}, FP={s_fp}, FN={s_fn}, TN={s_tn}")
    print(f"Precision: {s_prec:.4f}, Recall: {s_rec:.4f}")

eval_subset("Ideal: Nv=1, arm=none, geom=False, PA=0",
            lambda p, r: int(r['n_vortex_inj']) == 1 and float(r['a_los3_kms']) == 0.0 and r['geom_error_flag'].lower() == 'false' and float(r['pa_deg']) == 0.0)

eval_subset("Nv=1, arm=none, geom=False, all PA",
            lambda p, r: int(r['n_vortex_inj']) == 1 and float(r['a_los3_kms']) == 0.0 and r['geom_error_flag'].lower() == 'false')

eval_subset("Nv=1, arm=none, geom=True, all PA",
            lambda p, r: int(r['n_vortex_inj']) == 1 and float(r['a_los3_kms']) == 0.0 and r['geom_error_flag'].lower() == 'true')

eval_subset("Nv=1, arm_raw (a3 > 0), geom=False",
            lambda p, r: int(r['n_vortex_inj']) == 1 and float(r['a_los3_kms']) > 0.0 and r['arm_method'] == 'raw' and r['geom_error_flag'].lower() == 'false')

eval_subset("Nv=1, arm_fit (a3 > 0), geom=False",
            lambda p, r: int(r['n_vortex_inj']) == 1 and float(r['a_los3_kms']) > 0.0 and r['arm_method'] == 'arm_fit' and r['geom_error_flag'].lower() == 'false')

eval_subset("Nv >= 2, arm=none, geom=False",
            lambda p, r: int(r['n_vortex_inj']) >= 2 and float(r['a_los3_kms']) == 0.0 and r['geom_error_flag'].lower() == 'false')

eval_subset("Nv >= 2, any arm/geom",
            lambda p, r: int(r['n_vortex_inj']) >= 2)

# Inspect FP cells distribution
print(f"\nFP Cells Breakdown by Factor (Total FP = {len(fp_list)}):")
fp_models = {}
fp_nvs = {}
fp_geoms = {}
fp_arms = {}
fp_incs = {}
fp_fwhms = {}
fp_sigs = {}

for p, r in fp_list:
    fp_models[r['model_id']] = fp_models.get(r['model_id'], 0) + 1
    fp_nvs[r['n_vortex_inj']] = fp_nvs.get(r['n_vortex_inj'], 0) + 1
    fp_geoms[r['geom_error_flag']] = fp_geoms.get(r['geom_error_flag'], 0) + 1
    fp_arms[r['arm_method']] = fp_arms.get(r['arm_method'], 0) + 1
    fp_incs[r['inc_deg']] = fp_incs.get(r['inc_deg'], 0) + 1
    fp_fwhms[r['fwhm_kpc']] = fp_fwhms.get(r['fwhm_kpc'], 0) + 1
    fp_sigs[r['sigma_v_kms']] = fp_sigs.get(r['sigma_v_kms'], 0) + 1

print("Models:", fp_models)
print("Nv:", fp_nvs)
print("Geom:", fp_geoms)
print("Arm method:", fp_arms)
print("Inc:", fp_incs)
print("FWHM:", fp_fwhms)
print("Sigma_v:", fp_sigs)
