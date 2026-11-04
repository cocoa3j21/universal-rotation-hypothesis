import os, csv

pred_path = 'Labs/universal-rotation-hypothesis/output/cell_predictions_preregistered.csv'
res_path = 'Labs/universal-rotation-hypothesis/output/injection_recovery_summary.csv'

with open(pred_path, 'r', encoding='utf-8') as f:
    preds = list(csv.DictReader(f))
with open(res_path, 'r', encoding='utf-8') as f:
    results = list(csv.DictReader(f))

# Check length and pairing
assert len(preds) == len(results)

# Let's inspect all cells with vortex injection (n_vortex_inj > 0)
# and classify them under:
# Rule A: Actual = (cell_verdict == 'CELL_ROBUST_RECOVERED')
# Rule B: Actual = (recall_pos >= 0.50)

vortex_eval = []
for p, r in zip(preds, results):
    if int(r['n_vortex_inj']) > 0 and p['pred_class'] in ['RECOVERED', 'UNDETECTED']:
        vortex_eval.append((p, r))

print(f"Vortex evaluation cells: {len(vortex_eval)}")

# Under Rule A (ROBUST)
tp_A = fp_A = fn_A = tn_A = 0
for p, r in vortex_eval:
    p_rec = (p['pred_class'] == 'RECOVERED')
    a_rec = (r['cell_verdict'] == 'CELL_ROBUST_RECOVERED')
    if p_rec and a_rec: tp_A += 1
    elif p_rec and not a_rec: fp_A += 1
    elif not p_rec and a_rec: fn_A += 1
    else: tn_A += 1

print(f"Rule A (Actual = CELL_ROBUST_RECOVERED):")
print(f"  TP={tp_A}, FP={fp_A}, FN={fn_A}, TN={tn_A}")

# Under Rule B (recall_pos >= 0.50)
tp_B = fp_B = fn_B = tn_B = 0
discrepancies = []
for p, r in vortex_eval:
    p_rec = (p['pred_class'] == 'RECOVERED')
    rec_val = float(r['recall_pos'])
    a_rec = (rec_val >= 0.50)
    if p_rec and a_rec: tp_B += 1
    elif p_rec and not a_rec: fp_B += 1
    elif not p_rec and a_rec: fn_B += 1
    else: tn_B += 1

    # Check cells where Rule A and Rule B differ
    if (r['cell_verdict'] == 'CELL_ROBUST_RECOVERED') != (rec_val >= 0.50):
        discrepancies.append((p, r))

print(f"\nRule B (Actual = recall_pos >= 0.50):")
print(f"  TP={tp_B}, FP={fp_B}, FN={fn_B}, TN={tn_B}")

print(f"\nDiscrepant cells between Rule A and Rule B: {len(discrepancies)}")
for p, r in discrepancies:
    print(f"  model={r['model_id']}, Nv={r['n_vortex_inj']}, inc={r['inc_deg']}, fwhm={r['fwhm_kpc']}, sig={r['sigma_v_kms']}, geom={r['geom_error_flag']}, arm={r['arm_method']}, pa={r['pa_deg']}")
    print(f"    pred_class={p['pred_class']}, recall_pos={r['recall_pos']}, recall_multi={r['recall_multi']}, msel={r['rate_model_select']}, verdict={r['cell_verdict']}")
