import os, csv

project_dir = 'Labs/universal-rotation-hypothesis'
pred_path = os.path.join(project_dir, 'output', 'cell_predictions_preregistered.csv')
res_path = os.path.join(project_dir, 'output', 'injection_recovery_summary.csv')

with open(pred_path, 'r', encoding='utf-8') as f:
    preds = list(csv.DictReader(f))
with open(res_path, 'r', encoding='utf-8') as f:
    results = list(csv.DictReader(f))

for p, r in zip(preds, results):
    if (r['model_id'] == 'model_3' and int(r['n_vortex_inj']) == 1 and 
        float(r['a_los3_kms']) == 0.0 and r['geom_error_flag'].lower() == 'false' and 
        float(r['pa_deg']) == 0.0 and p['pred_class'] == 'RECOVERED' and float(r['recall_pos']) < 0.5):
        print(f"Cell: inc={r['inc_deg']}, fwhm={r['fwhm_kpc']}, sig={r['sigma_v_kms']}, pa={r['pa_deg']}")
        print(f"  pred: a50_pred={p['a50_pred_kms']}, snr_th={p['snr_th_stratum']}, eta_beam={p['eta_beam']}, eta_fit={p['eta_fit']}, a_inj={p['a_inj_physical_kms']}")
        print(f"  result: recall_pos={r['recall_pos']}, recall_multi={r['recall_multi']}, cell_verdict={r['cell_verdict']}")
