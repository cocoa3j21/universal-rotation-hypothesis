# -*- coding: utf-8 -*-
import csv

with open('output/h0_stratum_calibration.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

zero_fpr_rows = [r for r in rows if float(r['fpr_h0']) == 0.0]
print(f"Total strata with fpr_h0 == 0: {len(zero_fpr_rows)} / {len(rows)}")

for r in zero_fpr_rows:
    print(f"Stratum {r['stratum_id']}: model={r['model_id']}, inc={r['inc_deg']}, FWHM={r['fwhm_kpc']}, s={r['sigma_v']}, geom={r['geom_error_flag']}, a3={r['a_los3']}, meth={r['arm_method']}, n_trials={r['n_h0_trials']}, delta_ts_mf_th={r['delta_bic_th_stratum']}, fpr_upper={r['fpr_h0_upper95']}")
