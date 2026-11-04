# -*- coding: utf-8 -*-
"""
tools/summarize_v2_tables.py
Stage 3 v2 / Stage 4 v2 の完了後に以下を集計して表形式で出力する:
1. Model 2 (物理振幅 4.89 km/s) の節1基準 A50 表 (27条件) と v1 / v2 実測回収率の対照表
2. 物理許容半径 r_tol = xi_c + sigma_beam と Stage 3 v2 H0 偶然一致率 (f_chance_mean, f_chance_95, fpr_h0) の表
3. 幾何誤差ありセルにおける Matched Null 補正 (null_recall_matched, null_upper95) の効果集計
"""

import os
import csv
import json
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, "output")


def main():
    s3_csv = os.path.join(OUT_DIR, "v2_h0_calibration.csv")
    pred_csv = os.path.join(OUT_DIR, "v2_cell_predictions_preregistered.csv")
    s4_csv = os.path.join(OUT_DIR, "v2_injection_recovery_summary.csv")
    v1_m2_csv = os.path.join(OUT_DIR, "injection_recovery_model2_summary.csv")

    with open(s3_csv, "r", encoding="utf-8") as f:
        s3_rows = list(csv.DictReader(f))
    with open(pred_csv, "r", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))
    with open(s4_csv, "r", encoding="utf-8") as f:
        s4_rows = list(csv.DictReader(f))
    with open(v1_m2_csv, "r", encoding="utf-8") as f:
        v1_m2_rows = list(csv.DictReader(f))

    print("=" * 90)
    print("Table 1: 物理許容半径 r_tol = xi_c + sigma_beam と Stage 3 v2 H0 偶然一致率 (arm_method=none)")
    print("=" * 90)
    print(f"{'Model':>8} | {'FWHM':>4} | {'Inc':>4} | {'xi_c':>7} | {'s_beam':>7} | {'r_tol_kpc':>9} | {'r_tol_pix':>9} | {'SNR_th':>6} | {'fpr_h0':>6} | {'f_chance_mean':>13} | {'f_chance_95':>11} | {'f_area':>8}")
    print("-" * 120)
    seen = set()
    for r in s3_rows:
        if r["arm_method"] != "none" or float(r["sigma_v_kms"]) != 0.5:
            continue
        k = (r["model_id"], float(r["fwhm_kpc"]), float(r["inc_deg"]))
        if k in seen:
            continue
        seen.add(k)
        print(f"{r['model_id']:>8} | {float(r['fwhm_kpc']):4.1f} | {float(r['inc_deg']):4.0f} | {float(r['xi_c_kpc']):7.4f} | {float(r['sigma_beam_kpc']):7.4f} | {float(r['r_tol_kpc']):9.4f} | {float(r['r_tol_pix']):9.2f} | {float(r['snr_th_v2']):6.2f} | {float(r['fpr_h0']):6.4f} | {float(r['f_chance_mean']):13.6f} | {float(r['f_chance_95']):11.6f} | {float(r['f_area_ratio']):8.6f}")

    print("\n" + "=" * 90)
    print("Table 2: Model 2 (A_phys = 4.89 km/s) 節1基準 A50・比・v1/v2 実測回収率 (全27基本条件)")
    print("=" * 90)
    print(f"{'FWHM':>4} | {'sv':>4} | {'Inc':>4} | {'eta_b':>6} | {'A50(4.5)':>8} | {'Ratio':>6} | {'A50(cal)':>8} | {'Rat(cal)':>8} | {'PredClass':>10} | {'v1_MaxRec':>9} | {'v2_CleanRec':>11} | {'v2_MeanRec':>10} | {'v2_RobCells':>11}")
    print("-" * 130)

    for fwhm in [0.2, 0.5, 1.0]:
        for sv in [0.5, 1.0, 2.0]:
            for inc in [30.0, 50.0, 70.0]:
                # Match pred & s4 for model_2, nv > 0
                sub_p = [p for p in pred_rows if p["model_id"] == "model_2" and int(p["n_vortex_inj"]) > 0 and float(p["fwhm_kpc"]) == fwhm and float(p["sigma_v_kms"]) == sv and float(p["inc_deg"]) == inc and p["arm_method"] == "none"]
                p0 = sub_p[0]

                sub_s4 = [r for r in s4_rows if r["model_id"] == "model_2" and int(r["n_vortex_inj"]) > 0 and float(r["fwhm_kpc"]) == fwhm and float(r["sigma_v_kms"]) == sv and float(r["inc_deg"]) == inc]
                sub_s4_clean = [r for r in sub_s4 if int(r["n_vortex_inj"]) == 1 and float(r["a_los3_kms"]) == 0.0 and r["geom_error_flag"] == "False" and float(r["pa_deg"]) == 0.0]
                sub_v1 = [r for r in v1_m2_rows if int(r["n_vortex_inj"]) > 0 and float(r["fwhm_kpc"]) == fwhm and float(r["sigma_v_kms"]) == sv and float(r["inc_deg"]) == inc]

                v1_max = max(float(r["recall_pos"]) for r in sub_v1)
                v2_clean = float(sub_s4_clean[0]["recall_pos"]) if sub_s4_clean else 0.0
                v2_mean = float(np.mean([float(r["recall_pos"]) for r in sub_s4]))
                v2_rob = sum(1 for r in sub_s4 if r["cell_verdict"] == "CELL_ROBUST_RECOVERED")

                print(f"{fwhm:4.1f} | {sv:4.1f} | {inc:4.0f} | {float(p0['eta_beam_proj']):6.4f} | {float(p0['a50_sec1_kms']):8.2f} | {float(p0['ratio_sec1']):6.3f} | {float(p0['a50_cal_kms']):8.2f} | {float(p0['ratio_cal']):8.3f} | {p0['pred_class']:>10} | {v1_max:9.4f} | {v2_clean:11.4f} | {v2_mean:10.4f} | {v2_rob:4d}/{len(sub_s4):<4d}")

    print("\n" + "=" * 90)
    print("Table 3: 幾何誤差 (geom_error_flag) 別の Matched Null 補正効果集計 (全モデル)")
    print("=" * 90)
    for mid in ["model_1", "model_2", "model_3"]:
        for geom_str in ["False", "True"]:
            sub = [r for r in s4_rows if r["model_id"] == mid and int(r["n_vortex_inj"]) > 0 and r["geom_error_flag"] == geom_str]
            raw_ge50 = sum(1 for r in sub if float(r["recall_pos"]) >= 0.50)
            rob_rec = sum(1 for r in sub if r["cell_verdict"] == "CELL_ROBUST_RECOVERED")
            rej_null = sum(1 for r in sub if r["cell_verdict"] == "CELL_FAIL_MATCHED_NULL")
            mean_null = np.mean([float(r["null_recall_matched"]) for r in sub])
            max_null = max(float(r["null_recall_matched"]) for r in sub)
            print(f"  {mid} | geom={geom_str:5s} (N={len(sub):4d}): Raw>=0.50: {raw_ge50:4d}, Robust(recall>null_upper95): {rob_rec:4d}, Rejected_by_MatchedNull: {rej_null:4d} | null_recall mean={mean_null:.4f}, max={max_null:.4f}")


if __name__ == "__main__":
    main()
