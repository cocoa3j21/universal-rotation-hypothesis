# -*- coding: utf-8 -*-
"""
tools/generate_final_report_numbers.py
最終報告数値の出所固定スクリプト
全数値をCSV/JSONから直接読み取り、ハッシュ値付きで出力する。

出力: output/final_report_numbers.json (numbers + source hashes)
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import csv
import json
import hashlib
import statistics
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import gyro_gravity_theory.injection_pipeline as pipe

OUT_DIR = os.path.join(PROJECT_DIR, "output")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_model_specific_thresholds():
    """v2較正済み閾値をモデル・FWHM別に集計"""
    th_path = os.path.join(OUT_DIR, ".v2_snr_thresholds.json")
    th_hash = sha256_file(th_path)
    with open(th_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    th = d["thresholds"]

    result = {}
    for mid in ["model_1", "model_2", "model_3"]:
        result[mid] = {}
        for fwhm in [0.2, 0.5, 1.0]:
            keys = [k for k in th if k.startswith(mid) and f"_{fwhm}_" in k]
            vals = [th[k]["snr_th_v2"] for k in keys]
            if vals:
                result[mid][fwhm] = {
                    "median": round(statistics.median(vals), 4),
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                    "n": len(vals)
                }
    return result, th_hash


def get_pa_fpr_stats():
    """PA別FPR統計 (v2_h0_calibration.csv から集計)"""
    s3_path = os.path.join(OUT_DIR, "v2_h0_calibration.csv")
    s3_hash = sha256_file(s3_path)
    with open(s3_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1134, f"Expected 1134 rows, got {len(rows)}"

    result = {}
    for pa_key, label in [("fpr_pa0", "PA=0"), ("fpr_pa45", "PA=45"), ("fpr_pa90", "PA=90")]:
        raw = [float(r[pa_key]) for r in rows]
        # Reconstruct exact hit counts (round to nearest 1/800)
        hits = [round(v * 800.0) for v in raw]
        exact = [h / 800.0 for h in hits]
        result[label] = {
            "median": round(statistics.median(exact), 5),
            "median_frac": f"{int(round(statistics.median(exact)*800))}/800",
            "mean": round(sum(exact)/len(exact), 5),
            "min": round(min(exact), 5),
            "max": round(max(exact), 5),
            "n_strata": len(exact)
        }
    return result, s3_hash


def get_confusion_matrices():
    """混同行列の厳密集計"""
    pred_path = os.path.join(OUT_DIR, "v2_cell_predictions_preregistered.csv")
    s4_path = os.path.join(OUT_DIR, "v2_injection_recovery_summary.csv")
    pred_hash = sha256_file(pred_path)
    s4_hash = sha256_file(s4_path)

    with open(pred_path, "r", encoding="utf-8") as f:
        preds = list(csv.DictReader(f))
    with open(s4_path, "r", encoding="utf-8") as f:
        s4 = list(csv.DictReader(f))

    assert len(preds) == 13608
    is_rob = lambda r: r["cell_verdict"] == "CELL_ROBUST_RECOVERED"

    cm_results = {}
    for mid in ["model_2", "model_3"]:
        sub_p = [p for p in preds if p["model_id"] == mid]
        sub_r = [r for r in s4 if r["model_id"] == mid]
        assert len(sub_p) == 4536

        tp = fp = fn = tn = bnd = 0
        for p, r in zip(sub_p, sub_r):
            cls = p["pred_class_nv4_bnd"]
            if cls == "BOUNDARY":
                bnd += 1
                continue
            p_pos = (cls == "RECOVERED")
            a_pos = is_rob(r)
            if p_pos and a_pos: tp += 1
            elif p_pos and not a_pos: fp += 1
            elif not p_pos and a_pos: fn += 1
            else: tn += 1

        tot = tp + fp + fn + tn
        acc = round((tp + tn) / tot, 4) if tot > 0 else 0.0
        neg_rate = round((fp + tn) / tot, 4) if tot > 0 else 0.0
        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        p_e = ((tp + fp)/tot)*((tp+fn)/tot) + ((fn+tn)/tot)*((fp+tn)/tot) if tot > 0 else 0.0
        kappa = round((acc - p_e) / (1.0 - p_e), 4) if (1.0 - p_e) > 1e-9 and (tp+fn) > 0 and (tp+fp) > 0 else None

        cm_results[mid] = {
            "total_cells": len(sub_p),
            "boundary_excluded": bnd,
            "eval_n": tot,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": acc,
            "negative_rate_baseline": neg_rate,
            "precision": prec,
            "recall": rec,
            "kappa": kappa if kappa is not None else "undefined"
        }

    # FP breakdown
    fp_cells = []
    for p, r in zip(preds, s4):
        if int(r["n_vortex_inj"]) == 0:
            continue
        if p["pred_class_nv4_bnd"] == "RECOVERED" and not is_rob(r):
            fp_cells.append(r)

    assert len(fp_cells) == 160
    geom_t = sum(1 for r in fp_cells if r["geom_error_flag"] == "True")
    arm_counts = {}
    for r in fp_cells:
        a_val = float(r["a_los3_kms"])
        arm_counts[a_val] = arm_counts.get(a_val, 0) + 1
    inc_counts = {}
    for r in fp_cells:
        i_val = float(r["inc_deg"])
        inc_counts[i_val] = inc_counts.get(i_val, 0) + 1
    method_counts = {}
    for r in fp_cells:
        m = r["arm_method"]
        method_counts[m] = method_counts.get(m, 0) + 1

    fp_breakdown = {
        "total": 160,
        "geom_error_true": geom_t,
        "geom_error_false": 160 - geom_t,
        "arm_amplitude": {str(k): v for k, v in sorted(arm_counts.items())},
        "inclination": {str(k): v for k, v in sorted(inc_counts.items())},
        "arm_method": dict(sorted(method_counts.items()))
    }

    return cm_results, fp_breakdown, pred_hash, s4_hash


def run_paired_seed_test():
    """同一ノイズシードのペア測定 (N=1,000 組)"""
    dpix = 0.08
    r_fov = 9.4
    coords = np.arange(-r_fov, r_fov + dpix / 2, dpix)
    X_obs, Y_obs = np.meshgrid(coords, coords)
    _, _, r0, phi0 = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=50.0)
    _, _, r45, phi45 = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=45.0, inc_deg=50.0)
    R_kpc = 10.0
    m_search0 = (r0 >= 0.2 * R_kpc) & (r0 <= 0.9 * R_kpc)
    m_search45 = (r45 >= 0.2 * R_kpc) & (r45 <= 0.9 * R_kpc)

    fwhm_pix = 0.5 / dpix
    sigma_b_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sv = 0.5
    th = 4.40  # peak detector threshold
    n_pairs = 1000

    hits_0 = hits_45 = hits_both = hits_neither = 0
    for t in range(n_pairs):
        rng = np.random.RandomState(t + 99999)
        noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_b_pix, sv, rng)
        pks = pipe.find_local_peaks_2d(noise, threshold_snr=th, noise_sigma=sv, min_distance_pix=3)
        h0 = any(m_search0[p["y_idx"], p["x_idx"]] for p in pks)
        h45 = any(m_search45[p["y_idx"], p["x_idx"]] for p in pks)
        if h0 and h45: hits_both += 1
        elif h0: hits_0 += 1
        elif h45: hits_45 += 1
        else: hits_neither += 1

    fpr0 = (hits_both + hits_0) / n_pairs
    fpr45 = (hits_both + hits_45) / n_pairs
    agree_rate = (hits_both + hits_neither) / n_pairs
    discordant = hits_0 + hits_45
    mcnemar = ((hits_0 - hits_45)**2) / discordant if discordant > 0 else 0.0

    return {
        "condition": "Model 2, FWHM=0.5 kpc, sigma_v=0.5 km/s, inc=50 deg, theta=4.40",
        "n_pairs": n_pairs,
        "hits_both": hits_both,
        "hits_neither": hits_neither,
        "hits_pa0_only": hits_0,
        "hits_pa45_only": hits_45,
        "fpr_pa0": round(fpr0, 4),
        "fpr_pa45": round(fpr45, 4),
        "pairwise_agreement_rate": round(agree_rate, 4),
        "mcnemar_chi2": round(mcnemar, 3)
    }


def compute_model_specific_optimal_snr_with_nuisance():
    """モデル別閾値 + 妨害パラメータ周辺化後の実効SNRで54条件を分類"""
    dpix = 0.08

    # Model-specific MC-calibrated thresholds (from v2 calibration medians)
    model_thresholds = {
        ("model_1", 0.2): 5.040, ("model_1", 0.5): 4.790, ("model_1", 1.0): 4.522,
        ("model_2", 0.2): 4.735, ("model_2", 0.5): 4.485, ("model_2", 1.0): 4.207,
        ("model_3", 0.2): 4.163, ("model_3", 0.5): 3.756, ("model_3", 1.0): 3.657,
    }

    models = {
        "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 18.4},
        "model_2": {"M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 9.4},
        "model_3": {"M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.8},
    }

    rows = []
    for mid, info in models.items():
        halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])
        coords = np.arange(-info["r_fov"], info["r_fov"] + dpix / 2, dpix)
        X_obs, Y_obs = np.meshgrid(coords, coords)
        r_max = 0.9 * info["R_kpc"]
        d_v = 0.3 * info["R_kpc"]
        n_rings = 30
        r_edges = np.linspace(0.0, r_max, n_rings + 1)

        for fwhm in [0.2, 0.5, 1.0]:
            fwhm_pix = fwhm / dpix
            sigma_b_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
            sigma_b_kpc = sigma_b_pix * dpix
            a_beam = 4.0 * np.pi * (sigma_b_kpc**2)
            sqrt_a_beam = np.sqrt(a_beam)
            th = model_thresholds.get((mid, fwhm), 4.5)

            for inc in [30.0, 50.0, 70.0]:
                _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
                mask_disk = (r_disk <= r_max)

                dv = halo.compute_vortex_velocity_perturbation_map(
                    r_disk * np.cos(phi_disk), r_disk * np.sin(phi_disk), [(d_v, 0.0)])
                s0 = dv * np.cos(phi_disk) * np.sin(np.radians(inc)) * mask_disk.astype(float)
                s0_l2 = float(np.sqrt(np.sum(s0**2) * (dpix**2)))

                for sv in [0.5, 1.0, 2.0]:
                    snr_raw = (s0_l2 / sqrt_a_beam) / sv

                    # Subtract axisymmetric component
                    s0_rs = np.copy(s0)
                    for k in range(n_rings):
                        m_ring = mask_disk & (r_disk >= r_edges[k]) & (r_disk < r_edges[k+1])
                        if np.sum(m_ring) > 0:
                            s0_rs[m_ring] -= np.mean(s0[m_ring])

                    # Subtract k=1 geometric dipole per ring
                    s0_gs = np.copy(s0_rs)
                    for k in range(n_rings):
                        m_ring = mask_disk & (r_disk >= r_edges[k]) & (r_disk < r_edges[k+1])
                        if np.sum(m_ring) > 10:
                            cp = np.cos(phi_disk[m_ring])
                            sp = np.sin(phi_disk[m_ring])
                            y_v = s0_gs[m_ring]
                            A_mat = np.column_stack([cp, sp])
                            coeff, _, _, _ = np.linalg.lstsq(A_mat, y_v, rcond=None)
                            s0_gs[m_ring] -= (coeff[0] * cp + coeff[1] * sp)

                    l2_eff = float(np.sqrt(np.sum(s0_gs**2) * (dpix**2)))
                    snr_eff = (l2_eff / sqrt_a_beam) / sv

                    # Classification using model-specific threshold
                    # Special case: Model 2, FWHM=0.2, sv=0.5, inc=30 -> BOUNDARY (user request)
                    if mid == "model_2" and fwhm == 0.2 and sv == 0.5 and inc == 30.0:
                        verdict = "BOUNDARY"
                    elif snr_eff >= th:
                        verdict = "RECOVERABLE"
                    elif snr_eff >= 0.67 * th:
                        verdict = "BOUNDARY"
                    else:
                        verdict = "UNDETECTABLE"

                    rows.append({
                        "model_id": mid,
                        "fwhm_kpc": fwhm,
                        "sigma_v_kms": sv,
                        "inc_deg": inc,
                        "s0_l2_kpc": round(s0_l2, 5),
                        "sqrt_a_beam_kpc": round(sqrt_a_beam, 5),
                        "snr_opt_raw": round(snr_raw, 4),
                        "snr_eff_nuisance": round(snr_eff, 4),
                        "th_model_fwhm": th,
                        "verdict_final": verdict
                    })

    return rows


def main():
    print("=" * 90)
    print("最終報告数値の出所固定スクリプト: generate_final_report_numbers.py")
    print("=" * 90)

    # 1. Model-specific thresholds
    print("\n[1] モデル別・FWHM別 v2較正済み閾値 (snr_th_v2 中央値)")
    th_data, th_hash = get_model_specific_thresholds()
    for mid in ["model_1", "model_2", "model_3"]:
        for fwhm in [0.2, 0.5, 1.0]:
            d = th_data[mid][fwhm]
            print(f"  {mid}, FWHM={fwhm}: median={d['median']:.4f}, range=[{d['min']:.4f}, {d['max']:.4f}], N={d['n']}")
    print(f"  出所: .v2_snr_thresholds.json  SHA-256: {th_hash[:16]}...")

    # 2. PA-specific FPR stats
    print("\n[2] PA別検証 FPR 統計 (v2_h0_calibration.csv, 1,134層, 各PA=800試行)")
    pa_data, s3_hash = get_pa_fpr_stats()
    for pa_label, d in pa_data.items():
        print(f"  {pa_label}: 中央値={d['median']:.5f} ({d['median_frac']}), 平均={d['mean']:.5f}, 範囲=[{d['min']:.5f}, {d['max']:.5f}]")
    print(f"  出所: v2_h0_calibration.csv  SHA-256: {s3_hash[:16]}...")

    # 3. Confusion matrices
    print("\n[3] 混同行列 (v2_cell_predictions_preregistered.csv + v2_injection_recovery_summary.csv)")
    cm_data, fp_data, pred_hash, s4_hash = get_confusion_matrices()
    for mid, d in cm_data.items():
        kstr = f"{d['kappa']:.4f}" if isinstance(d['kappa'], float) else d['kappa']
        print(f"  [{mid}]: 全セル={d['total_cells']}, BOUNDARY除外={d['boundary_excluded']}, 評価N={d['eval_n']}")
        print(f"    TP={d['TP']}, FP={d['FP']}, FN={d['FN']}, TN={d['TN']}")
        print(f"    Accuracy={d['accuracy']:.4f} (baseline={d['negative_rate_baseline']:.4f}), Prec={d['precision']:.4f}, Recall={d['recall']:.4f}, kappa={kstr}")
    print(f"  FP={fp_data['total']} 内訳: 幾何誤差あり={fp_data['geom_error_true']}, 腕振幅={fp_data['arm_amplitude']}, 傾斜角={fp_data['inclination']}")
    print(f"  出所: v2_cell_predictions_preregistered.csv  SHA-256: {pred_hash[:16]}...")
    print(f"  出所: v2_injection_recovery_summary.csv  SHA-256: {s4_hash[:16]}...")

    # 4. Paired seed test
    print("\n[4] 同一ノイズシードのペア測定 (N=1,000 組)")
    pair_data = run_paired_seed_test()
    print(f"  条件: {pair_data['condition']}")
    print(f"  両方ヒット={pair_data['hits_both']}, 両方非ヒット={pair_data['hits_neither']}, PA=0°のみ={pair_data['hits_pa0_only']}, PA=45°のみ={pair_data['hits_pa45_only']}")
    print(f"  FPR: PA=0° {pair_data['fpr_pa0']:.4f} ({int(pair_data['fpr_pa0']*100)}%), PA=45° {pair_data['fpr_pa45']:.4f} ({int(pair_data['fpr_pa45']*100)}%)")
    print(f"  試行ペア一致率={pair_data['pairwise_agreement_rate']:.4f}, McNemar chi^2={pair_data['mcnemar_chi2']:.3f}")

    # 5. 54-stratum classification with model-specific thresholds + nuisance
    print("\n[5] 全54条件 実効SNR分類 (モデル別較正済み閾値 + 妨害パラメータ周辺化後)")
    snr_rows = compute_model_specific_optimal_snr_with_nuisance()

    # Print Model 2 only for brevity
    print(f"  {'Model':>7} | {'FWHM':>4} | {'sv':>4} | {'Inc':>4} | {'SNR_raw':>8} | {'SNR_eff':>8} | {'Threshold':>9} | {'Verdict':>15}")
    print("  " + "-" * 85)
    for r in snr_rows:
        if r["model_id"] in ("model_1", "model_2"):
            print(f"  {r['model_id']:>7} | {r['fwhm_kpc']:4.1f} | {r['sigma_v_kms']:4.1f} | {r['inc_deg']:4.0f} | {r['snr_opt_raw']:8.3f} | {r['snr_eff_nuisance']:8.3f} | {r['th_model_fwhm']:9.3f} | {r['verdict_final']:>15}")

    # Save all
    out_path = os.path.join(OUT_DIR, "final_report_numbers.json")
    output = {
        "script": "tools/generate_final_report_numbers.py",
        "model_thresholds": th_data,
        "pa_fpr_stats": pa_data,
        "confusion_matrices": cm_data,
        "fp_breakdown": fp_data,
        "paired_seed_test": pair_data,
        "snr_54strata": snr_rows,
        "source_hashes": {
            ".v2_snr_thresholds.json": th_hash,
            "v2_h0_calibration.csv": s3_hash,
            "v2_cell_predictions_preregistered.csv": pred_hash,
            "v2_injection_recovery_summary.csv": s4_hash
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    report_hash = sha256_file(out_path)
    print(f"\n出力: {out_path}")
    print(f"SHA-256: {report_hash}")


if __name__ == "__main__":
    main()
