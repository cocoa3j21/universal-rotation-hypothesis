# -*- coding: utf-8 -*-
"""
tools/update_final_assets_definitive.py
最新の全検証結果を反映して資産を確定・固定する:
1. v2_method_b_snr_54strata.csv の生成 (方法A vs 方法B 全54条件)
2. v2_positive_control_oversampled_result.json の生成
3. final_report_numbers.json の source_hashes を含む最新更新
4. SHA256SUMS.txt の完全更新 (全13ファイル)
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import hashlib
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import gyro_gravity_theory.injection_pipeline as pipe

OUT_DIR = os.path.join(PROJECT_DIR, "output")

def get_sha256(fname):
    fpath = os.path.join(OUT_DIR, fname)
    if not os.path.exists(fpath):
        return None
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def generate_method_b_table():
    model_thresholds = {
        ("model_1", 0.2): 5.040, ("model_1", 0.5): 4.790, ("model_1", 1.0): 4.522,
        ("model_2", 0.2): 4.735, ("model_2", 0.5): 4.485, ("model_2", 1.0): 4.207,
    }
    models = {
        "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 18.4},
        "model_2": {"M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 9.4},
    }
    dpix = 0.08
    lines = [
        "model_id,fwhm_kpc,sigma_v_kms,inc_deg,snr_opt_raw,snr_eff_method_a,r_a,verdict_a,snr_eff_method_b,r_b,verdict_b,threshold"
    ]

    for mid, info in models.items():
        halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])
        coords = np.arange(-info["r_fov"], info["r_fov"] + dpix / 2, dpix)
        X_obs, Y_obs = np.meshgrid(coords, coords)
        r_max = 0.9 * info["R_kpc"]
        r_min = 0.2 * info["R_kpc"]
        d_v = 0.3 * info["R_kpc"]
        n_rings = 30
        r_edges = np.linspace(0.0, r_max, n_rings + 1)

        for fwhm in [0.2, 0.5, 1.0]:
            fwhm_pix = fwhm / dpix
            sigma_b_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
            sigma_b_kpc = sigma_b_pix * dpix
            a_beam = 4.0 * np.pi * (sigma_b_kpc**2)
            sqrt_a_beam = np.sqrt(a_beam)
            th = model_thresholds[(mid, fwhm)]

            for inc in [30.0, 50.0, 70.0]:
                _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
                mask_disk = (r_disk <= r_max)
                mask_search = (r_disk >= r_min) & (r_disk <= r_max)

                dv = halo.compute_vortex_velocity_perturbation_map(
                    r_disk * np.cos(phi_disk), r_disk * np.sin(phi_disk), [(d_v, 0.0)])
                vlos_vort = dv * np.cos(phi_disk) * np.sin(np.radians(inc))

                s0_raw = vlos_vort * mask_disk.astype(float)
                l2_raw = float(np.sqrt(np.sum(s0_raw**2) * (dpix**2)))

                # Method A (Full disk, rings > 10 pix)
                s0_A = np.copy(s0_raw)
                for k in range(n_rings):
                    m = mask_disk & (r_disk >= r_edges[k]) & (r_disk < r_edges[k+1])
                    if np.sum(m) > 0: s0_A[m] -= np.mean(s0_raw[m])
                    if np.sum(m) > 10:
                        cp = np.cos(phi_disk[m]); sp = np.sin(phi_disk[m])
                        coeff, _, _, _ = np.linalg.lstsq(np.column_stack([cp, sp]), s0_A[m], rcond=None)
                        s0_A[m] -= (coeff[0]*cp + coeff[1]*sp)
                l2_A = float(np.sqrt(np.sum(s0_A**2) * (dpix**2)))

                # Method B (Search mask only + all rings projected)
                s0_B = vlos_vort * mask_search.astype(float)
                for k in range(n_rings):
                    m = mask_search & (r_disk >= r_edges[k]) & (r_disk < r_edges[k+1])
                    if np.sum(m) > 0: s0_B[m] -= np.mean(s0_B[m])
                    if np.sum(m) >= 3:
                        cp = np.cos(phi_disk[m]); sp = np.sin(phi_disk[m])
                        coeff, _, _, _ = np.linalg.lstsq(np.column_stack([cp, sp]), s0_B[m], rcond=None)
                        s0_B[m] -= (coeff[0]*cp + coeff[1]*sp)
                l2_B = float(np.sqrt(np.sum(s0_B**2) * (dpix**2)))

                for sv in [0.5, 1.0, 2.0]:
                    snr_raw = (l2_raw / sqrt_a_beam) / sv
                    snr_A = (l2_A / sqrt_a_beam) / sv
                    snr_B = (l2_B / sqrt_a_beam) / sv

                    r_A = snr_A / th
                    r_B = snr_B / th

                    v_A = "RECOVERABLE" if r_A >= 1.15 else ("BOUNDARY" if r_A >= 0.85 else "UNDETECTABLE")
                    v_B = "RECOVERABLE" if r_B >= 1.15 else ("BOUNDARY" if r_B >= 0.85 else "UNDETECTABLE")

                    lines.append(f"{mid},{fwhm},{sv},{inc},{snr_raw:.4f},{snr_A:.4f},{r_A:.4f},{v_A},{snr_B:.4f},{r_B:.4f},{v_B},{th:.4f}")

    csv_path = os.path.join(OUT_DIR, "v2_method_b_snr_54strata.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"作成完了: {csv_path}")


def generate_positive_control_json():
    pc_data = {
        "condition": "Model 2, FWHM=0.2 kpc, sigma_v=0.5 km/s, inc=50 deg",
        "grid": "oversampled dx=0.01 kpc, downsampled to dpix=0.08 kpc",
        "dilution_factor": 0.2122,
        "a50_empirical_kms": 8.797,
        "a50_analytic_kms": 11.160,
        "s_max_at_a50_kms": 1.866,
        "peak_snr_at_a50": 3.733,
        "inplane_phys_max_kms": 5.1563,
        "ratio_phys_to_a50": 0.5862,
        "notes": "面内物理最大振幅はA50の約0.59倍であり単一ピーク検出限界未満"
    }
    pc_path = os.path.join(OUT_DIR, "v2_positive_control_oversampled_result.json")
    with open(pc_path, "w", encoding="utf-8") as f:
        json.dump(pc_data, f, indent=2, ensure_ascii=False)
    print(f"作成完了: {pc_path}")


def update_final_report_json():
    base_json_path = os.path.join(OUT_DIR, "final_report_numbers.json")
    with open(base_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Update paired seed test condition explicitly
    data["paired_seed_test"]["condition"] = "Model 2, FWHM=0.5 kpc, sigma_v=0.5 km/s, inc=50 deg, theta=4.40"

    # 2. Add oversampled template detector results
    tmf_path = os.path.join(OUT_DIR, "v2_oversampled_template_detector_results.json")
    if os.path.exists(tmf_path):
        with open(tmf_path, "r", encoding="utf-8") as f:
            tmf_res = json.load(f)
        data["template_detector_oversampled"] = {
            "30_deg": {
                **tmf_res["30"],
                "snr_opt_raw": 5.280,
                "snr_eff_method_b": 4.442,
                "diff_opt_minus_inv": round(5.280 - tmf_res["30"]["snr_eff_inverted"], 3),
                "diff_b_minus_inv": round(4.442 - tmf_res["30"]["snr_eff_inverted"], 3)
            },
            "50_deg": {
                **tmf_res["50"],
                "snr_opt_raw": 6.865,
                "snr_eff_method_b": 5.837,
                "diff_opt_minus_inv": round(6.865 - tmf_res["50"]["snr_eff_inverted"], 3),
                "diff_b_minus_inv": round(5.837 - tmf_res["50"]["snr_eff_inverted"], 3)
            },
            "70_deg": {
                **tmf_res["70"],
                "snr_opt_raw": 5.834,
                "snr_eff_method_b": 5.034,
                "diff_opt_minus_inv": round(5.834 - tmf_res["70"]["snr_eff_inverted"], 3),
                "diff_b_minus_inv": round(5.034 - tmf_res["70"]["snr_eff_inverted"], 3)
            }
        }

    # 3. Add standard grid comparison from prior run (N=400)
    data["template_detector_standard_comparison"] = {
        "30_deg": {"max_s_std_kms": 0.9306, "rec_rate": 0.382, "ci95": [0.335, 0.431], "snr_eff_inv": 4.10, "verdict": "UNDETECTED"},
        "50_deg": {"max_s_std_kms": 1.2134, "rec_rate": 0.722, "ci95": [0.676, 0.765], "snr_eff_inv": 4.93, "verdict": "RECOVERED"},
        "70_deg": {"max_s_std_kms": 0.9623, "rec_rate": 0.482, "ci95": [0.433, 0.531], "snr_eff_inv": 4.14, "verdict": "BOUNDARY (CI includes 50%)"}
    }

    # 4. Add positive control oversampled
    pc_path = os.path.join(OUT_DIR, "v2_positive_control_oversampled_result.json")
    with open(pc_path, "r", encoding="utf-8") as f:
        data["positive_control_oversampled"] = json.load(f)

    # 5. Correct Model 3 stats and smoothing ratios
    data["model_3_stats"] = {
        "prediction_recall": 0.9817,
        "robust_recovery_cells": 539,
        "evaluated_cells": 3402,
        "robust_recovery_rate": 0.1584,
        "xi_inplane": 0.28,
        "note": "98.2%は事前予測の再現率であり、ロバスト回収率は15.8%にとどまる"
    }

    data["smoothing_ratios_current"] = {
        "model_1": 0.330,
        "model_2": 0.212,
        "model_3": 0.320
    }

    # 6. Update source hashes with all relevant files
    source_files = [
        "v2_h0_calibration.csv",
        ".v2_snr_thresholds.json",
        "v2_cell_predictions_preregistered.csv",
        "v2_injection_recovery_summary.csv",
        "v2_exact_optimal_snr_54strata.csv",
        "v2_audited_optimal_snr_54strata.csv",
        "v2_method_b_snr_54strata.csv",
        "v2_oversampled_template_detector_results.json",
        "v2_positive_control_oversampled_result.json"
    ]
    data["source_hashes"] = {}
    for sf in source_files:
        sha = get_sha256(sf)
        if sha:
            data["source_hashes"][sf] = sha

    with open(base_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"更新完了: {base_json_path}")


def update_sha256sums():
    target_files = [
        "v2_h0_calibration.csv",
        ".v2_snr_thresholds.json",
        "v2_cell_predictions_preregistered.csv",
        "v2_injection_recovery_summary.csv",
        "v2_exact_optimal_snr_54strata.csv",
        "v2_audited_optimal_snr_54strata.csv",
        "v2_method_b_snr_54strata.csv",
        "v2_oversampled_template_detector_results.json",
        "v2_positive_control_oversampled_result.json",
        "final_report_numbers.json",
        "cell_predictions_preregistered.csv",
        "h0_stratum_calibration.csv",
        "injection_recovery_summary.csv"
    ]

    sums_lines = []
    print("\n" + "=" * 90)
    print("SHA256SUMS 最新チェックサム (全13ファイル):")
    print("=" * 90)
    for fname in target_files:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            sha = h.hexdigest()
            sums_lines.append(f"{sha}  {fname}")
            print(f"{sha}  {fname}")

    sums_path = os.path.join(OUT_DIR, "SHA256SUMS.txt")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sums_lines) + "\n")
    print(f"\nSHA256SUMS.txt を更新完了: {sums_path}")


if __name__ == "__main__":
    generate_method_b_table()
    generate_positive_control_json()
    update_final_report_json()
    update_sha256sums()
