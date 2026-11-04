# -*- coding: utf-8 -*-
"""
tools/run_model2_h0_calibration.py — Model 2 専用 H0較正・層別プール帰無分布計算
(全378層 × 4,401試行 = 1,663,578試行)
"""

import os
import sys
import time
import json
import csv
import hashlib
from multiprocessing import Pool, cpu_count
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

MODELS = {
    "model_2": {
        "M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 6.0, "dpix": 0.125,
        "a_inj_physical": 4.89, "v0_rv": 107.88
    }
}

RESOLUTIONS_KPC = [0.2, 0.5, 1.0]
NOISES_KMS = [0.5, 1.0, 2.0]
GEOM_ERRORS = [False, True]
INCLINATIONS = [30.0, 50.0, 70.0]
POSITION_ANGLES = [0.0, 45.0, 90.0]

ARM_LEVELS = [{"a_los3": 0.0, "method": "none"}]
for a3 in [1.0, 3.0]:
    for meth in ["raw", "remove_k3", "arm_fit"]:
        ARM_LEVELS.append({"a_los3": a3, "method": meth})
assert len(ARM_LEVELS) == 7


def generate_strata():
    strata = []
    stratum_id = 0
    model_id = "model_2"
    for inc in INCLINATIONS:
        for fwhm in RESOLUTIONS_KPC:
            for sigma_v in NOISES_KMS:
                for geom_err in GEOM_ERRORS:
                    for arm in ARM_LEVELS:
                        cells = []
                        for pa in POSITION_ANGLES:
                            cells.append({
                                "model_id": model_id,
                                "inc_deg": inc,
                                "pa_deg": pa,
                                "fwhm_kpc": fwhm,
                                "sigma_v": sigma_v,
                                "geom_err": geom_err,
                                "a_los3": arm["a_los3"],
                                "arm_method": arm["method"]
                            })
                        strata.append({
                            "stratum_id": stratum_id,
                            "model_id": model_id,
                            "inc_deg": inc,
                            "fwhm_kpc": fwhm,
                            "sigma_v": sigma_v,
                            "geom_err": geom_err,
                            "a_los3": arm["a_los3"],
                            "arm_method": arm["method"],
                            "cells": cells
                        })
                        stratum_id += 1
    return strata


def process_stratum_h0(stratum):
    model_info = MODELS[stratum["model_id"]]
    halo = pipe.TFHaloModel(M_msun=model_info["M_msun"], R_kpc=model_info["R_kpc"], m_eV=model_info["m_eV"])

    dpix = model_info["dpix"]
    coords = np.arange(-model_info["r_fov"], model_info["r_fov"] + dpix/2, dpix)
    center_idx = len(coords) // 2
    X_obs, Y_obs = np.meshgrid(coords, coords)
    fwhm_pix = stratum["fwhm_kpc"] / dpix
    sigma_v = stratum["sigma_v"]

    r_disk_min = 0.2 * model_info["R_kpc"]
    r_disk_max = 0.9 * model_info["R_kpc"]
    d_vortex = 0.3 * model_info["R_kpc"]
    r_tol_kpc = pipe.compute_tolerance_radius(stratum["model_id"], stratum["fwhm_kpc"], model_info["R_kpc"])
    r_tol_pix = r_tol_kpc / dpix

    sample_pix_list = []
    n_samples = 8
    for s_idx in range(n_samples):
        phi_s = s_idx * (2.0 * np.pi / n_samples)
        xs_kpc = d_vortex * np.cos(phi_s)
        ys_kpc = d_vortex * np.sin(phi_s)
        px = center_idx + int(round(xs_kpc / dpix))
        py = center_idx + int(round(ys_kpc / dpix))
        sample_pix_list.append((px, py))

    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))

    n_cal_per_cell = 667
    n_val_per_cell = 800

    cell_contexts = []
    for cell in stratum["cells"]:
        inc_true = cell["inc_deg"]
        pa_true = cell["pa_deg"]
        v_sys_true = 100.0

        _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_true, inc_deg=inc_true)
        mask_disk = (r_disk_true <= model_info["R_kpc"] * 0.9)

        if cell["geom_err"]:
            delta_inc, delta_pa, delta_sys = 2.0, 2.0, 2.0
        else:
            delta_inc, delta_pa, delta_sys = 0.0, 0.0, 0.0

        inc_obs = inc_true + delta_inc
        pa_obs = pa_true + delta_pa
        v_sys_obs = v_sys_true + delta_sys
        _, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_obs, inc_deg=inc_obs)

        v0_true = halo.v_0(r_disk_true)
        v_los_unpert = pipe.project_vlos(v0_true, 0.0, phi_disk_true, inc_deg=inc_true, v_sys=v_sys_true)

        if cell["a_los3"] > 0:
            arm_residual_inj = cell["a_los3"] * np.cos(3.0 * phi_disk_true)
        else:
            arm_residual_inj = 0.0

        v_conv_signal = pipe.convolve_beam_normalized(v_los_unpert + arm_residual_inj, fwhm_pix=fwhm_pix, mask=mask_disk)
        mask_search = (r_disk_obs >= r_disk_min) & (r_disk_obs <= r_disk_max)

        cell_contexts.append({
            "cell": cell,
            "inc_obs": inc_obs,
            "v_sys_obs": v_sys_obs,
            "r_disk_obs": r_disk_obs,
            "phi_disk_obs": phi_disk_obs,
            "v_conv_signal": v_conv_signal,
            "mask_search": mask_search
        })

    # Step 1: Calibration (2001 trials)
    cal_max_snrs = []
    for ctx in cell_contexts:
        cell = ctx["cell"]
        inc_obs = ctx["inc_obs"]
        v_sys_obs = ctx["v_sys_obs"]
        r_disk_obs = ctx["r_disk_obs"]
        phi_disk_obs = ctx["phi_disk_obs"]
        v_conv_signal = ctx["v_conv_signal"]
        mask_search = ctx["mask_search"]

        for trial in range(n_cal_per_cell):
            seed_val = int(hashlib.md5(f"h0_m2_cal_{stratum['stratum_id']}_{cell['inc_deg']}_{cell['pa_deg']}_{trial}".encode()).hexdigest()[:8], 16)
            rng_cal = np.random.RandomState(seed_val)

            noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng_cal)
            v_obs = v_conv_signal + noise

            _, _, res_raw = pipe.fit_axisymmetric_disk_rings(
                r_disk_obs, phi_disk_obs, v_obs, inc_deg=inc_obs, v_sys=v_sys_obs, n_rings=15
            )

            meth = cell["arm_method"]
            if meth == "none" or meth == "raw":
                res_proc = res_raw
            elif meth == "remove_k3":
                res_proc, _ = pipe.remove_harmonic_k3_rings(r_disk_obs, phi_disk_obs, res_raw, n_rings=15)
            elif meth == "arm_fit":
                res_proc, _, _, _ = pipe.fit_spiral_arm_direct_c(r_disk_obs, phi_disk_obs, res_raw, inc_deg=inc_obs)
            else:
                res_proc = res_raw

            peaks = pipe.find_local_peaks_2d(res_proc, threshold_snr=1.0, noise_sigma=sigma_v, min_distance_pix=3)
            peaks_search = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]

            if len(peaks_search) > 0:
                cal_max_snrs.append(max(p["snr"] for p in peaks_search))
            else:
                cal_max_snrs.append(0.0)

    p95_snr = np.percentile(cal_max_snrs, 95.0)
    snr_threshold = float(max(4.0, p95_snr))

    # Step 2: Independent validation (2400 trials)
    stratum_matches = 0
    total_val_trials = n_val_per_cell * len(cell_contexts)
    val_delta_ts_mfs = []

    for ctx in cell_contexts:
        cell = ctx["cell"]
        inc_obs = ctx["inc_obs"]
        v_sys_obs = ctx["v_sys_obs"]
        r_disk_obs = ctx["r_disk_obs"]
        phi_disk_obs = ctx["phi_disk_obs"]
        v_conv_signal = ctx["v_conv_signal"]
        mask_search = ctx["mask_search"]

        for trial in range(n_val_per_cell):
            seed_val = int(hashlib.md5(f"h0_m2_val_{stratum['stratum_id']}_{cell['inc_deg']}_{cell['pa_deg']}_{trial}".encode()).hexdigest()[:8], 16)
            rng_val = np.random.RandomState(seed_val)

            noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng_val)
            v_obs = v_conv_signal + noise

            _, _, res_raw = pipe.fit_axisymmetric_disk_rings(
                r_disk_obs, phi_disk_obs, v_obs, inc_deg=inc_obs, v_sys=v_sys_obs, n_rings=15
            )

            meth = cell["arm_method"]
            if meth == "none" or meth == "raw":
                res_proc = res_raw
            elif meth == "remove_k3":
                res_proc, _ = pipe.remove_harmonic_k3_rings(r_disk_obs, phi_disk_obs, res_raw, n_rings=15)
            elif meth == "arm_fit":
                res_proc, _, _, _ = pipe.fit_spiral_arm_direct_c(r_disk_obs, phi_disk_obs, res_raw, inc_deg=inc_obs)
            else:
                res_proc = res_raw

            peaks = pipe.find_local_peaks_2d(res_proc, threshold_snr=snr_threshold, noise_sigma=sigma_v, min_distance_pix=3)
            peaks_search = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]

            mask_valid = mask_search & (~np.isnan(res_proc))
            _, _, _, delta_ts_mf = pipe.compute_empirical_matched_filter_stat(res_proc, sigma_beam_pix, mask_valid)
            val_delta_ts_mfs.append(delta_ts_mf)

            if len(peaks_search) > 0:
                is_hit = False
                for p_idx, (sx, sy) in enumerate(sample_pix_list):
                    for pk in peaks_search:
                        dist_sq = (pk["x_idx"] - sx)**2 + (pk["y_idx"] - sy)**2
                        if dist_sq <= r_tol_pix**2:
                            is_hit = True
                            break
                    if is_hit:
                        break
                if is_hit:
                    stratum_matches += 1

    p_fp = stratum_matches / float(total_val_trials)
    delta_ts_mf_th_stratum = float(np.percentile(val_delta_ts_mfs, 95.0))

    z_alpha = 1.95996
    denom = 1.0 + (z_alpha**2) / total_val_trials
    center_adj = (p_fp + (z_alpha**2) / (2.0 * total_val_trials)) / denom
    margin = (z_alpha * np.sqrt((p_fp * (1.0 - p_fp) + (z_alpha**2) / (4.0 * total_val_trials)) / total_val_trials)) / denom
    fpr_h0_upper95 = float(center_adj + margin)

    return {
        "stratum_id": stratum["stratum_id"],
        "model_id": stratum["model_id"],
        "inc_deg": stratum["inc_deg"],
        "fwhm_kpc": stratum["fwhm_kpc"],
        "sigma_v": stratum["sigma_v"],
        "geom_error_flag": stratum["geom_err"],
        "a_los3": stratum["a_los3"],
        "arm_method": stratum["arm_method"],
        "n_h0_trials": len(cal_max_snrs) + total_val_trials,
        "f_chance_95": p_fp,
        "f_chance_95_upper": fpr_h0_upper95,
        "f_chance_99": p_fp,
        "f_chance_99_upper": fpr_h0_upper95,
        "delta_bic_th_stratum": delta_ts_mf_th_stratum,
        "fpr_h0": p_fp,
        "fpr_h0_upper95": fpr_h0_upper95,
        "_snr_threshold": snr_threshold
    }


def main():
    print("=" * 80)
    print("STAGE 3 (MODEL 2): H0較正・層別帰無分布計算 (全378層 × 4,401試行)")
    print("=" * 80)

    strata = generate_strata()
    t0 = time.time()
    n_workers = min(cpu_count(), 16)
    print(f"並列ワーカー数: {n_workers} コア, 全378層")

    with Pool(processes=n_workers) as pool:
        results = pool.map(process_stratum_h0, strata)

    elapsed_sec = time.time() - t0
    print(f"Model 2 の全378層 H0較正完了: 所要時間 {elapsed_sec:.1f} 秒 ({elapsed_sec/60.0:.2f} 分)")

    out_dir = os.path.join(PROJECT_DIR, "output")
    out_path = os.path.join(out_dir, "h0_model2_calibration.csv")
    fieldnames = [
        "stratum_id", "model_id", "inc_deg", "fwhm_kpc", "sigma_v",
        "geom_error_flag", "a_los3", "arm_method",
        "n_h0_trials", "f_chance_95", "f_chance_95_upper",
        "f_chance_99", "f_chance_99_upper",
        "delta_bic_th_stratum", "fpr_h0", "fpr_h0_upper95"
    ]

    snr_dict = {}
    for r in results:
        stratum_key = f"{r['model_id']}_{r['inc_deg']}_{r['fwhm_kpc']}_{r['sigma_v']}_{r['geom_error_flag']}_{r['a_los3']}_{r['arm_method']}"
        snr_dict[stratum_key] = r["_snr_threshold"]

    cache_path = os.path.join(out_dir, ".model2_snr_thresholds.json")
    with open(cache_path, "w", encoding="utf-8") as f_json:
        json.dump(snr_dict, f_json, indent=2)

    csv_rows = [{k: r[k] for k in fieldnames} for r in results]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    n_passed = sum(1 for r in results if r["fpr_h0_upper95"] <= 0.070)
    print(f"Model 2 FPR上限達成率: {n_passed} / {len(results)} ({n_passed/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
