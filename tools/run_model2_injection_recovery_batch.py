# -*- coding: utf-8 -*-
"""
tools/run_model2_injection_recovery_batch.py — Model 2 専用本番注入回収バッチ
(全4,536セル × 100試行 = 453,600試行)
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

VORTEX_COUNTS = [0, 1, 2, 4]
INCLINATIONS = [30.0, 50.0, 70.0]
POSITION_ANGLES = [0.0, 45.0, 90.0]
RESOLUTIONS_KPC = [0.2, 0.5, 1.0]
NOISES_KMS = [0.5, 1.0, 2.0]
GEOM_ERRORS = [False, True]

ARM_LEVELS = [{"a_los3": 0.0, "method": "none"}]
for a3 in [1.0, 3.0]:
    for meth in ["raw", "remove_k3", "arm_fit"]:
        ARM_LEVELS.append({"a_los3": a3, "method": meth})


def generate_model2_cells():
    cells = []
    cell_idx = 0
    model_id = "model_2"
    for nv in VORTEX_COUNTS:
        for inc in INCLINATIONS:
            for pa in POSITION_ANGLES:
                for fwhm in RESOLUTIONS_KPC:
                    for sigma_v in NOISES_KMS:
                        for geom_err in GEOM_ERRORS:
                            for arm in ARM_LEVELS:
                                cell_id = f"m2_cell_{cell_idx:04d}_{nv}_{int(inc)}_{int(pa)}_{fwhm}_{sigma_v}_{geom_err}_{arm['a_los3']}_{arm['method']}"
                                cells.append({
                                    "cell_idx": cell_idx,
                                    "cell_id": cell_id,
                                    "model_id": model_id,
                                    "n_vortex_inj": nv,
                                    "inc_deg": inc,
                                    "pa_deg": pa,
                                    "fwhm_kpc": fwhm,
                                    "sigma_v_kms": sigma_v,
                                    "geom_error_flag": geom_err,
                                    "a_los3_kms": arm["a_los3"],
                                    "arm_method": arm["method"]
                                })
                                cell_idx += 1
    return cells


def load_calibration_data():
    out_dir = os.path.join(PROJECT_DIR, "output")
    json_path = os.path.join(out_dir, ".model2_snr_thresholds.json")
    csv_path = os.path.join(out_dir, "h0_model2_calibration.csv")

    with open(json_path, "r", encoding="utf-8") as f:
        snr_dict = json.load(f)

    table = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["model_id"],
                float(row["inc_deg"]),
                float(row["fwhm_kpc"]),
                float(row["sigma_v"]),
                row["geom_error_flag"].lower() == "true",
                float(row["a_los3"]),
                row["arm_method"]
            )
            stratum_key_str = f"{row['model_id']}_{row['inc_deg']}_{row['fwhm_kpc']}_{row['sigma_v']}_{row['geom_error_flag']}_{row['a_los3']}_{row['arm_method']}"
            snr_th_val = snr_dict.get(stratum_key_str, 4.0)

            table[key] = {
                "f_chance_95": float(row["f_chance_95"]),
                "f_chance_95_upper": float(row["f_chance_95_upper"]),
                "f_chance_99_pool": float(row["f_chance_99"]),
                "f_chance_99_upper": float(row["f_chance_99_upper"]),
                "delta_bic_th_stratum": float(row["delta_bic_th_stratum"]),
                "fpr_h0_val": float(row["fpr_h0"]),
                "fpr_h0_upper95": float(row["fpr_h0_upper95"]),
                "snr_threshold": snr_th_val
            }
    return table


def classify_cell(n_vortex_inj, fpr_h0_upper95, delta_fpr_arm_lower95,
                  recall_pos, recall_multi, rate_model_select, f_chance_95_upper):
    if fpr_h0_upper95 > 0.070:
        return "CELL_FAIL_FPR_UNCONTROLLED"
    if delta_fpr_arm_lower95 >= 0.03:
        return "CELL_DEGENERATE_ARM"
    if n_vortex_inj == 0:
        return "CELL_CONTROL_PASSED"
    if recall_pos <= f_chance_95_upper:
        return "CELL_UNDETECTED"
    if recall_pos < 0.50:
        return "CELL_MARGINAL"
    if (n_vortex_inj > 1) and (recall_multi < 0.50):
        return "CELL_POSITION_ONLY"
    if rate_model_select >= 0.50:
        return "CELL_ROBUST_RECOVERED"
    else:
        return "CELL_STRUCTURAL_ONLY"


def run_single_cell_simulation(cell_and_calib):
    cell, calib = cell_and_calib
    model_info = MODELS[cell["model_id"]]
    halo = pipe.TFHaloModel(M_msun=model_info["M_msun"], R_kpc=model_info["R_kpc"], m_eV=model_info["m_eV"])

    dpix = model_info["dpix"]
    coords = np.arange(-model_info["r_fov"], model_info["r_fov"] + dpix/2, dpix)
    center_idx = len(coords) // 2
    X_obs, Y_obs = np.meshgrid(coords, coords)
    fwhm_pix = cell["fwhm_kpc"] / dpix
    sigma_v = cell["sigma_v_kms"]

    r_disk_min = 0.2 * model_info["R_kpc"]
    r_disk_max = 0.9 * model_info["R_kpc"]

    inc_true = cell["inc_deg"]
    pa_true = cell["pa_deg"]
    v_sys_true = 100.0

    _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_true, inc_deg=inc_true)
    mask_disk = (r_disk_true <= model_info["R_kpc"] * 0.9)

    if cell["geom_error_flag"]:
        delta_inc, delta_pa, delta_sys = 2.0, 2.0, 2.0
    else:
        delta_inc, delta_pa, delta_sys = 0.0, 0.0, 0.0

    inc_obs = inc_true + delta_inc
    pa_obs = pa_true + delta_pa
    v_sys_obs = v_sys_true + delta_sys
    _, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_obs, inc_deg=inc_obs)

    v0_true = halo.v_0(r_disk_true)
    v_los_unpert = pipe.project_vlos(v0_true, 0.0, phi_disk_true, inc_deg=inc_true, v_sys=v_sys_true)

    n_v = cell["n_vortex_inj"]
    d_vortex = 0.3 * model_info["R_kpc"]

    if n_v == 0:
        vortex_pos_kpc = []
        vis_indices = []
    elif n_v == 1:
        vortex_pos_kpc = [(d_vortex, 0.0)]
        vis_indices = [0]
    elif n_v == 2:
        vortex_pos_kpc = [(d_vortex, 0.0), (-d_vortex, 0.0)]
        vis_indices = [0, 1]
    elif n_v == 4:
        vortex_pos_kpc = [(d_vortex, 0.0), (-d_vortex, 0.0), (0.0, d_vortex), (0.0, -d_vortex)]
        vis_indices = [0, 1]

    true_pix = []
    for xv, yv in vortex_pos_kpc:
        xo, yo = pipe.disk_to_obs(xv, yv, pa_deg=pa_true, inc_deg=inc_true)
        xi = center_idx + int(round(xo / dpix))
        yi = center_idx + int(round(yo / dpix))
        true_pix.append((xi, yi))

    r_tol_kpc = pipe.compute_tolerance_radius(cell["model_id"], cell["fwhm_kpc"], model_info["R_kpc"])
    r_tol_pix = r_tol_kpc / dpix

    if n_v > 0:
        dv_vortex_map = halo.compute_vortex_velocity_perturbation_map(
            r_disk_true * np.cos(phi_disk_true), r_disk_true * np.sin(phi_disk_true), vortex_pos_kpc
        )
        v_los_inj = dv_vortex_map * np.cos(phi_disk_true) * np.sin(np.radians(inc_true))
    else:
        v_los_inj = 0.0

    if cell["a_los3_kms"] > 0:
        arm_residual_inj = cell["a_los3_kms"] * np.cos(3.0 * phi_disk_true)
    else:
        arm_residual_inj = 0.0

    v_conv_signal = pipe.convolve_beam_normalized(v_los_unpert + v_los_inj + arm_residual_inj, fwhm_pix=fwhm_pix, mask=mask_disk)

    snr_th = calib.get("snr_threshold", 4.0)
    f_chance_95 = calib["f_chance_95"]
    f_chance_95_upper = calib["f_chance_95_upper"]
    f_chance_99_pool = calib["f_chance_99_pool"]
    f_chance_99_upper = calib["f_chance_99_upper"]
    delta_ts_mf_th_stratum = calib["delta_bic_th_stratum"]
    fpr_h0_val = calib["fpr_h0_val"]
    fpr_h0_upper95 = calib["fpr_h0_upper95"]

    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))

    n_trials = 100
    pos_matches_count = 0
    multi_matches_count = 0
    pos_matches_vis_count = 0
    multi_matches_vis_count = 0

    mf_select_count = 0
    mf_gt6_count = 0
    delta_ts_mfs = []
    arm_fp_count = 0

    trial_type_tag = "inj" if n_v > 0 else "h0_eval"
    mask_search = (r_disk_obs >= r_disk_min) & (r_disk_obs <= r_disk_max)

    for trial in range(n_trials):
        seed_trial = int(hashlib.md5(f"m2_{cell['cell_id']}_{trial_type_tag}_{trial}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed_trial)

        noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
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

        raw_peaks = pipe.find_local_peaks_2d(res_proc, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
        peaks = [p for p in raw_peaks if r_disk_min <= r_disk_obs[p["y_idx"], p["x_idx"]] <= r_disk_max]
        mask_valid = mask_search & (~np.isnan(res_proc))

        if n_v > 0:
            matches = pipe.evaluate_position_matching(peaks, true_pix, r_tol_pix=r_tol_pix)
            pos_matches_count += sum(matches)
            if all(matches) and len(peaks) == n_v:
                multi_matches_count += 1

            if len(vis_indices) > 0:
                vis_true_pix = [true_pix[idx] for idx in vis_indices]
                vis_matches = pipe.evaluate_position_matching(peaks, vis_true_pix, r_tol_pix=r_tol_pix)
                pos_matches_vis_count += sum(vis_matches)
                if all(vis_matches):
                    multi_matches_vis_count += 1

            _, _, _, delta_ts_mf = pipe.compute_empirical_matched_filter_stat(res_proc, sigma_beam_pix, mask_valid)
            delta_ts_mfs.append(delta_ts_mf)

            if delta_ts_mf > delta_ts_mf_th_stratum:
                mf_select_count += 1
            if delta_ts_mf >= 6.0:
                mf_gt6_count += 1
        else:
            if len(peaks) > 0:
                arm_fp_count += 1

    if n_v == 0 and cell["a_los3_kms"] > 0:
        fpr_arm = arm_fp_count / float(n_trials)
        z_a = 1.95996
        d_a = 1.0 + (z_a**2) / n_trials
        c_a = (fpr_arm + (z_a**2) / (2.0 * n_trials)) / d_a
        m_a = (z_a * np.sqrt((fpr_arm * (1.0 - fpr_arm) + (z_a**2) / (4.0 * n_trials)) / n_trials)) / d_a
        fpr_arm_lower95 = max(0.0, float(c_a - m_a))
        delta_fpr_arm_lower95 = fpr_arm_lower95 - fpr_h0_val
    else:
        fpr_arm = 0.0
        fpr_arm_lower95 = 0.0
        delta_fpr_arm_lower95 = 0.0

    if n_v > 0:
        recall_pos = pos_matches_count / float(n_trials * n_v)
        recall_multi = multi_matches_count / float(n_trials)
        recall_pos_vis = pos_matches_vis_count / float(n_trials * len(vis_indices)) if len(vis_indices) > 0 else 0.0
        recall_multi_vis = multi_matches_vis_count / float(n_trials)
        rate_model_select = mf_select_count / float(n_trials)
        rate_bic_gt6 = mf_gt6_count / float(n_trials)
        delta_bic_mean = float(np.mean(delta_ts_mfs)) if len(delta_ts_mfs) > 0 else 0.0
    else:
        recall_pos = 0.0
        recall_multi = 0.0
        recall_pos_vis = 0.0
        recall_multi_vis = 0.0
        rate_model_select = 0.0
        rate_bic_gt6 = 0.0
        delta_bic_mean = 0.0

    recall_multi_is_def = (recall_multi >= 0.50) if n_v > 1 else True
    is_sig = (rate_model_select >= 0.50)
    is_high_sig = (rate_bic_gt6 >= 0.50)
    is_deg_arm = (delta_fpr_arm_lower95 >= 0.03)

    verdict = classify_cell(
        n_vortex_inj=n_v,
        fpr_h0_upper95=fpr_h0_upper95,
        delta_fpr_arm_lower95=delta_fpr_arm_lower95,
        recall_pos=recall_pos,
        recall_multi=recall_multi,
        rate_model_select=rate_model_select,
        f_chance_95_upper=f_chance_95_upper
    )

    return {
        "model_id": cell["model_id"],
        "n_vortex_inj": n_v,
        "inc_deg": cell["inc_deg"],
        "pa_deg": cell["pa_deg"],
        "fwhm_kpc": cell["fwhm_kpc"],
        "sigma_v_kms": cell["sigma_v_kms"],
        "geom_error_flag": cell["geom_error_flag"],
        "a_los3_kms": cell["a_los3_kms"],
        "arm_method": cell["arm_method"],
        "fpr_h0_val": round(fpr_h0_val, 6),
        "fpr_h0_upper95": round(fpr_h0_upper95, 6),
        "fpr_arm": round(fpr_arm, 6),
        "fpr_arm_lower95": round(fpr_arm_lower95, 6),
        "delta_fpr_arm_lower95": round(delta_fpr_arm_lower95, 6),
        "f_chance_95": round(f_chance_95, 6),
        "f_chance_95_upper": round(f_chance_95_upper, 6),
        "f_chance_99_pool": round(f_chance_99_pool, 6),
        "f_chance_99_upper": round(f_chance_99_upper, 6),
        "recall_pos": round(recall_pos, 6),
        "recall_multi": round(recall_multi, 6),
        "recall_pos_vis": round(recall_pos_vis, 6),
        "recall_multi_vis": round(recall_multi_vis, 6),
        "recall_multi_is_definition": recall_multi_is_def,
        "delta_bic_th_stratum": round(delta_ts_mf_th_stratum, 4),
        "rate_model_select": round(rate_model_select, 4),
        "rate_bic_gt6": round(rate_bic_gt6, 4),
        "is_significant": is_sig,
        "is_high_significant": is_high_sig,
        "is_degenerate_arm": is_deg_arm,
        "cell_verdict": verdict
    }


def main():
    print("=" * 80)
    print("STAGE 4 (MODEL 2): 本番注入回収バッチ (全4,536セル × 100試行)")
    print("=" * 80)

    calib_table = load_calibration_data()
    cells = generate_model2_cells()
    print(f"Model 2 総セル数: {len(cells)} (渦注入セル: {sum(1 for c in cells if c['n_vortex_inj'] > 0)})")

    cell_args = []
    for c in cells:
        key = (
            c["model_id"],
            float(c["inc_deg"]),
            float(c["fwhm_kpc"]),
            float(c["sigma_v_kms"]),
            c["geom_error_flag"],
            float(c["a_los3_kms"]),
            c["arm_method"]
        )
        cal = calib_table[key]
        cell_args.append((c, cal))

    t0 = time.time()
    n_workers = min(cpu_count(), 16)
    print(f"並列ワーカー数: {n_workers} コア")

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single_cell_simulation, cell_args)

    elapsed_sec = time.time() - t0
    print(f"Model 2 本番バッチ完了: 所要時間 {elapsed_sec:.1f} 秒 ({elapsed_sec/60.0:.2f} 分)")

    out_dir = os.path.join(PROJECT_DIR, "output")
    out_csv = os.path.join(out_dir, "injection_recovery_model2_summary.csv")

    fieldnames = list(results[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} rows to {out_csv}")

    # Summary
    verdicts = {}
    vortex_results = [r for r in results if r["n_vortex_inj"] > 0]
    for r in results:
        v = r["cell_verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    print("\n--- Model 2 Verdict Distribution ---")
    for v, cnt in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {cnt} ({cnt/len(results)*100:.2f}%)")

    pos_recovered = sum(1 for r in vortex_results if r["recall_pos"] >= 0.50)
    print(f"\nModel 2 Vortex Injected Cells with recall_pos >= 0.50: {pos_recovered} / {len(vortex_results)} ({pos_recovered/len(vortex_results)*100:.2f}%)")


if __name__ == "__main__":
    main()
