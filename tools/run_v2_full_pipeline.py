# -*- coding: utf-8 -*-
"""
tools/run_v2_full_pipeline.py — 前方モデル減算パイプライン(v2) 完全実行
- 前方平滑化軸対称モデル減算
- 物理的許容半径 r_tol = xi_c + sigma_beam
- Matched Null による幾何系統誤差補正
- セル単位事前予測の固定とハッシュ記録
"""

import os
import sys
import time
import json
import csv
import hashlib
from datetime import datetime
from multiprocessing import Pool, cpu_count
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

MODELS = {
    "model_1": {
        "M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 12.0, "dpix": 0.08,
        "a_inj_physical": 1.12, "xi_c_kpc": 0.092442, "v0_rv": 76.29
    },
    "model_2": {
        "M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 6.0, "dpix": 0.08,
        "a_inj_physical": 4.89, "xi_c_kpc": 0.046221, "v0_rv": 107.88
    },
    "model_3": {
        "M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.2, "dpix": 0.08,
        "a_inj_physical": 30.80, "xi_c_kpc": 0.113218, "v0_rv": 62.29
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


def generate_all_cells():
    cells = []
    cell_idx = 0
    for model_id in ["model_1", "model_2", "model_3"]:
        for nv in VORTEX_COUNTS:
            for inc in INCLINATIONS:
                for pa in POSITION_ANGLES:
                    for fwhm in RESOLUTIONS_KPC:
                        for sigma_v in NOISES_KMS:
                            for geom_err in GEOM_ERRORS:
                                for arm in ARM_LEVELS:
                                    cell_id = f"v2_{model_id}_{cell_idx:05d}_{nv}_{int(inc)}_{int(pa)}_{fwhm}_{sigma_v}_{geom_err}_{arm['a_los3']}_{arm['method']}"
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


def compute_v2_predictions(cells):
    snr_th_v2 = 4.50
    pred_rows = []

    # Cache beam attenuation with proper disk projection
    beam_cache = {}

    for c in cells:
        model_id = c["model_id"]
        info = MODELS[model_id]
        fwhm = c["fwhm_kpc"]
        inc = c["inc_deg"]
        dpix = info["dpix"]

        key = (model_id, fwhm, inc)
        if key not in beam_cache:
            coords = np.arange(-info["r_fov"], info["r_fov"] + dpix/2, dpix)
            X_obs, Y_obs = np.meshgrid(coords, coords)
            fwhm_pix = fwhm / dpix
            d_v = 0.3 * info["R_kpc"]

            _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
            dist_sq = (r_disk * np.cos(phi_disk) - d_v)**2 + (r_disk * np.sin(phi_disk))**2
            v_raw = np.exp(-dist_sq / (2.0 * (0.1)**2))
            v_conv = pipe.convolve_beam_normalized(v_raw, fwhm_pix=fwhm_pix)
            eta_beam = float(np.nanmax(v_conv) / np.max(v_raw))
            beam_cache[key] = eta_beam
        else:
            eta_beam = beam_cache[key]

        meth = c["arm_method"]
        eta_fit = 0.996 if meth == "remove_k3" else (0.999 if meth == "arm_fit" else 1.0)
        phi_disk_rad = 0.0 # on major axis for single vortex / primary

        a50_pred = pipe.predict_a50(snr_th_v2, c["sigma_v_kms"], eta_beam, eta_fit, inc, phi_disk_rad)
        a_phys = info["a_inj_physical"]
        ratio = a_phys / a50_pred

        if c["n_vortex_inj"] == 0:
            p_class = "CONTROL"
        elif ratio >= 1.5:
            p_class = "RECOVERED"
        elif ratio <= 0.67:
            p_class = "UNDETECTED"
        else:
            p_class = "BOUNDARY"

        pred_rows.append({
            "cell_id": c["cell_id"],
            "model_id": model_id,
            "n_vortex_inj": c["n_vortex_inj"],
            "inc_deg": inc,
            "pa_deg": c["pa_deg"],
            "fwhm_kpc": fwhm,
            "sigma_v_kms": c["sigma_v_kms"],
            "geom_error_flag": c["geom_error_flag"],
            "a_los3_kms": c["a_los3_kms"],
            "arm_method": c["arm_method"],
            "snr_th_v2": snr_th_v2,
            "eta_beam": round(eta_beam, 4),
            "eta_fit": round(eta_fit, 4),
            "a50_pred_kms": round(a50_pred, 3),
            "a_inj_physical_kms": a_phys,
            "ratio_phys_to_a50": round(ratio, 3),
            "pred_class": p_class
        })

    return pred_rows


def run_cell_simulation_v2(cell):
    model_id = cell["model_id"]
    model_info = MODELS[model_id]
    halo = pipe.TFHaloModel(M_msun=model_info["M_msun"], R_kpc=model_info["R_kpc"], m_eV=model_info["m_eV"])

    dpix = model_info["dpix"]
    coords = np.arange(-model_info["r_fov"], model_info["r_fov"] + dpix/2, dpix)
    center_idx = len(coords) // 2
    X_obs, Y_obs = np.meshgrid(coords, coords)
    fwhm_pix = cell["fwhm_kpc"] / dpix
    sigma_v = cell["sigma_v_kms"]
    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))

    # Physical tolerance radius: r_tol = xi_c + sigma_beam
    xi_c_kpc = model_info["xi_c_kpc"]
    sigma_beam_kpc = sigma_beam_pix * dpix
    r_tol_kpc = xi_c_kpc + sigma_beam_kpc
    r_tol_pix = r_tol_kpc / dpix

    r_disk_min = 0.2 * model_info["R_kpc"]
    r_disk_max = 0.9 * model_info["R_kpc"]

    inc_true = cell["inc_deg"]
    pa_true = cell["pa_deg"]
    v_sys_true = 100.0

    if cell["geom_error_flag"]:
        delta_inc, delta_pa, delta_sys = 2.0, 2.0, 2.0
    else:
        delta_inc, delta_pa, delta_sys = 0.0, 0.0, 0.0

    inc_obs = inc_true + delta_inc
    pa_obs = pa_true + delta_pa
    v_sys_obs = v_sys_true + delta_sys

    _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_true, inc_deg=inc_true)
    _, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_obs, inc_deg=inc_obs)
    mask_disk = (r_disk_true <= model_info["R_kpc"] * 0.9)
    mask_search = (r_disk_obs >= r_disk_min) & (r_disk_obs <= r_disk_max)

    v0_true = halo.v_0(r_disk_true)
    v_los_unpert = pipe.project_vlos(v0_true, 0.0, phi_disk_true, inc_deg=inc_true, v_sys=v_sys_true)

    # v2: Forward Convolved Model of unperturbed disk (using observer's assumed geometry)
    v0_obs_model = halo.v_0(r_disk_obs)
    v_los_obs_model = pipe.project_vlos(v0_obs_model, 0.0, phi_disk_obs, inc_deg=inc_obs, v_sys=v_sys_obs)
    v_fwd_conv_model = pipe.convolve_beam_normalized(v_los_obs_model, fwhm_pix=fwhm_pix, mask=(r_disk_obs <= model_info["R_kpc"] * 0.9))

    n_v = cell["n_vortex_inj"]
    d_vortex = 0.3 * model_info["R_kpc"]

    if n_v == 0:
        vortex_pos_kpc = []
    elif n_v == 1:
        vortex_pos_kpc = [(d_vortex, 0.0)]
    elif n_v == 2:
        vortex_pos_kpc = [(d_vortex, 0.0), (-d_vortex, 0.0)]
    elif n_v == 4:
        vortex_pos_kpc = [(d_vortex, 0.0), (-d_vortex, 0.0), (0.0, d_vortex), (0.0, -d_vortex)]

    # True observation pixel coordinates (correctly projected via disk_to_obs)
    true_pix = []
    for xv, yv in vortex_pos_kpc:
        xo, yo = pipe.disk_to_obs(xv, yv, pa_deg=pa_true, inc_deg=inc_true)
        xi = center_idx + int(round(xo / dpix))
        yi = center_idx + int(round(yo / dpix))
        true_pix.append((xi, yi))

    # Vortex perturbation injection
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
    # Also prepare matched null signal (pure H0 with same arm and geometric distortion, no vortex)
    v_conv_signal_null = pipe.convolve_beam_normalized(v_los_unpert + arm_residual_inj, fwhm_pix=fwhm_pix, mask=mask_disk)

    snr_th = 4.50
    n_trials = 60
    pos_matches_count = 0
    null_matches_count = 0

    for trial in range(n_trials):
        seed_val = int(hashlib.md5(f"v2_{cell['cell_id']}_{trial}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed_val)
        noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)

        # 1. Forward model subtraction on signal
        res_v2 = (v_conv_signal + noise) - v_fwd_conv_model
        meth = cell["arm_method"]
        if meth == "remove_k3":
            res_v2, _ = pipe.remove_harmonic_k3_rings(r_disk_obs, phi_disk_obs, res_v2, n_rings=15)
        elif meth == "arm_fit":
            res_v2, _, _, _ = pipe.fit_spiral_arm_direct_c(r_disk_obs, phi_disk_obs, res_v2, inc_deg=inc_obs)

        peaks = pipe.find_local_peaks_2d(res_v2, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
        pks_search = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]

        if n_v > 0:
            matches = pipe.evaluate_position_matching(pks_search, true_pix, r_tol_pix=r_tol_pix)
            pos_matches_count += sum(matches)

        # 2. Matched null evaluation (H0 with same noise seed)
        if n_v > 0 and len(true_pix) > 0:
            res_null = (v_conv_signal_null + noise) - v_fwd_conv_model
            if meth == "remove_k3":
                res_null, _ = pipe.remove_harmonic_k3_rings(r_disk_obs, phi_disk_obs, res_null, n_rings=15)
            elif meth == "arm_fit":
                res_null, _, _, _ = pipe.fit_spiral_arm_direct_c(r_disk_obs, phi_disk_obs, res_null, inc_deg=inc_obs)
            pks_null = pipe.find_local_peaks_2d(res_null, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
            pks_null_search = [p for p in pks_null if mask_search[p["y_idx"], p["x_idx"]]]
            null_matches = pipe.evaluate_position_matching(pks_null_search, true_pix, r_tol_pix=r_tol_pix)
            null_matches_count += sum(null_matches)

    if n_v > 0:
        recall_pos = pos_matches_count / float(n_trials * n_v)
        null_recall = null_matches_count / float(n_trials * n_v)
        # Wilson upper 95% bound for null_recall
        z_a = 1.95996
        d_a = 1.0 + (z_a**2) / (n_trials * n_v)
        c_a = (null_recall + (z_a**2) / (2.0 * n_trials * n_v)) / d_a
        m_a = (z_a * np.sqrt((null_recall * (1.0 - null_recall) + (z_a**2) / (4.0 * n_trials * n_v)) / (n_trials * n_v))) / d_a
        null_upper95 = float(c_a + m_a)
    else:
        recall_pos = 0.0
        null_recall = 0.0
        null_upper95 = 0.0

    # Decision tree under v2:
    # Requires recall_pos >= 0.50 AND recall_pos > null_upper95 (strictly exceeds matched null)
    if n_v == 0:
        verdict = "CELL_CONTROL_PASSED"
    elif recall_pos < 0.50:
        if recall_pos <= null_upper95:
            verdict = "CELL_UNDETECTED"
        else:
            verdict = "CELL_MARGINAL"
    else:
        if recall_pos > null_upper95:
            verdict = "CELL_ROBUST_RECOVERED"
        else:
            verdict = "CELL_FAIL_GEOMETRIC_ARTIFACT"

    return {
        "cell_id": cell["cell_id"],
        "model_id": cell["model_id"],
        "n_vortex_inj": n_v,
        "inc_deg": cell["inc_deg"],
        "pa_deg": cell["pa_deg"],
        "fwhm_kpc": cell["fwhm_kpc"],
        "sigma_v_kms": cell["sigma_v_kms"],
        "geom_error_flag": cell["geom_error_flag"],
        "a_los3_kms": cell["a_los3_kms"],
        "arm_method": cell["arm_method"],
        "r_tol_kpc": round(r_tol_kpc, 4),
        "r_tol_pix": round(r_tol_pix, 2),
        "recall_pos": round(recall_pos, 4),
        "null_recall_matched": round(null_recall, 4),
        "null_upper95": round(null_upper95, 4),
        "cell_verdict": verdict
    }


def main():
    print("=" * 80)
    print("PIPELINE v2: 前方モデル減算・物理許容半径・Matched Null補正 完全バッチ")
    print("=" * 80)

    cells = generate_all_cells()
    print(f"全セル数: {len(cells)}")

    out_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Pre-register predictions file
    t_start = time.time()
    preds = compute_v2_predictions(cells)
    pred_csv = os.path.join(out_dir, "v2_cell_predictions_preregistered.csv")
    with open(pred_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        writer.writeheader()
        writer.writerows(preds)

    with open(pred_csv, "rb") as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()

    meta = {
        "filename": "v2_cell_predictions_preregistered.csv",
        "timestamp_iso": datetime.now().isoformat(),
        "sha256": pred_hash,
        "total_cells": len(preds),
        "snr_th_v2": 4.50,
        "pipeline_version": "v2_forward_model_subtraction"
    }
    with open(os.path.join(out_dir, "v2_cell_predictions_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"v2 予測ファイルを固定: {pred_csv}")
    print(f"SHA-256: {pred_hash}")

    # Step 2: Run simulation in parallel
    n_workers = min(cpu_count(), 16)
    print(f"並列ワーカー数: {n_workers} コア, 全{len(cells)}セル実行開始...")

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_cell_simulation_v2, cells)

    elapsed = time.time() - t_start
    print(f"v2 本番バッチ完了: 所要時間 {elapsed:.1f} 秒 ({elapsed/60.0:.2f} 分)")

    res_csv = os.path.join(out_dir, "v2_injection_recovery_summary.csv")
    with open(res_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(res_csv, "rb") as f:
        res_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"v2 結果サマリーを保存: {res_csv}")
    print(f"SHA-256: {res_hash}")

    # Step 3: Analysis & Confusion Matrix
    # Match pred and res
    tp = fp = fn = tn = 0
    boundary_count = 0
    vortex_results = [r for r in results if r["n_vortex_inj"] > 0]

    for p, r in zip(preds, results):
        if r["n_vortex_inj"] == 0:
            continue
        p_cls = p["pred_class"]
        if p_cls == "BOUNDARY":
            boundary_count += 1
            continue

        p_rec = (p_cls == "RECOVERED")
        a_rec = (r["cell_verdict"] == "CELL_ROBUST_RECOVERED") # recall >= 0.5 and > null_upper95

        if p_rec and a_rec: tp += 1
        elif p_rec and not a_rec: fp += 1
        elif not p_rec and a_rec: fn += 1
        else: tn += 1

    total_eval = tp + fp + fn + tn
    acc = (tp + tn) / total_eval if total_eval > 0 else 0
    all_undet = (fn + tn) / total_eval if total_eval > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n=== v2 Pipeline Confusion Matrix (Excluding {boundary_count} BOUNDARY cells, N={total_eval}) ===")
    print(f"TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(f"Accuracy: {acc:.4f} (All-UNDETECTED: {all_undet:.4f})")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}")

    # Model-by-model breakdown
    for mid in ["model_1", "model_2", "model_3"]:
        sub_res = [r for r in vortex_results if r["model_id"] == mid]
        rec_count = sum(1 for r in sub_res if r["cell_verdict"] == "CELL_ROBUST_RECOVERED")
        max_rec = max(r["recall_pos"] for r in sub_res)
        med_rec = np.median([r["recall_pos"] for r in sub_res])
        print(f"\n--- {mid} in v2 (Vortex Cells N={len(sub_res)}) ---")
        print(f"  ROBUST Recovered: {rec_count} / {len(sub_res)} ({rec_count/len(sub_res)*100:.2f}%)")
        print(f"  Recall_pos: max={max_rec:.4f}, median={med_rec:.4f}")


if __name__ == "__main__":
    main()
