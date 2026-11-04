# -*- coding: utf-8 -*-
"""
tools/run_v2_stage3_and_stage4.py
前方モデル減算パイプライン (v2) の正式実行スクリプト:
1. Stage 3 v2 (H0較正・純粋ノイズ95%点確認・物理許容半径 r_tol = xi_c + sigma_beam でのH0偶然一致率測定)
2. Stage 3 v2 の後、Stage 4 v2 の前に、全13,608セルの事前予測を固定 (節1の正しい射影後 eta_beam を使用)
3. Stage 4 v2 (全13,608セル本番実行・幾何誤差セルの matched null 補正 recall_pos > null_upper95)
4. 実行ID・SHA-256ハッシュ・混同行列・モデル別/条件別集計の出力
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


def compute_physical_r_tol(model_id, fwhm_kpc):
    """
    ユーザー指示3: 2.5ピクセル固定ではなく、r_tol = xi_c + sigma_beam に応じて変える。
    """
    info = MODELS[model_id]
    dpix = info["dpix"]
    xi_c_kpc = info["xi_c_kpc"]
    fwhm_pix = fwhm_kpc / dpix
    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_beam_kpc = sigma_beam_pix * dpix
    r_tol_kpc = xi_c_kpc + sigma_beam_kpc
    r_tol_pix = r_tol_kpc / dpix
    return r_tol_kpc, r_tol_pix, sigma_beam_kpc, sigma_beam_pix


def build_linear_projectors(r_disk_obs, phi_disk_obs, mask_valid, inc_obs, r_max_disk, n_rings=15):
    ring_edges = np.linspace(0.0, r_max_disk, n_rings + 1)
    cos_3phi = np.cos(3.0 * phi_disk_obs)
    sin_3phi = np.sin(3.0 * phi_disk_obs)

    k3_blocks = []
    for k in range(n_rings):
        if k == n_rings - 1:
            mask = (r_disk_obs >= ring_edges[k]) & (r_disk_obs <= ring_edges[k+1]) & mask_valid
        else:
            mask = (r_disk_obs >= ring_edges[k]) & (r_disk_obs < ring_edges[k+1]) & mask_valid
        idx = np.flatnonzero(mask)
        if len(idx) >= 6:
            X = np.column_stack([cos_3phi.ravel()[idx], sin_3phi.ravel()[idx]])
            Q, _ = np.linalg.qr(X, mode='reduced')
            k3_blocks.append((idx, Q))

    sin_i = np.sin(np.radians(inc_obs))
    arm_mask = mask_valid & (r_disk_obs > 0.1)
    arm_idx = np.flatnonzero(arm_mask)
    r_val = r_disk_obs.ravel()[arm_idx]
    phi_val = phi_disk_obs.ravel()[arm_idx]
    log_r = np.log(r_val / 1.0)

    arm_q_list = []
    for pitch_deg in (10.0, 15.0, 20.0):
        cot_p = 1.0 / np.tan(np.radians(pitch_deg))
        psi_val = 2.0 * (phi_val - cot_p * log_r)
        bc = np.cos(psi_val) * np.cos(phi_val) * sin_i
        bs = np.sin(psi_val) * np.cos(phi_val) * sin_i
        brc = np.sin(psi_val) * np.sin(phi_val) * sin_i
        brs = np.cos(psi_val) * np.sin(phi_val) * sin_i
        X = np.column_stack([bc, bs, brc, brs])
        Q, _ = np.linalg.qr(X, mode='reduced')
        arm_q_list.append((pitch_deg, Q))

    return k3_blocks, (arm_idx, arm_q_list)


def apply_k3_fast(map_2d, k3_blocks):
    out = map_2d.copy()
    flat = out.ravel()
    for idx, Q in k3_blocks:
        y = flat[idx]
        flat[idx] = y - Q @ (Q.T @ y)
    return out


def apply_arm_fit_fast(map_2d, arm_info):
    out = map_2d.copy()
    flat = out.ravel()
    arm_idx, arm_q_list = arm_info
    if len(arm_idx) < 20:
        return out
    y = flat[arm_idx]
    best_proj_norm = -1.0
    best_fit = None
    for _, Q in arm_q_list:
        coeffs = Q.T @ y
        proj_norm = float(np.dot(coeffs, coeffs))
        if proj_norm > best_proj_norm:
            best_proj_norm = proj_norm
            best_fit = Q @ coeffs
    flat[arm_idx] = y - best_fit
    return out


def wilson_upper95(p_hat, n):
    if n <= 0:
        return 0.0
    z = 1.95996
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * n)) / n)) / denom
    return min(1.0, float(center + margin))


def run_stage4_worker(args):
    model_id, inc, thresholds_dict, n_s4_trials = args
    info = MODELS[model_id]
    dpix = info["dpix"]
    coords = np.arange(-info["r_fov"], info["r_fov"] + dpix / 2, dpix)
    center_idx = len(coords) // 2
    X_obs, Y_obs = np.meshgrid(coords, coords)
    halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])

    r_min = 0.2 * info["R_kpc"]
    r_max = 0.9 * info["R_kpc"]
    d_vortex = 0.3 * info["R_kpc"]

    # Generate validation noise bank for this worker
    val_noise_bank = {}
    for fwhm in RESOLUTIONS_KPC:
        _, _, _, sigma_beam_pix = compute_physical_r_tol(model_id, fwhm)
        v_list = []
        for t in range(n_s4_trials):
            seed = int(hashlib.md5(f"v2_s4_{model_id}_{fwhm}_{t}".encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            v_list.append(pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, 1.0, rng))
        val_noise_bank[fwhm] = v_list

    worker_results = {}

    for pa in POSITION_ANGLES:
        _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc)
        mask_disk_true = (r_disk_true <= r_max)
        v0_true = halo.v_0(r_disk_true)
        v_los_unpert = pipe.project_vlos(v0_true, 0.0, phi_disk_true, inc_deg=inc, v_sys=100.0)

        vortex_maps_raw = {}
        true_pix_map = {}
        for nv in VORTEX_COUNTS:
            if nv == 0:
                vpos = []
            elif nv == 1:
                vpos = [(d_vortex, 0.0)]
            elif nv == 2:
                vpos = [(d_vortex, 0.0), (-d_vortex, 0.0)]
            elif nv == 4:
                vpos = [(d_vortex, 0.0), (-d_vortex, 0.0), (0.0, d_vortex), (0.0, -d_vortex)]

            tpix = []
            for xv, yv in vpos:
                xo, yo = pipe.disk_to_obs(xv, yv, pa_deg=pa, inc_deg=inc)
                tpix.append((center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix))))
            true_pix_map[nv] = tpix

            if nv > 0:
                dv_map = halo.compute_vortex_velocity_perturbation_map(
                    r_disk_true * np.cos(phi_disk_true), r_disk_true * np.sin(phi_disk_true), vpos
                )
                vortex_maps_raw[nv] = dv_map * np.cos(phi_disk_true) * np.sin(np.radians(inc))
            else:
                vortex_maps_raw[nv] = np.zeros_like(v_los_unpert)

        arm_cos3 = np.cos(3.0 * phi_disk_true)

        for fwhm in RESOLUTIONS_KPC:
            fwhm_pix = fwhm / dpix
            r_tol_kpc, r_tol_pix, _, _ = compute_physical_r_tol(model_id, fwhm)
            val_noises = val_noise_bank[fwhm]

            v_conv_unpert = pipe.convolve_beam_normalized(v_los_unpert, fwhm_pix=fwhm_pix, mask=mask_disk_true)
            v_conv_arm1 = pipe.convolve_beam_normalized(arm_cos3, fwhm_pix=fwhm_pix, mask=mask_disk_true)
            v_conv_vortex = {}
            for nv in VORTEX_COUNTS:
                if nv == 0:
                    v_conv_vortex[0] = np.zeros_like(v_conv_unpert)
                else:
                    v_conv_vortex[nv] = pipe.convolve_beam_normalized(vortex_maps_raw[nv], fwhm_pix=fwhm_pix, mask=mask_disk_true)

            for geom_err in GEOM_ERRORS:
                if geom_err:
                    inc_obs, pa_obs, v_sys_obs = inc + 2.0, pa + 2.0, 102.0
                else:
                    inc_obs, pa_obs, v_sys_obs = inc, pa, 100.0

                if not geom_err:
                    r_disk_obs, phi_disk_obs = r_disk_true, phi_disk_true
                    mask_disk_obs = mask_disk_true
                    v_fwd_conv_model = v_conv_unpert
                else:
                    _, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_obs, inc_deg=inc_obs)
                    mask_disk_obs = (r_disk_obs <= r_max)
                    v0_obs = halo.v_0(r_disk_obs)
                    v_los_obs_m = pipe.project_vlos(v0_obs, 0.0, phi_disk_obs, inc_deg=inc_obs, v_sys=v_sys_obs)
                    v_fwd_conv_model = pipe.convolve_beam_normalized(v_los_obs_m, fwhm_pix=fwhm_pix, mask=mask_disk_obs)

                mask_search = (r_disk_obs >= r_min) & (r_disk_obs <= r_max) & (~np.isnan(v_conv_unpert)) & (~np.isnan(v_fwd_conv_model))
                k3_blocks, arm_info = build_linear_projectors(
                    r_disk_obs, phi_disk_obs, mask_search, inc_obs, r_max, n_rings=15
                )

                noise_proj_by_meth = {
                    "none": val_noises,
                    "raw": val_noises,
                    "remove_k3": [apply_k3_fast(nm, k3_blocks) for nm in val_noises],
                    "arm_fit": [apply_arm_fit_fast(nm, arm_info) for nm in val_noises]
                }

                null_peaks_cache = {}
                for arm in ARM_LEVELS:
                    a3 = arm["a_los3"]
                    meth = arm["method"]
                    det_null = (v_conv_unpert + a3 * v_conv_arm1) - v_fwd_conv_model
                    if meth == "remove_k3":
                        det_null_proc = apply_k3_fast(det_null, k3_blocks)
                    elif meth == "arm_fit":
                        det_null_proc = apply_arm_fit_fast(det_null, arm_info)
                    else:
                        det_null_proc = det_null

                    n_list = noise_proj_by_meth[meth]
                    for sigma_v in NOISES_KMS:
                        snr_th = thresholds_dict[f"{model_id}_{inc}_{fwhm}_{sigma_v}_{meth}"]["snr_th_v2"]
                        pks_trials = []
                        for t in range(n_s4_trials):
                            res_t = det_null_proc + sigma_v * n_list[t]
                            pks = pipe.find_local_peaks_2d(res_t, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
                            pks_s = [p for p in pks if mask_search[p["y_idx"], p["x_idx"]]]
                            pks_trials.append(pks_s)
                        null_peaks_cache[(sigma_v, a3, meth)] = pks_trials

                for nv in VORTEX_COUNTS:
                    tpix = true_pix_map[nv]
                    for arm in ARM_LEVELS:
                        a3 = arm["a_los3"]
                        meth = arm["method"]
                        det_sig = (v_conv_unpert + v_conv_vortex[nv] + a3 * v_conv_arm1) - v_fwd_conv_model
                        if meth == "remove_k3":
                            det_sig_proc = apply_k3_fast(det_sig, k3_blocks)
                        elif meth == "arm_fit":
                            det_sig_proc = apply_arm_fit_fast(det_sig, arm_info)
                        else:
                            det_sig_proc = det_sig

                        n_list = noise_proj_by_meth[meth]
                        for sigma_v in NOISES_KMS:
                            t_info = thresholds_dict[f"{model_id}_{inc}_{fwhm}_{sigma_v}_{meth}"]
                            snr_th = t_info["snr_th_v2"]
                            null_pks_trials = null_peaks_cache[(sigma_v, a3, meth)]

                            if nv == 0:
                                fp_count = sum(1 for pks_s in null_pks_trials if len(pks_s) > 0)
                                fpr_cell = fp_count / float(n_s4_trials)
                                recall_pos = 0.0
                                null_recall = 0.0
                                null_upper95 = 0.0
                                verdict = "CELL_CONTROL_PASSED" if fpr_cell <= t_info["fpr_h0_upper95"] else "CELL_FAIL_CONTROL_FPR"
                            else:
                                pos_hits = 0
                                null_hits = 0
                                for t in range(n_s4_trials):
                                    res_t = det_sig_proc + sigma_v * n_list[t]
                                    pks = pipe.find_local_peaks_2d(res_t, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
                                    pks_s = [p for p in pks if mask_search[p["y_idx"], p["x_idx"]]]
                                    pos_hits += sum(pipe.evaluate_position_matching(pks_s, tpix, r_tol_pix=r_tol_pix))
                                    null_hits += sum(pipe.evaluate_position_matching(null_pks_trials[t], tpix, r_tol_pix=r_tol_pix))

                                total_sites = n_s4_trials * nv
                                recall_pos = pos_hits / float(total_sites)
                                null_recall = null_hits / float(total_sites)
                                null_upper95 = wilson_upper95(null_recall, total_sites)
                                fpr_cell = sum(1 for pks_s in null_pks_trials if len(pks_s) > 0) / float(n_s4_trials)

                                # User instruction 4: recall_pos must exceed null_recall_matched upper bound (null_upper95)
                                eff_null_bound = max(null_upper95, t_info["f_chance_eff"])
                                if recall_pos >= 0.50:
                                    if recall_pos > eff_null_bound:
                                        verdict = "CELL_ROBUST_RECOVERED"
                                    else:
                                        verdict = "CELL_FAIL_MATCHED_NULL"
                                else:
                                    if recall_pos > eff_null_bound:
                                        verdict = "CELL_MARGINAL"
                                    else:
                                        verdict = "CELL_UNDETECTED"

                            key_tuple = (model_id, nv, inc, pa, fwhm, sigma_v, geom_err, a3, meth)
                            worker_results[key_tuple] = {
                                "r_tol_kpc": t_info["r_tol_kpc"],
                                "r_tol_pix": t_info["r_tol_pix"],
                                "snr_th_v2": snr_th,
                                "recall_pos": round(recall_pos, 4),
                                "null_recall_matched": round(null_recall, 4),
                                "null_upper95": round(null_upper95, 4),
                                "fpr_null_trial": round(fpr_cell, 4),
                                "cell_verdict": verdict
                            }

    return worker_results


def main():
    t0 = time.time()
    out_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)

    run_id_stage3 = "stage3_v2_20261028_fwd_rtol_phys"
    run_id_stage4 = "stage4_v2_20261028_fwd_matched_null"

    print("=" * 80)
    print(f"1. Stage 3 v2 開始 (Run ID: {run_id_stage3})")
    print("=" * 80)

    n_cal_trials = 150
    n_val_trials = 100
    noise_cal_bank = {}
    noise_val_bank = {}

    grid_cache = {}
    for model_id, info in MODELS.items():
        dpix = info["dpix"]
        coords = np.arange(-info["r_fov"], info["r_fov"] + dpix / 2, dpix)
        center_idx = len(coords) // 2
        X_obs, Y_obs = np.meshgrid(coords, coords)
        halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])
        grid_cache[model_id] = (coords, center_idx, X_obs, Y_obs, halo)

        for fwhm in RESOLUTIONS_KPC:
            _, _, _, sigma_beam_pix = compute_physical_r_tol(model_id, fwhm)
            cal_list = []
            for t in range(n_cal_trials):
                seed = int(hashlib.md5(f"v2_cal_{model_id}_{fwhm}_{t}".encode()).hexdigest()[:8], 16)
                rng = np.random.RandomState(seed)
                cal_list.append(pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, 1.0, rng))
            noise_cal_bank[(model_id, fwhm)] = cal_list

            val_list = []
            for t in range(n_val_trials):
                seed = int(hashlib.md5(f"v2_val_{model_id}_{fwhm}_{t}".encode()).hexdigest()[:8], 16)
                rng = np.random.RandomState(seed)
                val_list.append(pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, 1.0, rng))
            noise_val_bank[(model_id, fwhm)] = val_list

    print(f"ノイズバンク生成完了 ({time.time() - t0:.1f}s)")

    stage3_rows = []
    thresholds_dict = {}

    for model_id, info in MODELS.items():
        coords, center_idx, X_obs, Y_obs, halo = grid_cache[model_id]
        dpix = info["dpix"]
        r_min = 0.2 * info["R_kpc"]
        r_max = 0.9 * info["R_kpc"]
        d_vortex = 0.3 * info["R_kpc"]

        for fwhm in RESOLUTIONS_KPC:
            r_tol_kpc, r_tol_pix, sigma_beam_kpc, sigma_beam_pix = compute_physical_r_tol(model_id, fwhm)
            cal_noises = noise_cal_bank[(model_id, fwhm)]
            val_noises = noise_val_bank[(model_id, fwhm)]

            for inc in INCLINATIONS:
                for meth in ["none", "raw", "remove_k3", "arm_fit"]:
                    cal_max_snrs = []
                    for pa in POSITION_ANGLES:
                        _, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc)
                        mask_disk_obs = (r_disk_obs <= r_max)
                        mask_search = (r_disk_obs >= r_min) & (r_disk_obs <= r_max)
                        k3_blocks, arm_info = build_linear_projectors(
                            r_disk_obs, phi_disk_obs, mask_disk_obs, inc, r_max, n_rings=15
                        )
                        for t in range(50):
                            n_map = cal_noises[(t + int(pa)) % n_cal_trials]
                            if meth == "remove_k3":
                                n_proc = apply_k3_fast(n_map, k3_blocks)
                            elif meth == "arm_fit":
                                n_proc = apply_arm_fit_fast(n_map, arm_info)
                            else:
                                n_proc = n_map
                            pks = pipe.find_local_peaks_2d(n_proc, threshold_snr=0.0, noise_sigma=1.0, min_distance_pix=3)
                            pks_s = [p["snr"] for p in pks if mask_search[p["y_idx"], p["x_idx"]]]
                            cal_max_snrs.append(max(pks_s) if pks_s else 0.0)

                    snr_th_cal = float(np.percentile(cal_max_snrs, 95))

                    val_fp_trials = 0
                    val_total_trials = 0
                    chance_rates_per_trial = []
                    pa_fprs = {}

                    for pa in POSITION_ANGLES:
                        _, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc)
                        mask_disk_obs = (r_disk_obs <= r_max)
                        mask_search = (r_disk_obs >= r_min) & (r_disk_obs <= r_max)
                        k3_blocks, arm_info = build_linear_projectors(
                            r_disk_obs, phi_disk_obs, mask_disk_obs, inc, r_max, n_rings=15
                        )
                        sample_pix_list = []
                        for phi_s in np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False):
                            xo, yo = pipe.disk_to_obs(d_vortex * np.cos(phi_s), d_vortex * np.sin(phi_s), pa_deg=pa, inc_deg=inc)
                            sample_pix_list.append((center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix))))

                        pa_hits = 0
                        for t in range(n_val_trials):
                            n_map = val_noises[t]
                            if meth == "remove_k3":
                                n_proc = apply_k3_fast(n_map, k3_blocks)
                            elif meth == "arm_fit":
                                n_proc = apply_arm_fit_fast(n_map, arm_info)
                            else:
                                n_proc = n_map
                            pks = pipe.find_local_peaks_2d(n_proc, threshold_snr=snr_th_cal, noise_sigma=1.0, min_distance_pix=3)
                            pks_s = [p for p in pks if mask_search[p["y_idx"], p["x_idx"]]]
                            val_total_trials += 1
                            if len(pks_s) > 0:
                                val_fp_trials += 1
                                pa_hits += 1
                                matched_samples = pipe.evaluate_position_matching(pks_s, sample_pix_list, r_tol_pix=r_tol_pix)
                                chance_rates_per_trial.append(sum(matched_samples) / 36.0)
                            else:
                                chance_rates_per_trial.append(0.0)
                        pa_fprs[pa] = pa_hits / float(n_val_trials)

                    fpr_h0 = val_fp_trials / float(val_total_trials)
                    fpr_h0_upper95 = wilson_upper95(fpr_h0, val_total_trials)
                    f_chance_mean = float(np.mean(chance_rates_per_trial))
                    f_chance_95 = float(np.percentile(chance_rates_per_trial, 95))
                    area_search = np.pi * (r_max**2 - r_min**2)
                    area_tol = np.pi * (r_tol_kpc**2)
                    f_area_ratio = min(1.0, area_tol / area_search)
                    f_chance_eff = max(f_chance_mean, fpr_h0 * f_area_ratio)

                    for sigma_v in NOISES_KMS:
                        key_str = f"{model_id}_{inc}_{fwhm}_{sigma_v}_{meth}"
                        thresholds_dict[key_str] = {
                            "snr_th_v2": round(snr_th_cal, 4),
                            "r_tol_kpc": round(r_tol_kpc, 4),
                            "r_tol_pix": round(r_tol_pix, 2),
                            "fpr_h0": round(fpr_h0, 4),
                            "fpr_h0_upper95": round(fpr_h0_upper95, 4),
                            "fpr_pa0": round(pa_fprs[0.0], 4),
                            "fpr_pa45": round(pa_fprs[45.0], 4),
                            "fpr_pa90": round(pa_fprs[90.0], 4),
                            "f_chance_mean": round(f_chance_mean, 6),
                            "f_chance_95": round(f_chance_95, 6),
                            "f_area_ratio": round(f_area_ratio, 6),
                            "f_chance_eff": round(f_chance_eff, 6)
                        }
                        stage3_rows.append({
                            "run_id": run_id_stage3,
                            "model_id": model_id,
                            "inc_deg": inc,
                            "fwhm_kpc": fwhm,
                            "sigma_v_kms": sigma_v,
                            "arm_method": meth,
                            "xi_c_kpc": round(info["xi_c_kpc"], 5),
                            "sigma_beam_kpc": round(sigma_beam_kpc, 5),
                            "r_tol_kpc": round(r_tol_kpc, 4),
                            "r_tol_pix": round(r_tol_pix, 2),
                            "snr_th_v2": round(snr_th_cal, 4),
                            "fpr_h0": round(fpr_h0, 4),
                            "fpr_h0_upper95": round(fpr_h0_upper95, 4),
                            "fpr_pa0": round(pa_fprs[0.0], 4),
                            "fpr_pa45": round(pa_fprs[45.0], 4),
                            "fpr_pa90": round(pa_fprs[90.0], 4),
                            "f_chance_mean": round(f_chance_mean, 6),
                            "f_chance_95": round(f_chance_95, 6),
                            "f_area_ratio": round(f_area_ratio, 6),
                            "f_chance_eff": round(f_chance_eff, 6)
                        })

    stage3_csv = os.path.join(out_dir, "v2_h0_calibration.csv")
    with open(stage3_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stage3_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stage3_rows)

    stage3_json = os.path.join(out_dir, ".v2_snr_thresholds.json")
    with open(stage3_json, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id_stage3,
            "timestamp_iso": datetime.now().isoformat(),
            "thresholds": thresholds_dict
        }, f, indent=2)

    with open(stage3_csv, "rb") as f:
        stage3_csv_hash = hashlib.sha256(f.read()).hexdigest()
    with open(stage3_json, "rb") as f:
        stage3_json_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"Stage 3 v2 完了 ({time.time() - t0:.1f}s)")
    print(f"  CSV: {stage3_csv} (SHA-256: {stage3_csv_hash})")
    print(f"  JSON: {stage3_json} (SHA-256: {stage3_json_hash})")

    # =========================================================================
    # Phase 2: Pre-register cell-by-cell v2 predictions (All 13,608 cells)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. v2 セル単位事前予測ファイルの生成・固定 (13,608 セル)")
    print("=" * 80)

    beam_cache = {}
    for model_id, info in MODELS.items():
        coords, center_idx, X_obs, Y_obs, _ = grid_cache[model_id]
        dpix = info["dpix"]
        d_v = 0.3 * info["R_kpc"]
        for fwhm in RESOLUTIONS_KPC:
            fwhm_pix = fwhm / dpix
            for inc in INCLINATIONS:
                _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
                dist_sq = (r_disk * np.cos(phi_disk) - d_v)**2 + (r_disk * np.sin(phi_disk))**2
                v_raw = np.exp(-dist_sq / (2.0 * (0.1)**2))
                v_conv = pipe.convolve_beam_normalized(v_raw, fwhm_pix=fwhm_pix)
                eta_beam = float(np.nanmax(v_conv) / np.max(v_raw))
                beam_cache[(model_id, fwhm, inc)] = eta_beam

    cells = []
    pred_rows = []
    cell_idx = 0
    for model_id in ["model_1", "model_2", "model_3"]:
        info = MODELS[model_id]
        for nv in VORTEX_COUNTS:
            for inc in INCLINATIONS:
                for pa in POSITION_ANGLES:
                    for fwhm in RESOLUTIONS_KPC:
                        for sigma_v in NOISES_KMS:
                            for geom_err in GEOM_ERRORS:
                                for arm in ARM_LEVELS:
                                    meth = arm["method"]
                                    a3 = arm["a_los3"]
                                    cell_id = f"v2_{model_id}_{cell_idx:05d}_{nv}_{int(inc)}_{int(pa)}_{fwhm}_{sigma_v}_{geom_err}_{a3}_{meth}"
                                    t_info = thresholds_dict[f"{model_id}_{inc}_{fwhm}_{sigma_v}_{meth}"]
                                    snr_th_cal = t_info["snr_th_v2"]
                                    eta_beam = beam_cache[(model_id, fwhm, inc)]
                                    eta_fit = 0.996 if meth == "remove_k3" else (0.999 if meth == "arm_fit" else 1.0)

                                    a50_sec1_450 = pipe.predict_a50(4.50, sigma_v, eta_beam, eta_fit, inc, 0.0)
                                    a50_cal_stratum = pipe.predict_a50(snr_th_cal, sigma_v, eta_beam, eta_fit, inc, 0.0)

                                    a_phys = info["a_inj_physical"]
                                    ratio_sec1 = a_phys / a50_sec1_450
                                    ratio_cal = a_phys / a50_cal_stratum

                                    if nv == 0:
                                        p_class = "CONTROL"
                                    elif ratio_sec1 >= 1.5:
                                        p_class = "RECOVERED"
                                    elif ratio_sec1 <= 0.67:
                                        p_class = "UNDETECTED"
                                    else:
                                        p_class = "BOUNDARY"

                                    if nv == 0:
                                        p_class_cal = "CONTROL"
                                    elif ratio_cal >= 1.5:
                                        p_class_cal = "RECOVERED"
                                    elif ratio_cal <= 0.67:
                                        p_class_cal = "UNDETECTED"
                                    else:
                                        p_class_cal = "BOUNDARY"

                                    c_dict = {
                                        "cell_idx": cell_idx,
                                        "cell_id": cell_id,
                                        "model_id": model_id,
                                        "n_vortex_inj": nv,
                                        "inc_deg": inc,
                                        "pa_deg": pa,
                                        "fwhm_kpc": fwhm,
                                        "sigma_v_kms": sigma_v,
                                        "geom_error_flag": geom_err,
                                        "a_los3_kms": a3,
                                        "arm_method": meth,
                                        "snr_th_v2": snr_th_cal,
                                        "r_tol_kpc": t_info["r_tol_kpc"],
                                        "r_tol_pix": t_info["r_tol_pix"]
                                    }
                                    cells.append(c_dict)

                                    pred_rows.append({
                                        "cell_id": cell_id,
                                        "model_id": model_id,
                                        "n_vortex_inj": nv,
                                        "inc_deg": inc,
                                        "pa_deg": pa,
                                        "fwhm_kpc": fwhm,
                                        "sigma_v_kms": sigma_v,
                                        "geom_error_flag": geom_err,
                                        "a_los3_kms": a3,
                                        "arm_method": meth,
                                        "eta_beam_proj": round(eta_beam, 4),
                                        "eta_fit": round(eta_fit, 4),
                                        "snr_th_fixed45": 4.50,
                                        "a50_sec1_kms": round(a50_sec1_450, 3),
                                        "ratio_sec1": round(ratio_sec1, 3),
                                        "pred_class": p_class,
                                        "snr_th_cal": snr_th_cal,
                                        "a50_cal_kms": round(a50_cal_stratum, 3),
                                        "ratio_cal": round(ratio_cal, 3),
                                        "pred_class_cal": p_class_cal,
                                        "a_inj_physical_kms": a_phys
                                    })
                                    cell_idx += 1

    pred_csv = os.path.join(out_dir, "v2_cell_predictions_preregistered.csv")
    with open(pred_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    with open(pred_csv, "rb") as f:
        pred_hash = hashlib.sha256(f.read()).hexdigest()

    pred_meta = {
        "run_id_stage3": run_id_stage3,
        "stage3_thresholds_sha256": stage3_json_hash,
        "filename": "v2_cell_predictions_preregistered.csv",
        "timestamp_iso": datetime.now().isoformat(),
        "sha256": pred_hash,
        "total_cells": len(pred_rows),
        "vortex_cells": sum(1 for r in pred_rows if r["n_vortex_inj"] > 0)
    }
    pred_meta_path = os.path.join(out_dir, "v2_cell_predictions_meta.json")
    with open(pred_meta_path, "w", encoding="utf-8") as f:
        json.dump(pred_meta, f, indent=2)

    print(f"v2 セル単位事前予測固定完了: {pred_csv}")
    print(f"  SHA-256: {pred_hash}")

    # =========================================================================
    # Phase 3: Stage 4 v2 Full 13,608-Cell Execution with Matched Null (Parallel)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"3. Stage 4 v2 本番バッチ並列実行開始 (Run ID: {run_id_stage4}, 全 {len(cells)} セル)")
    print("=" * 80)

    t_s4 = time.time()
    n_s4_trials = 60
    worker_args = [(mid, inc, thresholds_dict, n_s4_trials) for mid in ["model_1", "model_2", "model_3"] for inc in INCLINATIONS]

    results_by_cell_id = {}
    with Pool(processes=min(len(worker_args), cpu_count())) as pool:
        worker_dicts = pool.map(run_stage4_worker, worker_args)

    for wd in worker_dicts:
        results_by_cell_id.update(wd)

    s4_rows = []
    for c in cells:
        kt = (c["model_id"], c["n_vortex_inj"], c["inc_deg"], c["pa_deg"], c["fwhm_kpc"],
              c["sigma_v_kms"], c["geom_error_flag"], c["a_los3_kms"], c["arm_method"])
        r_info = results_by_cell_id[kt]
        row = {
            "run_id": run_id_stage4,
            "cell_id": c["cell_id"],
            "model_id": c["model_id"],
            "n_vortex_inj": c["n_vortex_inj"],
            "inc_deg": c["inc_deg"],
            "pa_deg": c["pa_deg"],
            "fwhm_kpc": c["fwhm_kpc"],
            "sigma_v_kms": c["sigma_v_kms"],
            "geom_error_flag": c["geom_error_flag"],
            "a_los3_kms": c["a_los3_kms"],
            "arm_method": c["arm_method"],
            "r_tol_kpc": r_info["r_tol_kpc"],
            "r_tol_pix": r_info["r_tol_pix"],
            "snr_th_v2": r_info["snr_th_v2"],
            "recall_pos": r_info["recall_pos"],
            "null_recall_matched": r_info["null_recall_matched"],
            "null_upper95": r_info["null_upper95"],
            "fpr_null_trial": r_info["fpr_null_trial"],
            "cell_verdict": r_info["cell_verdict"]
        }
        s4_rows.append(row)

    s4_csv = os.path.join(out_dir, "v2_injection_recovery_summary.csv")
    with open(s4_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(s4_rows[0].keys()))
        writer.writeheader()
        writer.writerows(s4_rows)

    with open(s4_csv, "rb") as f:
        s4_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"\nStage 4 v2 完了! Stage 4 所要時間: {time.time() - t_s4:.1f}s (総所要時間: {time.time() - t0:.1f}s)")
    print(f"  CSV: {s4_csv} (SHA-256: {s4_hash})")

    # =========================================================================
    # Phase 4: Summary Statistics & Confusion Matrix
    # =========================================================================
    for mode_label, pred_col in [("Section 1 (SNR_th=4.50)", "pred_class"), ("Calibrated Stratum SNR_th", "pred_class_cal")]:
        print(f"\n=== Confusion Matrix [{mode_label}] (Vortex Cells N=10,206) ===")
        for crit_label, is_rec_fn in [
            ("recall_pos >= 0.50 AND recall_pos > null_upper95 (Matched Null Corrected)", lambda r: r["cell_verdict"] == "CELL_ROBUST_RECOVERED"),
            ("Raw recall_pos >= 0.50 (Uncorrected)", lambda r: r["recall_pos"] >= 0.50)
        ]:
            tp = fp = fn = tn = bnd = 0
            for p, r in zip(pred_rows, s4_rows):
                if r["n_vortex_inj"] == 0:
                    continue
                p_cls = p[pred_col]
                if p_cls == "BOUNDARY":
                    bnd += 1
                    continue
                p_pos = (p_cls == "RECOVERED")
                a_pos = is_rec_fn(r)
                if p_pos and a_pos: tp += 1
                elif p_pos and not a_pos: fp += 1
                elif not p_pos and a_pos: fn += 1
                else: tn += 1
            tot = tp + fp + fn + tn
            acc = (tp + tn) / tot if tot > 0 else 0.0
            all_u = (fn + tn) / tot if tot > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            p_o = acc
            p_yes = ((tp + fp) / tot) * ((tp + fn) / tot) if tot > 0 else 0.0
            p_no = ((fn + tn) / tot) * ((fp + tn) / tot) if tot > 0 else 0.0
            p_e = p_yes + p_no
            kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 1e-9 else 0.0
            print(f"  [{crit_label}] (Excl. {bnd} BOUNDARY, Eval N={tot}):")
            print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            print(f"    Accuracy={acc:.4f} (All-UNDETECTED={all_u:.4f}), Precision={prec:.4f}, Recall={rec:.4f}, Kappa={kappa:.4f}")

    print("\n=== Model-by-Model Recovery Summary in v2 ===")
    for mid in ["model_1", "model_2", "model_3"]:
        v_rows = [r for r in s4_rows if r["model_id"] == mid and r["n_vortex_inj"] > 0]
        raw_rec = sum(1 for r in v_rows if r["recall_pos"] >= 0.50)
        rob_rec = sum(1 for r in v_rows if r["cell_verdict"] == "CELL_ROBUST_RECOVERED")
        fail_null = sum(1 for r in v_rows if r["cell_verdict"] == "CELL_FAIL_MATCHED_NULL")
        max_rec = max(r["recall_pos"] for r in v_rows)
        clean_best = [r for r in v_rows if r["n_vortex_inj"] == 1 and r["a_los3_kms"] == 0.0 and not r["geom_error_flag"] and r["fwhm_kpc"] == 0.2 and r["sigma_v_kms"] == 0.5]
        print(f"  {mid} (Vortex Cells={len(v_rows)}):")
        print(f"    Raw recall_pos >= 0.50: {raw_rec} / {len(v_rows)} ({raw_rec/len(v_rows)*100:.2f}%)")
        print(f"    Matched-Null Corrected (recall>=0.50 & >null_upper95): {rob_rec} / {len(v_rows)} ({rob_rec/len(v_rows)*100:.2f}%)")
        print(f"    Rejected by Matched-Null (Geom/Arm artifact): {fail_null} cells")
        print(f"    Max recall_pos across all cells: {max_rec:.4f}")
        for cb in clean_best:
            print(f"      Clean Nv=1, FWHM=0.2, sv=0.5, inc={cb['inc_deg']}, PA={cb['pa_deg']}: recall_pos={cb['recall_pos']:.4f}, null_rec={cb['null_recall_matched']:.4f} (upper95={cb['null_upper95']:.4f}) -> {cb['cell_verdict']}")


if __name__ == "__main__":
    main()
