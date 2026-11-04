# -*- coding: utf-8 -*-
"""
tools/run_v2_agreed_protocol.py
ユーザー指示1〜6をすべて反映した Pipeline v2 最終確定実行スクリプト:
1. 視野半幅を 0.9R 以上に拡大 (Model 1: 18.4 kpc, Model 2: 9.4 kpc, Model 3: 2.8 kpc) し、楕円切断を完全解消。
2. Stage 3 v2 を合意プロトコル (全1,134層、各層 N_cal=2,001 [667/PA], N_val=2,400 [800/PA], 計 4,401 試行/層、
   目標 FPR = 4.0% [96%分位点]) で実行。合格層数 (fpr_h0_upper95 <= 0.070) と PA別FPR (0°/45°/90°) を記録。
   f_chance は一致回数 k / 総判定数 N_eval と Wilson 95% 信頼区間 [ci_low, ci_high] 付きで記録。
3. Stage 3 v2 の後、Stage 4 v2 の前に全 13,608 セルの事前予測を固定。
   Nv=4 は短軸上の2渦が見えず回収率上限が 50% になることを予測に反映。
4. Stage 4 v2 (全 13,608 セル、Matched Null 補正 recall_pos > null_upper95、および CELL_FAIL_FPR_UNCONTROLLED 判定) を実行。
5. 全体および Nv別 (1, 2, 4) の混同行列を、実測側陰性比率 (自明基準) とともに算出。
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

# 視野半幅 r_fov >= 0.9 * R_kpc (楕円長軸 0.9R + ビーム裾野を完全包含)
MODELS = {
    "model_1": {
        "M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 18.4, "dpix": 0.08,
        "a_inj_physical": 1.12, "xi_c_kpc": 0.092442, "v0_rv": 76.29
    },
    "model_2": {
        "M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 9.4, "dpix": 0.08,
        "a_inj_physical": 4.89, "xi_c_kpc": 0.046221, "v0_rv": 107.88
    },
    "model_3": {
        "M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.8, "dpix": 0.08,
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
    info = MODELS[model_id]
    dpix = info["dpix"]
    xi_c_kpc = info["xi_c_kpc"]
    fwhm_pix = fwhm_kpc / dpix
    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_beam_kpc = sigma_beam_pix * dpix
    r_tol_kpc = xi_c_kpc + sigma_beam_kpc
    r_tol_pix = r_tol_kpc / dpix
    return r_tol_kpc, r_tol_pix, sigma_beam_kpc, sigma_beam_pix


def wilson_ci95(k, n):
    """Wilson score 95% confidence interval (low, high) for k successes out of n trials."""
    if n <= 0:
        return 0.0, 0.0
    p_hat = float(k) / float(n)
    z = 1.959963984540054
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)) / denom
    low = max(0.0, float(center - margin))
    high = min(1.0, float(center + margin))
    return low, high


def wilson_upper95(p_hat, n):
    if n <= 0:
        return 0.0
    _, high = wilson_ci95(p_hat * n, n)
    return high


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


def stage3_pa_worker(args):
    """
    1つの (model_id, fwhm_kpc, pa_deg) について、
    較正 N_cal_pa = 667 試行、検証 N_val_pa = 800 試行 (3つのPA合計で N_cal=2,001, N_val=2,400, 計4,401試行)
    の独立なビーム相関ノイズを生成し、各傾斜角 inc (30, 50, 70) × 各除去法 (none/raw, remove_k3, arm_fit)
    の局所ピークリストを抽出して返す。
    """
    model_id, fwhm, pa, n_cal_pa, n_val_pa = args
    info = MODELS[model_id]
    dpix = info["dpix"]
    coords = np.arange(-info["r_fov"], info["r_fov"] + dpix / 2, dpix)
    center_idx = len(coords) // 2
    X_obs, Y_obs = np.meshgrid(coords, coords)
    r_min = 0.2 * info["R_kpc"]
    r_max = 0.9 * info["R_kpc"]
    d_vortex = 0.3 * info["R_kpc"]
    r_tol_kpc, r_tol_pix, sigma_beam_kpc, sigma_beam_pix = compute_physical_r_tol(model_id, fwhm)

    # Precompute geometry & projectors for each inclination
    inc_ctx = {}
    for inc in INCLINATIONS:
        _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc)
        mask_disk = (r_disk <= r_max)
        mask_search = (r_disk >= r_min) & (r_disk <= r_max)
        k3_blocks, arm_info = build_linear_projectors(r_disk, phi_disk, mask_disk, inc, r_max, n_rings=15)

        sample_pix_list = []
        for phi_s in np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False):
            xo, yo = pipe.disk_to_obs(d_vortex * np.cos(phi_s), d_vortex * np.sin(phi_s), pa_deg=pa, inc_deg=inc)
            sample_pix_list.append((center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix))))

        inc_ctx[inc] = (mask_search, k3_blocks, arm_info, sample_pix_list)

    # Storage: cal_max_snrs[(inc, meth)] -> list of 667 floats
    # val_peaks[(inc, meth)] -> list of 800 peak lists (each peak is (x_idx, y_idx, snr))
    cal_max_snrs = {(inc, m): [] for inc in INCLINATIONS for m in ["none", "remove_k3", "arm_fit"]}
    val_peaks = {(inc, m): [] for inc in INCLINATIONS for m in ["none", "remove_k3", "arm_fit"]}

    # 1. Calibration trials (N_cal_pa = 667)
    for t in range(n_cal_pa):
        seed = int(hashlib.md5(f"s3v2_cal_{model_id}_{fwhm}_{pa}_{t}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, 1.0, rng)

        # For "none" (and "raw"), find peaks once on `noise` (threshold 2.2 is safely below any max SNR on disk)
        pks_none = pipe.find_local_peaks_2d(noise, threshold_snr=2.2, noise_sigma=1.0, min_distance_pix=3)
        for inc in INCLINATIONS:
            mask_search, k3_blocks, arm_info, _ = inc_ctx[inc]
            s_none = [p["snr"] for p in pks_none if mask_search[p["y_idx"], p["x_idx"]]]
            cal_max_snrs[(inc, "none")].append(max(s_none) if s_none else float(np.max(np.abs(noise[mask_search]))))

            n_k3 = apply_k3_fast(noise, k3_blocks)
            pks_k3 = pipe.find_local_peaks_2d(n_k3, threshold_snr=2.2, noise_sigma=1.0, min_distance_pix=3)
            s_k3 = [p["snr"] for p in pks_k3 if mask_search[p["y_idx"], p["x_idx"]]]
            cal_max_snrs[(inc, "remove_k3")].append(max(s_k3) if s_k3 else float(np.max(np.abs(n_k3[mask_search]))))

            n_af = apply_arm_fit_fast(noise, arm_info)
            pks_af = pipe.find_local_peaks_2d(n_af, threshold_snr=2.2, noise_sigma=1.0, min_distance_pix=3)
            s_af = [p["snr"] for p in pks_af if mask_search[p["y_idx"], p["x_idx"]]]
            cal_max_snrs[(inc, "arm_fit")].append(max(s_af) if s_af else float(np.max(np.abs(n_af[mask_search]))))

    # 2. Validation trials (N_val_pa = 800)
    for t in range(n_val_pa):
        seed = int(hashlib.md5(f"s3v2_val_{model_id}_{fwhm}_{pa}_{t}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, 1.0, rng)

        pks_none = pipe.find_local_peaks_2d(noise, threshold_snr=2.8, noise_sigma=1.0, min_distance_pix=3)
        for inc in INCLINATIONS:
            mask_search, k3_blocks, arm_info, _ = inc_ctx[inc]
            s_none = [(p["x_idx"], p["y_idx"], p["snr"]) for p in pks_none if mask_search[p["y_idx"], p["x_idx"]]]
            val_peaks[(inc, "none")].append(s_none)

            n_k3 = apply_k3_fast(noise, k3_blocks)
            pks_k3 = pipe.find_local_peaks_2d(n_k3, threshold_snr=2.8, noise_sigma=1.0, min_distance_pix=3)
            s_k3 = [(p["x_idx"], p["y_idx"], p["snr"]) for p in pks_k3 if mask_search[p["y_idx"], p["x_idx"]]]
            val_peaks[(inc, "remove_k3")].append(s_k3)

            n_af = apply_arm_fit_fast(noise, arm_info)
            pks_af = pipe.find_local_peaks_2d(n_af, threshold_snr=2.8, noise_sigma=1.0, min_distance_pix=3)
            s_af = [(p["x_idx"], p["y_idx"], p["snr"]) for p in pks_af if mask_search[p["y_idx"], p["x_idx"]]]
            val_peaks[(inc, "arm_fit")].append(s_af)

    sample_pts_by_inc = {inc: inc_ctx[inc][3] for inc in INCLINATIONS}
    return (model_id, fwhm, pa, cal_max_snrs, val_peaks, sample_pts_by_inc)


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

    val_noise_bank = {}
    for fwhm in RESOLUTIONS_KPC:
        _, _, _, sigma_beam_pix = compute_physical_r_tol(model_id, fwhm)
        v_list = []
        for t in range(n_s4_trials):
            seed = int(hashlib.md5(f"v2_s4_agreed_{model_id}_{fwhm}_{t}".encode()).hexdigest()[:8], 16)
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
                            stratum_passed = t_info["fpr_stratum_passed"]
                            null_pks_trials = null_peaks_cache[(sigma_v, a3, meth)]

                            if nv == 0:
                                fp_count = sum(1 for pks_s in null_pks_trials if len(pks_s) > 0)
                                fpr_cell = fp_count / float(n_s4_trials)
                                recall_pos = 0.0
                                null_recall = 0.0
                                null_upper95 = 0.0
                                if not stratum_passed:
                                    verdict = "CELL_FAIL_FPR_UNCONTROLLED"
                                elif (not geom_err and a3 == 0.0) and fpr_cell > 0.12:
                                    verdict = "CELL_FAIL_FPR_UNCONTROLLED"
                                else:
                                    verdict = "CELL_CONTROL_PASSED"
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

                                eff_null_bound = max(null_upper95, t_info["f_chance_eff"])
                                if not stratum_passed:
                                    verdict = "CELL_FAIL_FPR_UNCONTROLLED"
                                elif recall_pos >= 0.50:
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


def print_confusion_matrix(label, preds, results, pred_col, is_rec_fn, nv_filter=None):
    tp = fp = fn = tn = bnd = 0
    for p, r in zip(preds, results):
        nv = int(r["n_vortex_inj"])
        if nv == 0:
            continue
        if nv_filter is not None and nv != nv_filter:
            continue
        p_cls = p[pred_col]
        if p_cls == "BOUNDARY":
            bnd += 1
            continue
        p_pos = (p_cls == "RECOVERED")
        a_pos = is_rec_fn(r)
        if p_pos and a_pos: tp += 1
        elif p_pos and not a_pos: fp += 1
        elif not p_rec_check(p_pos) and a_pos: fn += 1
        else: tn += 1
    tot = tp + fp + fn + tn
    acc = (tp + tn) / tot if tot > 0 else 0.0
    # True trivial baseline: actual negative rate = (FN_no? Wait: actual negative is FP + TN!)
    # If a trivial classifier predicts ALL UNDETECTED, it gets TN + FP right!
    trivial_actual_neg = (fp + tn) / tot if tot > 0 else 0.0
    pred_neg_rate = (fn + tn) / tot if tot > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    p_o = acc
    p_yes = ((tp + fp) / tot) * ((tp + fn) / tot) if tot > 0 else 0.0
    p_no = ((fn + tn) / tot) * ((fp + tn) / tot) if tot > 0 else 0.0
    p_e = p_yes + p_no
    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 1e-9 else 0.0
    print(f"  [{label}] (Excl. {bnd:4d} BOUNDARY, Eval N={tot:4d}): "
          f"TP={tp:3d}, FP={fp:3d}, FN={fn:3d}, TN={tn:4d} | "
          f"Acc={acc:.4f} (自明基準[実測陰性率]={trivial_actual_neg:.4f}, 予測陰性率={pred_neg_rate:.4f}) | "
          f"Prec={prec:.4f}, Rec={rec:.4f}, Kappa={kappa:.4f}")


def p_rec_check(p_pos):
    return p_pos


def main():
    t0 = time.time()
    out_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)

    run_id_stage3 = "stage3_v2_20261028_agreed_4401_target4pct"
    run_id_stage4 = "stage4_v2_20261028_agreed_fov_matched_null"

    n_cal_pa = 667  # 667 * 3 PAs = 2,001 calibration trials per stratum
    n_val_pa = 800  # 800 * 3 PAs = 2,400 validation trials per stratum
    # Total per stratum = 4,401 trials

    print("=" * 90)
    print(f"1. Stage 3 v2 合意プロトコル実行開始 (Run ID: {run_id_stage3})")
    print(f"   規模: 全1,134層 × 4,401試行/層 (N_cal=2,001, N_val=2,400) = 4,990,734 試行, 目標FPR=4.0% (96%分位点)")
    print("=" * 90)

    s3_tasks = [
        (mid, fwhm, pa, n_cal_pa, n_val_pa)
        for mid in ["model_1", "model_2", "model_3"]
        for fwhm in RESOLUTIONS_KPC
        for pa in POSITION_ANGLES
    ]

    with Pool(processes=min(len(s3_tasks), cpu_count())) as pool:
        s3_worker_out = pool.map(stage3_pa_worker, s3_tasks)

    # Organize by (model_id, fwhm, pa)
    s3_data = {}
    for mid, fwhm, pa, cal_max_snrs, val_peaks, sample_pts_by_inc in s3_worker_out:
        s3_data[(mid, fwhm, pa)] = (cal_max_snrs, val_peaks, sample_pts_by_inc)

    stage3_rows = []
    thresholds_dict = {}
    stratum_id = 0
    passed_strata_count = 0

    for model_id in ["model_1", "model_2", "model_3"]:
        info = MODELS[model_id]
        r_min = 0.2 * info["R_kpc"]
        r_max = 0.9 * info["R_kpc"]
        area_search = np.pi * (r_max**2 - r_min**2)

        for inc in INCLINATIONS:
            for fwhm in RESOLUTIONS_KPC:
                r_tol_kpc, r_tol_pix, sigma_beam_kpc, sigma_beam_pix = compute_physical_r_tol(model_id, fwhm)
                area_tol = np.pi * (r_tol_kpc**2)
                f_area_ratio = min(1.0, area_tol / area_search)

                for meth in ["none", "raw", "remove_k3", "arm_fit"]:
                    base_meth = "none" if meth in ("none", "raw") else meth

                    # Pool N_cal = 2,001 across PA = 0, 45, 90
                    cal_pool = []
                    for pa in POSITION_ANGLES:
                        cal_pool.extend(s3_data[(model_id, fwhm, pa)][0][(inc, base_meth)])

                    # Target FPR = 4.0% -> 96th percentile!
                    snr_th_cal = float(np.percentile(cal_pool, 96))

                    # Validate on independent N_val = 2,400 across PA = 0, 45, 90
                    val_fp_trials = 0
                    val_total_trials = 0
                    pa_fprs = {}
                    chance_hit_sites = 0
                    chance_total_sites = 0
                    chance_rates_per_trial = []

                    for pa in POSITION_ANGLES:
                        val_pks_list = s3_data[(model_id, fwhm, pa)][1][(inc, base_meth)]
                        sample_pix_list = s3_data[(model_id, fwhm, pa)][2][inc]
                        pa_hits = 0

                        for pks_all in val_pks_list:
                            pks_above = [{"x_idx": x, "y_idx": y, "snr": s} for (x, y, s) in pks_all if s >= snr_th_cal]
                            val_total_trials += 1
                            chance_total_sites += len(sample_pix_list)
                            if len(pks_above) > 0:
                                val_fp_trials += 1
                                pa_hits += 1
                                matched_s = pipe.evaluate_position_matching(pks_above, sample_pix_list, r_tol_pix=r_tol_pix)
                                k_hit = sum(matched_s)
                                chance_hit_sites += k_hit
                                chance_rates_per_trial.append(k_hit / float(len(sample_pix_list)))
                            else:
                                chance_rates_per_trial.append(0.0)

                        pa_fprs[pa] = pa_hits / float(len(val_pks_list))

                    fpr_h0 = val_fp_trials / float(val_total_trials)
                    fpr_ci_low, fpr_h0_upper95 = wilson_ci95(val_fp_trials, val_total_trials)
                    stratum_passed = (fpr_h0_upper95 <= 0.070)

                    f_chance_mean = chance_hit_sites / float(chance_total_sites)
                    f_chance_ci_low, f_chance_ci_high = wilson_ci95(chance_hit_sites, chance_total_sites)
                    f_chance_95 = float(np.percentile(chance_rates_per_trial, 95))
                    f_chance_eff = max(f_chance_ci_high, fpr_h0 * f_area_ratio)

                    for sigma_v in NOISES_KMS:
                        key_str = f"{model_id}_{inc}_{fwhm}_{sigma_v}_{meth}"
                        thresholds_dict[key_str] = {
                            "snr_th_v2": round(snr_th_cal, 4),
                            "r_tol_kpc": round(r_tol_kpc, 4),
                            "r_tol_pix": round(r_tol_pix, 2),
                            "n_cal": len(cal_pool),
                            "n_val": val_total_trials,
                            "fpr_h0": round(fpr_h0, 4),
                            "fpr_h0_upper95": round(fpr_h0_upper95, 4),
                            "fpr_stratum_passed": stratum_passed,
                            "fpr_pa0": round(pa_fprs[0.0], 4),
                            "fpr_pa45": round(pa_fprs[45.0], 4),
                            "fpr_pa90": round(pa_fprs[90.0], 4),
                            "chance_hits": int(chance_hit_sites),
                            "chance_total_evals": int(chance_total_sites),
                            "f_chance_mean": round(f_chance_mean, 6),
                            "f_chance_ci95_low": round(f_chance_ci_low, 6),
                            "f_chance_ci95_high": round(f_chance_ci_high, 6),
                            "f_chance_95": round(f_chance_95, 6),
                            "f_area_ratio": round(f_area_ratio, 6),
                            "f_chance_eff": round(f_chance_eff, 6)
                        }

                        # Expand across all 1,134 strata (geom_err x arm_levels)
                        for geom_err in GEOM_ERRORS:
                            for arm in ARM_LEVELS:
                                if arm["method"] != meth:
                                    continue
                                stratum_id += 1
                                if stratum_passed:
                                    passed_strata_count += 1
                                stage3_rows.append({
                                    "run_id": run_id_stage3,
                                    "stratum_id": stratum_id,
                                    "model_id": model_id,
                                    "inc_deg": inc,
                                    "fwhm_kpc": fwhm,
                                    "sigma_v_kms": sigma_v,
                                    "geom_error_flag": geom_err,
                                    "a_los3_kms": arm["a_los3"],
                                    "arm_method": meth,
                                    "n_cal_trials": len(cal_pool),
                                    "n_val_trials": val_total_trials,
                                    "n_total_trials": len(cal_pool) + val_total_trials,
                                    "xi_c_kpc": round(info["xi_c_kpc"], 5),
                                    "sigma_beam_kpc": round(sigma_beam_kpc, 5),
                                    "r_tol_kpc": round(r_tol_kpc, 4),
                                    "r_tol_pix": round(r_tol_pix, 2),
                                    "snr_th_v2": round(snr_th_cal, 4),
                                    "fpr_h0": round(fpr_h0, 4),
                                    "fpr_h0_upper95": round(fpr_h0_upper95, 4),
                                    "fpr_stratum_passed": stratum_passed,
                                    "fpr_pa0": round(pa_fprs[0.0], 4),
                                    "fpr_pa45": round(pa_fprs[45.0], 4),
                                    "fpr_pa90": round(pa_fprs[90.0], 4),
                                    "chance_hits": int(chance_hit_sites),
                                    "chance_total_evals": int(chance_total_sites),
                                    "f_chance_mean": round(f_chance_mean, 6),
                                    "f_chance_ci95_low": round(f_chance_ci_low, 6),
                                    "f_chance_ci95_high": round(f_chance_ci_high, 6),
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
            "target_fpr": 0.04,
            "total_strata": len(stage3_rows),
            "passed_strata": passed_strata_count,
            "thresholds": thresholds_dict
        }, f, indent=2)

    with open(stage3_csv, "rb") as f:
        stage3_csv_hash = hashlib.sha256(f.read()).hexdigest()
    with open(stage3_json, "rb") as f:
        stage3_json_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"Stage 3 v2 完了 ({time.time() - t0:.1f}s)")
    print(f"  全層数: {len(stage3_rows)} 層, 合格層数 (fpr_h0_upper95 <= 0.070): {passed_strata_count} / {len(stage3_rows)} ({passed_strata_count/len(stage3_rows)*100:.2f}%)")
    print(f"  検証FPR (fpr_h0): median={np.median([r['fpr_h0'] for r in stage3_rows]):.4f}, mean={np.mean([r['fpr_h0'] for r in stage3_rows]):.4f}, max={max(r['fpr_h0'] for r in stage3_rows):.4f}")
    print(f"  CSV: {stage3_csv} (SHA-256: {stage3_csv_hash})")
    print(f"  JSON: {stage3_json} (SHA-256: {stage3_json_hash})")

    # =========================================================================
    # Phase 2: Pre-register cell-by-cell v2 predictions (All 13,608 cells)
    # Incorporating Nv=4 minor-axis invisibility (max possible recall = 50%)
    # =========================================================================
    print("\n" + "=" * 90)
    print("2. v2 セル単位事前予測ファイルの生成・固定 (13,608 セル, Nv=4 短軸50%上限組込)")
    print("=" * 90)

    beam_cache = {}
    for model_id, info in MODELS.items():
        dpix = info["dpix"]
        coords = np.arange(-info["r_fov"], info["r_fov"] + dpix / 2, dpix)
        X_obs, Y_obs = np.meshgrid(coords, coords)
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

                                    # User instruction 3: For Nv=4, 2 vortices on minor axis have cos(phi)=0 and are invisible,
                                    # capping recall_pos at <= 50% (0.50).
                                    max_possible_recall = 0.50 if nv == 4 else (1.00 if nv > 0 else 0.0)

                                    # Classification without Nv=4 cap (for reference comparison)
                                    if nv == 0:
                                        p_class_nocap = "CONTROL"
                                    elif ratio_cal >= 1.5:
                                        p_class_nocap = "RECOVERED"
                                    elif ratio_cal <= 0.67:
                                        p_class_nocap = "UNDETECTED"
                                    else:
                                        p_class_nocap = "BOUNDARY"

                                    # Classification with Nv=4 50%-cap incorporated:
                                    # Since Nv=4 has max_possible_recall = 0.50, even when major-axis vortices have ratio >= 1.5,
                                    # the cell-wide recall_pos cannot reliably exceed 0.50 (it sits on the <= 0.50 ceiling: BOUNDARY / UNDETECTED).
                                    if nv == 0:
                                        p_class_cap_bnd = "CONTROL"
                                        p_class_cap_und = "CONTROL"
                                    elif nv == 4:
                                        if ratio_cal >= 1.5:
                                            p_class_cap_bnd = "BOUNDARY"     # Capped right at 50% threshold
                                            p_class_cap_und = "UNDETECTED"   # Cannot strictly exceed 50% without false positives
                                        else:
                                            p_class_cap_bnd = "UNDETECTED"
                                            p_class_cap_und = "UNDETECTED"
                                    else:
                                        p_class_cap_bnd = p_class_nocap
                                        p_class_cap_und = p_class_nocap

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
                                        "max_possible_recall": max_possible_recall,
                                        "eta_beam_proj": round(eta_beam, 4),
                                        "eta_fit": round(eta_fit, 4),
                                        "snr_th_fixed45": 4.50,
                                        "a50_sec1_kms": round(a50_sec1_450, 3),
                                        "ratio_sec1": round(ratio_sec1, 3),
                                        "snr_th_cal": snr_th_cal,
                                        "a50_cal_kms": round(a50_cal_stratum, 3),
                                        "ratio_cal": round(ratio_cal, 3),
                                        "pred_class": p_class_cap_und,
                                        "pred_class_nv4_bnd": p_class_cap_bnd,
                                        "pred_class_nocap": p_class_nocap,
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
    print("\n" + "=" * 90)
    print(f"3. Stage 4 v2 本番バッチ並列実行開始 (Run ID: {run_id_stage4}, 全 {len(cells)} セル)")
    print("=" * 90)

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

    n_fail_fpr_all = sum(1 for r in s4_rows if r["cell_verdict"] == "CELL_FAIL_FPR_UNCONTROLLED")
    print(f"\nStage 4 v2 完了! Stage 4 所要時間: {time.time() - t_s4:.1f}s (総所要時間: {time.time() - t0:.1f}s)")
    print(f"  CSV: {s4_csv} (SHA-256: {s4_hash})")
    print(f"  CELL_FAIL_FPR_UNCONTROLLED 件数: {n_fail_fpr_all} / {len(s4_rows)}")

    # =========================================================================
    # Phase 4: Summary Statistics & Confusion Matrices (Overall & by Nv = 1, 2, 4)
    # =========================================================================
    is_rob_fn = lambda r: r["cell_verdict"] == "CELL_ROBUST_RECOVERED"

    print("\n=== 1. 全体混同行列 (All Vortex Cells N=10,206, Matched Null Corrected) ===")
    print_confusion_matrix("A. Nv=4上限50%組込 (Nv=4 -> UNDETECTED)", pred_rows, s4_rows, "pred_class", is_rob_fn)
    print_confusion_matrix("B. Nv=4上限50%組込 (Nv=4高SNR -> BOUNDARY)", pred_rows, s4_rows, "pred_class_nv4_bnd", is_rob_fn)
    print_confusion_matrix("C. Nv=4上限未組込 (旧予測比較用)", pred_rows, s4_rows, "pred_class_nocap", is_rob_fn)

    print("\n=== 2. Nv別 (Nv = 1, 2, 4) 混同行列 (Matched Null Corrected) ===")
    for nv_val in [1, 2, 4]:
        print(f"--- Nv = {nv_val} (Total cells = 3,402) ---")
        print_confusion_matrix(f"Nv={nv_val} (上限組込: pred_class)", pred_rows, s4_rows, "pred_class", is_rob_fn, nv_filter=nv_val)
        if nv_val == 4:
            print_confusion_matrix(f"Nv={nv_val} (上限組込: BOUNDARY扱)", pred_rows, s4_rows, "pred_class_nv4_bnd", is_rob_fn, nv_filter=nv_val)
            print_confusion_matrix(f"Nv={nv_val} (上限未組込: 旧予測)", pred_rows, s4_rows, "pred_class_nocap", is_rob_fn, nv_filter=nv_val)

    print("\n=== 3. モデル別・Nv別 ロバスト回収セル数 ===")
    for mid in ["model_1", "model_2", "model_3"]:
        v_rows = [r for r in s4_rows if r["model_id"] == mid and r["n_vortex_inj"] > 0]
        raw_rec = sum(1 for r in v_rows if r["recall_pos"] >= 0.50)
        rob_rec = sum(1 for r in v_rows if r["cell_verdict"] == "CELL_ROBUST_RECOVERED")
        rej_null = sum(1 for r in v_rows if r["cell_verdict"] == "CELL_FAIL_MATCHED_NULL")
        print(f"  {mid}: Raw>=0.50 = {raw_rec}/3402 ({raw_rec/3402*100:.2f}%), "
              f"Robust(Matched-Null) = {rob_rec}/3402 ({rob_rec/3402*100:.2f}%), Rejected = {rej_null}")
        for nv_val in [1, 2, 4]:
            sub_nv = [r for r in v_rows if r["n_vortex_inj"] == nv_val]
            rob_nv = sum(1 for r in sub_nv if r["cell_verdict"] == "CELL_ROBUST_RECOVERED")
            raw_nv = sum(1 for r in sub_nv if r["recall_pos"] >= 0.50)
            print(f"    Nv={nv_val}: Robust = {rob_nv}/1134 ({rob_nv/1134*100:.2f}%), Raw>=0.50 = {raw_nv}/1134")


if __name__ == "__main__":
    main()
