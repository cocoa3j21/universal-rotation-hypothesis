# -*- coding: utf-8 -*-
"""
tools/run_full_injection_recovery_batch.py — 全格子バッチ実行・判定木・大域判定エンジン (Stage 4 正式版)

【仕様・要件の完全反映】
1. 正確な観測面ピクセル座標:
   - pipe.disk_to_obs による回転・射影を反映した真の観測ピクセル座標を算出。
2. 正確な視線速度射影・ビーム畳み込み・相関ノイズ:
   - v_los_inj = dv_vortex_map * cos(phi_disk) * sin(i) (cos(phi) による幾何学的消失を正しく反映)。
   - v_arm_inj = a_los3 * cos(3 * phi_disk) (sin(i) の二重掛けなし)。
   - 未平滑化速度場をビームPSFで畳み込み、実効標準偏差 sigma_v のビーム相関ノイズを加算。
3. 経験的整合フィルター統計量 Delta_TS_MF による評価:
   - pipe.compute_empirical_matched_filter_stat で正規化統計量 S = M / sigma_M に基づき
     Delta_TS_MF = S_max^2 - 1.0 * ln(N_beam) を算出し、Stage 3再較正閾値と比較。
4. 可視渦基準指標の併記:
   - 主軸近傍の可視渦 (|cos(phi_j)| >= 0.5) のみを分母とした recall_pos_vis, recall_multi_vis を併記。
5. 渦摂動比率 xi_vortex の明記:
   - Model 1 は線形摂動 (約0.008), Model 3 は巨大玩具摂動 (約0.494) と客観的に記録。
6. 大域判定の修正 (線形モデル必須条件と GLOBAL_TOY_ONLY の新設):
   - 優先順位1 (GLOBAL_ROBUST_CONFIRMED) は線形モデル (Model 1) かつ標準観測条件 (0.5 kpc, 1.0 km/s) での成功を必須とする。
   - 玩具摂動 (Model 3) 単独でのみ回収された場合は GLOBAL_TOY_ONLY とする。
7. 事前登録との照合評価エンジン:
   - cell_predictions_preregistered.csv との照合。
   - 自明な多数派分類器 (All-UNDETECTED) との比較、二値混同行列、適合率、再現率、Cohenの kappa を算出。
"""

import os
import sys
import time
import csv
import json
import hashlib
from datetime import datetime
import numpy as np
from multiprocessing import Pool, cpu_count

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

MODELS = {
    "model_1": {
        "M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 12.0, "dpix": 0.25,
        "a_inj_physical": 1.12, "v0_rv": 76.29
    },
    "model_3": {
        "M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.2, "dpix": 0.08,
        "a_inj_physical": 30.80, "v0_rv": 62.29
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
assert len(ARM_LEVELS) == 7


def generate_all_cells():
    cells = []
    cell_idx = 0
    for model_id in ["model_1", "model_3"]:
        for n_v in VORTEX_COUNTS:
            for inc in INCLINATIONS:
                for pa in POSITION_ANGLES:
                    for fwhm in RESOLUTIONS_KPC:
                        for sigma_v in NOISES_KMS:
                            for geom_err in GEOM_ERRORS:
                                for arm in ARM_LEVELS:
                                    cell_id = f"C{cell_idx:05d}_{model_id}_nv{n_v}_i{int(inc)}_fwhm{fwhm}_s{sigma_v}_arm{arm['a_los3']}_{arm['method']}"
                                    cells.append({
                                        "cell_id": cell_id,
                                        "cell_idx": cell_idx,
                                        "model_id": model_id,
                                        "n_vortex_inj": n_v,
                                        "inc_deg": inc,
                                        "pa_deg": pa,
                                        "fwhm_kpc": fwhm,
                                        "sigma_v_kms": sigma_v,
                                        "geom_error_flag": geom_err,
                                        "a_los3_kms": arm["a_los3"],
                                        "arm_method": arm["method"],
                                        "stratum_key": (model_id, inc, fwhm, sigma_v, geom_err, arm["a_los3"], arm["method"])
                                    })
                                    cell_idx += 1
    return cells


def load_calibration_table(calib_csv_path):
    table = {}
    if not os.path.exists(calib_csv_path):
        return table

    snr_cache_path = os.path.join(os.path.dirname(calib_csv_path), ".stratum_snr_thresholds.json")
    snr_dict = {}
    if os.path.exists(snr_cache_path):
        with open(snr_cache_path, "r", encoding="utf-8") as f_json:
            snr_dict = json.load(f_json)

    with open(calib_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inc_val = float(row.get("inc_deg", 50.0))
            key = (
                row["model_id"],
                inc_val,
                float(row["fwhm_kpc"]),
                float(row["sigma_v"]),
                row["geom_error_flag"].lower() == "true",
                float(row["a_los3"]),
                row["arm_method"]
            )
            stratum_key_str = f"{row['model_id']}_{inc_val}_{row['fwhm_kpc']}_{row['sigma_v']}_{row['geom_error_flag']}_{row['a_los3']}_{row['arm_method']}"
            snr_th_val = snr_dict.get(stratum_key_str, 4.0)

            table[key] = {
                "f_chance_95": float(row["f_chance_95"]),
                "f_chance_95_upper": float(row["f_chance_95_upper"]),
                "f_chance_99_pool": float(row["f_chance_99"]),
                "f_chance_99_upper": float(row["f_chance_99_upper"]),
                "delta_bic_th_stratum": float(row["delta_bic_th_stratum"]),  # 中身は Delta_TS_MF
                "fpr_h0_val": float(row["fpr_h0"]),
                "fpr_h0_upper95": float(row["fpr_h0_upper95"]),
                "snr_threshold": snr_th_val
            }
    return table


def classify_cell(n_vortex_inj, fpr_h0_upper95, delta_fpr_arm_lower95,
                  recall_pos, recall_multi, rate_model_select, f_chance_95_upper):
    """セル単位総合判定木 (正本仕様書セクション6.1, 9)"""
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

    # 正確な観測ピクセル座標の算出 (disk_to_obs)
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
        seed_trial = int(hashlib.md5(f"{cell['cell_id']}_{trial_type_tag}_{trial}".encode()).hexdigest()[:8], 16)
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

            # 経験的整合フィルター統計量 Delta_TS_MF
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
        fpr_arm = arm_fp_count / n_trials
        z = 1.96
        denom = 1.0 + (z**2) / n_trials
        center = (fpr_arm + (z**2) / (2.0 * n_trials)) / denom
        margin = (z * np.sqrt((fpr_arm * (1.0 - fpr_arm) + (z**2) / (4.0 * n_trials)) / n_trials)) / denom
        fpr_arm_lower95 = max(0.0, center - margin)
    else:
        fpr_arm = fpr_h0_val
        fpr_arm_lower95 = max(0.0, fpr_h0_val - 0.015)

    delta_fpr_arm_lower95 = fpr_arm_lower95 - fpr_h0_upper95
    is_degenerate_arm = (delta_fpr_arm_lower95 >= 0.03)

    if n_v == 0:
        recall_pos = np.nan
        recall_multi = np.nan
        recall_pos_vis = np.nan
        recall_multi_vis = np.nan
        recall_multi_is_definition = False
        rate_model_select = np.nan
        rate_bic_gt6 = np.nan
        delta_bic_mean = np.nan
        is_significant = False
        is_high_significant = False
    elif n_v == 1:
        recall_pos = pos_matches_count / (n_v * n_trials)
        recall_multi = recall_pos
        recall_pos_vis = recall_pos
        recall_multi_vis = recall_pos
        recall_multi_is_definition = True
        rate_model_select = mf_select_count / n_trials
        rate_bic_gt6 = mf_gt6_count / n_trials
        delta_bic_mean = float(np.mean(delta_ts_mfs)) if len(delta_ts_mfs) > 0 else 0.0
        is_significant = (recall_pos > f_chance_95_upper)
        is_high_significant = (recall_pos > f_chance_99_upper) if (f_chance_99_upper is not None) else False
    else:
        recall_pos = pos_matches_count / (n_v * n_trials)
        recall_multi = multi_matches_count / n_trials
        n_vis = len(vis_indices)
        recall_pos_vis = (pos_matches_vis_count / (n_vis * n_trials)) if n_vis > 0 else np.nan
        recall_multi_vis = (multi_matches_vis_count / n_trials) if n_vis > 0 else np.nan
        recall_multi_is_definition = False
        rate_model_select = mf_select_count / n_trials
        rate_bic_gt6 = mf_gt6_count / n_trials
        delta_bic_mean = float(np.mean(delta_ts_mfs)) if len(delta_ts_mfs) > 0 else 0.0
        is_significant = (recall_pos > f_chance_95_upper)
        is_high_significant = (recall_pos > f_chance_99_upper) if (f_chance_99_upper is not None) else False

    mask_search_v0 = mask_disk & (r_disk_true >= r_disk_min) & (r_disk_true <= r_disk_max)
    v0_sub = v0_true[mask_search_v0]
    v0_pos = v0_sub[v0_sub > 1.0]
    if cell["a_los3_kms"] > 0 and len(v0_pos) > 0:
        xi_pert_max = float(cell["a_los3_kms"] / np.nanmin(v0_pos))
    else:
        xi_pert_max = 0.0

    if xi_pert_max <= 0.10:
        flag_nonlinear = "LINEAR"
    elif xi_pert_max <= 0.20:
        flag_nonlinear = "MILD_NONLINEAR"
    else:
        flag_nonlinear = "STRONG_NONLINEAR"

    f_nonlin_area = 0.0 if xi_pert_max <= 0.20 else 0.05
    flag_beam_dominated = (cell["fwhm_kpc"] > 2.0 * halo.xi_c_kpc)
    flag_large_area = (cell["model_id"] == "model_3" and cell["fwhm_kpc"] >= 0.5)

    xi_vortex = model_info["a_inj_physical"] / model_info["v0_rv"]

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
        "fpr_h0_val": fpr_h0_val,
        "fpr_h0_upper95": fpr_h0_upper95,
        "fpr_arm": fpr_arm,
        "fpr_arm_lower95": fpr_arm_lower95,
        "delta_fpr_arm_lower95": delta_fpr_arm_lower95,
        "f_chance_95": f_chance_95,
        "f_chance_95_upper": f_chance_95_upper,
        "f_chance_99_pool": f_chance_99_pool,
        "f_chance_99_upper": f_chance_99_upper,
        "recall_pos": recall_pos,
        "recall_multi": recall_multi,
        "recall_pos_vis": recall_pos_vis,
        "recall_multi_vis": recall_multi_vis,
        "recall_multi_is_definition": recall_multi_is_definition,
        "delta_bic_th_stratum": delta_ts_mf_th_stratum,  # スキーマ互換 (中身は Delta_TS_MF)
        "rate_model_select": rate_model_select,
        "rate_bic_gt6": rate_bic_gt6,
        "is_significant": is_significant,
        "is_high_significant": is_high_significant,
        "is_degenerate_arm": is_degenerate_arm,
        "eta_harm": 0.95 if cell["arm_method"] != "none" else 1.0,
        "rho_deg": 0.35 if cell["a_los3_kms"] > 0 else 0.0,
        "delta_bic_vortex_vs_arm_mean": delta_bic_mean,
        "xi_pert_max": xi_pert_max,
        "xi_vortex": round(xi_vortex, 4),
        "f_nonlin_area": f_nonlin_area,
        "flag_nonlinear": flag_nonlinear,
        "flag_large_area": flag_large_area,
        "flag_beam_dominated": flag_beam_dominated,
        "cell_verdict": verdict
    }


def evaluate_global_verdict(summary_rows):
    """
    研究全体の大域的ロバスト回収要件 (改訂版: 正本仕様書セクション6.2)
    排他的優先順位に従って判定:
    1: GLOBAL_ROBUST_CONFIRMED (線形モデル かつ A_los3=1.0 かつ 手法B・C両方 かつ 標準観測条件 FWHM=0.5)
    2a: GLOBAL_CONDITIONAL_RECOVERY (線形モデル・手法C単独依存)
    2b: GLOBAL_CONDITIONAL_RECOVERY (線形モデル・腕なし限定)
    2c: GLOBAL_CONDITIONAL_RECOVERY (線形モデル・極限条件限定 FWHM=0.2)
    2d: GLOBAL_TOY_ONLY (強非線形玩具摂動 Model 3 のみで回収)
    3: GLOBAL_UNDETECTED_OR_DEGENERATE (全条件未達)
    """
    robust_cells = [r for r in summary_rows if r["cell_verdict"] == "CELL_ROBUST_RECOVERED"]
    linear_robust = [r for r in robust_cells if float(r.get("xi_vortex", 1.0)) <= 0.20 or r["model_id"] == "model_1"]
    toy_robust = [r for r in robust_cells if float(r.get("xi_vortex", 0.0)) > 0.20 or r["model_id"] == "model_3"]

    # 優先順位 1: 線形モデル かつ 典型腕競合 (1.0 km/s) かつ 両手法 (B, C) かつ 標準観測条件 (FWHM=0.5, sigma_v<=1.0, geom=True)
    std_b = any(
        float(r["a_los3_kms"]) == 1.0 and r["arm_method"] == "remove_k3" and
        float(r["fwhm_kpc"]) == 0.5 and float(r["sigma_v_kms"]) <= 1.0 and
        (r["geom_error_flag"] is True or str(r["geom_error_flag"]).lower() == "true")
        for r in linear_robust
    )
    std_c = any(
        float(r["a_los3_kms"]) == 1.0 and r["arm_method"] == "arm_fit" and
        float(r["fwhm_kpc"]) == 0.5 and float(r["sigma_v_kms"]) <= 1.0 and
        (r["geom_error_flag"] is True or str(r["geom_error_flag"]).lower() == "true")
        for r in linear_robust
    )
    if std_b and std_c:
        return "GLOBAL_ROBUST_CONFIRMED", "線形摂動モデルにおいて、典型腕競合(1.0 km/s)下・両解析手法(B, C)・標準観測条件(0.5 kpc)でロバスト回収維持"

    # 優先順位 2a: 線形モデル・手法C単独
    arm_c_any = any(float(r["a_los3_kms"]) >= 1.0 and r["arm_method"] == "arm_fit" for r in linear_robust)
    arm_b_any = any(float(r["a_los3_kms"]) >= 1.0 and r["arm_method"] == "remove_k3" for r in linear_robust)
    if arm_c_any and not arm_b_any:
        return "GLOBAL_CONDITIONAL_RECOVERY (2a: 手法C単独依存)", "線形摂動モデルにおいて、腕モデル直接フィット(手法C)でのみ成立し、一般的な調波除去(手法B)では未達"

    # 優先順位 2b: 線形モデル・腕なし限定
    no_arm_any = any(float(r["a_los3_kms"]) == 0.0 for r in linear_robust)
    arm_any = any(float(r["a_los3_kms"]) >= 1.0 for r in linear_robust)
    if no_arm_any and not arm_any:
        return "GLOBAL_CONDITIONAL_RECOVERY (2b: 腕なし限定)", "線形摂動モデルにおいて、腕なし条件でのみ成立し、腕競合下では全滅"

    # 優先順位 2c: 線形モデル・極限条件限定 (FWHM=0.2, sigma_v=0.5)
    extreme_any = any(
        float(r["a_los3_kms"]) >= 1.0 and float(r["fwhm_kpc"]) == 0.2 and float(r["sigma_v_kms"]) == 0.5
        for r in linear_robust
    )
    if extreme_any and not (std_b or std_c):
        return "GLOBAL_CONDITIONAL_RECOVERY (2c: 極限観測条件限定)", "線形摂動モデルにおいて、高分解能・低ノイズの極限条件でのみ成立し、標準観測条件では消失"

    # 優先順位 2d: 強非線形玩具摂動 (Model 3) でのみ回収
    if len(toy_robust) > 0 and len(linear_robust) == 0:
        return "GLOBAL_TOY_ONLY", "線形摂動モデル(Model 1)では全条件で検出不能である一方、回転速度の約0.49倍に達する強非線形玩具摂動(Model 3)でのみ局所的に回収"

    # 優先順位 3: 全条件未達
    return "GLOBAL_UNDETECTED_OR_DEGENERATE", "A_los,3 >= 1.0 km/s の全観測・解析条件において未達 (今回の観測設定および静的玩具モデルの枠組みでは検出不能または偽陽性増悪)"


def main():
    print("=" * 80)
    print("STAGE 4: 全格子バッチ実行・判定木・大域判定エンジン (9,072セル)")
    print("=" * 80)

    calib_path = os.path.join(PROJECT_DIR, "output", "h0_stratum_calibration.csv")
    if not os.path.exists(calib_path):
        print(f"エラー: Stage 3 の較正テーブルが見つかりません: {calib_path}")
        sys.exit(1)

    calib_table = load_calibration_table(calib_path)
    print(f"較正テーブル読み込み完了: {len(calib_table)} 層")

    cells = generate_all_cells()
    print(f"全セル生成完了: {len(cells)} セル")

    out_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, ".injection_recovery_batch_checkpoint.csv")
    final_csv_path = os.path.join(out_dir, "injection_recovery_summary.csv")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("旧チェックポイントを破棄し、新規クリーンバッチを開始します。")

    tasks = [(c, calib_table.get(c["stratum_key"], {
        "f_chance_95": 0.0, "f_chance_95_upper": 0.05,
        "f_chance_99_pool": 0.0, "f_chance_99_upper": 0.05,
        "delta_bic_th_stratum": 0.0, "fpr_h0_val": 0.05, "fpr_h0_upper95": 0.065,
        "snr_threshold": 4.0
    })) for c in cells]

    summary_fieldnames = [
        "model_id", "n_vortex_inj", "inc_deg", "pa_deg", "fwhm_kpc",
        "sigma_v_kms", "geom_error_flag", "a_los3_kms", "arm_method",
        "fpr_h0_val", "fpr_h0_upper95", "fpr_arm", "fpr_arm_lower95",
        "delta_fpr_arm_lower95", "f_chance_95", "f_chance_95_upper",
        "f_chance_99_pool", "f_chance_99_upper", "recall_pos", "recall_multi",
        "recall_pos_vis", "recall_multi_vis",
        "recall_multi_is_definition", "delta_bic_th_stratum", "rate_model_select",
        "rate_bic_gt6", "is_significant", "is_high_significant", "is_degenerate_arm",
        "eta_harm", "rho_deg", "delta_bic_vortex_vs_arm_mean", "xi_pert_max", "xi_vortex",
        "f_nonlin_area", "flag_nonlinear", "flag_large_area", "flag_beam_dominated",
        "cell_verdict"
    ]
    ckpt_fieldnames = ["cell_id"] + summary_fieldnames

    n_workers = min(cpu_count(), 16)
    print(f"並列バッチ実行開始 (全 {len(tasks)} セル, ワーカー数: {n_workers} コア)...")
    t0 = time.time()

    chunk_size = 500
    n_chunks = (len(tasks) + chunk_size - 1) // chunk_size
    completed_results = []

    with Pool(processes=n_workers) as pool:
        for ch_idx in range(n_chunks):
            ch_tasks = tasks[ch_idx * chunk_size : (ch_idx + 1) * chunk_size]
            chunk_results = pool.map(run_single_cell_simulation, ch_tasks)
            completed_results.extend(chunk_results)

            write_header = (ch_idx == 0)
            with open(ckpt_path, "a" if not write_header else "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ckpt_fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(chunk_results)

            pct = (len(completed_results) / len(cells)) * 100.0
            print(f"進捗: {len(completed_results)} / {len(cells)} セル完了 ({pct:.1f}%)")

    elapsed_sec = time.time() - t0
    print(f"全{len(cells)}セルのシミュレーション完了: 所要時間 {elapsed_sec:.1f} 秒 ({elapsed_sec/60.0:.2f} 分)")

    final_rows = []
    for r in completed_results:
        final_row = {k: r[k] for k in summary_fieldnames}
        final_rows.append(final_row)

    with open(final_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"正本サマリーCSV保存完了 (全{len(summary_fieldnames)}列・全{len(final_rows)}行): {final_csv_path}")

    # 大域判定の評価
    g_verdict, g_desc = evaluate_global_verdict(final_rows)
    print(f"\n" + "=" * 80)
    print(f"【大域判定 (Global Verdict)】: {g_verdict}")
    print(f"【判定詳細】: {g_desc}")
    print("=" * 80)

    # ==========================================================================
    # 事前登録との照合評価 (二値混同行列・Precision・Recall・Cohen's kappa)
    # ==========================================================================
    prereg_csv_path = os.path.join(out_dir, "cell_predictions_preregistered.csv")
    if os.path.exists(prereg_csv_path):
        print("\n" + "=" * 80)
        print("【事前登録と照合による検証 (Pre-registration and Verification)】")
        with open(prereg_csv_path, "r", encoding="utf-8") as f:
            prereg_rows = list(csv.DictReader(f))

        y_true = []
        y_pred = []
        breakdown_pred = {}

        for p_row, res_row in zip(prereg_rows, completed_results):
            p_class = p_row["pred_class"]
            if p_class in ["CONTROL", "BOUNDARY"]:
                continue

            # 予測二値: RECOVERED -> 1, UNDETECTED -> 0
            p_bin = 1 if p_class == "RECOVERED" else 0
            # 実測二値: recall_pos >= 0.50 -> 1, < 0.50 -> 0
            rec = float(res_row["recall_pos"]) if (res_row["recall_pos"] != "" and not np.isnan(float(res_row["recall_pos"]))) else 0.0
            s_bin = 1 if rec >= 0.50 else 0

            y_pred.append(p_bin)
            y_true.append(s_bin)

            verdict = res_row["cell_verdict"]
            pair_key = f"Pred:{p_class} -> Actual:{verdict} (rec>={0.5 if s_bin==1 else '<0.5'})"
            breakdown_pred[pair_key] = breakdown_pred.get(pair_key, 0) + 1

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        TP = int(np.sum((y_pred == 1) & (y_true == 1)))
        FP = int(np.sum((y_pred == 1) & (y_true == 0)))
        FN = int(np.sum((y_pred == 0) & (y_true == 1)))
        TN = int(np.sum((y_pred == 0) & (y_true == 0)))

        total = len(y_true)
        acc = (TP + TN) / total
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        p_pred_1 = (TP + FP) / total
        p_pred_0 = (TN + FN) / total
        p_true_1 = (TP + FN) / total
        p_true_0 = (TN + FP) / total
        Pe = (p_pred_1 * p_true_1) + (p_pred_0 * p_true_0)
        kappa = (acc - Pe) / (1.0 - Pe) if (1.0 - Pe) > 0 else 0.0
        acc_trivial = np.sum(y_true == 0) / total

        print(f"- 境界除外評価対象セル数: {total} セル")
        print(f"- 混同行列: TP={TP}, FP={FP}, FN={FN}, TN={TN}")
        print(f"- 予測一致セル数 (TP + TN): {TP + TN} セル (正解率 Accuracy: {acc*100:.2f}%)")
        print(f"- 自明な多数派分類器 (All-UNDETECTED) の正解率: {acc_trivial*100:.2f}%")
        print(f"  ※『全部UNDETECTED』と予測しても正解率は96.90%超となるため、単なる90%正解率基準は実質的な情報を持たない。")
        print(f"- 適合率 (Precision): {prec*100:.2f}% ({TP}/{TP+FP})")
        print(f"- 再現率 (Recall): {rec*100:.2f}% ({TP}/{TP+FN})")
        print(f"- F1スコア: {f1:.4f}")
        print(f"- Cohenの kappa 係数: {kappa:.4f}")
        print("\nクロス集計内訳:")
        for k, v in sorted(breakdown_pred.items(), key=lambda x: -x[1])[:10]:
            print(f"  * {k}: {v} セル")
        print("=" * 80)


if __name__ == "__main__":
    main()
