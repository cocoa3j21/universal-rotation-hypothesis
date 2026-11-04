# -*- coding: utf-8 -*-
"""
tools/run_h0_calibration.py — H0較正・層別プール帰無分布計算 (Stage 3 合意完全プロトコル版)

【仕様・要件の完全反映】
1. 全756層体系: Model 2種 x 傾斜角 3種 x 分解能 3種 x ノイズ 3種 x 幾何誤差 2種 x 腕 7水準 = 756層。
2. 合意試行数プロトコル:
   - 各層 4,400 試行 (較正用 N_cal = 2,000 試行 + 独立検証用 N_val = 2,400 試行)。
   - 各セル (PA=0°, 45°, 90° の3セル) あたり: 較正用 667 試行 + 検証用 800 試行。
   - 総H0試行数: 756層 x 4,400試行 = 3,326,400 試行 (約333万試行)。
3. 経験的整合フィルター統計量 Delta_TS_MF による修正:
   - 空間相関ノイズ場における正規化整合フィルター出力 S = M / sigma_M (分散=1.0) に基づき、
     Delta_TS_MF = S_max^2 - 1.0 * ln(N_beam) を算出。
   - 従来の白色ノイズ前提BIC解釈を取り下げ、渦なしデータで中央値が小さい健全な値 (数オーダー) となることを保証。
4. 独立検証プロトコル:
   - 較正用2,000試行で目標棄却率 alpha = 4.0% (96.0パーセンタイル点) から層別SNR閾値を決定。
   - 独立検証用2,400試行で標本FPRおよびWilson 95%信頼区間上限 fpr_h0_upper95 を評価 (標本標準誤差 sigma ≈ 0.40%)。
   - FPR管理が成立した層数と合格率を客観的にそのまま報告。
"""

import os
import sys
import time
import csv
import json
import hashlib
import numpy as np
from multiprocessing import Pool, cpu_count

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

MODELS = {
    "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 12.0, "dpix": 0.25},
    "model_3": {"M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.2, "dpix": 0.08}
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
    for model_id in ["model_1", "model_3"]:
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
    """
    1つの層における合意プロトコルH0較正を実行する。
    - 較正用 N_cal = 2,001 試行 (各セル 667 試行 x 3セル)
    - 独立検証用 N_val = 2,400 試行 (各セル 800 試行 x 3セル)
    - 計 4,401 試行 / 層
    """
    model_info = MODELS[stratum["model_id"]]
    halo = pipe.TFHaloModel(M_msun=model_info["M_msun"], R_kpc=model_info["R_kpc"], m_eV=model_info["m_eV"])

    dpix = model_info["dpix"]
    coords = np.arange(-model_info["r_fov"], model_info["r_fov"] + dpix/2, dpix)
    X_obs, Y_obs = np.meshgrid(coords, coords)
    fwhm_pix = stratum["fwhm_kpc"] / dpix
    sigma_v = stratum["sigma_v"]

    r_disk_min = 0.2 * model_info["R_kpc"]
    r_disk_max = 0.9 * model_info["R_kpc"]

    r_tol_kpc = pipe.compute_tolerance_radius(stratum["model_id"], stratum["fwhm_kpc"], model_info["R_kpc"])
    r_tol_pix = r_tol_kpc / dpix

    area_search = np.pi * (r_disk_max**2 - r_disk_min**2)
    area_tol = np.pi * (r_tol_kpc**2)
    f_area_ratio = min(1.0, area_tol / area_search)

    # 空間アンサンブル用の円周上36サンプル点 (d = 0.3 R)
    d_vortex = 0.3 * model_info["R_kpc"]
    center_idx = len(coords) // 2
    phi_samples = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    sample_pix_list = []
    for phi_s in phi_samples:
        xs_kpc = d_vortex * np.cos(phi_s)
        ys_kpc = d_vortex * np.sin(phi_s)
        px = center_idx + int(round(xs_kpc / dpix))
        py = center_idx + int(round(ys_kpc / dpix))
        sample_pix_list.append((px, py))

    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))

    n_cal_per_cell = 667
    n_val_per_cell = 800

    # 事前セットアップ (各セルの固定信号の事前平滑化キャッシュ)
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

    # ==========================================================================
    # ステップ 1: 較正セット (N_cal = 2,001 試行) による層別 SNR 閾値の決定
    # ==========================================================================
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
            seed_val = int(hashlib.md5(f"h0_cal_{stratum['stratum_id']}_{cell['inc_deg']}_{cell['pa_deg']}_{trial}".encode()).hexdigest()[:8], 16)
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

            mask_valid = mask_search & (~np.isnan(res_proc))
            if np.sum(mask_valid) > 0:
                max_snr = float(np.max(np.abs(res_proc[mask_valid])) / sigma_v)
            else:
                max_snr = 0.0
            cal_max_snrs.append(max_snr)

    # 目標棄却率 alpha = 4.0% (96.0%パーセンタイル点)
    snr_threshold = max(3.5, float(np.percentile(cal_max_snrs, 96.0)))

    # ==========================================================================
    # ステップ 2: 独立検証セット (N_val = 2,400 試行) による評価
    # ==========================================================================
    pool_f_chance = []
    pool_delta_ts_mf = []
    n_fp_val = 0
    n_val_total = 0

    for ctx in cell_contexts:
        cell = ctx["cell"]
        inc_obs = ctx["inc_obs"]
        v_sys_obs = ctx["v_sys_obs"]
        r_disk_obs = ctx["r_disk_obs"]
        phi_disk_obs = ctx["phi_disk_obs"]
        v_conv_signal = ctx["v_conv_signal"]
        mask_search = ctx["mask_search"]

        for trial in range(n_val_per_cell):
            seed_val = int(hashlib.md5(f"h0_val_{stratum['stratum_id']}_{cell['inc_deg']}_{cell['pa_deg']}_{trial}".encode()).hexdigest()[:8], 16)
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

            mask_valid = mask_search & (~np.isnan(res_proc))
            n_val_total += 1

            # 較正済み閾値による局所ピーク抽出
            raw_peaks = pipe.find_local_peaks_2d(res_proc, threshold_snr=snr_threshold, noise_sigma=sigma_v, min_distance_pix=3)
            peaks = [p for p in raw_peaks if r_disk_min <= r_disk_obs[p["y_idx"], p["x_idx"]] <= r_disk_max]

            # 1. 空間アンサンブル偶然一致率
            if len(peaks) > 0:
                match_count = 0
                for sp in sample_pix_list:
                    min_dist = min(np.sqrt((p["x_idx"] - sp[0])**2 + (p["y_idx"] - sp[1])**2) for p in peaks)
                    if min_dist <= r_tol_pix:
                        match_count += 1
                f_chance_trial = match_count / len(sample_pix_list)
                n_fp_val += 1
            else:
                f_chance_trial = 0.0
            pool_f_chance.append(f_chance_trial)

            # 2. 経験的整合フィルター統計量 Delta_TS_MF
            _, _, _, delta_ts_mf = pipe.compute_empirical_matched_filter_stat(res_proc, sigma_beam_pix, mask_valid)
            pool_delta_ts_mf.append(delta_ts_mf)

    # --- 統計量算出 ---
    arr_chance = np.array(pool_f_chance)
    arr_ts_mf = np.array(pool_delta_ts_mf)

    p_fp = n_fp_val / n_val_total
    geom_floor = p_fp * f_area_ratio

    f_chance_95 = max(float(np.percentile(arr_chance, 95)), geom_floor)
    f_chance_99_pool = max(float(np.percentile(arr_chance, 99)), geom_floor * 1.5)

    B = 2000
    n_sample = len(arr_chance)
    boot_95 = []
    boot_99 = []
    rng_b = np.random.RandomState(42 + stratum["stratum_id"])
    for _ in range(B):
        idx = rng_b.randint(0, n_sample, size=n_sample)
        sample = arr_chance[idx]
        boot_95.append(max(float(np.percentile(sample, 95)), geom_floor))
        boot_99.append(max(float(np.percentile(sample, 99)), geom_floor * 1.5))

    f_chance_95_upper = min(1.0, float(np.percentile(boot_95, 95)))
    f_chance_99_upper = min(1.0, float(np.percentile(boot_99, 95)))

    # 経験的整合フィルター統計量の 95% 点閾値
    delta_ts_mf_th_stratum = float(np.percentile(arr_ts_mf, 95))

    # Wilson 95% 信頼区間上限 (独立検証セット N_val = 2,400 による)
    z = 1.96
    denom = 1.0 + (z**2) / n_val_total
    center = (p_fp + (z**2) / (2.0 * n_val_total)) / denom
    margin = (z * np.sqrt((p_fp * (1.0 - p_fp) + (z**2) / (4.0 * n_val_total)) / n_val_total)) / denom
    fpr_h0_upper95 = min(1.0, float(center + margin))

    total_stratum_trials = len(cal_max_snrs) + n_val_total

    return {
        "stratum_id": stratum["stratum_id"],
        "model_id": stratum["model_id"],
        "inc_deg": stratum["inc_deg"],
        "fwhm_kpc": stratum["fwhm_kpc"],
        "sigma_v": stratum["sigma_v"],
        "geom_error_flag": stratum["geom_err"],
        "a_los3": stratum["a_los3"],
        "arm_method": stratum["arm_method"],
        "n_h0_trials": total_stratum_trials,
        "f_chance_95": f_chance_95,
        "f_chance_95_upper": f_chance_95_upper,
        "f_chance_99": f_chance_99_pool,
        "f_chance_99_upper": f_chance_99_upper,
        "delta_bic_th_stratum": delta_ts_mf_th_stratum,  # スキーマ互換性維持 (中身は Delta_TS_MF)
        "fpr_h0": p_fp,
        "fpr_h0_upper95": fpr_h0_upper95,
        "_snr_threshold": snr_threshold  # 内部連携用
    }


def main():
    print("=" * 80)
    print("STAGE 3: H0較正・層別プール帰無分布計算 (合意完全プロトコル: 各層4,400試行・約333万試行版)")
    print("=" * 80)

    strata = generate_strata()
    n_trials_per_stratum = 4401
    total_h0_trials = len(strata) * n_trials_per_stratum
    print(f"生成層数: {len(strata)} 層 (各層{n_trials_per_stratum}試行 [2001較正+2400独立検証], 総H0試行数: {total_h0_trials:,} 試行)")

    t0 = time.time()
    n_workers = min(cpu_count(), 16)
    print(f"並列ワーカー数: {n_workers} コア")

    with Pool(processes=n_workers) as pool:
        results = pool.map(process_stratum_h0, strata)

    elapsed_sec = time.time() - t0
    print(f"全{len(strata)}層のH0較正完了: 所要時間 {elapsed_sec:.1f} 秒 ({elapsed_sec/60.0:.2f} 分)")

    out_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h0_stratum_calibration.csv")
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

    cache_path = os.path.join(out_dir, ".stratum_snr_thresholds.json")
    with open(cache_path, "w", encoding="utf-8") as f_json:
        json.dump(snr_dict, f_json, indent=2)

    csv_rows = []
    n_passed = 0
    for r in results:
        row = {k: r[k] for k in fieldnames}
        csv_rows.append(row)
        if r["fpr_h0_upper95"] <= 0.070:
            n_passed += 1

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    pass_rate = (n_passed / len(results)) * 100.0
    print(f"\n【Stage 3 検証結果サマリー】")
    print(f"- 較正テーブル保存完了: {out_path} (全{len(fieldnames)}列・全{len(csv_rows)}層)")
    print(f"- 内部SNR閾値キャッシュ保存完了: {cache_path}")
    print(f"- 主合格条件 (fpr_h0_upper95 <= 0.070) 達成層数: {n_passed} / {len(results)} 層 (合格率 {pass_rate:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
