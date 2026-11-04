# -*- coding: utf-8 -*-
"""
tools/run_pilot_benchmark.py — パイロット実測ベンチマーク (Stage 2)

【目的】
正本仕様書セクション2.3に基づき、主要な物理パラメータ層を網羅する
層別代表10セルにおいて、手法A, B, Cを含めて独立に計測を実施する。

1. 1試行あたりの所要時間 (t_A, t_B, t_C, t_pilot_mean) の実測。
2. A_los,3 = 0.5 km/s の早期除外判定:
   「パイロット代表10セルの全セルにおいて、位置回収率が H0 偶然一致率の 95% 上限以下
   (recall_pos <= f_chance_95_upper) であった場合に限り、本番全格子の A_los,3 = 0.5 km/s を一括除外」。
3. 実測結果に基づく全格子 (16,848セル) か縮小格子かの客観的決定。

※宇宙論・物理学・数理検証専用。
※成功を前提とせず、検出不能・全滅となった場合もそのまま記録する。
"""

import os
import sys
import time
import csv
import numpy as np

# パス設定
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe


# ==============================================================================
# パイロット代表10セルの定義 (正本セクション2.3)
# ==============================================================================
PILOT_CELLS = [
    # Model 1 代表 5セル (M=10^11 Msun, R=20 kpc)
    {
        "cell_id": "P01", "model_id": "model_1", "M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22,
        "inc_deg": 70.0, "pa_deg": 45.0, "fwhm_kpc": 0.2, "sigma_v": 0.5, "geom_err": False,
        "desc": "Model 1: 高傾斜(短軸幾何短縮大) x 高分解能 x 低ノイズ"
    },
    {
        "cell_id": "P02", "model_id": "model_1", "M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22,
        "inc_deg": 50.0, "pa_deg": 45.0, "fwhm_kpc": 0.5, "sigma_v": 1.0, "geom_err": False,
        "desc": "Model 1: 中間傾斜 x 標準分解能 x 標準ノイズ (基準条件)"
    },
    {
        "cell_id": "P03", "model_id": "model_1", "M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22,
        "inc_deg": 30.0, "pa_deg": 45.0, "fwhm_kpc": 0.5, "sigma_v": 0.5, "geom_err": False,
        "desc": "Model 1: 低傾斜 x 標準分解能 x 低ノイズ"
    },
    {
        "cell_id": "P04", "model_id": "model_1", "M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22,
        "inc_deg": 50.0, "pa_deg": 45.0, "fwhm_kpc": 1.0, "sigma_v": 1.0, "geom_err": False,
        "desc": "Model 1: 中間傾斜 x 低分解能(ビーム平滑化支配) x 標準ノイズ"
    },
    {
        "cell_id": "P05", "model_id": "model_1", "M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22,
        "inc_deg": 50.0, "pa_deg": 45.0, "fwhm_kpc": 0.5, "sigma_v": 1.0, "geom_err": True,
        "desc": "Model 1: 幾何系統誤差摂動 (+2°, +2°, +2 km/s)"
    },

    # Model 3 代表 5セル (M=10^10 Msun, R=3 kpc, コンパクト)
    {
        "cell_id": "P06", "model_id": "model_3", "M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22,
        "inc_deg": 50.0, "pa_deg": 45.0, "fwhm_kpc": 0.2, "sigma_v": 0.5, "geom_err": False,
        "desc": "Model 3: コンパクト x 高分解能 x 低ノイズ"
    },
    {
        "cell_id": "P07", "model_id": "model_3", "M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22,
        "inc_deg": 50.0, "pa_deg": 45.0, "fwhm_kpc": 0.5, "sigma_v": 1.0, "geom_err": False,
        "desc": "Model 3: コンパクト x 標準分解能 x 標準ノイズ"
    },
    {
        "cell_id": "P08", "model_id": "model_3", "M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22,
        "inc_deg": 30.0, "pa_deg": 45.0, "fwhm_kpc": 0.5, "sigma_v": 0.5, "geom_err": False,
        "desc": "Model 3: コンパクト x 低傾斜 x 低ノイズ"
    },
    {
        "cell_id": "P09", "model_id": "model_3", "M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22,
        "inc_deg": 70.0, "pa_deg": 45.0, "fwhm_kpc": 0.5, "sigma_v": 1.0, "geom_err": False,
        "desc": "Model 3: コンパクト x 高傾斜(短軸分解能劣化) x 標準ノイズ"
    },
    {
        "cell_id": "P10", "model_id": "model_3", "M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22,
        "inc_deg": 50.0, "pa_deg": 45.0, "fwhm_kpc": 0.5, "sigma_v": 1.0, "geom_err": True,
        "desc": "Model 3: コンパクト x 幾何系統誤差摂動"
    },
]


def run_single_cell_benchmark(cell_conf, n_benchmark_trials=25):
    """
    1つのセルについて、計算時間と A_los,3 = 0.5 km/s の回収率を測定する。
    """
    halo = pipe.TFHaloModel(M_msun=cell_conf["M_msun"], R_kpc=cell_conf["R_kpc"], m_eV=cell_conf["m_eV"])

    # 観測グリッド設定 (モデルのサイズに応じる)
    # Model 1: R=20 kpc -> grid ±12 kpc, step = 0.15 kpc
    # Model 3: R=3 kpc  -> grid ±2.2 kpc, step = 0.05 kpc
    if cell_conf["model_id"] == "model_1":
        r_fov = 12.0
        dpix = 0.15
    else:
        r_fov = 2.2
        dpix = 0.05

    coords = np.arange(-r_fov, r_fov + dpix/2, dpix)
    X_obs, Y_obs = np.meshgrid(coords, coords)
    fwhm_pix = cell_conf["fwhm_kpc"] / dpix

    # 円盤座標
    inc_true = cell_conf["inc_deg"]
    pa_true = cell_conf["pa_deg"]
    v_sys_true = 100.0

    _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_true, inc_deg=inc_true)
    mask_disk = (r_disk_true <= cell_conf["R_kpc"] * 0.9)

    # 幾何誤差
    if cell_conf["geom_err"]:
        delta_inc = 2.0
        delta_pa = 2.0
        delta_sys = 2.0
    else:
        delta_inc = 0.0
        delta_pa = 0.0
        delta_sys = 0.0

    # 観測者が仮定する幾何
    inc_obs = inc_true + delta_inc
    pa_obs = pa_true + delta_pa
    v_sys_obs = v_sys_true + delta_sys
    _, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_obs, inc_deg=inc_obs)

    # 非摂動真の視線速度
    v0_true = halo.v_0(r_disk_true)
    v_los_unpert = pipe.project_vlos(v0_true, 0.0, phi_disk_true, inc_deg=inc_true, v_sys=v_sys_true)

    # 渦注入位置: Nv=1, d = 0.3 * R
    d_vortex = 0.3 * cell_conf["R_kpc"]
    vortex_pos_kpc = [(d_vortex, 0.0)]
    # ピクセル座標
    center_idx = len(coords) // 2
    true_pix = [(center_idx + int(round(d_vortex / dpix)), center_idx)]

    # 許容半径
    r_tol_kpc = max(halo.xi_c_kpc, cell_conf["fwhm_kpc"] / 2.0)
    if cell_conf["model_id"] == "model_3":
        r_tol_kpc = min(r_tol_kpc, 0.15 * cell_conf["R_kpc"])
    r_tol_pix = r_tol_kpc / dpix

    # 渦による速度欠損マップ
    dv_vortex_map = halo.compute_vortex_velocity_perturbation_map(
        r_disk_true * np.cos(phi_disk_true), r_disk_true * np.sin(phi_disk_true), vortex_pos_kpc
    )

    # 腕競合 (A_los,3 = 0.5 km/s)
    # k=3 調波残差を注入
    arm_residual_inj = 0.5 * np.cos(3.0 * phi_disk_true) * np.sin(np.radians(inc_true))

    # --- 時間計測 (手法A, B, C) ---
    times_A = []
    times_B = []
    times_C = []

    h0_matches = []
    inj_matches_A = []
    inj_matches_B = []
    inj_matches_C = []

    sigma_v = cell_conf["sigma_v"]

    for trial in range(n_benchmark_trials):
        noise = np.random.normal(0.0, sigma_v, size=X_obs.shape)

        # 探索領域マスク: 0.2 R_disk <= r <= 0.9 R_disk
        r_disk_min = 0.2 * cell_conf["R_kpc"]
        r_disk_max = 0.9 * cell_conf["R_kpc"]

        # 1. H0 (対照群: 渦なし・ノイズのみ)
        v_obs_h0 = v_los_unpert + noise
        v_conv_h0 = pipe.convolve_beam_normalized(v_obs_h0, fwhm_pix=fwhm_pix, mask=mask_disk)
        # 軸対称フィット減算
        _, _, res_h0 = pipe.fit_axisymmetric_disk_rings(
            r_disk_obs, phi_disk_obs, v_conv_h0, inc_deg=inc_obs, v_sys=v_sys_obs, n_rings=15
        )
        raw_peaks_h0 = pipe.find_local_peaks_2d(res_h0, threshold_snr=3.0, noise_sigma=sigma_v, min_distance_pix=3)
        # 探索領域内のピークのみに制限
        peaks_h0 = []
        for p in raw_peaks_h0:
            r_p = r_disk_obs[p["y_idx"], p["x_idx"]]
            if r_disk_min <= r_p <= r_disk_max:
                peaks_h0.append(p)
        match_h0 = pipe.evaluate_position_matching(peaks_h0, true_pix, r_tol_pix=r_tol_pix)
        h0_matches.append(any(match_h0) if len(match_h0) > 0 else False)

        # 2. 渦注入 + A_los,3 = 0.5 km/s 腕競合データ
        v_obs_inj = v_los_unpert + dv_vortex_map * np.sin(np.radians(inc_true)) + arm_residual_inj + noise
        v_conv_inj = pipe.convolve_beam_normalized(v_obs_inj, fwhm_pix=fwhm_pix, mask=mask_disk)

        # 手法A (Raw残差)
        t0 = time.perf_counter()
        _, _, res_A = pipe.fit_axisymmetric_disk_rings(
            r_disk_obs, phi_disk_obs, v_conv_inj, inc_deg=inc_obs, v_sys=v_sys_obs, n_rings=15
        )
        raw_peaks_A = pipe.find_local_peaks_2d(res_A, threshold_snr=3.0, noise_sigma=sigma_v, min_distance_pix=3)
        peaks_A = [p for p in raw_peaks_A if r_disk_min <= r_disk_obs[p["y_idx"], p["x_idx"]] <= r_disk_max]
        t_A = time.perf_counter() - t0
        times_A.append(t_A)
        match_A = pipe.evaluate_position_matching(peaks_A, true_pix, r_tol_pix=r_tol_pix)
        inj_matches_A.append(any(match_A) if len(match_A) > 0 else False)

        # 手法B (remove_k3)
        t0 = time.perf_counter()
        res_B, _ = pipe.remove_harmonic_k3_rings(r_disk_obs, phi_disk_obs, res_A, n_rings=15)
        raw_peaks_B = pipe.find_local_peaks_2d(res_B, threshold_snr=3.0, noise_sigma=sigma_v, min_distance_pix=3)
        peaks_B = [p for p in raw_peaks_B if r_disk_min <= r_disk_obs[p["y_idx"], p["x_idx"]] <= r_disk_max]
        t_B = time.perf_counter() - t0 + t_A
        times_B.append(t_B)
        match_B = pipe.evaluate_position_matching(peaks_B, true_pix, r_tol_pix=r_tol_pix)
        inj_matches_B.append(any(match_B) if len(match_B) > 0 else False)

        # 手法C (arm_fit)
        t0 = time.perf_counter()
        res_C, _, _, _ = pipe.fit_spiral_arm_direct_c(r_disk_obs, phi_disk_obs, res_A, inc_deg=inc_obs)
        raw_peaks_C = pipe.find_local_peaks_2d(res_C, threshold_snr=3.0, noise_sigma=sigma_v, min_distance_pix=3)
        peaks_C = [p for p in raw_peaks_C if r_disk_min <= r_disk_obs[p["y_idx"], p["x_idx"]] <= r_disk_max]
        t_C = time.perf_counter() - t0 + t_A
        times_C.append(t_C)
        match_C = pipe.evaluate_position_matching(peaks_C, true_pix, r_tol_pix=r_tol_pix)
        inj_matches_C.append(any(match_C) if len(match_C) > 0 else False)

    # 統計集計
    t_mean_A = np.mean(times_A)
    t_mean_B = np.mean(times_B)
    t_mean_C = np.mean(times_C)
    t_cell_mean = (t_mean_A + t_mean_B + t_mean_C) / 3.0

    # H0 偶然一致率 (Wilson 95% 上限、厳格な境界クリッピング)
    n_h0_success = np.sum(h0_matches)
    n_tot = len(h0_matches)
    p_hat = n_h0_success / n_tot
    if p_hat >= 1.0 - 1e-6:
        f_chance_95_upper = 1.0
    else:
        z = 1.96
        denom = 1.0 + (z**2) / n_tot
        center = (p_hat + (z**2) / (2.0 * n_tot)) / denom
        margin = (z * np.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * n_tot)) / n_tot)) / denom
        f_chance_95_upper = min(1.0, center + margin)

    # 回収率 (手法A, B, C の平均または最良)
    recall_A = np.mean(inj_matches_A)
    recall_B = np.mean(inj_matches_B)
    recall_C = np.mean(inj_matches_C)
    recall_max = max(recall_A, recall_B, recall_C)

    # 有意性判定: f_chance_95_upper < 1.0 かつ recall_max > f_chance_95_upper
    is_significant_cell = (recall_max > f_chance_95_upper) and (f_chance_95_upper < 0.95)

    return {
        "cell_id": cell_conf["cell_id"],
        "model_id": cell_conf["model_id"],
        "t_A_sec": t_mean_A,
        "t_B_sec": t_mean_B,
        "t_C_sec": t_mean_C,
        "t_mean_sec": t_cell_mean,
        "f_chance_hat": p_hat,
        "f_chance_95_upper": f_chance_95_upper,
        "recall_A": recall_A,
        "recall_B": recall_B,
        "recall_C": recall_C,
        "recall_max": recall_max,
        "is_significant": is_significant_cell
    }


def main():
    print("=" * 80)
    print("STAGE 2: パイロット実測ベンチマーク (層別代表10セル)")
    print("=" * 80)

    results = []
    for cell in PILOT_CELLS:
        print(f"計測中: {cell['cell_id']} ({cell['desc']})...", end="", flush=True)
        res = run_single_cell_benchmark(cell, n_benchmark_trials=20)
        results.append(res)
        print(f" 完了 [t_mean={res['t_mean_sec']*1000:.1f}ms, recall_max={res['recall_max']:.2f}, f_chance_95_up={res['f_chance_95_upper']:.2f}, 有意={res['is_significant']}]")

    # 全体集計
    t_A_overall = float(np.mean([r["t_A_sec"] for r in results]))
    t_B_overall = float(np.mean([r["t_B_sec"] for r in results]))
    t_C_overall = float(np.mean([r["t_C_sec"] for r in results]))
    t_pilot_mean = float(np.mean([r["t_mean_sec"] for r in results]))

    print("\n" + "=" * 80)
    print("ベンチマーク計測結果集計")
    print("=" * 80)
    print(f"手法A 平均時間: {t_A_overall*1000:.2f} ms / 試行")
    print(f"手法B 平均時間: {t_B_overall*1000:.2f} ms / 試行")
    print(f"手法C 平均時間: {t_C_overall*1000:.2f} ms / 試行")
    print(f"代表10セル加重平均時間: {t_pilot_mean*1000:.2f} ms / 試行")

    # 16,848 セル（各100試行 = 1,684,800 試行）の推定時間
    total_cpu_hours = (t_pilot_mean * 1684800) / 3600.0
    wall_clock_16cores_min = (total_cpu_hours / 16.0) * 60.0
    print(f"本番全16,848セル 推定総CPU時間: {total_cpu_hours:.2f} 時間 (16コア並列で約 {wall_clock_16cores_min:.1f} 分)")

    # 早期除外判定の評価
    # 「パイロット代表10セルの全セルにおいて recall_pos <= f_chance_95_upper の場合に限り A_los,3 = 0.5 km/s を除外」
    any_significant = any(r["is_significant"] for r in results)
    all_insignificant = not any_significant

    print("\n" + "-" * 80)
    print("【A_los,3 = 0.5 km/s の客観的除外判定】")
    if all_insignificant:
        print(">> 判定結果: パイロット代表10セル【全セルで有意なピーク検出なし】(全セル未達)")
        print(">> 確定処置: 本番全格子において A_los,3 = 0.5 km/s 条件を一括除外する (第1段階縮小適用)")
        grid_verdict = "PRUNED_GRID_STAGE1"
    else:
        sig_cells = [r["cell_id"] for r in results if r["is_significant"]]
        print(f">> 判定結果: {len(sig_cells)} セル ({sig_cells}) において有意な回収を確認 (recall_pos > f_chance_95_upper)")
        print(">> 確定処置: A_los,3 = 0.5 km/s は除外せず維持する")
        grid_verdict = "FULL_GRID"

    # 計算時間判定
    print("\n" + "-" * 80)
    print("【計算時間と格子規模判定】")
    if t_pilot_mean > 0.045:
        print(f">> 平均時間 {t_pilot_mean:.3f}s > 0.045s: 計算時間超過のため格子縮小プロトコルを発動")
        grid_choice = "PRUNED_GRID"
    else:
        print(f">> 平均時間 {t_pilot_mean:.3f}s <= 0.045s: 計算時間基準をクリア")
        grid_choice = "FULL_GRID" if grid_verdict == "FULL_GRID" else "PRUNED_GRID_STAGE1"

    print(f"\n最終採用格子方針: 【{grid_choice}】")

    # CSV出力
    out_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pilot_benchmark_results.csv")
    fieldnames = list(results[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"ベンチマーク詳細結果を保存: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
