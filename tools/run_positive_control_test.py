# -*- coding: utf-8 -*-
"""
tools/run_positive_control_test.py — 陽性対照試験・健全性ベンチマークおよび感度限界測定エンジン

【目的】
正本仕様書セクション1.3に基づき、パイプラインが「大振幅の渦信号を確実に100%回収できること」を
数学的・実験的に立証し、各観測・解析条件における50%回収閾値振幅 A_50 (Detection Limit) を測定する。

【試験内容】
1. 健全性ベンチマーク (Sanity Check):
   ビームなし・ノイズなし (FWHM=0, sigma_v=0) で全振幅 A_inj in {2, 5, 10, 20, 50} km/s において
   主軸上 (theta_v=0°) の位置回収率が厳密に 100% (recall_pos = 1.0000) となることを検証。
2. 幾何学的投影と陰性対照 (Negative Control):
   theta_v in {0°, 45°, 90°} において、回収感度が |cos(theta_v)| の理論予測に従って減衰すること、
   特に theta_v=90° では信号が視線速度場から消失し回収率が 0 になることを確認。
3. ビーム平滑化減衰率 eta_beam および腕処理減衰率 eta_fit、実効SNRの測定。
4. ノイズ存在下での回収率曲線 Recall(A_inj) と 50%回収閾値 A_50 の算出。

※宇宙論・物理学・数理検証専用。
※成功を前提とせず、得られた統計的限界を客観的に記録する。
"""

import os
import sys
import csv
import hashlib
import numpy as np

# パス設定
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe


# 対象モデル定義
MODELS = {
    "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 12.0, "dpix": 0.08},
    "model_3": {"M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.2, "dpix": 0.08}
}

INJ_AMPLITUDES = [2.0, 5.0, 10.0, 20.0, 50.0]  # km/s (面内最大速度欠損)
THETA_V_LIST = [0.0, 45.0, 90.0]                # deg (0: 陽性対照最良, 45: 中間, 90: 陰性対照)


def run_positive_control_benchmark():
    print("=" * 80)
    print("POSITIVE CONTROL BENCHMARK: Injection-Recovery Pipeline Verification")
    print("=" * 80)

    results = []

    # --------------------------------------------------------------------------
    # 1. 健全性ベンチマーク (Ideal Sanity Check: FWHM=0, sigma_v=0, PA=0, i=50)
    # --------------------------------------------------------------------------
    print("\n[Phase 1] Ideal Sanity Check (FWHM=0, sigma_v=0, no noise, no beam)...")
    sanity_all_passed = True

    for model_id in ["model_1", "model_3"]:
        info = MODELS[model_id]
        dpix = info["dpix"]
        coords = np.arange(-info["r_fov"], info["r_fov"] + dpix/2, dpix)
        center_idx = len(coords) // 2
        X_obs, Y_obs = np.meshgrid(coords, coords)

        d_v = 0.3 * info["R_kpc"]
        r_min = 0.2 * info["R_kpc"]
        r_max = 0.9 * info["R_kpc"]
        inc_deg = 50.0
        pa_deg = 0.0

        _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_deg, inc_deg=inc_deg)
        mask_search = (r_disk_true >= r_min) & (r_disk_true <= r_max)

        r_tol_kpc = pipe.compute_tolerance_radius(model_id, fwhm_kpc=0.2, r_disk_kpc=info["R_kpc"])
        r_tol_pix = r_tol_kpc / dpix

        for a_inj in INJ_AMPLITUDES:
            for theta_v in THETA_V_LIST:
                theta_rad = np.radians(theta_v)
                xv_kpc = d_v * np.cos(theta_rad)
                yv_kpc = d_v * np.sin(theta_rad)

                # 観測面での真のピクセル位置
                xo, yo = pipe.disk_to_obs(xv_kpc, yv_kpc, pa_deg=pa_deg, inc_deg=inc_deg)
                true_pix = [(center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix)))]

                # 人工的ガウシアン渦テンプレートの直接面内注入 (最大振幅 a_inj)
                sigma_vortex_kpc = 0.15 * info["R_kpc"]
                sigma_vortex_pix = sigma_vortex_kpc / dpix
                dist_sq_kpc = (r_disk_true * np.cos(phi_disk_true) - xv_kpc)**2 + (r_disk_true * np.sin(phi_disk_true) - yv_kpc)**2
                dv_map = a_inj * np.exp(-dist_sq_kpc / (2.0 * sigma_vortex_kpc**2))

                # 正確な視線速度投影
                v_los_inj = dv_map * np.cos(phi_disk_true) * np.sin(np.radians(inc_deg))

                # ピーク抽出 (ノイズなしなので閾値は微小値)
                peaks = pipe.find_local_peaks_2d(v_los_inj, threshold_snr=1.0, noise_sigma=0.01, min_distance_pix=3)
                valid_peaks = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]

                matches = pipe.evaluate_position_matching(valid_peaks, true_pix, r_tol_pix=r_tol_pix)
                recall = 1.0 if any(matches) else 0.0

                # 理論的期待
                # theta_v=0: 100% 回収
                # theta_v=90: 陰性対照 (cos(90)=0 により信号が消失し 0% 回収)
                if theta_v == 0.0:
                    status = "PASS" if recall == 1.0 else "FAIL"
                    if recall != 1.0:
                        sanity_all_passed = False
                elif theta_v == 90.0:
                    status = "PASS (Negative Control Confirmed)" if recall == 0.0 else "UNEXPECTED"
                else:
                    status = f"Recall={recall:.1f}"

                print(f"  [{model_id}] A_inj={a_inj:4.1f} km/s, theta_v={theta_v:4.1f}° | recall={recall:.2f} -> {status}")

                results.append({
                    "phase": "sanity_check",
                    "model_id": model_id,
                    "a_inj_kms": a_inj,
                    "theta_v_deg": theta_v,
                    "fwhm_kpc": 0.0,
                    "sigma_v_kms": 0.0,
                    "arm_method": "none",
                    "eta_beam": 1.0,
                    "eta_fit": 1.0,
                    "a_eff_kms": a_inj * np.abs(np.cos(theta_rad)) * np.sin(np.radians(inc_deg)),
                    "snr_eff": np.inf if a_inj > 0 and theta_v != 90.0 else 0.0,
                    "recall_pos": recall,
                    "note": status
                })

    assert sanity_all_passed, "Sanity Check FAILED: theta_v=0° で 100% 回収が達成できませんでした。"
    print("\n>>> Sanity Check SUCCESS: 全ての theta_v=0° で 100% 回収を確認 (検証した範囲でバグは見つからなかった)。")

    # --------------------------------------------------------------------------
    # 2. 現実的観測条件下での減衰率 (eta_beam, eta_fit) および感度限界 A_50 測定
    # --------------------------------------------------------------------------
    print("\n[Phase 2] Realistic Observation & Completeness Scan...")

    inc_deg = 50.0
    pa_deg = 45.0  # 一般的な傾斜・位置角
    theta_v = 0.0  # 最良投影条件 (主軸上)

    for model_id in ["model_1", "model_3"]:
        info = MODELS[model_id]
        dpix = info["dpix"]
        coords = np.arange(-info["r_fov"], info["r_fov"] + dpix/2, dpix)
        center_idx = len(coords) // 2
        X_obs, Y_obs = np.meshgrid(coords, coords)

        d_v = 0.3 * info["R_kpc"]
        r_min = 0.2 * info["R_kpc"]
        r_max = 0.9 * info["R_kpc"]

        _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_deg, inc_deg=inc_deg)
        mask_search = (r_disk_true >= r_min) & (r_disk_true <= r_max)

        xo, yo = pipe.disk_to_obs(d_v, 0.0, pa_deg=pa_deg, inc_deg=inc_deg)
        true_pix = [(center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix)))]

        for fwhm in [0.2, 0.5, 1.0]:
            fwhm_pix = fwhm / dpix
            sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
            r_tol_kpc = pipe.compute_tolerance_radius(model_id, fwhm_kpc=fwhm, r_disk_kpc=info["R_kpc"])
            r_tol_pix = r_tol_kpc / dpix

            for sigma_v in [0.5, 1.0, 2.0]:
                for arm_method in ["none", "remove_k3", "arm_fit"]:
                    recalls_for_amp = []
                    eta_beam_val = 1.0
                    eta_fit_val = 1.0

                    for a_inj in INJ_AMPLITUDES:
                        # 渦プロファイル (回復長スケール)
                        sigma_vortex_kpc = max(0.1, pipe.compute_tolerance_radius(model_id, 0.2, info["R_kpc"]))
                        dist_sq_kpc = (r_disk_true * np.cos(phi_disk_true) - d_v)**2 + (r_disk_true * np.sin(phi_disk_true))**2
                        dv_map = a_inj * np.exp(-dist_sq_kpc / (2.0 * sigma_vortex_kpc**2))

                        v_los_unconv = dv_map * np.cos(phi_disk_true) * np.sin(np.radians(inc_deg))
                        max_unconv = np.max(np.abs(v_los_unconv[mask_search]))

                        # ビーム畳み込み
                        v_conv = pipe.convolve_beam_normalized(v_los_unconv, fwhm_pix=fwhm_pix, mask=mask_search)
                        max_conv = np.nanmax(np.abs(v_conv[mask_search]))
                        eta_beam_val = max_conv / max_unconv if max_unconv > 1e-6 else 1.0

                        # 腕処理による減衰
                        if arm_method == "remove_k3":
                            v_proc, _ = pipe.remove_harmonic_k3_rings(r_disk_true, phi_disk_true, v_conv, n_rings=15)
                        elif arm_method == "arm_fit":
                            v_proc, _, _, _ = pipe.fit_spiral_arm_direct_c(r_disk_true, phi_disk_true, v_conv, inc_deg=inc_deg)
                        else:
                            v_proc = v_conv

                        max_proc = np.nanmax(np.abs(v_proc[mask_search]))
                        eta_fit_val = max_proc / max_conv if max_conv > 1e-6 else 1.0

                        a_eff = a_inj * eta_beam_val * eta_fit_val * np.sin(np.radians(inc_deg))
                        snr_eff = a_eff / sigma_v

                        # 50試行のノイズ注入回収試験 (高速モンテカルロ)
                        n_trials_mc = 50
                        matched_mc = 0
                        snr_threshold = 4.0  # 固定基準 (事前登録)

                        for trial in range(n_trials_mc):
                            seed = int(hashlib.md5(f"pos_{model_id}_{fwhm}_{sigma_v}_{arm_method}_{a_inj}_{trial}".encode()).hexdigest()[:8], 16)
                            rng = np.random.RandomState(seed)
                            noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix=sigma_beam_pix, sigma_v=sigma_v, rng=rng)
                            res_trial = v_proc + noise

                            peaks = pipe.find_local_peaks_2d(res_trial, threshold_snr=snr_threshold, noise_sigma=sigma_v, min_distance_pix=3)
                            valid_p = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
                            m = pipe.evaluate_position_matching(valid_p, true_pix, r_tol_pix=r_tol_pix)
                            if any(m):
                                matched_mc += 1

                        recall_mc = matched_mc / n_trials_mc
                        recalls_for_amp.append((a_inj, recall_mc))

                        results.append({
                            "phase": "realistic_scan",
                            "model_id": model_id,
                            "a_inj_kms": a_inj,
                            "theta_v_deg": theta_v,
                            "fwhm_kpc": fwhm,
                            "sigma_v_kms": sigma_v,
                            "arm_method": arm_method,
                            "eta_beam": eta_beam_val,
                            "eta_fit": eta_fit_val,
                            "a_eff_kms": a_eff,
                            "snr_eff": snr_eff,
                            "recall_pos": recall_mc,
                            "note": f"SNR={snr_eff:.2f}"
                        })

                    # A_50 (50% 回収閾値) の線形内挿
                    a_50 = np.nan
                    for idx in range(len(recalls_for_amp) - 1):
                        a1, r1 = recalls_for_amp[idx]
                        a2, r2 = recalls_for_amp[idx+1]
                        if r1 <= 0.50 <= r2 and (r2 - r1) > 1e-4:
                            a_50 = a1 + (0.50 - r1) * (a2 - a1) / (r2 - r1)
                            break
                        elif r1 >= 0.50:
                            a_50 = a1
                            break

                    print(f"  [{model_id}] FWHM={fwhm:.1f}kpc, sigma_v={sigma_v:.1f}km/s, meth={arm_method:9s} | eta_b={eta_beam_val:.2f}, eta_fit={eta_fit_val:.2f} -> A_50 = {a_50:.2f} km/s")

    # CSV 出力
    out_csv = os.path.join(PROJECT_DIR, "output", "positive_control_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["phase", "model_id", "a_inj_kms", "theta_v_deg", "fwhm_kpc", "sigma_v_kms",
                      "arm_method", "eta_beam", "eta_fit", "a_eff_kms", "snr_eff", "recall_pos", "note"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n>>> Positive Control Results saved to: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    run_positive_control_benchmark()
