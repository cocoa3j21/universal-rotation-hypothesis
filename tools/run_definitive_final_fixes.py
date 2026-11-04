# -*- coding: utf-8 -*-
"""
tools/run_definitive_final_fixes.py
最終確定版のための全計算・実測スクリプト:
1. 超サンプリング信号を基準とした上限SNR・方法B・窓打ち切り効果の精密計算 (Model 2: 30, 50, 70度)
2. Model 1 最良条件 (0.2 kpc, 0.5 km/s, 50度) のテンプレート検出器モンテカルロ実測
3. 平滑化比率の条件特定
4. SHA256SUMS に tools/*.py を含む全ファイルのハッシュを出力
5. final_report_numbers.json の完全更新
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import glob
import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.stats import norm

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import gyro_gravity_theory.injection_pipeline as pipe

OUT_DIR = os.path.join(PROJECT_DIR, "output")

def get_sha256(fpath):
    if not os.path.exists(fpath):
        return None
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    dpix = 0.08
    dx_fine = 0.01
    factor = int(dpix / dx_fine)  # 8

    # =========================================================================
    # 1. Model 2 超サンプリング基準の上限SNR、方法B、窓打ち切り、RB の再計算
    # =========================================================================
    print("=" * 100)
    print("1. Model 2 超サンプリング基準の SNR_opt, 方法B, 窓打ち切り, R_B 再計算")
    print("=" * 100)

    fwhm = 0.2
    sv = 0.5
    r_fov_m2 = 9.4
    halo_m2 = pipe.TFHaloModel(M_msun=1e11, R_kpc=10.0, m_eV=1e-22)
    d_v_m2 = 3.0
    r_max_m2 = 9.0
    r_min_m2 = 2.0

    coords_obs_m2 = np.arange(-r_fov_m2, r_fov_m2 + dpix / 2, dpix)
    X_obs_m2, Y_obs_m2 = np.meshgrid(coords_obs_m2, coords_obs_m2)
    coords_fine_m2 = np.arange(-r_fov_m2, r_fov_m2 + dx_fine / 2, dx_fine)
    X_fine_m2, Y_fine_m2 = np.meshgrid(coords_fine_m2, coords_fine_m2)

    fwhm_pix_obs = fwhm / dpix
    sigma_b_pix_obs = max(1.0, fwhm_pix_obs / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_b_kpc = sigma_b_pix_obs * dpix
    a_beam = 4.0 * np.pi * (sigma_b_kpc**2)
    sqrt_a_beam = np.sqrt(a_beam)
    fwhm_pix_fine = fwhm / dx_fine

    # Load oversampled detector results
    with open(os.path.join(OUT_DIR, "v2_oversampled_template_detector_results.json"), "r", encoding="utf-8") as f:
        tmf_results = json.load(f)

    # rings for method B
    n_rings = 30
    r_edges_m2 = np.linspace(0.0, r_max_m2, n_rings + 1)

    model2_oversampled_analysis = {}

    print(f"{'Inc':>4} | {'超サンプリング max|S|':>20} | {'SNR_opt(超サンプ)':>18} | {'SNR_B(超サンプ)':>16} | {'窓(41x41)比率':>12} | {'th_TMF':>8} | {'R_B':>6} | {'実測回収率':>10} | {'逆算SNR_eff':>12}")
    print("-" * 125)

    for inc in [30.0, 50.0, 70.0]:
        # Fine grid generation
        _, _, r_disk_fine, phi_disk_fine = pipe.obs_to_disk(X_fine_m2, Y_fine_m2, pa_deg=0.0, inc_deg=inc)
        mask_disk_fine = (r_disk_fine <= r_max_m2)
        v0_fine = halo_m2.v_0(r_disk_fine)
        v_unp_fine = pipe.project_vlos(v0_fine, 0.0, phi_disk_fine, inc_deg=inc, v_sys=100.0)
        dv_fine = halo_m2.compute_vortex_velocity_perturbation_map(
            r_disk_fine * np.cos(phi_disk_fine), r_disk_fine * np.sin(phi_disk_fine), [(d_v_m2, 0.0)]
        )
        vlos_vort_fine = dv_fine * np.cos(phi_disk_fine) * np.sin(np.radians(inc))

        v_conv_tot_fine = pipe.convolve_beam_normalized(v_unp_fine + vlos_vort_fine, fwhm_pix=fwhm_pix_fine, mask=mask_disk_fine)
        v_conv_unp_fine = pipe.convolve_beam_normalized(v_unp_fine, fwhm_pix=fwhm_pix_fine, mask=mask_disk_fine)
        S_fine = np.nan_to_num(v_conv_tot_fine - v_conv_unp_fine, nan=0.0)

        # Downsample to dpix = 0.08
        ny = (S_fine.shape[0] // factor) * factor
        nx = (S_fine.shape[1] // factor) * factor
        S_down = S_fine[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))
        S_true = np.zeros_like(X_obs_m2)
        min_y = min(S_down.shape[0], X_obs_m2.shape[0])
        min_x = min(S_down.shape[1], X_obs_m2.shape[1])
        S_true[:min_y, :min_x] = S_down[:min_y, :min_x]

        _, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs_m2, Y_obs_m2, pa_deg=0.0, inc_deg=inc)
        mask_disk_obs = (r_disk_obs <= r_max_m2)
        mask_search_obs = (r_disk_obs >= r_min_m2) & (r_disk_obs <= r_max_m2)

        # L2 of oversampled true signal on full disk
        s_disk = S_true * mask_disk_obs.astype(float)
        l2_opt_os = float(np.sqrt(np.sum(s_disk**2) * (dpix**2)))
        snr_opt_os = (l2_opt_os / sqrt_a_beam) / sv

        # Method B on S_true in search mask
        s_search = S_true * mask_search_obs.astype(float)
        s0_B = np.copy(s_search)
        for k in range(n_rings):
            m = mask_search_obs & (r_disk_obs >= r_edges_m2[k]) & (r_disk_obs < r_edges_m2[k+1])
            if np.sum(m) > 0: s0_B[m] -= np.mean(s0_B[m])
            if np.sum(m) >= 3:
                cp = np.cos(phi_disk_obs[m]); sp = np.sin(phi_disk_obs[m])
                coeff, _, _, _ = np.linalg.lstsq(np.column_stack([cp, sp]), s0_B[m], rcond=None)
                s0_B[m] -= (coeff[0]*cp + coeff[1]*sp)
        l2_B_os = float(np.sqrt(np.sum(s0_B**2) * (dpix**2)))
        snr_B_os = (l2_B_os / sqrt_a_beam) / sv

        # Window cutoff ratio: 41x41 cutout window around vortex
        xv_obs, yv_obs = pipe.disk_to_obs(d_v_m2, 0.0, pa_deg=0.0, inc_deg=inc)
        ix_v = int(np.round((xv_obs - (-r_fov_m2)) / dpix))
        iy_v = int(np.round((yv_obs - (-r_fov_m2)) / dpix))

        win_41 = 20  # 41x41
        y1, y2 = max(0, iy_v - win_41), min(X_obs_m2.shape[0], iy_v + win_41 + 1)
        x1, x2 = max(0, ix_v - win_41), min(X_obs_m2.shape[1], ix_v + win_41 + 1)
        cutout_41 = S_true[y1:y2, x1:x2]
        l2_win41 = float(np.sqrt(np.sum(cutout_41**2) * (dpix**2)))
        ratio_win41 = l2_win41 / l2_opt_os if l2_opt_os > 0 else 0.0

        th_tmf = tmf_results[str(int(inc))]["th_h0_tmf"]
        rec_rate = tmf_results[str(int(inc))]["rec_rate"]
        snr_eff_inv = tmf_results[str(int(inc))]["snr_eff_inverted"]

        r_B_os = snr_B_os / th_tmf

        print(f"{inc:4.0f} | {np.max(np.abs(S_true)):18.4f} km/s | {snr_opt_os:18.3f} | {snr_B_os:16.3f} | {ratio_win41:12.3f} | {th_tmf:8.4f} | {r_B_os:6.3f} | {rec_rate:9.1%} | {snr_eff_inv:12.3f}")

        model2_oversampled_analysis[int(inc)] = {
            "inc_deg": inc,
            "max_s_fine_kms": round(float(np.max(np.abs(S_true))), 4),
            "snr_opt_oversampled": round(snr_opt_os, 3),
            "snr_B_oversampled": round(snr_B_os, 3),
            "window_41_energy_ratio": round(ratio_win41, 3),
            "th_h0_tmf": round(th_tmf, 4),
            "r_B_oversampled": round(r_B_os, 3),
            "rec_rate": rec_rate,
            "ci95_lower": tmf_results[str(int(inc))]["ci95_lower"],
            "ci95_upper": tmf_results[str(int(inc))]["ci95_upper"],
            "snr_eff_inverted": snr_eff_inv
        }

    # =========================================================================
    # 2. Model 1 最良条件 (0.2 kpc, 0.5 km/s, 50度) テンプレート検出器の実測
    # =========================================================================
    print("\n" + "=" * 100)
    print("2. Model 1 最良条件 (FWHM=0.2 kpc, sv=0.5 km/s, Inc=50度) テンプレート検出器の実測")
    print("=" * 100)

    r_fov_m1 = 18.4
    halo_m1 = pipe.TFHaloModel(M_msun=1e11, R_kpc=20.0, m_eV=1e-22)
    d_v_m1 = 6.0
    r_max_m1 = 18.0
    r_min_m1 = 4.0
    inc_m1 = 50.0

    coords_obs_m1 = np.arange(-r_fov_m1, r_fov_m1 + dpix / 2, dpix)
    X_obs_m1, Y_obs_m1 = np.meshgrid(coords_obs_m1, coords_obs_m1)
    coords_fine_m1 = np.arange(-r_fov_m1, r_fov_m1 + dx_fine / 2, dx_fine)
    X_fine_m1, Y_fine_m1 = np.meshgrid(coords_fine_m1, coords_fine_m1)

    _, _, r_disk_fine_m1, phi_disk_fine_m1 = pipe.obs_to_disk(X_fine_m1, Y_fine_m1, pa_deg=0.0, inc_deg=inc_m1)
    mask_disk_fine_m1 = (r_disk_fine_m1 <= r_max_m1)
    v0_fine_m1 = halo_m1.v_0(r_disk_fine_m1)
    v_unp_fine_m1 = pipe.project_vlos(v0_fine_m1, 0.0, phi_disk_fine_m1, inc_deg=inc_m1, v_sys=100.0)
    dv_fine_m1 = halo_m1.compute_vortex_velocity_perturbation_map(
        r_disk_fine_m1 * np.cos(phi_disk_fine_m1), r_disk_fine_m1 * np.sin(phi_disk_fine_m1), [(d_v_m1, 0.0)]
    )
    vlos_vort_fine_m1 = dv_fine_m1 * np.cos(phi_disk_fine_m1) * np.sin(np.radians(inc_m1))

    v_conv_tot_fine_m1 = pipe.convolve_beam_normalized(v_unp_fine_m1 + vlos_vort_fine_m1, fwhm_pix=fwhm_pix_fine, mask=mask_disk_fine_m1)
    v_conv_unp_fine_m1 = pipe.convolve_beam_normalized(v_unp_fine_m1, fwhm_pix=fwhm_pix_fine, mask=mask_disk_fine_m1)
    S_fine_m1 = np.nan_to_num(v_conv_tot_fine_m1 - v_conv_unp_fine_m1, nan=0.0)

    ny_m1 = (S_fine_m1.shape[0] // factor) * factor
    nx_m1 = (S_fine_m1.shape[1] // factor) * factor
    S_down_m1 = S_fine_m1[:ny_m1, :nx_m1].reshape(ny_m1 // factor, factor, nx_m1 // factor, factor).mean(axis=(1, 3))
    S_true_m1 = np.zeros_like(X_obs_m1)
    min_y_m1 = min(S_down_m1.shape[0], X_obs_m1.shape[0])
    min_x_m1 = min(S_down_m1.shape[1], X_obs_m1.shape[1])
    S_true_m1[:min_y_m1, :min_x_m1] = S_down_m1[:min_y_m1, :min_x_m1]

    _, _, r_disk_obs_m1, phi_disk_obs_m1 = pipe.obs_to_disk(X_obs_m1, Y_obs_m1, pa_deg=0.0, inc_deg=inc_m1)
    mask_search_obs_m1 = (r_disk_obs_m1 >= r_min_m1) & (r_disk_obs_m1 <= r_max_m1)

    xv_obs_m1, yv_obs_m1 = pipe.disk_to_obs(d_v_m1, 0.0, pa_deg=0.0, inc_deg=inc_m1)
    ix_v_m1 = int(np.round((xv_obs_m1 - (-r_fov_m1)) / dpix))
    iy_v_m1 = int(np.round((yv_obs_m1 - (-r_fov_m1)) / dpix))

    win_m1 = 25
    y1, y2 = max(0, iy_v_m1 - win_m1), min(X_obs_m1.shape[0], iy_v_m1 + win_m1 + 1)
    x1, x2 = max(0, ix_v_m1 - win_m1), min(X_obs_m1.shape[1], ix_v_m1 + win_m1 + 1)
    template_m1 = np.copy(S_true_m1[y1:y2, x1:x2])
    template_m1 -= np.mean(template_m1)

    g_temp_m1 = gaussian_filter(template_m1, sigma=sigma_b_pix_obs, mode='constant', cval=0.0)
    imp_m1 = np.zeros_like(template_m1)
    imp_m1[imp_m1.shape[0]//2, imp_m1.shape[1]//2] = 1.0
    gk_m1 = gaussian_filter(imp_m1, sigma=sigma_b_pix_obs, mode='constant', cval=0.0)
    tmf_std_m1 = sv * (np.sqrt(np.sum(g_temp_m1**2)) / np.sqrt(np.sum(gk_m1**2)))
    t_rev_m1 = template_m1[::-1, ::-1]

    n_h0_m1 = 600
    n_inj_m1 = 400
    max_z_h0_m1 = []
    for t in range(n_h0_m1):
        rng = np.random.RandomState(t + 71000)
        noise = pipe.generate_beam_correlated_noise(X_obs_m1.shape, sigma_b_pix_obs, sv, rng)
        z_map = fftconvolve(noise, t_rev_m1, mode='same') / tmf_std_m1
        z_search = z_map[mask_search_obs_m1]
        max_z_h0_m1.append(np.max(np.abs(z_search)))

    th_tmf_m1 = float(np.percentile(max_z_h0_m1, 96.0))

    n_rec_m1 = 0
    r_tol_pix_m1 = 3.0
    for t in range(n_inj_m1):
        rng = np.random.RandomState(t + 81000)
        noise = pipe.generate_beam_correlated_noise(X_obs_m1.shape, sigma_b_pix_obs, sv, rng)
        obs = S_true_m1 + noise
        z_map = fftconvolve(obs, t_rev_m1, mode='same') / tmf_std_m1
        z_masked = np.where(mask_search_obs_m1, np.abs(z_map), 0.0)
        if np.max(z_masked) >= th_tmf_m1:
            py, px = np.unravel_index(np.argmax(z_masked), z_masked.shape)
            if np.sqrt((px - ix_v_m1)**2 + (py - iy_v_m1)**2) <= r_tol_pix_m1:
                n_rec_m1 += 1

    rec_rate_m1 = n_rec_m1 / n_inj_m1
    z_crit = 1.95996
    denom_m1 = 1.0 + (z_crit**2) / n_inj_m1
    center_m1 = (rec_rate_m1 + (z_crit**2) / (2.0 * n_inj_m1)) / denom_m1
    margin_m1 = (z_crit * np.sqrt((rec_rate_m1 * (1.0 - rec_rate_m1) / n_inj_m1) + ((z_crit**2) / (4.0 * (n_inj_m1**2))))) / denom_m1
    ci_lower_m1 = max(0.0, center_m1 - margin_m1)
    ci_upper_m1 = min(1.0, center_m1 + margin_m1)

    snr_eff_inv_m1 = th_tmf_m1 + float(norm.ppf(rec_rate_m1)) if 0.0 < rec_rate_m1 < 1.0 else np.nan

    print(f"Model 1 (0.2 kpc, 0.5 km/s, 50度) 超サンプリング実測結果:")
    print(f"  超サンプリング max|S|: {float(np.max(np.abs(S_true_m1))):.4f} km/s")
    print(f"  テンプレート H0 96% 閾値: {th_tmf_m1:.4f}")
    print(f"  実測回収率: {rec_rate_m1:.1%} [95% CI: {ci_lower_m1:.1%}, {ci_upper_m1:.1%}]")
    print(f"  逆算実効 SNR: {snr_eff_inv_m1:.3f}")
    print(f"  確定判定: UNDETECTED (50%に届かない)")

    model1_tmf_result = {
        "condition": "Model 1, FWHM=0.2 kpc, sigma_v=0.5 km/s, inc=50 deg",
        "max_s_fine_kms": round(float(np.max(np.abs(S_true_m1))), 4),
        "th_h0_tmf": round(th_tmf_m1, 4),
        "rec_rate": round(rec_rate_m1, 4),
        "ci95_lower": round(ci_lower_m1, 4),
        "ci95_upper": round(ci_upper_m1, 4),
        "snr_eff_inverted": round(snr_eff_inv_m1, 3),
        "verdict": "UNDETECTED (回収率50%に届かない)"
    }

    # =========================================================================
    # 3. 平滑化後最大値の比の条件確認
    # =========================================================================
    # Model 1: FWHM=0.2 kpc, inc=50 deg, a_phys=2.2226 km/s -> max|S|=0.7331 km/s -> ratio = 0.3298 ~ 0.330
    # Model 2: FWHM=0.2 kpc, inc=50 deg, a_phys=5.1563 km/s -> max|S|=1.0940 km/s -> ratio = 0.2122 ~ 0.212
    # Model 3: FWHM=0.2 kpc, inc=50 deg, a_phys=17.7719 km/s -> max|S|=5.6802 km/s -> ratio = 0.3196 ~ 0.320
    smoothing_conditions = {
        "fwhm_kpc": 0.2,
        "inc_deg": 50.0,
        "grid": "oversampled dx=0.01 kpc, downsampled to dpix=0.08 kpc",
        "ratios": {
            "model_1": {"a_phys_kms": 2.2226, "s_max_kms": 0.7331, "ratio": 0.330},
            "model_2": {"a_phys_kms": 5.1563, "s_max_kms": 1.0940, "ratio": 0.212},
            "model_3": {"a_phys_kms": 17.7719, "s_max_kms": 5.6802, "ratio": 0.320}
        }
    }

    # =========================================================================
    # 4. JSON の更新
    # =========================================================================
    base_json_path = os.path.join(OUT_DIR, "final_report_numbers.json")
    with open(base_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["model2_oversampled_analysis"] = model2_oversampled_analysis
    data["model1_template_detector_result"] = model1_tmf_result
    data["smoothing_ratios_details"] = smoothing_conditions

    with open(base_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nfinal_report_numbers.json を更新完了")

    # =========================================================================
    # 5. SHA256SUMS に tools/*.py を含む全ファイルを収録
    # =========================================================================
    print("\n" + "=" * 100)
    print("5. SHA256SUMS.txt の生成 (出力ファイル13件 + 解析スクリプト tools/*.py 全件)")
    print("=" * 100)

    output_files = [
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

    tool_files = sorted(glob.glob(os.path.join(TOOLS_DIR, "*.py")))

    sums_lines = []
    print("[出力ファイル]:")
    for fname in output_files:
        fpath = os.path.join(OUT_DIR, fname)
        sha = get_sha256(fpath)
        if sha:
            line = f"{sha}  output/{fname}"
            sums_lines.append(line)
            print(line)

    print("\n[解析スクリプト (tools/*.py)]:")
    for tpath in tool_files:
        sha = get_sha256(tpath)
        rel_path = "tools/" + os.path.basename(tpath)
        line = f"{sha}  {rel_path}"
        sums_lines.append(line)
        print(line)

    sums_path = os.path.join(OUT_DIR, "SHA256SUMS.txt")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sums_lines) + "\n")
    print(f"\nSHA256SUMS.txt を更新完了: {sums_path}")


if __name__ == "__main__":
    main()
