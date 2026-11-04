# -*- coding: utf-8 -*-
"""
tools/run_final_definitive_analysis.py
最終承認用 決着スクリプト:
1. 周辺化後 SNR の 2通りの方法 (方法A: 穏健射影, 方法B: 保守的解析射影 [-14%]) による全54条件一覧
2. SHA256SUMS.txt の自動生成と出力ファイル完全検証
3. 双極子テンプレート走査整合フィルター検出器の実装と Model 2 最良3条件の実測回収率 (暫定の確定化)
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import gyro_gravity_theory.injection_pipeline as pipe

OUT_DIR = os.path.join(PROJECT_DIR, "output")


def generate_sha256sums_file():
    print("=" * 100)
    print("1. SHA256SUMS ファイルの生成 (出力ファイル全部の完全チェックサム)")
    print("=" * 100)

    target_files = [
        "v2_h0_calibration.csv",
        ".v2_snr_thresholds.json",
        "v2_cell_predictions_preregistered.csv",
        "v2_injection_recovery_summary.csv",
        "v2_exact_optimal_snr_54strata.csv",
        "v2_audited_optimal_snr_54strata.csv",
        "final_report_numbers.json",
        "cell_predictions_preregistered.csv",
        "h0_stratum_calibration.csv",
        "injection_recovery_summary.csv"
    ]

    sums_lines = []
    for fname in target_files:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            sha = h.hexdigest()
            sums_lines.append(f"{sha}  {fname}")
            print(f"{sha}  {fname}")

    sums_path = os.path.join(OUT_DIR, "SHA256SUMS.txt")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sums_lines) + "\n")
    print(f"\nSHA256SUMS.txt を作成完了: {sums_path}")
    return sums_lines


def compute_two_method_snr_table():
    print("\n" + "=" * 100)
    print("2. 周辺化後 SNR の 2通りの方法 (方法A: 穏健射影 vs 方法B: 保守的解析射影) による全54条件一覧")
    print("=" * 100)

    # Classification rule: R = SNR_eff / th
    # R >= 1.15: RECOVERABLE, 0.85 <= R < 1.15: BOUNDARY, R < 0.85: UNDETECTABLE
    model_thresholds = {
        ("model_1", 0.2): 5.040, ("model_1", 0.5): 4.790, ("model_1", 1.0): 4.522,
        ("model_2", 0.2): 4.735, ("model_2", 0.5): 4.485, ("model_2", 1.0): 4.207,
    }

    models = {
        "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 18.4},
        "model_2": {"M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 9.4},
    }

    dpix = 0.08
    results_54 = []

    print(f"{'Model':>7} | {'FWHM':>4} | {'sv':>4} | {'Inc':>4} | {'SNR_raw':>7} | {'SNR_eff(A)':>10} | {'R(A)':>6} | {'判定(A)':>12} | {'SNR_eff(B)':>10} | {'R(B)':>6} | {'判定(B)':>12}")
    print("-" * 110)

    for mid, info in models.items():
        halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])
        coords = np.arange(-info["r_fov"], info["r_fov"] + dpix / 2, dpix)
        X_obs, Y_obs = np.meshgrid(coords, coords)
        r_max = 0.9 * info["R_kpc"]
        r_min = 0.2 * info["R_kpc"]
        d_v = 0.3 * info["R_kpc"]
        n_rings = 30
        r_edges = np.linspace(0.0, r_max, n_rings + 1)

        for fwhm in [0.2, 0.5, 1.0]:
            fwhm_pix = fwhm / dpix
            sigma_b_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
            sigma_b_kpc = sigma_b_pix * dpix
            a_beam = 4.0 * np.pi * (sigma_b_kpc**2)
            sqrt_a_beam = np.sqrt(a_beam)
            th = model_thresholds[(mid, fwhm)]

            for inc in [30.0, 50.0, 70.0]:
                _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
                mask_disk = (r_disk <= r_max)
                mask_search = (r_disk >= r_min) & (r_disk <= r_max)

                dv = halo.compute_vortex_velocity_perturbation_map(
                    r_disk * np.cos(phi_disk), r_disk * np.sin(phi_disk), [(d_v, 0.0)])
                vlos_vort = dv * np.cos(phi_disk) * np.sin(np.radians(inc))

                # Unprojected signal
                s0_raw = vlos_vort * mask_disk.astype(float)
                l2_raw = float(np.sqrt(np.sum(s0_raw**2) * (dpix**2)))

                # Method A (穏健射影): Full disk s0, stable rings > 10 pixels
                s0_A = np.copy(s0_raw)
                for k in range(n_rings):
                    m = mask_disk & (r_disk >= r_edges[k]) & (r_disk < r_edges[k+1])
                    if np.sum(m) > 0: s0_A[m] -= np.mean(s0_raw[m])
                    if np.sum(m) > 10:
                        cp = np.cos(phi_disk[m]); sp = np.sin(phi_disk[m])
                        coeff, _, _, _ = np.linalg.lstsq(np.column_stack([cp, sp]), s0_A[m], rcond=None)
                        s0_A[m] -= (coeff[0]*cp + coeff[1]*sp)
                l2_A = float(np.sqrt(np.sum(s0_A**2) * (dpix**2)))

                # Method B (保守的解析射影): Search mask only (r >= 0.2R) + all rings projected
                s0_B = vlos_vort * mask_search.astype(float)
                for k in range(n_rings):
                    m = mask_search & (r_disk >= r_edges[k]) & (r_disk < r_edges[k+1])
                    if np.sum(m) > 0: s0_B[m] -= np.mean(s0_B[m])
                    if np.sum(m) >= 3:
                        cp = np.cos(phi_disk[m]); sp = np.sin(phi_disk[m])
                        coeff, _, _, _ = np.linalg.lstsq(np.column_stack([cp, sp]), s0_B[m], rcond=None)
                        s0_B[m] -= (coeff[0]*cp + coeff[1]*sp)
                l2_B = float(np.sqrt(np.sum(s0_B**2) * (dpix**2)))

                for sv in [0.5, 1.0, 2.0]:
                    snr_raw = (l2_raw / sqrt_a_beam) / sv
                    snr_A = (l2_A / sqrt_a_beam) / sv
                    snr_B = (l2_B / sqrt_a_beam) / sv

                    r_A = snr_A / th
                    r_B = snr_B / th

                    v_A = "RECOVERABLE" if r_A >= 1.15 else ("BOUNDARY" if r_A >= 0.85 else "UNDETECTABLE")
                    v_B = "RECOVERABLE" if r_B >= 1.15 else ("BOUNDARY" if r_B >= 0.85 else "UNDETECTABLE")

                    results_54.append({
                        "model": mid, "fwhm": fwhm, "sv": sv, "inc": inc,
                        "snr_raw": round(snr_raw, 3),
                        "snr_A": round(snr_A, 3), "r_A": round(r_A, 4), "verdict_A": v_A,
                        "snr_B": round(snr_B, 3), "r_B": round(r_B, 4), "verdict_B": v_B,
                        "threshold": th
                    })

                    print(f"{mid:>7} | {fwhm:4.1f} | {sv:4.1f} | {inc:4.0f} | {snr_raw:7.3f} | {snr_A:10.3f} | {r_A:6.3f} | {v_A:>12} | {snr_B:10.3f} | {r_B:6.3f} | {v_B:>12}")

    return results_54


def run_dipole_template_matched_filter_mc():
    print("\n" + "=" * 100)
    print("3. 双極子テンプレート走査 整合フィルター検出器の実装と Model 2 最良3条件の実測回収率")
    print("=" * 100)

    # Model 2 best 3 conditions: FWHM=0.2 kpc, sigma_v=0.5 km/s, inc in [30, 50, 70] deg
    # Clean subset: no arms (a3=0), no geometric error (geom=False)
    dpix = 0.08
    r_fov = 9.4
    fwhm = 0.2
    sv = 0.5
    fwhm_pix = fwhm / dpix
    sigma_b_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_b_kpc = sigma_b_pix * dpix
    a_beam = 4.0 * np.pi * (sigma_b_kpc**2)

    halo = pipe.TFHaloModel(M_msun=1e11, R_kpc=10.0, m_eV=1e-22)
    coords = np.arange(-r_fov, r_fov + dpix / 2, dpix)
    X_obs, Y_obs = np.meshgrid(coords, coords)
    r_max = 9.0
    r_min = 2.0
    d_v = 3.0

    # Test each inclination
    n_h0_trials = 800
    n_inj_trials = 400

    print(f"Model 2 双極子テンプレート走査 整合フィルター検出器 (H0: N={n_h0_trials}, H1: N={n_inj_trials})")
    print(f"{'Inc':>4} | {'H0 96%閾値(4%FPR)':>18} | {'物理注入 max|S|':>15} | {'最適SNR_opt':>12} | {'TMF実測回収率':>14} | {'実測判定':>14}")
    print("-" * 85)

    for inc in [30.0, 50.0, 70.0]:
        _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
        mask_disk = (r_disk <= r_max)
        mask_search = (r_disk >= r_min) & (r_disk <= r_max)

        v0 = halo.v_0(r_disk)
        v_unp = pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc, v_sys=100.0)
        dv = halo.compute_vortex_velocity_perturbation_map(r_disk * np.cos(phi_disk), r_disk * np.sin(phi_disk), [(d_v, 0.0)])
        vlos_vort = dv * np.cos(phi_disk) * np.sin(np.radians(inc))

        # True convolved template signal S(x, y)
        v_conv_tot = pipe.convolve_beam_normalized(v_unp + vlos_vort, fwhm_pix=fwhm_pix, mask=mask_disk)
        v_conv_unp = pipe.convolve_beam_normalized(v_unp, fwhm_pix=fwhm_pix, mask=mask_disk)
        S_true = np.nan_to_num(v_conv_tot - v_conv_unp, nan=0.0)
        max_s_val = float(np.max(np.abs(S_true)))

        # Prewhitened optimal SNR
        s0_raw = vlos_vort * mask_disk.astype(float)
        l2_raw = float(np.sqrt(np.sum(s0_raw**2) * (dpix**2)))
        snr_opt = (l2_raw / np.sqrt(a_beam)) / sv

        # Matched filter template centered at vortex position (d_v, 0.0)
        # We extract a local cutout template of size 41x41 pixels
        # Vortex obs position: pa=0 -> x_obs = d_v, y_obs = 0
        xv_obs, yv_obs = pipe.disk_to_obs(d_v, 0.0, pa_deg=0.0, inc_deg=inc)
        # Find pixel indices
        ix_v = int(np.round((xv_obs - (-r_fov)) / dpix))
        iy_v = int(np.round((yv_obs - (-r_fov)) / dpix))

        win = 25  # half size
        y1, y2 = max(0, iy_v - win), min(X_obs.shape[0], iy_v + win + 1)
        x1, x2 = max(0, ix_v - win), min(X_obs.shape[1], ix_v + win + 1)
        template = np.copy(S_true[y1:y2, x1:x2])
        # Demean and normalize template inner product variance under beam noise
        template = template - np.mean(template)
        # Convolution of template with Gaussian beam kernel for noise variance
        g_temp = gaussian_filter(template, sigma=sigma_b_pix, mode='constant', cval=0.0)
        # Variance of (template * noise):
        # noise = (G * w) * (sv / ||G||_2) => Var = sv^2 * ||g_temp||_2^2 / ||G||_2^2
        imp = np.zeros_like(template)
        imp[imp.shape[0]//2, imp.shape[1]//2] = 1.0
        gk = gaussian_filter(imp, sigma=sigma_b_pix, mode='constant', cval=0.0)
        gk_norm = float(np.sqrt(np.sum(gk**2)))
        g_temp_norm = float(np.sqrt(np.sum(g_temp**2)))
        tmf_std = sv * (g_temp_norm / gk_norm) if gk_norm > 0 else 1.0

        # Reversed template for 2D correlation/convolution
        t_rev = template[::-1, ::-1]

        # 1. H0 calibration: measure max_{search} Z(x,y)
        from scipy.signal import fftconvolve
        max_z_h0 = []
        for t in range(n_h0_trials):
            rng = np.random.RandomState(t + 11000)
            noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_b_pix, sv, rng)
            # 2D correlation via FFT
            z_map = fftconvolve(noise, t_rev, mode='same') / tmf_std
            z_search = z_map[mask_search]
            max_z_h0.append(np.max(np.abs(z_search)))

        # 96th percentile threshold (target FPR = 4.0%)
        th_tmf = float(np.percentile(max_z_h0, 96.0))

        # 2. H1 Injection-Recovery: inject S_true + noise and test detection
        n_rec = 0
        r_tol_pix = 3.0  # within 3 pixels of true vortex
        for t in range(n_inj_trials):
            rng = np.random.RandomState(t + 22000)
            noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_b_pix, sv, rng)
            obs = S_true + noise
            z_map = fftconvolve(obs, t_rev, mode='same') / tmf_std
            # Find peak in search mask
            z_masked = np.where(mask_search, np.abs(z_map), 0.0)
            max_val = np.max(z_masked)
            if max_val >= th_tmf:
                py, px = np.unravel_index(np.argmax(z_masked), z_masked.shape)
                dist_pix = np.sqrt((px - ix_v)**2 + (py - iy_v)**2)
                if dist_pix <= r_tol_pix:
                    n_rec += 1

        rec_rate = n_rec / n_inj_trials
        verdict = "RECOVERED (>=50%)" if rec_rate >= 0.50 else "UNDETECTED (<50%)"

        print(f"{inc:4.0f} | {th_tmf:18.4f} | {max_s_val:15.4f} | {snr_opt:12.3f} | {rec_rate:13.1%} | {verdict:>14}")


if __name__ == "__main__":
    generate_sha256sums_file()
    compute_two_method_snr_table()
    run_dipole_template_matched_filter_mc()
