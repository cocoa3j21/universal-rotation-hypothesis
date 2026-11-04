# -*- coding: utf-8 -*-
"""
tools/run_oversampled_template_detector.py
超サンプリング格子 (dx=0.01 kpc) による真の注入場 S_true を生成し、
双極子テンプレート走査整合フィルター検出器を再実行する。
- 30度, 50度, 70度の注入ピーク値
- H0閾値 (4% FPR)
- 実測回収率 + 95%信頼区間
- 実測回収率からの逆算実効SNR
- 上限SNRおよび方法B周辺化後SNRとの比較
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
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

def main():
    dpix = 0.08
    dx_fine = 0.01
    factor = int(dpix / dx_fine)  # 8
    r_fov = 9.4
    fwhm = 0.2
    sv = 0.5
    fwhm_pix_obs = fwhm / dpix
    sigma_b_pix_obs = max(1.0, fwhm_pix_obs / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_b_kpc = sigma_b_pix_obs * dpix
    a_beam = 4.0 * np.pi * (sigma_b_kpc**2)

    halo = pipe.TFHaloModel(M_msun=1e11, R_kpc=10.0, m_eV=1e-22)
    coords_obs = np.arange(-r_fov, r_fov + dpix / 2, dpix)
    X_obs, Y_obs = np.meshgrid(coords_obs, coords_obs)
    r_max = 9.0
    r_min = 2.0
    d_v = 3.0

    coords_fine = np.arange(-r_fov, r_fov + dx_fine / 2, dx_fine)
    X_fine, Y_fine = np.meshgrid(coords_fine, coords_fine)
    fwhm_pix_fine = fwhm / dx_fine

    n_h0_trials = 800
    n_inj_trials = 500  # N=500 for tighter CI

    results = {}

    print("=" * 100)
    print("超サンプリング注入 (dx=0.01 kpc) による双極子テンプレート検出器の再実測")
    print("=" * 100)
    print(f"{'Inc':>4} | {'超サンプリング max|S|':>20} | {'通常格子 max|S|':>17} | {'H0 96%閾値':>11} | {'実測回収率 (95% CI)':>24} | {'逆算SNR_eff':>12} | {'判定':>14}")
    print("-" * 115)

    for inc in [30.0, 50.0, 70.0]:
        # Fine grid generation
        _, _, r_disk_fine, phi_disk_fine = pipe.obs_to_disk(X_fine, Y_fine, pa_deg=0.0, inc_deg=inc)
        mask_disk_fine = (r_disk_fine <= r_max)
        v0_fine = halo.v_0(r_disk_fine)
        v_unp_fine = pipe.project_vlos(v0_fine, 0.0, phi_disk_fine, inc_deg=inc, v_sys=100.0)
        dv_fine = halo.compute_vortex_velocity_perturbation_map(
            r_disk_fine * np.cos(phi_disk_fine), r_disk_fine * np.sin(phi_disk_fine), [(d_v, 0.0)]
        )
        vlos_vort_fine = dv_fine * np.cos(phi_disk_fine) * np.sin(np.radians(inc))

        v_conv_tot_fine = pipe.convolve_beam_normalized(v_unp_fine + vlos_vort_fine, fwhm_pix=fwhm_pix_fine, mask=mask_disk_fine)
        v_conv_unp_fine = pipe.convolve_beam_normalized(v_unp_fine, fwhm_pix=fwhm_pix_fine, mask=mask_disk_fine)
        S_fine = np.nan_to_num(v_conv_tot_fine - v_conv_unp_fine, nan=0.0)

        # Downsample to dpix = 0.08 kpc
        ny = (S_fine.shape[0] // factor) * factor
        nx = (S_fine.shape[1] // factor) * factor
        S_down = S_fine[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))
        S_true = np.zeros_like(X_obs)
        min_y = min(S_down.shape[0], X_obs.shape[0])
        min_x = min(S_down.shape[1], X_obs.shape[1])
        S_true[:min_y, :min_x] = S_down[:min_y, :min_x]

        # Standard grid signal for comparison
        _, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
        mask_disk_obs = (r_disk_obs <= r_max)
        mask_search_obs = (r_disk_obs >= r_min) & (r_disk_obs <= r_max)
        v0_obs = halo.v_0(r_disk_obs)
        v_unp_obs = pipe.project_vlos(v0_obs, 0.0, phi_disk_obs, inc_deg=inc, v_sys=100.0)
        dv_obs = halo.compute_vortex_velocity_perturbation_map(
            r_disk_obs * np.cos(phi_disk_obs), r_disk_obs * np.sin(phi_disk_obs), [(d_v, 0.0)]
        )
        vlos_vort_obs = dv_obs * np.cos(phi_disk_obs) * np.sin(np.radians(inc))
        v_conv_tot_obs = pipe.convolve_beam_normalized(v_unp_obs + vlos_vort_obs, fwhm_pix=fwhm_pix_obs, mask=mask_disk_obs)
        v_conv_unp_obs = pipe.convolve_beam_normalized(v_unp_obs, fwhm_pix=fwhm_pix_obs, mask=mask_disk_obs)
        S_std = np.nan_to_num(v_conv_tot_obs - v_conv_unp_obs, nan=0.0)

        max_s_fine = float(np.max(np.abs(S_true)))
        max_s_std = float(np.max(np.abs(S_std)))

        # Cutout template for matched filter
        xv_obs, yv_obs = pipe.disk_to_obs(d_v, 0.0, pa_deg=0.0, inc_deg=inc)
        ix_v = int(np.round((xv_obs - (-r_fov)) / dpix))
        iy_v = int(np.round((yv_obs - (-r_fov)) / dpix))

        win = 25
        y1, y2 = max(0, iy_v - win), min(X_obs.shape[0], iy_v + win + 1)
        x1, x2 = max(0, ix_v - win), min(X_obs.shape[1], ix_v + win + 1)
        template = np.copy(S_true[y1:y2, x1:x2])
        template -= np.mean(template)

        g_temp = gaussian_filter(template, sigma=sigma_b_pix_obs, mode='constant', cval=0.0)
        imp = np.zeros_like(template)
        imp[imp.shape[0]//2, imp.shape[1]//2] = 1.0
        gk = gaussian_filter(imp, sigma=sigma_b_pix_obs, mode='constant', cval=0.0)
        gk_norm = float(np.sqrt(np.sum(gk**2)))
        g_temp_norm = float(np.sqrt(np.sum(g_temp**2)))
        tmf_std = sv * (g_temp_norm / gk_norm) if gk_norm > 0 else 1.0
        t_rev = template[::-1, ::-1]

        # H0 calibration
        max_z_h0 = []
        for t in range(n_h0_trials):
            rng = np.random.RandomState(t + 15000)
            noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_b_pix_obs, sv, rng)
            z_map = fftconvolve(noise, t_rev, mode='same') / tmf_std
            z_search = z_map[mask_search_obs]
            max_z_h0.append(np.max(np.abs(z_search)))

        th_tmf = float(np.percentile(max_z_h0, 96.0))

        # H1 Injection-Recovery
        n_rec = 0
        r_tol_pix = 3.0
        for t in range(n_inj_trials):
            rng = np.random.RandomState(t + 35000)
            noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_b_pix_obs, sv, rng)
            obs = S_true + noise
            z_map = fftconvolve(obs, t_rev, mode='same') / tmf_std
            z_masked = np.where(mask_search_obs, np.abs(z_map), 0.0)
            if np.max(z_masked) >= th_tmf:
                py, px = np.unravel_index(np.argmax(z_masked), z_masked.shape)
                if np.sqrt((px - ix_v)**2 + (py - iy_v)**2) <= r_tol_pix:
                    n_rec += 1

        rec_rate = n_rec / n_inj_trials
        # Wilson score interval for 95% CI
        z_crit = 1.95996
        denom = 1.0 + (z_crit**2) / n_inj_trials
        center = (rec_rate + (z_crit**2) / (2.0 * n_inj_trials)) / denom
        margin = (z_crit * np.sqrt((rec_rate * (1.0 - rec_rate) / n_inj_trials) + ((z_crit**2) / (4.0 * (n_inj_trials**2))))) / denom
        ci_lower = max(0.0, center - margin)
        ci_upper = min(1.0, center + margin)

        # Invert SNR_eff from recovery rate: rec_rate = Phi(SNR_eff - th_tmf)
        # SNR_eff = th_tmf + norm.ppf(rec_rate)
        if 0.0 < rec_rate < 1.0:
            snr_eff_inv = th_tmf + float(norm.ppf(rec_rate))
        else:
            snr_eff_inv = np.nan

        # Verdict
        if ci_lower >= 0.50:
            verdict = "RECOVERED"
        elif ci_upper < 0.50:
            verdict = "UNDETECTED"
        else:
            verdict = "BOUNDARY"  # CI includes 50%

        print(f"{inc:4.0f} | {max_s_fine:18.4f} km/s | {max_s_std:15.4f} km/s | {th_tmf:11.4f} | {rec_rate:6.1%} [{ci_lower:5.1%}, {ci_upper:5.1%}] | {snr_eff_inv:12.3f} | {verdict:>14}")

        results[int(inc)] = {
            "inc_deg": inc,
            "max_s_fine_kms": round(max_s_fine, 4),
            "max_s_std_kms": round(max_s_std, 4),
            "th_h0_tmf": round(th_tmf, 4),
            "rec_rate": round(rec_rate, 4),
            "ci95_lower": round(ci_lower, 4),
            "ci95_upper": round(ci_upper, 4),
            "snr_eff_inverted": round(snr_eff_inv, 3),
            "verdict": verdict
        }

    out_file = os.path.join(OUT_DIR, "v2_oversampled_template_detector_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果を保存しました: {out_file}")

if __name__ == "__main__":
    main()
