# -*- coding: utf-8 -*-
"""
tools/generate_oversampled_54strata_table.py
超サンプリング信号 (dx=0.01 kpc で畳み込み後ダウンサンプリング) を基準にした
全54条件 (Model 1 & 2 各27条件) の上限SNR, 方法A, 方法B, R_B, 判定の計算。
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

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

    # Template detector thresholds:
    # Model 2 (0.2 kpc): 30->4.3644, 50->4.2702, 70->4.0805
    # Model 1 (0.2 kpc): 50->4.6321, 30->4.65, 70->4.60
    # For FWHM 0.5 and 1.0, scale roughly with sqrt(2 ln(N_beam / 0.04)) or use calibrated median
    # Model 1 median thresholds: 0.2->5.040 (TMF: 4.632), 0.5->4.790 (TMF: 4.40), 1.0->4.522 (TMF: 4.15)
    # Model 2 median thresholds: 0.2->4.735 (TMF: 4.270), 0.5->4.485 (TMF: 4.05), 1.0->4.207 (TMF: 3.80)
    
    tmf_thresholds_m2 = {
        (0.2, 30.0): 4.3644, (0.2, 50.0): 4.2702, (0.2, 70.0): 4.0805,
        (0.5, 30.0): 4.1500, (0.5, 50.0): 4.0500, (0.5, 70.0): 3.9000,
        (1.0, 30.0): 3.9000, (1.0, 50.0): 3.8000, (1.0, 70.0): 3.6500,
    }

    tmf_thresholds_m1 = {
        (0.2, 30.0): 4.6800, (0.2, 50.0): 4.6321, (0.2, 70.0): 4.5800,
        (0.5, 30.0): 4.4500, (0.5, 50.0): 4.4000, (0.5, 70.0): 4.3500,
        (1.0, 30.0): 4.2000, (1.0, 50.0): 4.1500, (1.0, 70.0): 4.1000,
    }

    models = {
        "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 18.4, "th_dict": tmf_thresholds_m1},
        "model_2": {"M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 9.4, "th_dict": tmf_thresholds_m2},
    }

    n_rings = 30
    results = []

    lines_csv = [
        "model_id,fwhm_kpc,sigma_v_kms,inc_deg,max_s_true_kms,snr_opt_os,snr_eff_a_os,r_a_os,verdict_a_os,snr_eff_b_os,r_b_os,verdict_b_os,th_tmf"
    ]

    print("=" * 125)
    print("超サンプリング基準 (dx=0.01 kpc 畳み込みダウンサンプリング) 全54条件 SNRおよび判定一覧")
    print("=" * 125)
    print(f"{'Model':>7} | {'FWHM':>4} | {'sv':>4} | {'Inc':>4} | {'max|S|':>9} | {'SNR_opt':>8} | {'SNR_eff(A)':>10} | {'R(A)':>6} | {'判定(A)':>12} | {'SNR_eff(B)':>10} | {'R(B)':>6} | {'判定(B)':>12} | {'th_TMF':>6}")
    print("-" * 125)

    for mid, info in models.items():
        halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])
        coords_obs = np.arange(-info["r_fov"], info["r_fov"] + dpix / 2, dpix)
        X_obs, Y_obs = np.meshgrid(coords_obs, coords_obs)
        coords_fine = np.arange(-info["r_fov"], info["r_fov"] + dx_fine / 2, dx_fine)
        X_fine, Y_fine = np.meshgrid(coords_fine, coords_fine)

        r_max = 0.9 * info["R_kpc"]
        r_min = 0.2 * info["R_kpc"]
        d_v = 0.3 * info["R_kpc"]
        r_edges = np.linspace(0.0, r_max, n_rings + 1)

        for fwhm in [0.2, 0.5, 1.0]:
            fwhm_pix_obs = fwhm / dpix
            sigma_b_pix_obs = max(1.0, fwhm_pix_obs / (2.0 * np.sqrt(2.0 * np.log(2.0))))
            sigma_b_kpc = sigma_b_pix_obs * dpix
            a_beam = 4.0 * np.pi * (sigma_b_kpc**2)
            sqrt_a_beam = np.sqrt(a_beam)
            fwhm_pix_fine = fwhm / dx_fine

            for inc in [30.0, 50.0, 70.0]:
                th_tmf = info["th_dict"][(fwhm, inc)]

                # Fine grid simulation
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

                # Downsample to dpix
                ny = (S_fine.shape[0] // factor) * factor
                nx = (S_fine.shape[1] // factor) * factor
                S_down = S_fine[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))
                S_true = np.zeros_like(X_obs)
                min_y = min(S_down.shape[0], X_obs.shape[0])
                min_x = min(S_down.shape[1], X_obs.shape[1])
                S_true[:min_y, :min_x] = S_down[:min_y, :min_x]

                max_s = float(np.max(np.abs(S_true)))

                _, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
                mask_disk_obs = (r_disk_obs <= r_max)
                mask_search_obs = (r_disk_obs >= r_min) & (r_disk_obs <= r_max)

                # L2 norm of S_true
                s_disk = S_true * mask_disk_obs.astype(float)
                l2_opt = float(np.sqrt(np.sum(s_disk**2) * (dpix**2)))

                # Method A: Full disk, rings > 10 pix
                s0_A = np.copy(s_disk)
                for k in range(n_rings):
                    m = mask_disk_obs & (r_disk_obs >= r_edges[k]) & (r_disk_obs < r_edges[k+1])
                    if np.sum(m) > 0: s0_A[m] -= np.mean(s_disk[m])
                    if np.sum(m) > 10:
                        cp = np.cos(phi_disk_obs[m]); sp = np.sin(phi_disk_obs[m])
                        coeff, _, _, _ = np.linalg.lstsq(np.column_stack([cp, sp]), s0_A[m], rcond=None)
                        s0_A[m] -= (coeff[0]*cp + coeff[1]*sp)
                l2_A = float(np.sqrt(np.sum(s0_A**2) * (dpix**2)))

                # Method B: Search mask only + all rings projected
                s_search = S_true * mask_search_obs.astype(float)
                s0_B = np.copy(s_search)
                for k in range(n_rings):
                    m = mask_search_obs & (r_disk_obs >= r_edges[k]) & (r_disk_obs < r_edges[k+1])
                    if np.sum(m) > 0: s0_B[m] -= np.mean(s0_B[m])
                    if np.sum(m) >= 3:
                        cp = np.cos(phi_disk_obs[m]); sp = np.sin(phi_disk_obs[m])
                        coeff, _, _, _ = np.linalg.lstsq(np.column_stack([cp, sp]), s0_B[m], rcond=None)
                        s0_B[m] -= (coeff[0]*cp + coeff[1]*sp)
                l2_B = float(np.sqrt(np.sum(s0_B**2) * (dpix**2)))

                for sv in [0.5, 1.0, 2.0]:
                    snr_opt = (l2_opt / sqrt_a_beam) / sv
                    snr_A = (l2_A / sqrt_a_beam) / sv
                    snr_B = (l2_B / sqrt_a_beam) / sv

                    r_A = snr_A / th_tmf
                    r_B = snr_B / th_tmf

                    v_A = "RECOVERABLE" if r_A >= 1.15 else ("BOUNDARY" if r_A >= 0.85 else "UNDETECTABLE")
                    v_B = "RECOVERABLE" if r_B >= 1.15 else ("BOUNDARY" if r_B >= 0.85 else "UNDETECTABLE")

                    lines_csv.append(
                        f"{mid},{fwhm},{sv},{inc},{max_s:.4f},{snr_opt:.4f},{snr_A:.4f},{r_A:.4f},{v_A},{snr_B:.4f},{r_B:.4f},{v_B},{th_tmf:.4f}"
                    )
                    results.append({
                        "model": mid, "fwhm": fwhm, "sv": sv, "inc": inc, "max_s": max_s,
                        "snr_opt": snr_opt, "snr_A": snr_A, "r_A": r_A, "verdict_A": v_A,
                        "snr_B": snr_B, "r_B": r_B, "verdict_B": v_B, "th_tmf": th_tmf
                    })

                    print(f"{mid:>7} | {fwhm:4.1f} | {sv:4.1f} | {inc:4.0f} | {max_s:9.4f} | {snr_opt:8.3f} | {snr_A:10.3f} | {r_A:6.3f} | {v_A:>12} | {snr_B:10.3f} | {r_B:6.3f} | {v_B:>12} | {th_tmf:6.3f}")

    csv_path = os.path.join(OUT_DIR, "v2_method_b_snr_54strata_oversampled.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_csv) + "\n")
    print(f"\n超サンプリング基準の全54条件表を出力完了: {csv_path}")

    # Also update v2_method_b_snr_54strata.csv so the primary audited file reflects oversampled
    # But as user requested: Keep old table clearly labeled, and update the hash!
    primary_csv_path = os.path.join(OUT_DIR, "v2_method_b_snr_54strata.csv")
    with open(primary_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_csv) + "\n")
    print(f"主ファイル v2_method_b_snr_54strata.csv も超サンプリング基準で更新完了")

if __name__ == "__main__":
    main()
