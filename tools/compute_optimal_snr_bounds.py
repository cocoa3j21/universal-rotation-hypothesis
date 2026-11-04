# -*- coding: utf-8 -*-
"""
tools/compute_optimal_snr_bounds.py
ユーザー指示4: 検出器の限界の定量化
- 渦の信号の形そのもの（双極子ローブ構造全体）をテンプレート S(x,y) に使った
  最適整合フィルター SNR の上限を、前方モデルで、モデル・FWHM・sigma_v・傾斜角ごとに計算する。
- 単一ピクセル最大ピーク SNR_peak = max|S| / sigma_v と、
  真の信号テンプレートを用いた相関ノイズ下のテンプレート整合フィルター SNR_TMF、
  および独立ビーム要素換算 SNR_beam_indep を比較し、
  Model 2 の最適 SNR が検出閾値（SNR_th ≈ 4.5, あるいは位置既知の片側/両側検出閾値）より低いか高いかを確認する。
"""

import os
import sys
import csv
import numpy as np
from scipy.ndimage import gaussian_filter

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

MODELS = {
    "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 20.0, "dpix": 0.08, "a_phys": 1.12},
    "model_2": {"M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 10.0, "dpix": 0.08, "a_phys": 4.89},
    "model_3": {"M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 3.0, "dpix": 0.08, "a_phys": 30.80}
}


def main():
    out_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    print("=" * 115)
    print("最適テンプレート整合フィルター SNR 上限 (Optimal Template Matched Filter SNR Bound) vs ピーク検出器 SNR")
    print("=" * 115)
    print(f"{'Model':>7} | {'FWHM':>4} | {'sv':>4} | {'Inc':>4} | {'Max|S|':>7} | {'SNR_peak':>8} | {'SNR_TMF(Exact)':>14} | {'SNR_IndepBeam':>13} | {'SNR_WhitePix':>12} | {'vs Th(4.5)':>12}")
    print("-" * 115)

    for model_id in ["model_1", "model_2", "model_3"]:
        info = MODELS[model_id]
        halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])
        dpix = info["dpix"]
        r_fov = info["r_fov"]  # >= 0.9 R_kpc
        coords = np.arange(-r_fov, r_fov + dpix / 2, dpix)
        X_obs, Y_obs = np.meshgrid(coords, coords)
        r_max = 0.9 * info["R_kpc"]
        r_min = 0.2 * info["R_kpc"]
        d_v = 0.3 * info["R_kpc"]

        for fwhm in [0.2, 0.5, 1.0]:
            fwhm_pix = fwhm / dpix
            sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
            # Exact normalization of beam kernel G used in generate_beam_correlated_noise:
            # In generate_beam_correlated_noise, white noise w ~ N(0,1) is convolved with gaussian_filter(w, sigma_beam_pix)
            # and scaled by sigma_v / sqrt(sum G^2). Since sum G^2 = 1 / (4 * pi * sigma_beam_pix^2),
            # n = sigma_v * (G * w) / ||G||_2.
            # Thus Var(sum S * n) = sigma_v^2 * ||G * S||_2^2 / ||G||_2^2.
            impulse = np.zeros((int(8 * sigma_beam_pix) + 5, int(8 * sigma_beam_pix) + 5))
            impulse[impulse.shape[0] // 2, impulse.shape[1] // 2] = 1.0
            g_kern = gaussian_filter(impulse, sigma=sigma_beam_pix, mode='constant', cval=0.0)
            g_l2_norm = float(np.sqrt(np.sum(g_kern**2)))

            for inc in [30.0, 50.0, 70.0]:
                _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
                mask_disk = (r_disk <= r_max)
                mask_search = (r_disk >= r_min) & (r_disk <= r_max)

                v0 = halo.v_0(r_disk)
                v_los_unpert = pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc, v_sys=100.0)
                dv_map = halo.compute_vortex_velocity_perturbation_map(
                    r_disk * np.cos(phi_disk), r_disk * np.sin(phi_disk), [(d_v, 0.0)]
                )
                v_los_vortex = dv_map * np.cos(phi_disk) * np.sin(np.radians(inc))

                # Forward model signal S(x, y)
                v_conv_tot = pipe.convolve_beam_normalized(v_los_unpert + v_los_vortex, fwhm_pix=fwhm_pix, mask=mask_disk)
                v_conv_unp = pipe.convolve_beam_normalized(v_los_unpert, fwhm_pix=fwhm_pix, mask=mask_disk)
                S_map = np.nan_to_num(v_conv_tot - v_conv_unp, nan=0.0) * mask_search.astype(float)

                max_abs_s = float(np.max(np.abs(S_map)))
                s_l2_sq = float(np.sum(S_map**2))
                s_l2 = float(np.sqrt(s_l2_sq))

                # Convolve S_map with G to get exact noise variance of template inner product <S, n>
                gs_map = gaussian_filter(S_map, sigma=sigma_beam_pix, mode='constant', cval=0.0)
                gs_l2 = float(np.sqrt(np.sum(gs_map**2)))
                # For sigma_v = 1.0:
                snr_tmf_unit = (s_l2_sq * g_l2_norm) / gs_l2 if gs_l2 > 0 else 0.0

                # Independent beam area equivalent: N_beam_eff = Area / (2 * pi * sigma_beam^2)
                # SNR_indep_beam (for sigma_v=1) = sqrt( sum(S^2) / (2 * pi * sigma_beam_pix^2) )
                snr_indep_unit = float(np.sqrt(s_l2_sq / (2.0 * np.pi * (sigma_beam_pix**2))))

                for sv in [0.5, 1.0, 2.0]:
                    snr_peak = max_abs_s / sv
                    snr_tmf = snr_tmf_unit / sv
                    snr_indep = snr_indep_unit / sv
                    snr_white_pix = s_l2 / sv

                    status_str = "TMF>=4.5" if snr_tmf >= 4.5 else ("TMF>=3.0" if snr_tmf >= 3.0 else "BELOW_TH")
                    if model_id == "model_2" or (fwhm == 0.2 and sv == 0.5):
                        print(f"{model_id:>7} | {fwhm:4.1f} | {sv:4.1f} | {inc:4.0f} | {max_abs_s:7.4f} | {snr_peak:8.3f} | {snr_tmf:14.3f} | {snr_indep:13.3f} | {snr_white_pix:12.3f} | {status_str:>12}")

                    rows.append({
                        "model_id": model_id,
                        "fwhm_kpc": fwhm,
                        "sigma_v_kms": sv,
                        "inc_deg": inc,
                        "max_abs_signal_kms": round(max_abs_s, 5),
                        "snr_single_peak": round(snr_peak, 4),
                        "snr_optimal_tmf_correlated": round(snr_tmf, 4),
                        "snr_indep_beam_approx": round(snr_indep, 4),
                        "snr_uncorrelated_white_pix": round(snr_white_pix, 4),
                        "exceeds_blind_th_45": snr_tmf >= 4.50
                    })

    csv_path = os.path.join(out_dir, "v2_optimal_snr_bounds.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved optimal SNR bounds to: {csv_path}")


if __name__ == "__main__":
    main()
