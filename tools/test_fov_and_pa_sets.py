# -*- coding: utf-8 -*-
"""
tools/test_fov_and_pa_sets.py
ユーザー指示2の検証:
1. 旧視野 (r_fov = 6.0 kpc < 0.9 R = 9.0 kpc) では楕円長軸が視野端で切断され、
   PA=0°セット (長軸上限 6.0 kpc) と PA=45°セット (対角線長軸上限 8.49 kpc) で領域が異なっていたことを確認。
2. 視野半幅を r_fov = 1.0 * R_kpc (>= 0.9 R) に拡大し、r_max = 0.9 * R_kpc に統一したとき、
   PA=0°セット (0, 90, 180, 270 deg) と PA=45°セット (45, 135, 225, 315 deg) の
   渦なしデータの統計および FPR が統計誤差の範囲で一致することを確認する。
"""

import os
import sys
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe


def test_fov_truncation_and_symmetry():
    R_kpc = 10.0
    r_max_disk = 0.9 * R_kpc  # 9.0 kpc
    r_min_disk = 0.2 * R_kpc  # 2.0 kpc
    halo = pipe.TFHaloModel(M_msun=1e11, R_kpc=R_kpc, m_eV=1e-22)
    dpix = 0.08
    fwhm_kpc = 0.5
    fwhm_pix = fwhm_kpc / dpix
    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_v = 0.5
    inc_deg = 50.0

    for fov_label, r_fov in [("旧視野 (r_fov=6.0 kpc < 0.9R=9.0 kpc: 楕円切断あり)", 6.0),
                             ("修正視野 (r_fov=10.0 kpc >= 0.9R=9.0 kpc: 楕円完全包含)", 10.0)]:
        print("=" * 85)
        print(f"[{fov_label}]")
        print("=" * 85)
        coords = np.arange(-r_fov, r_fov + dpix / 2, dpix)
        X_obs, Y_obs = np.meshgrid(coords, coords)

        for pa in [0.0, 45.0, 90.0]:
            _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc_deg)
            mask_disk = (r_disk <= r_max_disk)
            mask_search = (r_disk >= r_min_disk) & (r_disk <= r_max_disk)
            n_pix_disk = int(np.sum(mask_disk))
            n_pix_search = int(np.sum(mask_search))
            max_r_in_mask = float(np.max(r_disk[mask_disk]))

            v0 = halo.v_0(r_disk)
            v_los = pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc_deg, v_sys=100.0)
            v_conv = pipe.convolve_beam_normalized(v_los, fwhm_pix=fwhm_pix, mask=mask_disk)
            _, _, res_v1 = pipe.fit_axisymmetric_disk_rings(
                r_disk, phi_disk, v_conv, inc_deg=inc_deg, v_sys=100.0, r_max=r_max_disk, n_rings=15
            )
            rms_v1 = float(np.sqrt(np.nanmean(res_v1[mask_search]**2)))
            pks_v1 = pipe.find_local_peaks_2d(res_v1, threshold_snr=0.0, noise_sigma=sigma_v, min_distance_pix=3)
            max_snr_v1 = max([p["snr"] for p in pks_v1 if mask_search[p["y_idx"], p["x_idx"]]])

            print(f"  PA={pa:4.0f}°: n_pix_disk={n_pix_disk:5d}, n_pix_search={n_pix_search:5d}, "
                  f"v1 Noiseless RMS={rms_v1:.4f} km/s, MaxSNR={max_snr_v1:.2f}")

    # Now run Monte Carlo trials comparing 0-deg set (0, 90, 180, 270) vs 45-deg set (45, 135, 225, 315)
    # with r_fov = 10.0 kpc (>= 0.9R = 9.0 kpc)
    print("\n" + "=" * 85)
    print("修正視野 (r_fov=10.0 kpc >= 0.9R) における 0°セット vs 45°セット 回転対称性検証")
    print("=" * 85)
    r_fov = 10.0
    coords = np.arange(-r_fov, r_fov + dpix / 2, dpix)
    X_obs, Y_obs = np.meshgrid(coords, coords)

    n_trials_per_pa = 600  # 2,400 trials per 4-PA set
    all_pas = [0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]

    # Precompute masks and noiseless residuals per PA
    pa_ctx = {}
    for pa in all_pas:
        _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc_deg)
        mask_disk = (r_disk <= r_max_disk)
        mask_search = (r_disk >= r_min_disk) & (r_disk <= r_max_disk)
        v0 = halo.v_0(r_disk)
        v_los = pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc_deg, v_sys=100.0)
        v_conv = pipe.convolve_beam_normalized(v_los, fwhm_pix=fwhm_pix, mask=mask_disk)
        _, _, res_v1_nl = pipe.fit_axisymmetric_disk_rings(
            r_disk, phi_disk, v_conv, inc_deg=inc_deg, v_sys=100.0, r_max=r_max_disk, n_rings=15
        )
        pa_ctx[pa] = (r_disk, phi_disk, mask_disk, mask_search, res_v1_nl)

    # Collect max SNRs for v2 (forward model subtraction -> pure noise in mask_search)
    # and v1 (ring subtraction)
    snrs_v2 = {pa: [] for pa in all_pas}
    snrs_v1 = {pa: [] for pa in all_pas}

    for pa in all_pas:
        r_disk, phi_disk, mask_disk, mask_search, res_v1_nl = pa_ctx[pa]
        for t in range(n_trials_per_pa):
            rng = np.random.RandomState(int(pa * 10000 + t + 12345))
            noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)

            # v2: res = noise on mask_search
            pks2 = pipe.find_local_peaks_2d(noise, threshold_snr=0.0, noise_sigma=sigma_v, min_distance_pix=3)
            s2 = [p["snr"] for p in pks2 if mask_search[p["y_idx"], p["x_idx"]]]
            snrs_v2[pa].append(max(s2) if s2 else 0.0)

            # v1: res = res_v1_nl + ring_subtracted(noise) -> approximate or exact
            res1 = res_v1_nl + noise
            pks1 = pipe.find_local_peaks_2d(res1, threshold_snr=0.0, noise_sigma=sigma_v, min_distance_pix=3)
            s1 = [p["snr"] for p in pks1 if mask_search[p["y_idx"], p["x_idx"]]]
            snrs_v1[pa].append(max(s1) if s1 else 0.0)

    for pipe_name, snr_dict in [("Pipeline v2 (前方モデル減算, A_los3=0)", snrs_v2),
                                ("Pipeline v1 (リング減算, r_max=0.9R, r_fov=1.0R)", snrs_v1)]:
        print(f"\n[{pipe_name}]")
        # Calibrate 96th percentile (target FPR = 4.0%) on first 200 trials of each PA, validate on next 400
        cal_pool = []
        for pa in all_pas:
            cal_pool.extend(snr_dict[pa][:200])
        th96 = float(np.percentile(cal_pool, 96))
        print(f"  Pooled 96th-percentile Threshold (Target FPR=4.0%): SNR_th = {th96:.4f}")

        set0_hits = 0
        set0_tot = 0
        set45_hits = 0
        set45_tot = 0

        for pa in all_pas:
            val_arr = np.array(snr_dict[pa][200:])
            hits = int(np.sum(val_arr >= th96))
            tot = len(val_arr)
            fpr = hits / tot
            se = np.sqrt(fpr * (1.0 - fpr) / tot)
            print(f"    PA={pa:5.1f}°: MeanMaxSNR={np.mean(val_arr):.3f}±{np.std(val_arr)/np.sqrt(tot):.3f}, "
                  f"P96={np.percentile(val_arr, 96):.3f}, Val FPR={fpr:.4f}±{se:.4f} ({hits}/{tot})")
            if pa in [0.0, 90.0, 180.0, 270.0]:
                set0_hits += hits
                set0_tot += tot
            else:
                set45_hits += hits
                set45_tot += tot

        fpr0 = set0_hits / set0_tot
        fpr45 = set45_hits / set45_tot
        diff = abs(fpr0 - fpr45)
        se_diff = np.sqrt(fpr0 * (1.0 - fpr0) / set0_tot + fpr45 * (1.0 - fpr45) / set45_tot)
        z_score = diff / se_diff if se_diff > 0 else 0.0
        print(f"  --> 0°セット (0,90,180,270) FPR = {fpr0:.4f} ({set0_hits}/{set0_tot})")
        print(f"  --> 45°セット (45,135,225,315) FPR = {fpr45:.4f} ({set45_hits}/{set45_tot})")
        print(f"  --> セット間差 |FPR_0 - FPR_45| = {diff:.4f} (SE = {se_diff:.4f}, Z = {z_score:.2f} sigma)")


if __name__ == "__main__":
    test_fov_truncation_and_symmetry()
