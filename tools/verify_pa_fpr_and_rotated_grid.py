# -*- coding: utf-8 -*-
import os, sys, hashlib
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

def main():
    halo = pipe.TFHaloModel(M_msun=1e11, R_kpc=10.0, m_eV=1e-22)
    dpix = 0.125
    r_fov = 6.0
    coords = np.arange(-r_fov, r_fov + dpix/2, dpix)
    X_obs, Y_obs = np.meshgrid(coords, coords)
    fwhm_kpc = 0.5
    fwhm_pix = fwhm_kpc / dpix
    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_v = 0.5
    inc_deg = 50.0
    r_min, r_max_disk = 0.2 * 10.0, 0.9 * 10.0

    # =========================================================================
    # 1. Stage 3 の fpr_h0 と同じ定義 (トライアル単位: len(peaks) > 0) による PA別検証FPR
    # =========================================================================
    print("=" * 80)
    print("1. Stage 3 と同一定義 (トライアル単位 len(peaks)>0) による PA別検証FPR")
    print("=" * 80)

    for mode_label, rmax_arg in [("v1旧実装 (r_max=None 四隅バグあり)", None), ("r_max修正後 (r_max=0.9*R_kpc)", r_max_disk)]:
        print(f"\n[{mode_label}]")
        for a_los3, arm_method in [(0.0, "none"), (1.0, "raw"), (3.0, "remove_k3"), (3.0, "arm_fit")]:
            # Calibration (667 trials per PA = 2001 trials)
            cal_snrs = []
            contexts = {}
            for pa in [0.0, 45.0, 90.0]:
                _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc_deg)
                mask_disk = (r_disk <= r_max_disk)
                mask_search = (r_disk >= r_min) & (r_disk <= r_max_disk)
                v0 = halo.v_0(r_disk)
                v_los = pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc_deg, v_sys=100.0)
                arm_inj = a_los3 * np.cos(3.0 * phi_disk) if a_los3 > 0 else 0.0
                v_conv = pipe.convolve_beam_normalized(v_los + arm_inj, fwhm_pix=fwhm_pix, mask=mask_disk)
                contexts[pa] = (r_disk, phi_disk, v_conv, mask_search)

                for t in range(300): # 300 trials per PA for fast exact test
                    rng = np.random.RandomState(int(pa * 10000 + t))
                    noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
                    _, _, res_raw = pipe.fit_axisymmetric_disk_rings(
                        r_disk, phi_disk, v_conv + noise, inc_deg=inc_deg, v_sys=100.0, r_max=rmax_arg, n_rings=15
                    )
                    if arm_method in ["none", "raw"]:
                        res_proc = res_raw
                    elif arm_method == "remove_k3":
                        res_proc, _ = pipe.remove_harmonic_k3_rings(r_disk, phi_disk, res_raw, r_max=rmax_arg, n_rings=15)
                    elif arm_method == "arm_fit":
                        res_proc, _, _, _ = pipe.fit_spiral_arm_direct_c(r_disk, phi_disk, res_raw, inc_deg=inc_deg)

                    peaks = pipe.find_local_peaks_2d(res_proc, threshold_snr=1.0, noise_sigma=sigma_v, min_distance_pix=3)
                    pks_s = [p["snr"] for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
                    cal_snrs.append(max(pks_s) if len(pks_s) > 0 else 0.0)

            snr_th = max(4.0, float(np.percentile(cal_snrs, 95.0)))

            # Validation (400 trials per PA)
            pa_fpr = {}
            for pa in [0.0, 45.0, 90.0]:
                r_disk, phi_disk, v_conv, mask_search = contexts[pa]
                fp_cnt = 0
                n_val = 400
                for t in range(n_val):
                    rng = np.random.RandomState(int(500000 + pa * 10000 + t))
                    noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
                    _, _, res_raw = pipe.fit_axisymmetric_disk_rings(
                        r_disk, phi_disk, v_conv + noise, inc_deg=inc_deg, v_sys=100.0, r_max=rmax_arg, n_rings=15
                    )
                    if arm_method in ["none", "raw"]:
                        res_proc = res_raw
                    elif arm_method == "remove_k3":
                        res_proc, _ = pipe.remove_harmonic_k3_rings(r_disk, phi_disk, res_raw, r_max=rmax_arg, n_rings=15)
                    elif arm_method == "arm_fit":
                        res_proc, _, _, _ = pipe.fit_spiral_arm_direct_c(r_disk, phi_disk, res_raw, inc_deg=inc_deg)

                    peaks = pipe.find_local_peaks_2d(res_proc, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
                    pks_s = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
                    if len(pks_s) > 0:
                        fp_cnt += 1
                pa_fpr[pa] = fp_cnt / n_val

            mean_fpr = np.mean(list(pa_fpr.values()))
            print(f"  a3={a_los3}, {arm_method:9s} | Pooled SNR_th={snr_th:5.2f} | "
                  f"FPR(PA=0)={pa_fpr[0.0]:.4f}, FPR(PA=45)={pa_fpr[45.0]:.4f}, FPR(PA=90)={pa_fpr[90.0]:.4f} | "
                  f"Stratum Total FPR={mean_fpr:.4f}")

    # =========================================================================
    # 2. 90度ずつ回転対称性テスト (PA = 0, 90, 180, 270 deg) - 渦なし統計の一致確認
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. 90度回転対称性テスト (PA = 0, 90, 180, 270 deg, N=400 trials/PA)")
    print("=" * 80)
    # Using fixed threshold from PA=0 calibration (or v2 forward model)
    for pipe_name in ["v1 (Ring fit, r_max=9.0)", "v2 (Forward Convolved Model)"]:
        print(f"\n[{pipe_name}]")
        snrs_by_pa = {}
        for pa in [0.0, 90.0, 180.0, 270.0]:
            _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc_deg)
            mask_disk = (r_disk <= r_max_disk)
            mask_search = (r_disk >= r_min) & (r_disk <= r_max_disk)
            v0 = halo.v_0(r_disk)
            v_los = pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc_deg, v_sys=100.0)
            v_conv = pipe.convolve_beam_normalized(v_los, fwhm_pix=fwhm_pix, mask=mask_disk)

            max_snrs = []
            for t in range(400):
                rng = np.random.RandomState(int(900000 + pa * 1000 + t))
                noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
                if pipe_name.startswith("v1"):
                    _, _, res = pipe.fit_axisymmetric_disk_rings(
                        r_disk, phi_disk, v_conv + noise, inc_deg=inc_deg, v_sys=100.0, r_max=r_max_disk, n_rings=15
                    )
                else:
                    res = (v_conv + noise) - v_conv
                peaks = pipe.find_local_peaks_2d(res, threshold_snr=1.0, noise_sigma=sigma_v, min_distance_pix=3)
                pks_s = [p["snr"] for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
                max_snrs.append(max(pks_s) if len(pks_s) > 0 else 0.0)
            snrs_by_pa[pa] = np.array(max_snrs)

        th_ref = float(np.percentile(snrs_by_pa[0.0], 95.0))
        for pa in [0.0, 90.0, 180.0, 270.0]:
            arr = snrs_by_pa[pa]
            fpr_at_ref = np.mean(arr >= th_ref)
            se_fpr = np.sqrt(fpr_at_ref * (1.0 - fpr_at_ref) / len(arr))
            print(f"  PA={pa:5.1f} deg: MeanMaxSNR={np.mean(arr):.3f}±{np.std(arr)/np.sqrt(len(arr)):.3f}, "
                  f"P95={np.percentile(arr, 95.0):.3f}, FPR(at th={th_ref:.2f})={fpr_at_ref:.4f}±{se_fpr:.4f}")

    # =========================================================================
    # 3. 画素格子を45度回転させた対照実験 (モアレ仮説 vs r_max四隅バグの直接検証)
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. 画素格子を45度回転させた対照実験 (PA=45° on Standard Grid vs PA=0° on 45°-Rotated Grid)")
    print("=" * 80)
    # Compare:
    # Case 1: Galaxy at PA=45° on standard grid (X_obs, Y_obs)
    # Case 2: Galaxy at PA=0° on 45°-rotated grid (X_rot, Y_rot) -> relative angle between galaxy and grid axes is 45°!
    # Case 3: Galaxy at PA=45° on 45°-rotated grid -> galaxy axes aligned with rotated grid axes (equivalent to PA=0° on standard grid)!
    theta_grid = np.radians(45.0)
    X_rot = X_obs * np.cos(theta_grid) + Y_obs * np.sin(theta_grid)
    Y_rot = -X_obs * np.sin(theta_grid) + Y_obs * np.cos(theta_grid)

    for label, Xg, Yg, pa_gal in [
        ("Standard Grid (0°), Galaxy PA=0°  (相対角 0°)", X_obs, Y_obs, 0.0),
        ("Standard Grid (0°), Galaxy PA=45° (相対角 45°)", X_obs, Y_obs, 45.0),
        ("Rotated Grid (45°), Galaxy PA=45° (相対角 0°)", X_rot, Y_rot, 45.0),
    ]:
        _, _, r_disk, phi_disk = pipe.obs_to_disk(Xg, Yg, pa_deg=pa_gal, inc_deg=inc_deg)
        mask_disk = (r_disk <= r_max_disk)
        mask_search = (r_disk >= r_min) & (r_disk <= r_max_disk)
        v0 = halo.v_0(r_disk)
        v_los = pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc_deg, v_sys=100.0)
        arm_inj = 3.0 * np.cos(3.0 * phi_disk)
        # Note: gaussian_filter acts along array axes (which are rotated by 45° in Rotated Grid)
        v_conv = pipe.convolve_beam_normalized(v_los + arm_inj, fwhm_pix=fwhm_pix, mask=mask_disk)

        _, _, res_old = pipe.fit_axisymmetric_disk_rings(r_disk, phi_disk, v_conv, inc_deg=inc_deg, v_sys=100.0, r_max=None, n_rings=15)
        _, _, res_fix = pipe.fit_axisymmetric_disk_rings(r_disk, phi_disk, v_conv, inc_deg=inc_deg, v_sys=100.0, r_max=r_max_disk, n_rings=15)

        rms_old = np.sqrt(np.nanmean(res_old[mask_search]**2))
        rms_fix = np.sqrt(np.nanmean(res_fix[mask_search]**2))
        pks_old = [p["snr"] for p in pipe.find_local_peaks_2d(res_old, 1.0, sigma_v, 3) if mask_search[p["y_idx"], p["x_idx"]]]
        pks_fix = [p["snr"] for p in pipe.find_local_peaks_2d(res_fix, 1.0, sigma_v, 3) if mask_search[p["y_idx"], p["x_idx"]]]
        print(f"  {label}:")
        print(f"    r_max=None (旧): r_max_grid={np.nanmax(r_disk):.2f} kpc, RMS={rms_old:.4f}, MaxSNR={max(pks_old, default=0):.2f}")
        print(f"    r_max=9.00 (修): r_max_grid={r_max_disk:.2f} kpc, RMS={rms_fix:.4f}, MaxSNR={max(pks_fix, default=0):.2f}")

if __name__ == "__main__":
    main()
