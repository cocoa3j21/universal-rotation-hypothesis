# -*- coding: utf-8 -*-
import os, sys, hashlib
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

# 1. 回転対称性テスト (PA = 0, 90, 180, 270 deg および 45, 135, 225, 315 deg)
# 渦なしデータで、同じノイズ乱数または独立乱数に対する統計が一致するか確認する。
def test_rotational_symmetry():
    print("=" * 80)
    print("1. 回転対称性テスト (PA = 0, 90, 180, 270 deg / 45, 135, 225, 315 deg)")
    print("=" * 80)
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
    r_min, r_max = 0.2 * 10.0, 0.9 * 10.0

    for a_los3, arm_method in [(0.0, "none"), (3.0, "raw"), (3.0, "remove_k3"), (3.0, "arm_fit")]:
        print(f"\n--- Condition: a_los3={a_los3}, arm_method={arm_method} ---")
        # First check noiseless residual max SNR across PAs
        for pa in [0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]:
            _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc_deg)
            mask_disk = (r_disk <= 10.0 * 0.9)
            mask_search = (r_disk >= r_min) & (r_disk <= r_max)
            v0 = halo.v_0(r_disk)
            v_los = pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc_deg, v_sys=100.0)
            arm_inj = a_los3 * np.cos(3.0 * phi_disk) if a_los3 > 0 else 0.0
            v_conv = pipe.convolve_beam_normalized(v_los + arm_inj, fwhm_pix=fwhm_pix, mask=mask_disk)

            _, _, res_raw = pipe.fit_axisymmetric_disk_rings(
                r_disk, phi_disk, v_conv, inc_deg=inc_deg, v_sys=100.0, n_rings=15
            )
            if arm_method in ["none", "raw"]:
                res_proc = res_raw
            elif arm_method == "remove_k3":
                res_proc, _ = pipe.remove_harmonic_k3_rings(r_disk, phi_disk, res_raw, n_rings=15)
            elif arm_method == "arm_fit":
                res_proc, _, _, _ = pipe.fit_spiral_arm_direct_c(r_disk, phi_disk, res_raw, inc_deg=inc_deg)

            peaks = pipe.find_local_peaks_2d(res_proc, threshold_snr=1.0, noise_sigma=sigma_v, min_distance_pix=3)
            peaks_s = [p for p in peaks if r_min <= r_disk[p["y_idx"], p["x_idx"]] <= r_max]
            max_snr_nl = max((p["snr"] for p in peaks_s), default=0.0)
            rms_nl = np.sqrt(np.nanmean(res_proc[mask_search]**2))
            print(f"  PA={pa:5.1f} deg (Noiseless): RMS={rms_nl:.4f} km/s, MaxPeakSNR={max_snr_nl:.4f} (n_peaks={len(peaks_s)})")

if __name__ == "__main__":
    test_rotational_symmetry()
