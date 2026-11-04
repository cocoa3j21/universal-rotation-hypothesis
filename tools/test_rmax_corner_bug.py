# -*- coding: utf-8 -*-
import os, sys
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

halo = pipe.TFHaloModel(M_msun=1e11, R_kpc=10.0, m_eV=1e-22)
dpix = 0.125
r_fov = 6.0
coords = np.arange(-r_fov, r_fov + dpix/2, dpix)
X_obs, Y_obs = np.meshgrid(coords, coords)
fwhm_kpc = 0.5
fwhm_pix = fwhm_kpc / dpix
sigma_v = 0.5
inc_deg = 50.0
r_min, r_max_disk = 0.2 * 10.0, 0.9 * 10.0

print("=== 検証A: 旧仕様 (r_max=None -> np.nanmax(r_disk) バグあり) vs 修正後 (r_max=r_max_disk) ===")
for mode in ["OLD (r_max=None: corner bug)", "FIXED (r_max=0.9*R_kpc)"]:
    print(f"\n[{mode}]")
    for a_los3, arm_method in [(0.0, "none"), (3.0, "raw"), (3.0, "remove_k3")]:
        vals = []
        for pa in [0.0, 45.0, 90.0]:
            _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc_deg)
            mask_disk = (r_disk <= r_max_disk)
            mask_search = (r_disk >= r_min) & (r_disk <= r_max_disk)
            v0 = halo.v_0(r_disk)
            v_los = pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc_deg, v_sys=100.0)
            arm_inj = a_los3 * np.cos(3.0 * phi_disk) if a_los3 > 0 else 0.0
            v_conv = pipe.convolve_beam_normalized(v_los + arm_inj, fwhm_pix=fwhm_pix, mask=mask_disk)

            rmax_arg = None if mode.startswith("OLD") else r_max_disk
            _, _, res_raw = pipe.fit_axisymmetric_disk_rings(
                r_disk, phi_disk, v_conv, inc_deg=inc_deg, v_sys=100.0, r_max=rmax_arg, n_rings=15
            )
            if arm_method == "remove_k3":
                res_proc, _ = pipe.remove_harmonic_k3_rings(r_disk, phi_disk, res_raw, r_max=rmax_arg, n_rings=15)
            else:
                res_proc = res_raw

            rms_nl = np.sqrt(np.nanmean(res_proc[mask_search]**2))
            peaks = pipe.find_local_peaks_2d(res_proc, threshold_snr=1.0, noise_sigma=sigma_v, min_distance_pix=3)
            peaks_s = [p for p in peaks if r_min <= r_disk[p["y_idx"], p["x_idx"]] <= r_max_disk]
            max_snr = max((p["snr"] for p in peaks_s), default=0.0)
            vals.append(f"PA={pa:4.1f}: RMS={rms_nl:.4f}, MaxSNR={max_snr:.2f} (r_max_used={np.nanmax(r_disk) if rmax_arg is None else rmax_arg:.2f})")
        print(f"  a3={a_los3}, {arm_method:9s} -> " + " | ".join(vals))
