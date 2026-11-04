# -*- coding: utf-8 -*-
import os, sys
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

MODELS = {
    "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "a_phys": 1.12, "xi_c": 0.092442, "r_fov": 18.4},
    "model_2": {"M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "a_phys": 4.89, "xi_c": 0.046221, "r_fov": 9.4},
    "model_3": {"M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "a_phys": 30.80, "xi_c": 0.113218, "r_fov": 2.8}
}

dpix = 0.08
fwhm = 0.2
inc_deg = 50.0

for mid, info in MODELS.items():
    halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])
    d_v = 0.3 * info["R_kpc"]
    v0 = halo.v_0(d_v)
    m_def = halo.compute_vortex_mass_deficit(d_v)
    
    # 2D grid
    coords = np.arange(-info["r_fov"], info["r_fov"] + dpix/2, dpix)
    X_obs, Y_obs = np.meshgrid(coords, coords)
    _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc_deg)
    mask_disk = (r_disk <= 0.9 * info["R_kpc"])
    
    # Unperturbed & Vortex
    v0_grid = halo.v_0(r_disk)
    v_los_unp = pipe.project_vlos(v0_grid, 0.0, phi_disk, inc_deg=inc_deg, v_sys=100.0)
    dv_map = halo.compute_vortex_velocity_perturbation_map(
        r_disk * np.cos(phi_disk), r_disk * np.sin(phi_disk), [(d_v, 0.0)]
    )
    # Physical amplitude in grid
    max_dv_map = np.max(np.abs(dv_map))
    
    # Line of sight unconvolved vortex signal: v_los_vortex = dv_map * cos(phi) * sin(inc)
    v_los_vortex = dv_map * np.cos(phi_disk) * np.sin(np.radians(inc_deg))
    max_unconv_vlos = np.max(np.abs(v_los_vortex))
    
    # Convolved
    fwhm_pix = fwhm / dpix
    v_conv_tot = pipe.convolve_beam_normalized(v_los_unp + v_los_vortex, fwhm_pix=fwhm_pix, mask=mask_disk)
    v_conv_unp = pipe.convolve_beam_normalized(v_los_unp, fwhm_pix=fwhm_pix, mask=mask_disk)
    s_map = np.nan_to_num(v_conv_tot - v_conv_unp, nan=0.0)
    max_conv_s = np.max(np.abs(s_map))
    
    ratio_to_phys = max_conv_s / (info["a_phys"] * np.sin(np.radians(inc_deg)))
    ratio_to_unconv = max_conv_s / max_unconv_vlos
    
    print(f"\n=== {mid} (R={info['R_kpc']} kpc, xi_c={info['xi_c']:.4f} kpc, d_v={d_v:.2f} kpc) ===")
    print(f"  v0(0.3R) = {v0:.2f} km/s, M_def = {m_def:.3e} kg")
    print(f"  Nominal a_phys = {info['a_phys']:.2f} km/s, Actual max |dv_map| = {max_dv_map:.4f} km/s")
    print(f"  Unconvolved max |v_los,vortex| = {max_unconv_vlos:.4f} km/s (cf. a_phys*sin i = {info['a_phys']*np.sin(np.radians(inc_deg)):.4f})")
    print(f"  Convolved max |S| (FWHM=0.2) = {max_conv_s:.4f} km/s")
    print(f"  Ratio max|S| / (a_phys * sin i) = {ratio_to_phys:.4f}")
    print(f"  Ratio max|S| / max_unconv = {ratio_to_unconv:.4f}")
