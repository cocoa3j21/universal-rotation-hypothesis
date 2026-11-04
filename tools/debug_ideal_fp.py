import os, sys
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/src')
import numpy as np
import gyro_gravity_theory.injection_pipeline as pipe

# Model 3 parameters
M_msun = 1e10
R_kpc = 3.0
m_eV = 1e-22
r_fov = 2.2
dpix = 0.08
fwhm = 0.2
sigma_v = 0.5
inc_deg = 30.0
pa_deg = 0.0
v_sys_true = 100.0

halo = pipe.TFHaloModel(M_msun=M_msun, R_kpc=R_kpc, m_eV=m_eV)
coords = np.arange(-r_fov, r_fov + dpix/2, dpix)
X_obs, Y_obs = np.meshgrid(coords, coords)
center_idx = len(coords) // 2

r_min = 0.2 * R_kpc
r_max = 0.9 * R_kpc
d_vortex = 0.3 * R_kpc # 0.9 kpc
sigma_vortex = max(0.1, pipe.compute_tolerance_radius("model_3", fwhm, R_kpc))
r_tol_kpc = pipe.compute_tolerance_radius("model_3", fwhm, R_kpc)
r_tol_pix = r_tol_kpc / dpix

_, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_deg, inc_deg=inc_deg)
mask_disk = (r_disk_true <= R_kpc * 0.9)
mask_search = (r_disk_true >= r_min) & (r_disk_true <= r_max)

fwhm_pix = fwhm / dpix
sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))

# True vortex position
# In pipeline: vortex placed at d_vortex on disk
# For Nv=1, what phi is used?
phi_vortex_disk = 0.0
xv_kpc = d_vortex * np.cos(phi_vortex_disk)
yv_kpc = d_vortex * np.sin(phi_vortex_disk)
xo, yo = pipe.disk_to_obs(xv_kpc, yv_kpc, pa_deg=pa_deg, inc_deg=inc_deg)
true_pix = (center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix)))

# Let's inspect the injection logic in run_full_injection_recovery_batch.py
print(f"True vortex pix: {true_pix}, r_tol_pix: {r_tol_pix:.2f}")
