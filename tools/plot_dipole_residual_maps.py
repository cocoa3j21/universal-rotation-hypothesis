import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

# Model 3 parameters
M_msun = 1e10
R_kpc = 3.0
m_eV = 1e-22
r_fov = 2.2
dpix = 0.08
fwhm = 0.2
inc_deg = 30.0
pa_deg = 0.0

halo = pipe.TFHaloModel(M_msun=M_msun, R_kpc=R_kpc, m_eV=m_eV)
coords = np.arange(-r_fov, r_fov + dpix/2, dpix)
center_idx = len(coords) // 2
X_obs, Y_obs = np.meshgrid(coords, coords)
fwhm_pix = fwhm / dpix

d_v = 0.3 * R_kpc # 0.9 kpc
vortex_pos_kpc = [(d_v, 0.0)]
xo, yo = pipe.disk_to_obs(d_v, 0.0, pa_deg=pa_deg, inc_deg=inc_deg)
true_pix = (center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix)))

_, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_deg, inc_deg=inc_deg)
mask_disk = (r_disk_true <= R_kpc * 0.9)

# 1. Unconvolved vortex perturbation
dv_map = halo.compute_vortex_velocity_perturbation_map(
    r_disk_true * np.cos(phi_disk_true), r_disk_true * np.sin(phi_disk_true), vortex_pos_kpc
)
v_los_unconv = dv_map * np.cos(phi_disk_true) * np.sin(np.radians(inc_deg))

# 2. Convolved vortex perturbation
v_los_conv = pipe.convolve_beam_normalized(v_los_unconv, fwhm_pix=fwhm_pix, mask=mask_disk)

r_tol_kpc = pipe.compute_tolerance_radius("model_3", fwhm, R_kpc)
r_tol_pix = r_tol_kpc / dpix

# Zoom-in window around vortex (patch of 31x31 pixels)
wx, wy = true_pix
half_w = 12
x_slice = slice(wx - half_w, wx + half_w + 1)
y_slice = slice(wy - half_w, wy + half_w + 1)

patch_unconv = v_los_unconv[y_slice, x_slice]
patch_conv = v_los_conv[y_slice, x_slice]

center_patch = (half_w, half_w)

# Find peaks within the immediate neighborhood around the vortex (radius 8 pixels)
Y_p, X_p = np.ogrid[:patch_unconv.shape[0], :patch_unconv.shape[1]]
dist_from_core = np.sqrt((X_p - center_patch[0])**2 + (Y_p - center_patch[1])**2)
local_mask = (dist_from_core <= 8.0)

patch_unconv_masked = np.where(local_mask, patch_unconv, 0.0)
patch_conv_masked = np.where(local_mask, patch_conv, 0.0)

p_pos_unconv = np.unravel_index(np.argmax(patch_unconv_masked), patch_unconv.shape)
p_neg_unconv = np.unravel_index(np.argmin(patch_unconv_masked), patch_unconv.shape)
p_pos_conv = np.unravel_index(np.argmax(patch_conv_masked), patch_conv.shape)
p_neg_conv = np.unravel_index(np.argmin(patch_conv_masked), patch_conv.shape)

center_patch = (half_w, half_w)
dist_pos_unconv = np.sqrt((p_pos_unconv[1] - center_patch[1])**2 + (p_pos_unconv[0] - center_patch[0])**2)
dist_pos_conv = np.sqrt((p_pos_conv[1] - center_patch[1])**2 + (p_pos_conv[0] - center_patch[0])**2)

print(f"True vortex center in patch: {center_patch}")
print(f"Unconvolved: max={np.max(patch_unconv):.2f} km/s at {p_pos_unconv}, dist={dist_pos_unconv:.2f} pix, min={np.min(patch_unconv):.2f} km/s at {p_neg_unconv}")
print(f"Convolved (FWHM=0.2kpc): max={np.max(patch_conv):.2f} km/s at {p_pos_conv}, dist={dist_pos_conv:.2f} pix, min={np.min(patch_conv):.2f} km/s at {p_neg_conv}")
print(f"Tolerance radius: {r_tol_pix:.2f} pix ({r_tol_kpc:.3f} kpc)")

# Plotting dark theme
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

vmax = max(np.max(np.abs(patch_unconv)), np.max(np.abs(patch_conv)))

im0 = axes[0].imshow(patch_unconv, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
axes[0].set_title(f"Model 3 Unconvolved Vortex Perturbation\n(Pure Dipole, Peak dist = {dist_pos_unconv:.2f} pix)", fontsize=12)
circle0 = plt.Circle(center_patch, r_tol_pix, color='yellow', fill=False, linestyle='--', linewidth=2, label=f'r_tol = {r_tol_pix:.2f} pix')
axes[0].add_patch(circle0)
axes[0].plot(center_patch[0], center_patch[1], 'yx', markersize=10, markeredgewidth=2, label='Vortex Core (True)')
axes[0].plot(p_pos_unconv[1], p_pos_unconv[0], 'go', markersize=8, label=f'Positive Peak ({dist_pos_unconv:.2f} pix)')
axes[0].plot(p_neg_unconv[1], p_neg_unconv[0], 'mo', markersize=8, label='Negative Peak')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_xlabel("Relative X [pix]")
axes[0].set_ylabel("Relative Y [pix]")
plt.colorbar(im0, ax=axes[0], label="v_los [km/s]")

im1 = axes[1].imshow(patch_conv, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
axes[1].set_title(f"Model 3 Beam-Convolved Perturbation (FWHM=0.2 kpc)\n(Smearing pushes peak outward to {dist_pos_conv:.2f} pix > r_tol)", fontsize=12)
circle1 = plt.Circle(center_patch, r_tol_pix, color='yellow', fill=False, linestyle='--', linewidth=2, label=f'r_tol = {r_tol_pix:.2f} pix')
axes[1].add_patch(circle1)
axes[1].plot(center_patch[0], center_patch[1], 'yx', markersize=10, markeredgewidth=2, label='Vortex Core (True)')
axes[1].plot(p_pos_conv[1], p_pos_conv[0], 'ro', markersize=8, label=f'Shifted Peak ({dist_pos_conv:.2f} pix > r_tol)')
axes[1].plot(p_neg_conv[1], p_neg_conv[0], 'mo', markersize=8, label='Negative Peak')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].set_xlabel("Relative X [pix]")
axes[1].set_ylabel("Relative Y [pix]")
plt.colorbar(im1, ax=axes[1], label="v_los [km/s]")

plt.tight_layout()
out_png = os.path.join(PROJECT_DIR, "output", "dipole_residual_structure.png")
plt.savefig(out_png, dpi=200)
plt.close()
print(f"Saved dipole visualization to {out_png}")
