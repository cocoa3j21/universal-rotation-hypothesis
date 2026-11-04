import os, sys
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/src')
import numpy as np
from scipy.signal import correlate2d
import gyro_gravity_theory.injection_pipeline as pipe

M_msun = 1e10
R_kpc = 3.0
m_eV = 1e-22
r_fov = 2.2
dpix = 0.08
fwhm = 0.2
sigma_v = 0.5
pa_deg = 0.0

halo = pipe.TFHaloModel(M_msun=M_msun, R_kpc=R_kpc, m_eV=m_eV)
coords = np.arange(-r_fov, r_fov + dpix/2, dpix)
center_idx = len(coords) // 2
X_obs, Y_obs = np.meshgrid(coords, coords)
fwhm_pix = fwhm / dpix
sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))

r_min = 0.2 * R_kpc
r_max = 0.9 * R_kpc
d_v = 0.3 * R_kpc
r_tol_kpc = pipe.compute_tolerance_radius("model_3", fwhm, R_kpc)
r_tol_pix = r_tol_kpc / dpix

inc_deg = 30.0
snr_th = 9.2679

_, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_deg, inc_deg=inc_deg)
mask_disk = (r_disk_true <= R_kpc * 0.9)
mask_search = (r_disk_true >= r_min) & (r_disk_true <= r_max)

v0_true = halo.v_0(r_disk_true)
v_los_unpert = pipe.project_vlos(v0_true, 0.0, phi_disk_true, inc_deg=inc_deg, v_sys=100.0)

vortex_pos_kpc = [(d_v, 0.0)]
xo, yo = pipe.disk_to_obs(d_v, 0.0, pa_deg=pa_deg, inc_deg=inc_deg)
true_pix = (center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix)))

# Inject vortex
dv_map = halo.compute_vortex_velocity_perturbation_map(
    r_disk_true * np.cos(phi_disk_true), r_disk_true * np.sin(phi_disk_true), vortex_pos_kpc
)
v_los_inj = dv_map * np.cos(phi_disk_true) * np.sin(np.radians(inc_deg))
v_conv_signal = pipe.convolve_beam_normalized(v_los_unpert + v_los_inj, fwhm_pix=fwhm_pix, mask=mask_disk)

# Create Dipole Template (centered at 0)
# A small patch of the vortex perturbation projected and beam-convolved
patch_size = 21
patch_coords = (np.arange(patch_size) - patch_size // 2) * dpix
P_X, P_Y = np.meshgrid(patch_coords, patch_coords)
# On disk at origin
dv_template = halo.compute_vortex_velocity_perturbation_map(P_X, P_Y, [(0.0, 0.0)])
# Projected with inc_deg
P_r = np.sqrt(P_X**2 + P_Y**2)
P_phi = np.arctan2(P_Y, P_X)
template_raw = dv_template * np.cos(P_phi) * np.sin(np.radians(inc_deg))
template_conv = pipe.convolve_beam_normalized(template_raw, fwhm_pix=fwhm_pix)
# Normalize template
template_norm = template_conv / np.sqrt(np.sum(template_conv**2))

print(f"True vortex pix: {true_pix}")
print(f"Tolerance radius: {r_tol_pix:.2f} pix")

# Run 50 trials comparing methods
n_trials = 50
matches_monopole = 0
matches_dipole = 0
matches_monopole_wide_tol = 0  # tol = 2.5 pix

for trial in range(n_trials):
    rng = np.random.RandomState(trial)
    noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
    v_obs = v_conv_signal + noise

    _, _, res_raw = pipe.fit_axisymmetric_disk_rings(
        r_disk_true, phi_disk_true, v_obs, inc_deg=inc_deg, v_sys=100.0, n_rings=15
    )

    # Method 1: Monopole peak detector (current pipeline)
    raw_peaks = pipe.find_local_peaks_2d(res_raw, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
    peaks = [p for p in raw_peaks if mask_search[p["y_idx"], p["x_idx"]]]
    matched_m = pipe.evaluate_position_matching(peaks, [true_pix], r_tol_pix=r_tol_pix)
    if any(matched_m):
        matches_monopole += 1

    matched_m_wide = pipe.evaluate_position_matching(peaks, [true_pix], r_tol_pix=2.5)
    if any(matched_m_wide):
        matches_monopole_wide_tol += 1

    # Method 2: Dipole Matched Filter
    # Convolve res_raw with template_norm
    res_clean = np.where(np.isnan(res_raw), 0.0, res_raw)
    mf_map = correlate2d(res_clean, template_norm, mode='same')
    # Find peak of MF in search mask
    mf_masked = np.where(mask_search, np.abs(mf_map), -np.inf)
    max_idx = np.unravel_index(np.argmax(mf_masked), mf_masked.shape)
    dist_dipole = np.sqrt((max_idx[1] - true_pix[0])**2 + (max_idx[0] - true_pix[1])**2)
    if dist_dipole <= r_tol_pix:
        matches_dipole += 1

print(f"\n--- Post-hoc Analysis on Model 3 (inc=30 deg, sigma_v=0.5 km/s, N=50 trials) ---")
print(f"1. Monopole detector (Pipeline standard, r_tol=1.42 pix): Recall = {matches_monopole / n_trials:.3f}")
print(f"2. Monopole detector with widened tolerance (r_tol=2.5 pix): Recall = {matches_monopole_wide_tol / n_trials:.3f}")
print(f"3. Dipole Matched Filter (Post-hoc template, r_tol=1.42 pix): Recall = {matches_dipole / n_trials:.3f}")
