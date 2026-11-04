import os, sys
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/src')
import numpy as np
import gyro_gravity_theory.injection_pipeline as pipe

# Model 3, inc=50, fwhm=0.5, geom=True, pa=45.0
M_msun = 1e10
R_kpc = 3.0
m_eV = 1e-22
r_fov = 2.2
dpix = 0.08
fwhm = 0.5
sigma_v = 0.5
inc_true = 50.0
pa_true = 45.0
v_sys_true = 100.0

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

# Geometric errors
delta_inc, delta_pa, delta_sys = 2.0, 2.0, 2.0
inc_obs = inc_true + delta_inc
pa_obs = pa_true + delta_pa
v_sys_obs = v_sys_true + delta_sys

_, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_true, inc_deg=inc_true)
_, _, r_disk_obs, phi_disk_obs = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_obs, inc_deg=inc_obs)
mask_disk = (r_disk_true <= R_kpc * 0.9)
mask_search = (r_disk_obs >= r_min) & (r_disk_obs <= r_max)

v0_true = halo.v_0(r_disk_true)
v_los_unpert = pipe.project_vlos(v0_true, 0.0, phi_disk_true, inc_deg=inc_true, v_sys=v_sys_true)
v_conv_signal_h0 = pipe.convolve_beam_normalized(v_los_unpert, fwhm_pix=fwhm_pix, mask=mask_disk)

# Injection location for Nv=1
xo, yo = pipe.disk_to_obs(d_v, 0.0, pa_deg=pa_true, inc_deg=inc_true)
true_pix = (center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix)))

# Let's check noiseless residual map of H0 (geometric distortion only)
_, _, res_h0_noiseless = pipe.fit_axisymmetric_disk_rings(
    r_disk_obs, phi_disk_obs, v_conv_signal_h0, inc_deg=inc_obs, v_sys=v_sys_obs, n_rings=15
)

# Inspect residual value at true_pix
val_at_true_pix = res_h0_noiseless[true_pix[1], true_pix[0]]
peak_h0_idx = np.unravel_index(np.argmax(np.where(mask_search, np.abs(res_h0_noiseless), 0.0)), res_h0_noiseless.shape)
dist_geom_to_inj = np.sqrt((peak_h0_idx[1] - true_pix[0])**2 + (peak_h0_idx[0] - true_pix[1])**2)

print(f"=== FN Investigation: Model 3, inc=50, fwhm=0.5, geom=True, PA=45 ===")
print(f"True injection pix: {true_pix}")
print(f"Noiseless H0 residual at injection pix: {val_at_true_pix:.2f} km/s (SNR = {val_at_true_pix / sigma_v:.2f})")
print(f"Noiseless H0 geometric peak location (x, y): ({peak_h0_idx[1]}, {peak_h0_idx[0]}), max abs = {res_h0_noiseless[peak_h0_idx]:.2f} km/s")
print(f"Distance between geometric residual peak and vortex injection position: {dist_geom_to_inj:.2f} pix (tol = {r_tol_pix:.2f} pix)")

# Measure False Recovery Rate at the injection position in H0 (100 trials)
snr_th = 6.26 # stratum threshold for this stratum
h0_hits = 0
n_trials = 100
for t in range(n_trials):
    rng = np.random.RandomState(t)
    noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
    v_obs = v_conv_signal_h0 + noise
    _, _, res = pipe.fit_axisymmetric_disk_rings(r_disk_obs, phi_disk_obs, v_obs, inc_deg=inc_obs, v_sys=v_sys_obs, n_rings=15)
    peaks = pipe.find_local_peaks_2d(res, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
    pks_search = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
    matched = pipe.evaluate_position_matching(pks_search, [true_pix], r_tol_pix=r_tol_pix)
    if any(matched):
        h0_hits += 1

print(f"H0 False Positive Recovery at injection position (Pure geometric artifact): {h0_hits} / {n_trials} ({h0_hits / n_trials * 100:.1f}%)")
