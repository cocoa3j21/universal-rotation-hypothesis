import os, sys
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/src')
import numpy as np
import gyro_gravity_theory.injection_pipeline as pipe

M_msun = 1e11
R_kpc = 20.0
m_eV = 1e-22
r_fov = 12.0
dpix = 0.25
fwhm = 0.2
sigma_v = 0.5
inc_deg = 30.0
pa_deg = 0.0

halo = pipe.TFHaloModel(M_msun=M_msun, R_kpc=R_kpc, m_eV=m_eV)
coords = np.arange(-r_fov, r_fov + dpix/2, dpix)
center_idx = len(coords) // 2
X_obs, Y_obs = np.meshgrid(coords, coords)
fwhm_pix = fwhm / dpix
sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))

r_min = 0.2 * R_kpc
r_max = 0.9 * R_kpc

_, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_deg, inc_deg=inc_deg)
mask_disk = (r_disk_true <= R_kpc * 0.9)
mask_search = (r_disk_true >= r_min) & (r_disk_true <= r_max)

v0_true = halo.v_0(r_disk_true)
v_los_unpert = pipe.project_vlos(v0_true, 0.0, phi_disk_true, inc_deg=inc_deg, v_sys=100.0)
v_conv_signal = pipe.convolve_beam_normalized(v_los_unpert, fwhm_pix=fwhm_pix, mask=mask_disk)

n_trials = 200

# Experiment 1: Pure Correlated Noise only (No galaxy signal, no subtraction)
snr_pure_noise = []
for t in range(n_trials):
    rng = np.random.RandomState(t)
    noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
    peaks = pipe.find_local_peaks_2d(noise, threshold_snr=1.0, noise_sigma=sigma_v, min_distance_pix=3)
    pks_s = [p["snr"] for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
    snr_pure_noise.append(max(pks_s) if len(pks_s) > 0 else 0.0)

# Experiment 2: Current pipeline (Simple ring fit subtraction on v_conv_signal + noise)
snr_current_pipeline = []
res_noiseless_current = []
for t in range(n_trials):
    rng = np.random.RandomState(t)
    noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
    v_obs = v_conv_signal + noise
    _, _, res = pipe.fit_axisymmetric_disk_rings(r_disk_true, phi_disk_true, v_obs, inc_deg=inc_deg, v_sys=100.0, n_rings=15)
    peaks = pipe.find_local_peaks_2d(res, threshold_snr=1.0, noise_sigma=sigma_v, min_distance_pix=3)
    pks_s = [p["snr"] for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
    snr_current_pipeline.append(max(pks_s) if len(pks_s) > 0 else 0.0)

# Check noiseless residual in current pipeline
_, _, res_nl = pipe.fit_axisymmetric_disk_rings(r_disk_true, phi_disk_true, v_conv_signal, inc_deg=inc_deg, v_sys=100.0, n_rings=15)
max_nl_current = np.nanmax(np.abs(res_nl[mask_search]))
rms_nl_current = np.sqrt(np.nanmean(res_nl[mask_search]**2))

# Experiment 3: Forward Convolved Model Subtraction
# True unperturbed beam-convolved model is subtracted (ideal forward model)
snr_forward_model = []
for t in range(n_trials):
    rng = np.random.RandomState(t)
    noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
    v_obs = v_conv_signal + noise
    # Forward model subtraction: subtract v_conv_signal
    res_fwd = v_obs - v_conv_signal
    peaks = pipe.find_local_peaks_2d(res_fwd, threshold_snr=1.0, noise_sigma=sigma_v, min_distance_pix=3)
    pks_s = [p["snr"] for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
    snr_forward_model.append(max(pks_s) if len(pks_s) > 0 else 0.0)

print(f"=== Threshold Investigation: Model 1, inc=30 deg, fwhm=0.2 kpc, sigma_v=0.5 km/s (N={n_trials}) ===")
print(f"1. Pure Correlated Noise Only:")
print(f"   Max SNR 95th percentile: {np.percentile(snr_pure_noise, 95.0):.2f}")
print(f"   Median: {np.median(snr_pure_noise):.2f}, Max: {np.max(snr_pure_noise):.2f}")

print(f"\n2. Current Pipeline (Simple Ring Subtraction without beam forward model):")
print(f"   Noiseless residual in search area: RMS = {rms_nl_current:.3f} km/s, Max = {max_nl_current:.3f} km/s")
print(f"   Max SNR 95th percentile (Calibrated SNR_th): {np.percentile(snr_current_pipeline, 95.0):.2f}")
print(f"   Median: {np.median(snr_current_pipeline):.2f}, Max: {np.max(snr_current_pipeline):.2f}")

print(f"\n3. Forward Convolved Model Subtraction (Beam forward model included):")
print(f"   Max SNR 95th percentile: {np.percentile(snr_forward_model, 95.0):.2f}")
print(f"   Median: {np.median(snr_forward_model):.2f}, Max: {np.max(snr_forward_model):.2f}")
