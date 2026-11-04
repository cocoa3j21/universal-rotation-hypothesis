import os, sys
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/src')
import numpy as np
import gyro_gravity_theory.injection_pipeline as pipe

# Model 1
M_msun = 1e11
R_kpc = 20.0
m_eV = 1e-22
r_fov = 12.0
dpix = 0.08  # use fine pixel
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
d_v = 0.3 * R_kpc # 6.0 kpc
sigma_vortex_kpc = max(0.1, pipe.compute_tolerance_radius("model_1", 0.2, R_kpc))
r_tol_kpc = pipe.compute_tolerance_radius("model_1", fwhm, R_kpc)
r_tol_pix = r_tol_kpc / dpix

amp_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]

print("=== Model 1, FWHM=0.2 kpc, sigma_v=0.5 km/s Positive Control Scan across Inclinations ===")

for inc_deg in [30.0, 50.0, 70.0]:
    _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_deg, inc_deg=inc_deg)
    mask_disk = (r_disk_true <= R_kpc * 0.9)
    mask_search = (r_disk_true >= r_min) & (r_disk_true <= r_max)

    xo, yo = pipe.disk_to_obs(d_v, 0.0, pa_deg=pa_deg, inc_deg=inc_deg)
    true_pix = [(center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix)))]

    dist_sq_kpc = (r_disk_true * np.cos(phi_disk_true) - d_v)**2 + (r_disk_true * np.sin(phi_disk_true))**2

    # Theoretical A50
    # Beam attenuation
    v_raw = np.exp(-dist_sq_kpc / (2.0 * sigma_vortex_kpc**2))
    v_conv = pipe.convolve_beam_normalized(v_raw, fwhm_pix=fwhm_pix)
    eta_beam = float(np.nanmax(v_conv) / np.max(v_raw))
    eta_fit = 1.0
    snr_th_ref = 4.0
    a50_theory = pipe.predict_a50(snr_th_ref, sigma_v, eta_beam, eta_fit, inc_deg, phi_disk_rad=0.0)

    # Let's scan injection amplitudes
    recalls = []
    n_trials = 50
    for a_inj in amp_grid:
        dv_map = a_inj * np.exp(-dist_sq_kpc / (2.0 * sigma_vortex_kpc**2))
        v_los_unconv = dv_map * np.cos(phi_disk_true) * np.sin(np.radians(inc_deg))
        v_conv_signal = pipe.convolve_beam_normalized(v_los_unconv, fwhm_pix=fwhm_pix, mask=mask_disk)

        match_count = 0
        for trial in range(n_trials):
            rng = np.random.RandomState(int(inc_deg * 1000 + a_inj * 100 + trial))
            noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
            v_obs = v_conv_signal + noise

            raw_peaks = pipe.find_local_peaks_2d(v_obs, threshold_snr=snr_th_ref, noise_sigma=sigma_v, min_distance_pix=3)
            peaks = [p for p in raw_peaks if mask_search[p["y_idx"], p["x_idx"]]]
            matches = pipe.evaluate_position_matching(peaks, true_pix, r_tol_pix=r_tol_pix)
            if any(matches):
                match_count += 1
        rec = match_count / n_trials
        recalls.append((a_inj, rec))

    # Interpolate empirical A50 (where recall crosses 0.50)
    amps = [r[0] for r in recalls]
    recs = [r[1] for r in recalls]
    a50_emp = np.interp(0.50, recs, amps) if (min(recs) < 0.50 < max(recs)) else np.nan

    print(f"\nInc = {inc_deg} deg (sin i = {np.sin(np.radians(inc_deg)):.4f}):")
    print(f"  eta_beam = {eta_beam:.4f}, Theoretical A50 (SNR=4.0) = {a50_theory:.3f} km/s")
    print(f"  Empirical scan (amp -> recall): {[(round(a,1), round(r,2)) for a, r in recalls]}")
    print(f"  Empirical A50 = {a50_emp:.3f} km/s")
    print(f"  Ratio Emp / Theory = {a50_emp / a50_theory:.3f}")
