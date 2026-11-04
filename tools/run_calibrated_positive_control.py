import os, sys
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/src')
import numpy as np
import gyro_gravity_theory.injection_pipeline as pipe

# Test Model 3 across calibrated thresholds
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
sigma_vortex = max(0.1, pipe.compute_tolerance_radius("model_3", fwhm, R_kpc))
r_tol_kpc = pipe.compute_tolerance_radius("model_3", fwhm, R_kpc)
r_tol_pix = r_tol_kpc / dpix

amp_grid = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0]

cases = [
    {"inc": 30.0, "snr_th_calib": 9.2679, "snr_th_ref": 4.0},
    {"inc": 50.0, "snr_th_calib": 18.0901, "snr_th_ref": 4.0},
    {"inc": 70.0, "snr_th_calib": 42.5000, "snr_th_ref": 4.0}
]

print("=== Positive Control with Calibrated Thresholds vs Reference (SNR=4.0) ===")

for c in cases:
    inc_deg = c["inc"]
    snr_calib = c["snr_th_calib"]
    snr_ref = c["snr_th_ref"]

    _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_deg, inc_deg=inc_deg)
    mask_disk = (r_disk_true <= R_kpc * 0.9)
    mask_search = (r_disk_true >= r_min) & (r_disk_true <= r_max)

    xo, yo = pipe.disk_to_obs(d_v, 0.0, pa_deg=pa_deg, inc_deg=inc_deg)
    true_pix = [(center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix)))]
    dist_sq_kpc = (r_disk_true * np.cos(phi_disk_true) - d_v)**2 + (r_disk_true * np.sin(phi_disk_true))**2

    # Beam attenuation
    v_raw = np.exp(-dist_sq_kpc / (2.0 * sigma_vortex**2))
    v_conv = pipe.convolve_beam_normalized(v_raw, fwhm_pix=fwhm_pix)
    eta_beam = float(np.nanmax(v_conv) / np.max(v_raw))

    a50_pred_ref = pipe.predict_a50(snr_ref, sigma_v, eta_beam, 1.0, inc_deg)
    a50_pred_calib = pipe.predict_a50(snr_calib, sigma_v, eta_beam, 1.0, inc_deg)

    # Scan for both thresholds
    for snr_val, label, pred_val in [(snr_ref, "Reference (SNR=4.0)", a50_pred_ref), (snr_calib, f"Calibrated (SNR={snr_calib:.2f})", a50_pred_calib)]:
        recalls = []
        n_trials = 40
        for a_inj in amp_grid:
            dv_map = a_inj * np.exp(-dist_sq_kpc / (2.0 * sigma_vortex**2))
            v_los = dv_map * np.cos(phi_disk_true) * np.sin(np.radians(inc_deg))
            v_sig = pipe.convolve_beam_normalized(v_los, fwhm_pix=fwhm_pix, mask=mask_disk)

            matches = 0
            for t in range(n_trials):
                rng = np.random.RandomState(int(inc_deg * 1000 + snr_val * 10 + a_inj + t))
                noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
                v_obs = v_sig + noise

                peaks = pipe.find_local_peaks_2d(v_obs, threshold_snr=snr_val, noise_sigma=sigma_v, min_distance_pix=3)
                pks_search = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
                matched = pipe.evaluate_position_matching(pks_search, true_pix, r_tol_pix=r_tol_pix)
                if any(matched):
                    matches += 1
            recalls.append((a_inj, matches / n_trials))

        amps = [r[0] for r in recalls]
        recs = [r[1] for r in recalls]
        a50_emp = np.interp(0.50, recs, amps) if (min(recs) < 0.50 < max(recs)) else (">60" if max(recs) < 0.50 else "<5")
        print(f"Inc = {inc_deg} deg | {label}:")
        print(f"  Predicted A50 = {pred_val:.3f} km/s, Empirical A50 = {a50_emp}")
        print(f"  Scan: {[(a, round(r, 2)) for a, r in recalls[:6]]}")
