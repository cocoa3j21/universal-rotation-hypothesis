import os, sys
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/src')
import numpy as np
import gyro_gravity_theory.injection_pipeline as pipe

# Model 1, Model 2, Model 3 parameters
MODELS = {
    "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 12.0, "dpix": 0.08, "a_inj": 1.12},
    "model_2": {"M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 6.0, "dpix": 0.08, "a_inj": 4.89},
    "model_3": {"M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.2, "dpix": 0.08, "a_inj": 30.80}
}

print("=== Pipeline v2: Forward Convolved Model Subtraction Positive Control Scan ===")

results_v2_pos = []

for model_id in ["model_1", "model_2", "model_3"]:
    info = MODELS[model_id]
    halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])
    dpix = info["dpix"]
    coords = np.arange(-info["r_fov"], info["r_fov"] + dpix/2, dpix)
    center_idx = len(coords) // 2
    X_obs, Y_obs = np.meshgrid(coords, coords)

    fwhm = 0.2
    fwhm_pix = fwhm / dpix
    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_v = 0.5
    inc_deg = 50.0
    pa_deg = 0.0

    r_min = 0.2 * info["R_kpc"]
    r_max = 0.9 * info["R_kpc"]
    d_v = 0.3 * info["R_kpc"]
    # v2 tolerance radius: covers beam smearing of dipole lobe (2.5 pix = 0.20 kpc)
    r_tol_pix = 2.5

    _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_deg, inc_deg=inc_deg)
    mask_disk = (r_disk_true <= info["R_kpc"] * 0.9)
    mask_search = (r_disk_true >= r_min) & (r_disk_true <= r_max)

    v0_true = halo.v_0(r_disk_true)
    v_los_unpert = pipe.project_vlos(v0_true, 0.0, phi_disk_true, inc_deg=inc_deg, v_sys=100.0)
    # Forward convolved model (unperturbed)
    v_fwd_model = pipe.convolve_beam_normalized(v_los_unpert, fwhm_pix=fwhm_pix, mask=mask_disk)

    xo, yo = pipe.disk_to_obs(d_v, 0.0, pa_deg=pa_deg, inc_deg=inc_deg)
    true_pix = [(center_idx + int(round(xo / dpix)), center_idx + int(round(yo / dpix)))]
    dist_sq_kpc = (r_disk_true * np.cos(phi_disk_true) - d_v)**2 + (r_disk_true * np.sin(phi_disk_true))**2

    # In v2, forward model subtraction eliminates beam smearing residual:
    # SNR threshold under pure noise is 4.47 (use 4.50)
    snr_th_v2 = 4.50

    # Calculate theoretical A50 under v2
    v_raw = np.exp(-dist_sq_kpc / (2.0 * (0.1)**2))
    v_conv = pipe.convolve_beam_normalized(v_raw, fwhm_pix=fwhm_pix)
    eta_beam = float(np.nanmax(v_conv) / np.max(v_raw))
    a50_pred_v2 = pipe.predict_a50(snr_th_v2, sigma_v, eta_beam, 1.0, inc_deg)

    # Empirical scan
    amp_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0]
    recalls = []
    n_trials = 50
    for a_inj in amp_grid:
        dv_map = a_inj * np.exp(-dist_sq_kpc / (2.0 * (0.1)**2))
        v_los_inj = dv_map * np.cos(phi_disk_true) * np.sin(np.radians(inc_deg))
        v_total_conv = pipe.convolve_beam_normalized(v_los_unpert + v_los_inj, fwhm_pix=fwhm_pix, mask=mask_disk)

        matches = 0
        for t in range(n_trials):
            rng = np.random.RandomState(int(a_inj * 100 + t))
            noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
            v_obs = v_total_conv + noise

            # Forward model subtraction in v2:
            res_v2 = v_obs - v_fwd_model

            peaks = pipe.find_local_peaks_2d(res_v2, threshold_snr=snr_th_v2, noise_sigma=sigma_v, min_distance_pix=3)
            pks_search = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
            matched = pipe.evaluate_position_matching(pks_search, true_pix, r_tol_pix=r_tol_pix)
            if any(matched):
                matches += 1
        recalls.append((a_inj, matches / n_trials))

    amps = [r[0] for r in recalls]
    recs = [r[1] for r in recalls]
    a50_emp = np.interp(0.50, recs, amps) if (min(recs) < 0.50 < max(recs)) else np.nan
    ratio_phys = info["a_inj"] / a50_emp

    print(f"\nModel: {model_id} (Physical Amp = {info['a_inj']} km/s):")
    print(f"  Theoretical A50 (SNR=4.5) = {a50_pred_v2:.3f} km/s")
    print(f"  Empirical A50 (v2, r_tol=2.5pix) = {a50_emp:.3f} km/s")
    print(f"  Scan: {[(a, round(r, 2)) for a, r in recalls]}")
    print(f"  Physical Amp / Empirical A50 = {ratio_phys:.3f}")
    if ratio_phys >= 1.5:
        print("  Status: RECOVERABLE (A_phys >= 1.5 A50)")
    elif ratio_phys <= 0.67:
        print("  Status: UNDETECTABLE (A_phys <= 0.67 A50)")
    else:
        print("  Status: BOUNDARY")
