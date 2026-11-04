import os, sys
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/tools')
import run_model2_h0_calibration as m2cal
import numpy as np

strata = m2cal.generate_strata()
print(f"Total strata: {len(strata)}")

# Let's inspect a sample of strata across PA
# In run_model2_h0_calibration.py, validation was done across cell_contexts (which are PA=0, 45, 90)
# Let's write a function to evaluate PA-specific FPR for representative strata
sample_strata = [s for s in strata if s["inc_deg"] == 50.0 and s["fwhm_kpc"] == 0.5 and s["sigma_v"] == 0.5 and not s["geom_err"]]
print(f"Sample strata for inc=50, fwhm=0.5, sig=0.5, geom=False: {len(sample_strata)}")

# Evaluate PA-specific FPR on these strata
pa_fprs = {0.0: [], 45.0: [], 90.0: []}

# Let's run a test on arm_levels (none, raw a3=1, remove_k3 a3=1, arm_fit a3=1)
for s in sample_strata:
    model_info = m2cal.MODELS[s["model_id"]]
    halo = m2cal.pipe.TFHaloModel(M_msun=model_info["M_msun"], R_kpc=model_info["R_kpc"], m_eV=model_info["m_eV"])
    dpix = model_info["dpix"]
    coords = np.arange(-model_info["r_fov"], model_info["r_fov"] + dpix/2, dpix)
    center_idx = len(coords) // 2
    X_obs, Y_obs = np.meshgrid(coords, coords)
    fwhm_pix = s["fwhm_kpc"] / dpix
    sigma_v = s["sigma_v"]
    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    r_tol_pix = m2cal.pipe.compute_tolerance_radius(s["model_id"], s["fwhm_kpc"], model_info["R_kpc"]) / dpix

    # sample_pix_list
    d_vortex = 0.3 * model_info["R_kpc"]
    sample_pix_list = []
    for s_idx in range(8):
        phi_s = s_idx * (2.0 * np.pi / 8)
        px = center_idx + int(round(d_vortex * np.cos(phi_s) / dpix))
        py = center_idx + int(round(d_vortex * np.sin(phi_s) / dpix))
        sample_pix_list.append((px, py))

    # Evaluate each cell in stratum
    snr_th = 6.0 # representative threshold
    for cell in s["cells"]:
        pa = cell["pa_deg"]
        inc = cell["inc_deg"]
        _, _, r_disk, phi_disk = m2cal.pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc)
        mask_disk = (r_disk <= model_info["R_kpc"] * 0.9)
        mask_search = (r_disk >= 0.2 * model_info["R_kpc"]) & (r_disk <= 0.9 * model_info["R_kpc"])

        v0 = halo.v_0(r_disk)
        v_los = m2cal.pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc, v_sys=100.0)
        arm_sig = cell["a_los3"] * np.cos(3.0 * phi_disk) if cell["a_los3"] > 0 else 0.0
        v_conv = m2cal.pipe.convolve_beam_normalized(v_los + arm_sig, fwhm_pix=fwhm_pix, mask=mask_disk)

        hits = 0
        n_trials = 100
        for t in range(n_trials):
            rng = np.random.RandomState(int(pa * 100 + t))
            noise = m2cal.pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
            v_obs = v_conv + noise
            _, _, res_raw = m2cal.pipe.fit_axisymmetric_disk_rings(r_disk, phi_disk, v_obs, inc_deg=inc, v_sys=100.0, n_rings=15)
            meth = cell["arm_method"]
            if meth == "none" or meth == "raw": res = res_raw
            elif meth == "remove_k3": res, _ = m2cal.pipe.remove_harmonic_k3_rings(r_disk, phi_disk, res_raw, n_rings=15)
            elif meth == "arm_fit": res, _, _, _ = m2cal.pipe.fit_spiral_arm_direct_c(r_disk, phi_disk, res_raw, inc_deg=inc)
            else: res = res_raw

            peaks = m2cal.pipe.find_local_peaks_2d(res, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
            pks = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]
            hit = False
            for sx, sy in sample_pix_list:
                for pk in pks:
                    if (pk["x_idx"] - sx)**2 + (pk["y_idx"] - sy)**2 <= r_tol_pix**2:
                        hit = True; break
                if hit: break
            if hit: hits += 1
        pa_fprs[pa].append(hits / n_trials)

print("=== Stage 3 FPR Breakdown by PA (Sample strata, 100 trials/stratum) ===")
for pa, vals in pa_fprs.items():
    print(f"PA = {pa:4.1f} deg: mean FPR = {np.mean(vals):.4f}, median = {np.median(vals):.4f}, max = {np.max(vals):.4f}")
