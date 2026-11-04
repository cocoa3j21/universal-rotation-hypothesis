import os, sys
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/src')
import numpy as np
import gyro_gravity_theory.injection_pipeline as pipe

# Model 2 parameters
M_msun = 1e11
R_kpc = 10.0
m_eV = 1e-22
r_fov = 6.0
dpix = 0.08
fwhm = 0.5
sigma_v = 0.5
inc_deg = 50.0

halo = pipe.TFHaloModel(M_msun=M_msun, R_kpc=R_kpc, m_eV=m_eV)
coords = np.arange(-r_fov, r_fov + dpix/2, dpix)
center_idx = len(coords) // 2
X_obs, Y_obs = np.meshgrid(coords, coords)
fwhm_pix = fwhm / dpix
sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))

# Physical tolerance radius: xi_c + sigma_beam
xi_c_kpc = 0.04622  # Model 2
sigma_beam_kpc = sigma_beam_pix * dpix
r_tol_kpc = xi_c_kpc + sigma_beam_kpc
r_tol_pix = r_tol_kpc / dpix

d_vortex = 0.3 * R_kpc
n_samples = 8
n_trials = 200
snr_th = 4.50

print(f"=== Rotational Symmetry Test across PAs (N={n_trials} trials/PA) ===")
print(f"xi_c = {xi_c_kpc:.4f} kpc, sigma_beam = {sigma_beam_kpc:.4f} kpc, r_tol = {r_tol_kpc:.4f} kpc ({r_tol_pix:.2f} pix)")

pa_list = [0.0, 45.0, 90.0, 135.0, 180.0, 270.0]
results = {}

for pa in pa_list:
    _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc_deg)
    mask_disk = (r_disk <= R_kpc * 0.9)
    mask_search = (r_disk >= 0.2 * R_kpc) & (r_disk <= 0.9 * R_kpc)

    # Correct projection of sample points: disk_to_obs!
    sample_pix_list = []
    for s_idx in range(n_samples):
        phi_s = s_idx * (2.0 * np.pi / n_samples)
        xs_kpc = d_vortex * np.cos(phi_s)
        ys_kpc = d_vortex * np.sin(phi_s)
        xo, yo = pipe.disk_to_obs(xs_kpc, ys_kpc, pa_deg=pa, inc_deg=inc_deg)
        px = center_idx + int(round(xo / dpix))
        py = center_idx + int(round(yo / dpix))
        sample_pix_list.append((px, py))

    # Pure noise trials (H0 null distribution)
    hits = 0
    max_snrs = []
    for t in range(n_trials):
        rng = np.random.RandomState(int(pa * 1000 + t))
        noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
        peaks = pipe.find_local_peaks_2d(noise, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
        pks = [p for p in peaks if mask_search[p["y_idx"], p["x_idx"]]]

        if len(pks) > 0:
            max_snrs.append(max(p["snr"] for p in pks))
        else:
            max_snrs.append(0.0)

        hit = False
        for sx, sy in sample_pix_list:
            for pk in pks:
                if (pk["x_idx"] - sx)**2 + (pk["y_idx"] - sy)**2 <= r_tol_pix**2:
                    hit = True; break
            if hit: break
        if hit: hits += 1

    fpr = hits / n_trials
    p95_snr = np.percentile(max_snrs, 95.0)
    results[pa] = {"fpr": fpr, "p95_snr": p95_snr, "mean_snr": np.mean(max_snrs)}
    print(f"PA = {pa:5.1f} deg: FPR = {fpr:.4f} ({hits}/{n_trials}), 95th SNR = {p95_snr:.2f}, Mean SNR = {np.mean(max_snrs):.2f}")

# Check statistical consistency
fpr_vals = [r["fpr"] for r in results.values()]
print(f"\nOverall FPR across PAs: mean={np.mean(fpr_vals):.4f}, std={np.std(fpr_vals):.4f}")
print(f"Standard binomial error: {np.sqrt(np.mean(fpr_vals)*(1-np.mean(fpr_vals))/n_trials):.4f}")
