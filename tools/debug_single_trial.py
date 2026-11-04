import os, sys
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/src')
import numpy as np
import gyro_gravity_theory.injection_pipeline as pipe

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
center_idx = len(coords) // 2
X_obs, Y_obs = np.meshgrid(coords, coords)
fwhm_pix = fwhm / dpix

r_disk_min = 0.2 * R_kpc
r_disk_max = 0.9 * R_kpc
d_vortex = 0.3 * R_kpc # 0.9 kpc

_, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_deg, inc_deg=inc_deg)
mask_disk = (r_disk_true <= R_kpc * 0.9)
v0_true = halo.v_0(r_disk_true)
v_los_unpert = pipe.project_vlos(v0_true, 0.0, phi_disk_true, inc_deg=inc_deg, v_sys=v_sys_true)

vortex_pos_kpc = [(d_vortex, 0.0)]
true_pix = []
for xv, yv in vortex_pos_kpc:
    xo, yo = pipe.disk_to_obs(xv, yv, pa_deg=pa_deg, inc_deg=inc_deg)
    xi = center_idx + int(round(xo / dpix))
    yi = center_idx + int(round(yo / dpix))
    true_pix.append((xi, yi))

dv_vortex_map = halo.compute_vortex_velocity_perturbation_map(
    r_disk_true * np.cos(phi_disk_true), r_disk_true * np.sin(phi_disk_true), vortex_pos_kpc
)
v_los_inj = dv_vortex_map * np.cos(phi_disk_true) * np.sin(np.radians(inc_deg))

v_conv_signal = pipe.convolve_beam_normalized(v_los_unpert + v_los_inj, fwhm_pix=fwhm_pix, mask=mask_disk)

snr_th = 9.2679
sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
r_tol_kpc = pipe.compute_tolerance_radius("model_3", fwhm, R_kpc)
r_tol_pix = r_tol_kpc / dpix

print(f"True pix: {true_pix[0]}, r_tol_pix: {r_tol_pix:.2f}")

# Run 10 trials
matches_list = []
for trial in range(10):
    rng = np.random.RandomState(trial)
    noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
    v_obs = v_conv_signal + noise

    _, _, res_raw = pipe.fit_axisymmetric_disk_rings(
        r_disk_true, phi_disk_true, v_obs, inc_deg=inc_deg, v_sys=v_sys_true, n_rings=15
    )

    raw_peaks = pipe.find_local_peaks_2d(res_raw, threshold_snr=snr_th, noise_sigma=sigma_v, min_distance_pix=3)
    peaks = [p for p in raw_peaks if r_disk_min <= r_disk_true[p["y_idx"], p["x_idx"]] <= r_disk_max]
    matches = pipe.evaluate_position_matching(peaks, true_pix, r_tol_pix=r_tol_pix)
    matches_list.append(sum(matches))
    print(f"Trial {trial}: {len(peaks)} peaks found. Matches: {matches}")
    for pk in peaks:
        dist = np.sqrt((pk['x_idx'] - true_pix[0][0])**2 + (pk['y_idx'] - true_pix[0][1])**2)
        print(f"  peak at ({pk['x_idx']}, {pk['y_idx']}), snr={pk['snr']:.2f}, val={pk['value']:.2f}, dist_to_true={dist:.2f} pix (tol={r_tol_pix:.2f})")

print(f"Recall: {np.mean(matches_list)}")
