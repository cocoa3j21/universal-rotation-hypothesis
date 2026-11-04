# -*- coding: utf-8 -*-
"""
tools/export_a50_grid_table.py — モデル×FWHM×sigma_v×傾斜角ごとの理論A50およびA50/v0完全一覧表出力
"""

import os
import sys
import csv
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

MODELS = {
    "model_1": {
        "M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 12.0, "dpix": 0.08,
        "a_inj_physical": 1.12, "rv_kpc": 6.0, "v0_rv": 76.29
    },
    "model_3": {
        "M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.2, "dpix": 0.08,
        "a_inj_physical": 30.80, "rv_kpc": 0.9, "v0_rv": 62.29
    }
}

RESOLUTIONS_KPC = [0.2, 0.5, 1.0]
NOISES_KMS = [0.5, 1.0, 2.0]
INCLINATIONS = [30.0, 50.0, 70.0]


def compute_beam_attenuation(model_id, fwhm_kpc):
    info = MODELS[model_id]
    dpix = info["dpix"]
    coords = np.arange(-info["r_fov"], info["r_fov"] + dpix/2, dpix)
    X_obs, Y_obs = np.meshgrid(coords, coords)
    d_v = 0.3 * info["R_kpc"]
    sigma_vortex_kpc = max(0.1, pipe.compute_tolerance_radius(model_id, 0.2, info["R_kpc"]))
    dist_sq_kpc = (X_obs - d_v)**2 + Y_obs**2
    v_raw = np.exp(-dist_sq_kpc / (2.0 * sigma_vortex_kpc**2))
    fwhm_pix = fwhm_kpc / dpix
    v_conv = pipe.convolve_beam_normalized(v_raw, fwhm_pix=fwhm_pix)
    peak_raw = np.max(v_raw)
    peak_conv = np.nanmax(v_conv)
    return float(peak_conv / peak_raw) if peak_raw > 0 else 1.0


def main():
    out_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "a50_grid_table.csv")

    rows = []
    snr_th_ref = 4.0
    eta_fit_ref = 0.995

    for model_id in ["model_1", "model_3"]:
        info = MODELS[model_id]
        a_inj_phys = info["a_inj_physical"]
        v0_val = info["v0_rv"]

        for fwhm in RESOLUTIONS_KPC:
            eta_beam = compute_beam_attenuation(model_id, fwhm)

            for sigma_v in NOISES_KMS:
                for inc in INCLINATIONS:
                    a50_pred = pipe.predict_a50(
                        snr_th=snr_th_ref,
                        sigma_v=sigma_v,
                        eta_beam=eta_beam,
                        eta_fit=eta_fit_ref,
                        inc_deg=inc,
                        phi_disk_rad=0.0
                    )

                    ratio_a50_v0 = a50_pred / v0_val

                    if a_inj_phys >= 1.5 * a50_pred:
                        pred_class = "RECOVERED"
                    elif a_inj_phys <= 0.67 * a50_pred:
                        pred_class = "UNDETECTED"
                    else:
                        pred_class = "BOUNDARY"

                    rows.append({
                        "model_id": model_id,
                        "fwhm_kpc": fwhm,
                        "sigma_v": sigma_v,
                        "inc_deg": inc,
                        "v0_rv_kms": v0_val,
                        "snr_th_ref": snr_th_ref,
                        "eta_beam": round(eta_beam, 4),
                        "eta_fit": round(eta_fit_ref, 4),
                        "a50_pred_kms": round(a50_pred, 3),
                        "a50_over_v0": round(ratio_a50_v0, 4),
                        "a_inj_physical_kms": round(a_inj_phys, 2),
                        "ratio_phys_to_a50": round(a_inj_phys / a50_pred, 3),
                        "pred_class": pred_class
                    })

    fieldnames = [
        "model_id", "fwhm_kpc", "sigma_v", "inc_deg", "v0_rv_kms", "snr_th_ref",
        "eta_beam", "eta_fit", "a50_pred_kms", "a50_over_v0", "a_inj_physical_kms",
        "ratio_phys_to_a50", "pred_class"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
