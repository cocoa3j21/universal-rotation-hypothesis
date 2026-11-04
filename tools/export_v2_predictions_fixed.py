import os
import sys
import csv
import json
import hashlib
from datetime import datetime
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

MODELS = {
    "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 12.0, "dpix": 0.08, "a_inj": 1.12, "v0_rv": 76.29},
    "model_2": {"M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 6.0, "dpix": 0.08, "a_inj": 4.89, "v0_rv": 107.88},
    "model_3": {"M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.2, "dpix": 0.08, "a_inj": 30.80, "v0_rv": 62.29}
}

RESOLUTIONS_KPC = [0.2, 0.5, 1.0]
NOISES_KMS = [0.5, 1.0, 2.0]
INCLINATIONS = [30.0, 50.0, 70.0]

rows = []
snr_th_v2 = 4.50

for model_id in ["model_1", "model_2", "model_3"]:
    info = MODELS[model_id]
    dpix = info["dpix"]
    coords = np.arange(-info["r_fov"], info["r_fov"] + dpix/2, dpix)
    X_obs, Y_obs = np.meshgrid(coords, coords)
    d_v = 0.3 * info["R_kpc"]

    for fwhm in RESOLUTIONS_KPC:
        fwhm_pix = fwhm / dpix
        dist_sq = (X_obs - d_v)**2 + Y_obs**2
        v_raw = np.exp(-dist_sq / (2.0 * (0.1)**2))
        v_conv = pipe.convolve_beam_normalized(v_raw, fwhm_pix=fwhm_pix)
        eta_beam = float(np.nanmax(v_conv) / np.max(v_raw))

        for sigma_v in NOISES_KMS:
            for inc in INCLINATIONS:
                a50_pred = pipe.predict_a50(snr_th_v2, sigma_v, eta_beam, 1.0, inc)
                ratio = info["a_inj"] / a50_pred
                if ratio >= 1.5:
                    pred_class = "RECOVERED"
                elif ratio <= 0.67:
                    pred_class = "UNDETECTED"
                else:
                    pred_class = "BOUNDARY"

                rows.append({
                    "model_id": model_id,
                    "fwhm_kpc": fwhm,
                    "sigma_v_kms": sigma_v,
                    "inc_deg": inc,
                    "snr_th_v2": snr_th_v2,
                    "eta_beam": round(eta_beam, 4),
                    "a50_pred_kms": round(a50_pred, 3),
                    "a_inj_physical_kms": info["a_inj"],
                    "v0_rv_kms": info["v0_rv"],
                    "ratio_phys_to_a50": round(ratio, 3),
                    "pred_class": pred_class
                })

out_dir = os.path.join(PROJECT_DIR, "output")
out_csv = os.path.join(out_dir, "v2_predictions_fixed.csv")
fieldnames = list(rows[0].keys())

with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(out_csv, "rb") as f:
    sha256_hash = hashlib.sha256(f.read()).hexdigest()

meta = {
    "filename": "v2_predictions_fixed.csv",
    "timestamp_utc": datetime.utcnow().isoformat(),
    "sha256": sha256_hash,
    "total_conditions": len(rows),
    "snr_th_v2": snr_th_v2,
    "r_tol_pix_v2": 2.5
}
with open(os.path.join(out_dir, "v2_predictions_fixed_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Exported {len(rows)} v2 predictions to {out_csv}")
print(f"SHA-256: {sha256_hash}")
