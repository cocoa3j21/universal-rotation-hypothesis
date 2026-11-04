# -*- coding: utf-8 -*-
"""
tools/fix_tables_and_hashes.py
旧表 (通常格子基準) を v2_method_b_snr_54strata.csv に復元し、
新表 (超サンプリング基準) を v2_method_b_snr_54strata_oversampled.csv に保存。
SHA256SUMS.txt を再生成して別個のハッシュとして収録する。
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import glob
import hashlib
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
DOCS_DIR = os.path.join(PROJECT_DIR, "docs")
OUT_DIR = os.path.join(PROJECT_DIR, "output")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import gyro_gravity_theory.injection_pipeline as pipe

def get_sha256(fpath):
    if not os.path.exists(fpath):
        return None
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def generate_standard_grid_table():
    # Model 1 & 2 standard grid (dpix = 0.08 kpc analytic signal)
    model_thresholds = {
        ("model_1", 0.2): 5.040, ("model_1", 0.5): 4.790, ("model_1", 1.0): 4.522,
        ("model_2", 0.2): 4.735, ("model_2", 0.5): 4.485, ("model_2", 1.0): 4.207,
    }
    models = {
        "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 18.4},
        "model_2": {"M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 9.4},
    }
    dpix = 0.08
    lines = [
        "model_id,fwhm_kpc,sigma_v_kms,inc_deg,snr_opt_raw,snr_eff_method_a,r_a,verdict_a,snr_eff_method_b,r_b,verdict_b,threshold"
    ]

    for mid, info in models.items():
        halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])
        coords = np.arange(-info["r_fov"], info["r_fov"] + dpix / 2, dpix)
        X_obs, Y_obs = np.meshgrid(coords, coords)
        r_max = 0.9 * info["R_kpc"]
        r_min = 0.2 * info["R_kpc"]
        d_v = 0.3 * info["R_kpc"]
        n_rings = 30
        r_edges = np.linspace(0.0, r_max, n_rings + 1)

        for fwhm in [0.2, 0.5, 1.0]:
            fwhm_pix = fwhm / dpix
            sigma_b_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
            sigma_b_kpc = sigma_b_pix * dpix
            a_beam = 4.0 * np.pi * (sigma_b_kpc**2)
            sqrt_a_beam = np.sqrt(a_beam)
            th = model_thresholds[(mid, fwhm)]

            for inc in [30.0, 50.0, 70.0]:
                _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
                mask_disk = (r_disk <= r_max)
                mask_search = (r_disk >= r_min) & (r_disk <= r_max)

                dv = halo.compute_vortex_velocity_perturbation_map(
                    r_disk * np.cos(phi_disk), r_disk * np.sin(phi_disk), [(d_v, 0.0)])
                vlos_vort = dv * np.cos(phi_disk) * np.sin(np.radians(inc))

                s0_raw = vlos_vort * mask_disk.astype(float)
                l2_raw = float(np.sqrt(np.sum(s0_raw**2) * (dpix**2)))

                # Method A (Full disk, rings > 10 pix)
                s0_A = np.copy(s0_raw)
                for k in range(n_rings):
                    m = mask_disk & (r_disk >= r_edges[k]) & (r_disk < r_edges[k+1])
                    if np.sum(m) > 0: s0_A[m] -= np.mean(s0_raw[m])
                    if np.sum(m) > 10:
                        cp = np.cos(phi_disk[m]); sp = np.sin(phi_disk[m])
                        coeff, _, _, _ = np.linalg.lstsq(np.column_stack([cp, sp]), s0_A[m], rcond=None)
                        s0_A[m] -= (coeff[0]*cp + coeff[1]*sp)
                l2_A = float(np.sqrt(np.sum(s0_A**2) * (dpix**2)))

                # Method B (Search mask only + all rings projected)
                s0_B = vlos_vort * mask_search.astype(float)
                for k in range(n_rings):
                    m = mask_search & (r_disk >= r_edges[k]) & (r_disk < r_edges[k+1])
                    if np.sum(m) > 0: s0_B[m] -= np.mean(s0_B[m])
                    if np.sum(m) >= 3:
                        cp = np.cos(phi_disk[m]); sp = np.sin(phi_disk[m])
                        coeff, _, _, _ = np.linalg.lstsq(np.column_stack([cp, sp]), s0_B[m], rcond=None)
                        s0_B[m] -= (coeff[0]*cp + coeff[1]*sp)
                l2_B = float(np.sqrt(np.sum(s0_B**2) * (dpix**2)))

                for sv in [0.5, 1.0, 2.0]:
                    snr_raw = (l2_raw / sqrt_a_beam) / sv
                    snr_A = (l2_A / sqrt_a_beam) / sv
                    snr_B = (l2_B / sqrt_a_beam) / sv

                    r_A = snr_A / th
                    r_B = snr_B / th

                    v_A = "RECOVERABLE" if r_A >= 1.15 else ("BOUNDARY" if r_A >= 0.85 else "UNDETECTABLE")
                    v_B = "RECOVERABLE" if r_B >= 1.15 else ("BOUNDARY" if r_B >= 0.85 else "UNDETECTABLE")

                    lines.append(f"{mid},{fwhm},{sv},{inc},{snr_raw:.4f},{snr_A:.4f},{r_A:.4f},{v_A},{snr_B:.4f},{r_B:.4f},{v_B},{th:.4f}")

    old_csv_path = os.path.join(OUT_DIR, "v2_method_b_snr_54strata.csv")
    with open(old_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"旧表 (通常格子基準) を復元保存完了: {old_csv_path}")


def update_json_details():
    json_path = os.path.join(OUT_DIR, "final_report_numbers.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Update interpolation interpretation
    data["r_b_interpolation"]["interpretation_detail"] = {
        "oversampled_grid": "内挿値 0.91 は境界帯 (0.85 <= R < 1.15) の内側に位置し、境界帯が50%の位置を適切に挟んでいる",
        "standard_grid": "通常格子の内挿値 1.21 は境界帯の外側 (上側) にあり、旧基準 (R>=1.15を回収可能) は楽観的な分類を与えていた"
    }

    # 2. Threshold sources note
    data["threshold_sources"] = {
        "empirical_mc": {
            "model_2_fwhm0.2_sv0.5": {"30_deg": 4.3644, "50_deg": 4.2702, "70_deg": 4.0805},
            "model_1_fwhm0.2_sv0.5": {"50_deg": 4.6321}
        },
        "interpolated_extrapolated": "上記以外はビーム数依存性に基づく補間・外挿値 (0.05の倍数)",
        "insensitivity_note": "比率 R が 0.4 以下の条件は大半を占め、分類 (UNDETECTABLE) は閾値の微小な変動に一切依存しない"
    }

    # 3. Model 1 vs Model 2 method B reduction ratio
    data["method_b_reduction_ratio"] = {
        "model_2": "通常格子の約 0.67 〜 0.78 倍に低下",
        "model_1": "通常格子の約 0.87 〜 0.90 倍に低下"
    }

    # 4. Inverted SNR vs Method B ratio
    data["snr_b_to_inverted_ratio"] = {
        "30_deg": {"snr_B": 3.472, "snr_inv": 3.908, "ratio": 0.888},
        "50_deg": {"snr_B": 4.370, "snr_inv": 4.760, "ratio": 0.918},
        "70_deg": {"snr_B": 3.391, "snr_inv": 3.780, "ratio": 0.897},
        "note": "超サンプリングに基準を揃えた後、方法Bは逆算有効SNRの約 0.89 〜 0.92 倍 (約 10% 低め・保守的)"
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"final_report_numbers.json を更新完了")


def regenerate_sha256sums():
    sums_lines = []

    # 1. Output files
    output_files = [
        "v2_h0_calibration.csv",
        ".v2_snr_thresholds.json",
        "v2_cell_predictions_preregistered.csv",
        "v2_injection_recovery_summary.csv",
        "v2_exact_optimal_snr_54strata.csv",
        "v2_audited_optimal_snr_54strata.csv",
        "v2_method_b_snr_54strata.csv",               # Old table (standard grid)
        "v2_method_b_snr_54strata_oversampled.csv",   # New table (oversampled)
        "v2_oversampled_template_detector_results.json",
        "v2_positive_control_oversampled_result.json",
        "final_report_numbers.json",
        "cell_predictions_preregistered.csv",
        "h0_stratum_calibration.csv",
        "injection_recovery_summary.csv"
    ]

    print("=" * 100)
    print("SHA256SUMS.txt の完全再生成 (旧表 + 新表 + 成果物 + tools/*.py + src/* + docs/*)")
    print("=" * 100)

    for fname in output_files:
        fpath = os.path.join(OUT_DIR, fname)
        sha = get_sha256(fpath)
        if sha:
            line = f"{sha}  output/{fname}"
            sums_lines.append(line)
            print(line)

    # 2. Tools scripts
    tool_files = sorted(glob.glob(os.path.join(TOOLS_DIR, "*.py")))
    for tpath in tool_files:
        sha = get_sha256(tpath)
        sums_lines.append(f"{sha}  tools/{os.path.basename(tpath)}")

    # 3. Src files
    src_files = sorted(glob.glob(os.path.join(SRC_DIR, "**", "*.py"), recursive=True))
    for spath in src_files:
        sha = get_sha256(spath)
        rel = os.path.relpath(spath, PROJECT_DIR).replace("\\", "/")
        sums_lines.append(f"{sha}  {rel}")

    # 4. Docs specification
    doc_path = os.path.join(DOCS_DIR, "INJECTION_RECOVERY_TEST_SPEC_V7.md")
    if os.path.exists(doc_path):
        sha = get_sha256(doc_path)
        sums_lines.append(f"{sha}  docs/INJECTION_RECOVERY_TEST_SPEC_V7.md")

    sums_path = os.path.join(OUT_DIR, "SHA256SUMS.txt")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sums_lines) + "\n")
    print(f"\n全 {len(sums_lines)} ファイルのチェックサムを SHA256SUMS.txt に書き出しました: {sums_path}")


if __name__ == "__main__":
    generate_standard_grid_table()
    update_json_details()
    regenerate_sha256sums()
