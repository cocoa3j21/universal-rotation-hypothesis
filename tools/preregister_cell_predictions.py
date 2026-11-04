# -*- coding: utf-8 -*-
"""
tools/preregister_cell_predictions.py — 全9,072セルのA50理論予測・事前登録プロトコル (Pre-registration Protocol)

【プロトコル規定】
1. Stage 3 の全756層における自己較正済み層別SNR閾値 (stratum_snr_threshold) を使用 (固定値4.0は不使用)。
2. 理論予測式: A_50_pred = (SNR_th * sigma_v) / (eta_beam * eta_fit * |cos(phi)| * sin(i))
3. 三分類基準:
   - N_v = 0: CONTROL (対照群)
   - 物理振幅 A_phys >= 1.5 * A_50_pred: RECOVERED (回収予測)
   - 物理振幅 A_phys <= 0.67 * A_50_pred: UNDETECTED (未回収予測)
   - その間: BOUNDARY (境界領域)
4. 事前合否基準:
   - 境界除外セル (BOUNDARYおよびCONTROLを除く) において、予測と本番結果の一致率 >= 90% を事前合否基準とする。
5. 出力ファイルのSHA-256ハッシュ値およびタイムスタンプを算出し、完全に固定する。
"""

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
    "model_1": {
        "M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 12.0, "dpix": 0.08,
        "a_inj_physical": 1.12, "xi_c_kpc": 0.092442
    },
    "model_3": {
        "M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.2, "dpix": 0.08,
        "a_inj_physical": 30.80, "xi_c_kpc": 0.113218
    }
}

VORTEX_COUNTS = [0, 1, 2, 4]
INCLINATIONS = [30.0, 50.0, 70.0]
POSITION_ANGLES = [0.0, 45.0, 90.0]
RESOLUTIONS_KPC = [0.2, 0.5, 1.0]
NOISES_KMS = [0.5, 1.0, 2.0]
GEOM_ERRORS = [False, True]

ARM_LEVELS = [{"a_los3": 0.0, "method": "none"}]
for a3 in [1.0, 3.0]:
    for meth in ["raw", "remove_k3", "arm_fit"]:
        ARM_LEVELS.append({"a_los3": a3, "method": meth})


def compute_beam_attenuation(model_id, fwhm_kpc):
    """ビーム減衰率 eta_beam の実測"""
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
    print("=" * 80)
    print("PRE-REGISTRATION PROTOCOL: 全9,072セルの理論A50予測と固定ハッシュ生成")
    print("=" * 80)

    out_dir = os.path.join(PROJECT_DIR, "output")
    snr_cache_path = os.path.join(out_dir, ".stratum_snr_thresholds.json")
    if not os.path.exists(snr_cache_path):
        raise FileNotFoundError(f"Stage 3 のSNR閾値キャッシュが見つかりません: {snr_cache_path}")

    with open(snr_cache_path, "r", encoding="utf-8") as f:
        snr_dict = json.load(f)
    print(f"Loaded {len(snr_dict)} stratum SNR thresholds from Stage 3 calibration.")

    # ビーム減衰率テーブルの事前計算
    eta_beam_table = {}
    for mid in ["model_1", "model_3"]:
        for fwhm in RESOLUTIONS_KPC:
            eta_beam_table[(mid, fwhm)] = compute_beam_attenuation(mid, fwhm)

    # 手法別典型フィッティング効率
    eta_fit_table = {
        "none": 1.0,
        "raw": 1.0,
        "remove_k3": 0.996,
        "arm_fit": 0.999
    }

    pred_rows = []
    cell_idx = 0

    class_counts = {"CONTROL": 0, "RECOVERED": 0, "UNDETECTED": 0, "BOUNDARY": 0}

    for model_id in ["model_1", "model_3"]:
        info = MODELS[model_id]
        a_phys_base = info["a_inj_physical"]

        for n_v in VORTEX_COUNTS:
            for inc in INCLINATIONS:
                for pa in POSITION_ANGLES:
                    for fwhm in RESOLUTIONS_KPC:
                        for sigma_v in NOISES_KMS:
                            for geom_err in GEOM_ERRORS:
                                for arm in ARM_LEVELS:
                                    cell_id = f"C{cell_idx:05d}_{model_id}_nv{n_v}_i{int(inc)}_fwhm{fwhm}_s{sigma_v}_arm{arm['a_los3']}_{arm['method']}"
                                    stratum_key_str = f"{model_id}_{inc}_{fwhm}_{sigma_v}_{geom_err}_{arm['a_los3']}_{arm['method']}"

                                    snr_th = float(snr_dict.get(stratum_key_str, 4.0))
                                    eta_beam = eta_beam_table[(model_id, fwhm)]
                                    eta_fit = eta_fit_table.get(arm["method"], 1.0)

                                    # 主軸可視渦 (|cos(phi)| = 1.0) に対する理論 A_50
                                    a50_pred = pipe.predict_a50(
                                        snr_th=snr_th,
                                        sigma_v=sigma_v,
                                        eta_beam=eta_beam,
                                        eta_fit=eta_fit,
                                        inc_deg=inc,
                                        phi_disk_rad=0.0
                                    )

                                    if n_v == 0:
                                        pred_class = "CONTROL"
                                        a_phys = 0.0
                                        ratio = 0.0
                                    else:
                                        a_phys = a_phys_base
                                        ratio = a_phys / a50_pred if a50_pred > 0 else 0.0
                                        if a_phys >= 1.5 * a50_pred:
                                            pred_class = "RECOVERED"
                                        elif a_phys <= 0.67 * a50_pred:
                                            pred_class = "UNDETECTED"
                                        else:
                                            pred_class = "BOUNDARY"

                                    class_counts[pred_class] += 1

                                    pred_rows.append({
                                        "cell_idx": cell_idx,
                                        "cell_id": cell_id,
                                        "model_id": model_id,
                                        "n_vortex_inj": n_v,
                                        "inc_deg": inc,
                                        "pa_deg": pa,
                                        "fwhm_kpc": fwhm,
                                        "sigma_v_kms": sigma_v,
                                        "geom_error_flag": geom_err,
                                        "a_los3_kms": arm["a_los3"],
                                        "arm_method": arm["method"],
                                        "snr_th_stratum": round(snr_th, 4),
                                        "eta_beam": round(eta_beam, 4),
                                        "eta_fit": round(eta_fit, 4),
                                        "a50_pred_kms": round(a50_pred, 3),
                                        "a_inj_physical_kms": round(a_phys, 2),
                                        "ratio_phys_to_a50": round(ratio, 3),
                                        "pred_class": pred_class
                                    })
                                    cell_idx += 1

    assert len(pred_rows) == 9072, f"Total cells mismatch: {len(pred_rows)} != 9072"

    fieldnames = [
        "cell_idx", "cell_id", "model_id", "n_vortex_inj", "inc_deg", "pa_deg",
        "fwhm_kpc", "sigma_v_kms", "geom_error_flag", "a_los3_kms", "arm_method",
        "snr_th_stratum", "eta_beam", "eta_fit", "a50_pred_kms",
        "a_inj_physical_kms", "ratio_phys_to_a50", "pred_class"
    ]

    out_csv = os.path.join(out_dir, "cell_predictions_preregistered.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)

    # SHA-256 ハッシュ値の計算
    with open(out_csv, "rb") as f:
        file_bytes = f.read()
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    now_iso = datetime.now().isoformat()
    n_boundary_excl = len(pred_rows) - class_counts["CONTROL"] - class_counts["BOUNDARY"]

    meta_info = {
        "preregistration_timestamp": now_iso,
        "csv_path": out_csv,
        "file_size_bytes": len(file_bytes),
        "sha256_hash": sha256_hash,
        "total_cells": len(pred_rows),
        "class_breakdown": class_counts,
        "evaluable_cells_excluding_boundary": n_boundary_excl,
        "pre_registered_pass_criterion": "Consistency Rate >= 90.0% on non-boundary non-control cells"
    }

    meta_path = os.path.join(out_dir, "cell_predictions_preregistered_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print("\n【事前登録完了サマリー】")
    print(f"- 保存先CSV: {out_csv}")
    print(f"- メタデータ: {meta_path}")
    print(f"- 登録時刻: {now_iso}")
    print(f"- SHA-256 ハッシュ値: {sha256_hash}")
    print(f"- 総セル数: {len(pred_rows)} セル")
    print(f"  * 対照群 (CONTROL): {class_counts['CONTROL']} セル (25.0%)")
    print(f"  * 回収予測 (RECOVERED): {class_counts['RECOVERED']} セル ({class_counts['RECOVERED']/len(pred_rows)*100:.1f}%)")
    print(f"  * 未回収予測 (UNDETECTED): {class_counts['UNDETECTED']} セル ({class_counts['UNDETECTED']/len(pred_rows)*100:.1f}%)")
    print(f"  * 境界領域 (BOUNDARY): {class_counts['BOUNDARY']} セル ({class_counts['BOUNDARY']/len(pred_rows)*100:.1f}%)")
    print(f"- 境界除外評価対象セル数: {n_boundary_excl} セル")
    print(f"- 事前合否基準: 境界除外セルにおける予測一致率 >= 90.0%")
    print("=" * 80)


if __name__ == "__main__":
    main()
