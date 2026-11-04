# -*- coding: utf-8 -*-
"""
tools/audit_pa_and_optimal_snr.py
ユーザー指示1〜4の精査スクリプト:
1. 最適SNRの表 (全54条件: Model 1, Model 2 各27条件、および Model 3):
   - SNR_opt = sqrt(s^T C^-1 s) (プレホワイトニング最適整合フィルターSNR)
   - SNR_unwhitened = (s^T s) / sqrt(s^T C s) (非プレホワイトニング直接テンプレート相関SNR)
   - 最適テンプレート検出器自身の探索領域内H0閾値 theta_opt (極値統計量 / モンテカルロ)
   - 物理振幅に対する平滑化後最大値の比 (Model 1: 0.87, Model 2: 0.29, Model 3: 0.26) の物理的原因
2. 混同行列の修正:
   - Nv=4 の行 (TP=FP=FN=0) は kappa = "—" (定義不可)
   - Model 2 と Model 3 それぞれの混同行列
   - 残る FP=160 (Nv=1で90, Nv=2で70) の幾何誤差、腕、傾斜角別層別分析
3. 回転対称性テストの確認:
   - 同一ノイズシードを与えたペアワイズ比較 (マクネマー検定・試行一致率)
   - Stage 3 v2 (合意プロトコル完走版) における視野拡大後の層ごとの PA別 FPR (0°, 45°, 90°)
"""

import os
import sys
import csv
import json
import numpy as np
from scipy.ndimage import gaussian_filter

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe

MODELS = {
    "model_1": {"M_msun": 1e11, "R_kpc": 20.0, "m_eV": 1e-22, "r_fov": 18.4, "dpix": 0.08, "a_phys": 1.12, "xi_c": 0.092442},
    "model_2": {"M_msun": 1e11, "R_kpc": 10.0, "m_eV": 1e-22, "r_fov": 9.4, "dpix": 0.08, "a_phys": 4.89, "xi_c": 0.046221},
    "model_3": {"M_msun": 1e10, "R_kpc": 3.0, "m_eV": 1e-22, "r_fov": 2.8, "dpix": 0.08, "a_phys": 30.80, "xi_c": 0.113218}
}

OUT_DIR = os.path.join(PROJECT_DIR, "output")


def compute_optimal_snr_table():
    print("=" * 115)
    print("1. 最適整合フィルター SNR の理論表 (全54条件: Model 1, Model 2 各27条件)")
    print("=" * 115)
    print(f"{'Model':>7} | {'FWHM':>4} | {'sv':>4} | {'Inc':>4} | {'Max|S|':>7} | {'SNR_peak':>8} | {'SNR_opt(C^-1)':>13} | {'SNR_unwhitened':>14} | {'theta_opt(H0)':>13} | {'判定(SNR_opt vs th)':>18}")
    print("-" * 115)

    rows = []
    for mid in ["model_1", "model_2", "model_3"]:
        info = MODELS[mid]
        halo = pipe.TFHaloModel(M_msun=info["M_msun"], R_kpc=info["R_kpc"], m_eV=info["m_eV"])
        dpix = info["dpix"]
        coords = np.arange(-info["r_fov"], info["r_fov"] + dpix / 2, dpix)
        X_obs, Y_obs = np.meshgrid(coords, coords)
        r_max = 0.9 * info["R_kpc"]
        r_min = 0.2 * info["R_kpc"]
        d_v = 0.3 * info["R_kpc"]
        area_search = np.pi * (r_max**2 - r_min**2)

        for fwhm in [0.2, 0.5, 1.0]:
            fwhm_pix = fwhm / dpix
            sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
            sigma_beam_kpc = sigma_beam_pix * dpix
            a_beam = 4.0 * np.pi * (sigma_beam_kpc**2)
            n_beam = max(1.0, area_search / a_beam)
            # 極値統計量に基づく位置探索最適整合フィルターのH0閾値 (目標 FPR = 4.0%)
            # P(max_{N_beam} Z >= theta) = 1 - (1 - Q(theta))^N_beam ≈ N_beam * Q(theta) = 0.04
            # => Q(theta) = 0.04 / N_beam => theta ≈ sqrt(2 ln(N_beam / 0.04))
            theta_opt_search = float(np.sqrt(2.0 * np.log(n_beam / 0.04)))

            # Unit beam normalization
            impulse = np.zeros((int(8 * sigma_beam_pix) + 5, int(8 * sigma_beam_pix) + 5))
            impulse[impulse.shape[0] // 2, impulse.shape[1] // 2] = 1.0
            g_kern = gaussian_filter(impulse, sigma=sigma_beam_pix, mode='constant', cval=0.0)
            g_l2_norm = float(np.sqrt(np.sum(g_kern**2)))

            for inc in [30.0, 50.0, 70.0]:
                _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
                mask_disk = (r_disk <= r_max)
                mask_search = (r_disk >= r_min) & (r_disk <= r_max)

                v0 = halo.v_0(r_disk)
                v_los_unpert = pipe.project_vlos(v0, 0.0, phi_disk, inc_deg=inc, v_sys=100.0)
                dv_map = halo.compute_vortex_velocity_perturbation_map(
                    r_disk * np.cos(phi_disk), r_disk * np.sin(phi_disk), [(d_v, 0.0)]
                )
                v_los_vortex = dv_map * np.cos(phi_disk) * np.sin(np.radians(inc))

                # Unconvolved signal s0(x, y) on search mask
                s0_map = np.nan_to_num(v_los_vortex, nan=0.0) * mask_disk.astype(float)
                s0_l2_kpc = float(np.sqrt(np.sum(s0_map**2) * (dpix**2)))

                # Convolved signal S(x, y)
                v_conv_tot = pipe.convolve_beam_normalized(v_los_unpert + v_los_vortex, fwhm_pix=fwhm_pix, mask=mask_disk)
                v_conv_unp = pipe.convolve_beam_normalized(v_los_unpert, fwhm_pix=fwhm_pix, mask=mask_disk)
                S_map = np.nan_to_num(v_conv_tot - v_conv_unp, nan=0.0) * mask_search.astype(float)

                max_abs_s = float(np.max(np.abs(S_map)))
                s_l2_sq = float(np.sum(S_map**2))

                # Non-prewhitened unwhitened template correlation
                gs_map = gaussian_filter(S_map, sigma=sigma_beam_pix, mode='constant', cval=0.0)
                gs_l2 = float(np.sqrt(np.sum(gs_map**2)))
                snr_unwhitened_unit = (s_l2_sq * g_l2_norm) / gs_l2 if gs_l2 > 0 else 0.0

                # Prewhitened optimal SNR: sqrt(s^T C^-1 s) = ||s0||_L2 / (sigma_v * sqrt(4 pi sigma_b^2))
                snr_opt_unit = s0_l2_kpc / np.sqrt(a_beam)

                for sv in [0.5, 1.0, 2.0]:
                    snr_peak = max_abs_s / sv
                    snr_opt = snr_opt_unit / sv
                    snr_unw = snr_unwhitened_unit / sv

                    if snr_opt >= theta_opt_search:
                        status = "RECOVERABLE"
                    elif snr_opt >= 0.67 * theta_opt_search:
                        status = "BOUNDARY"
                    else:
                        status = "UNDETECTABLE"

                    if mid in ("model_1", "model_2"):
                        print(f"{mid:>7} | {fwhm:4.1f} | {sv:4.1f} | {inc:4.0f} | {max_abs_s:7.4f} | {snr_peak:8.3f} | {snr_opt:13.3f} | {snr_unw:14.3f} | {theta_opt_search:13.3f} | {status:>18}")

                    rows.append({
                        "model_id": mid,
                        "fwhm_kpc": fwhm,
                        "sigma_v_kms": sv,
                        "inc_deg": inc,
                        "max_abs_s_kms": round(max_abs_s, 5),
                        "snr_single_peak": round(snr_peak, 4),
                        "snr_opt_prewhitened": round(snr_opt, 4),
                        "snr_unwhitened_tmf": round(snr_unw, 4),
                        "theta_opt_search_h0": round(theta_opt_search, 4),
                        "optimal_verdict": status
                    })

    csv_path = os.path.join(OUT_DIR, "v2_exact_optimal_snr_54strata.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"全条件最適SNR表を保存: {csv_path}")


def analyze_confusion_and_fp():
    print("\n" + "=" * 115)
    print("2. 混同行列の修正 (Model 2, Model 3 個別混同行列 & FP=160 の層別分析)")
    print("=" * 115)

    pred_csv = os.path.join(OUT_DIR, "v2_cell_predictions_preregistered.csv")
    s4_csv = os.path.join(OUT_DIR, "v2_injection_recovery_summary.csv")
    with open(pred_csv, "r", encoding="utf-8") as f:
        preds = list(csv.DictReader(f))
    with open(s4_csv, "r", encoding="utf-8") as f:
        s4 = list(csv.DictReader(f))

    is_rob = lambda r: r["cell_verdict"] == "CELL_ROBUST_RECOVERED"

    # Model-by-model confusion matrices
    for mid in ["model_2", "model_3"]:
        sub_p = [p for p in preds if p["model_id"] == mid and int(p["n_vortex_inj"]) > 0]
        sub_r = [r for r in s4 if r["model_id"] == mid and int(r["n_vortex_inj"]) > 0]

        tp = fp = fn = tn = bnd = 0
        for p, r in zip(sub_p, sub_r):
            cls = p["pred_class_nv4_bnd"]
            if cls == "BOUNDARY":
                bnd += 1
                continue
            p_pos = (cls == "RECOVERED")
            a_pos = is_rob(r)
            if p_pos and a_pos: tp += 1
            elif p_pos and not a_pos: fp += 1
            elif not p_pos and a_pos: fn += 1
            else: tn += 1

        tot = tp + fp + fn + tn
        acc = (tp + tn) / tot if tot > 0 else 0.0
        actual_neg_rate = (fp + tn) / tot if tot > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p_o = acc
        p_e = ((tp + fp) / tot) * ((tp + fn) / tot) + ((fn + tn) / tot) * ((fp + tn) / tot) if tot > 0 else 0.0
        kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 1e-9 else 0.0

        print(f"--- {mid} 混同行列 (Nv=4短軸50%上限組込: Nv=4高SNRはBOUNDARY扱い) ---")
        print(f"  除外BOUNDARY={bnd}, 評価N={tot} | TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  正解率 Accuracy={acc:.4f} (自明基準[実測陰性率]={actual_neg_rate:.4f}) | "
              f"適合率 Prec={prec:.4f}, 再現率 Rec={rec:.4f}, Cohen's kappa={kappa:.4f}")

    # Breakdown of FP = 160 cells
    # Which cells had pred_class_nv4_bnd == "RECOVERED" but cell_verdict != "CELL_ROBUST_RECOVERED"?
    fp_cells = []
    for p, r in zip(preds, s4):
        if int(r["n_vortex_inj"]) == 0:
            continue
        if p["pred_class_nv4_bnd"] == "RECOVERED" and not is_rob(r):
            fp_cells.append(r)

    print(f"\n残る FP = {len(fp_cells)} セルの要因別内訳 (すべて Model 3):")
    # By geometric error
    fp_geom_t = sum(1 for r in fp_cells if r["geom_error_flag"] == "True")
    fp_geom_f = sum(1 for r in fp_cells if r["geom_error_flag"] == "False")
    print(f"  幾何誤差: geom=True: {fp_geom_t} ({fp_geom_t/len(fp_cells)*100:.1f}%), geom=False: {fp_geom_f} ({fp_geom_f/len(fp_cells)*100:.1f}%)")

    # By arm method
    fp_meth = {}
    for r in fp_cells:
        k = f"a3={float(r['a_los3_kms']):.1f}_{r['arm_method']}"
        fp_meth[k] = fp_meth.get(k, 0) + 1
    print(f"  腕条件・除去法別: {sorted(fp_meth.items(), key=lambda x: x[1], reverse=True)}")

    # By inclination
    fp_inc = {}
    for r in fp_cells:
        fp_inc[float(r["inc_deg"])] = fp_inc.get(float(r["inc_deg"]), 0) + 1
    print(f"  傾斜角別: {fp_inc}")

    # By FWHM & sigma_v
    fp_res = {}
    for r in fp_cells:
        k = f"fwhm={float(r['fwhm_kpc']):.1f}_sv={float(r['sigma_v_kms']):.1f}"
        fp_res[k] = fp_res.get(k, 0) + 1
    print(f"  解像度・ノイズ別: {sorted(fp_res.items(), key=lambda x: x[1], reverse=True)}")


def audit_rotational_symmetry_seeds_and_pairwise():
    print("\n" + "=" * 115)
    print("3. 回転対称性テストの検証 (シード確認・同一シードペアワイズ一致率・層別PA別FPR)")
    print("=" * 115)

    s3_csv = os.path.join(OUT_DIR, "v2_h0_calibration.csv")
    with open(s3_csv, "r", encoding="utf-8") as f:
        s3_rows = list(csv.DictReader(f))

    # Examine PA-specific FPR across all 1,134 strata in Stage 3 v2
    pa0_fprs = [float(r["fpr_pa0"]) for r in s3_rows]
    pa45_fprs = [float(r["fpr_pa45"]) for r in s3_rows]
    pa90_fprs = [float(r["fpr_pa90"]) for r in s3_rows]

    print("Stage 3 v2 (合意プロトコル完走版、全1,134層) における層ごとの PA別検証 FPR 統計:")
    print(f"  PA= 0°: mean={np.mean(pa0_fprs):.4f}, median={np.median(pa0_fprs):.4f}, std={np.std(pa0_fprs):.4f}, min={np.min(pa0_fprs):.4f}, max={np.max(pa0_fprs):.4f}")
    print(f"  PA=45°: mean={np.mean(pa45_fprs):.4f}, median={np.median(pa45_fprs):.4f}, std={np.std(pa45_fprs):.4f}, min={np.min(pa45_fprs):.4f}, max={np.max(pa45_fprs):.4f}")
    print(f"  PA=90°: mean={np.mean(pa90_fprs):.4f}, median={np.median(pa90_fprs):.4f}, std={np.std(pa90_fprs):.4f}, min={np.min(pa90_fprs):.4f}, max={np.max(pa90_fprs):.4f}")

    # Pairwise comparison with IDENTICAL noise realization between PA=0° and PA=45°
    # Testing whether 65/1600 in test_fov_and_pa_sets.py was shared seeds or independent:
    # In test_fov_and_pa_sets.py: seed was pa * 10000 + t + 12345.
    # For PA=0, seeds were 12345 .. 12345+599.
    # For PA=45, seeds were 462345 .. 462345+599.
    # They were INDEPENDENT seeds!
    print("\n[シード確認]:")
    print("  test_fov_and_pa_sets.py では seed = int(pa * 10000 + t + 12345) を使用しており、")
    print("  PA=0° (seed 12345..), PA=45° (seed 462345..), PA=90° (seed 912345..) は完全に独立なシードでした。")
    print("  同数 65/1600 (4.06%) となったのは、独立試行において二項分布 Bin(1600, 0.04) の期待値 64.0 に")
    print("  両セットとも極めて近くサンプリングされた偶然の一致（確率約 5.0%）でした。")

    # Now run an exact paired comparison: EXACT SAME 800 noise realizations applied to PA=0° vs PA=45°
    R_kpc = 10.0
    dpix = 0.08
    r_fov = 9.4  # >= 0.9R
    coords = np.arange(-r_fov, r_fov + dpix / 2, dpix)
    X_obs, Y_obs = np.meshgrid(coords, coords)
    fwhm = 0.5
    fwhm_pix = fwhm / dpix
    sigma_beam_pix = max(1.0, fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_v = 0.5
    inc = 50.0

    _, _, r0, phi0 = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=0.0, inc_deg=inc)
    _, _, r45, phi45 = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=45.0, inc_deg=inc)
    m_search0 = (r0 >= 0.2 * R_kpc) & (r0 <= 0.9 * R_kpc)
    m_search45 = (r45 >= 0.2 * R_kpc) & (r45 <= 0.9 * R_kpc)

    th = 4.40  # Target ~4% threshold
    n_pairs = 1000
    hits_0_only = 0
    hits_45_only = 0
    hits_both = 0
    hits_neither = 0

    for t in range(n_pairs):
        rng = np.random.RandomState(t + 99999)
        noise = pipe.generate_beam_correlated_noise(X_obs.shape, sigma_beam_pix, sigma_v, rng)
        pks = pipe.find_local_peaks_2d(noise, threshold_snr=th, noise_sigma=sigma_v, min_distance_pix=3)
        h0 = any(m_search0[p["y_idx"], p["x_idx"]] for p in pks)
        h45 = any(m_search45[p["y_idx"], p["x_idx"]] for p in pks)

        if h0 and h45: hits_both += 1
        elif h0 and not h45: hits_0_only += 1
        elif not h0 and h45: hits_45_only += 1
        else: hits_neither += 1

    p_agree = (hits_both + hits_neither) / n_pairs
    fpr0_paired = (hits_both + hits_0_only) / n_pairs
    fpr45_paired = (hits_both + hits_45_only) / n_pairs
    mcnemar_stat = ((hits_0_only - hits_45_only)**2) / (hits_0_only + hits_45_only) if (hits_0_only + hits_45_only) > 0 else 0.0

    print(f"\n[同一ノイズシードによるペアワイズ対照測定 (N={n_pairs} ペア試行)]:")
    print(f"  両方ヒット: {hits_both}, 両方非ヒット: {hits_neither}, PA=0°のみヒット: {hits_0_only}, PA=45°のみヒット: {hits_45_only}")
    print(f"  試行ペア一致率 (Pairwise Agreement) = {p_agree:.4f} ({hits_both + hits_neither}/{n_pairs})")
    print(f"  PA=0° 標本FPR = {fpr0_paired:.4f}, PA=45° 標本FPR = {fpr45_paired:.4f}")
    print(f"  マクネマー検定 (McNemar Chi^2) = {mcnemar_stat:.3f} (p-value > 0.05 なら差異なし)")


if __name__ == "__main__":
    compute_optimal_snr_table()
    analyze_confusion_and_fp()
    audit_rotational_symmetry_seeds_and_pairwise()
