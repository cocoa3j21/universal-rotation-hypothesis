"""
tools/feasibility_tf_vortices.py — 自己重力TFハローにおける量子渦成立可能性評価スクリプト

本スクリプトは、自己重力トーマス・フェルミ（TF）平衡BECハロー（Zinner型反発2体相互作用模型）において、
観測可能な渦構造（マクロ中空円筒）を生む条件が同時に成立するかを評価する。

※既存の監査済みコード（src/配下）は変更せず、独立した実験用スクリプトとして動作する。
※外部公開・コミット・pushは行わない。
"""

import os
import sys
import numpy as np

# 物理定数（SI単位系）
HBAR = 1.054571817e-34    # J s
C = 299792458.0           # m/s
EV = 1.602176634e-19      # J
KPC = 3.085677581e19      # m
M_SUN = 1.98847e30        # kg
G = 6.67430e-11           # m^3 kg^-1 s^-2


def evaluate_point(m_eV: float, R_kpc: float, M_msun: float, q_val: float):
    """
    1つのパラメータ点における従属量および成立条件の評価。
    """
    # 1. 独立入力のSI換算
    m_kg = m_eV * EV / (C ** 2)
    R_m = R_kpc * KPC
    M_kg = M_msun * M_SUN
    q = q_val

    # 2. 自己重力TF平衡条件による従属量
    # Böhmer & Harko (2007): R = pi * sqrt(hbar^2 a_s / (G m^3))
    a_s = G * (m_kg ** 3) * (R_m ** 2) / ((np.pi ** 2) * (HBAR ** 2))
    g_int = 4.0 * np.pi * (HBAR ** 2) * a_s / m_kg

    # 中心密度（n=1ポリトロープ球）
    rho_c = np.pi * M_kg / (4.0 * (R_m ** 3))
    n_c = rho_c / m_kg

    # 中心回復長（healing length）
    xi_c = 1.0 / np.sqrt(8.0 * np.pi * a_s * n_c)
    # 解析解との整合検算: xi_c == HBAR / (m_kg * sqrt(2 * G * M_kg / R_m))
    v_esc = np.sqrt(2.0 * G * M_kg / R_m)

    # 3. 回転角速度および臨界角速度
    Omega = np.sqrt(q * G * M_kg / (R_m ** 3))
    if R_m > xi_c:
        # Zinner (2011) 基本式（原文係数ではなくSI基本定数から計算）
        Omega_c = (HBAR / (m_kg * (R_m ** 2))) * np.log(1.46 * R_m / xi_c)
    else:
        Omega_c = np.nan

    # 4. 渦の特性量（ファインマン則）
    kappa = 2.0 * np.pi * HBAR / m_kg
    n_v = 2.0 * Omega / kappa
    d_v = np.sqrt(2.0 / (np.sqrt(3.0) * n_v))
    N_v = n_v * np.pi * (R_m ** 2)

    # 5. 各種無次元比
    xi_over_R = xi_c / R_m
    xi_over_d = xi_c / d_v
    d_over_R = d_v / R_m
    dilute_param = n_c * (a_s ** 3)
    non_rel_param = v_esc / C

    # 6. 近似成立および判定条件
    tf_valid = (xi_over_R <= 0.1)
    vortex_enters = (not np.isnan(Omega_c)) and (Omega > Omega_c)
    dilute_lattice = (xi_over_d <= 0.2)
    size_filter = (xi_c / KPC >= 0.1)      # サイズ選別フィルタ（0.1 kpc以上）
    dilute_gas = (dilute_param < 1e-3)
    non_rel = (non_rel_param < 1e-2)
    small_rot = (q <= 0.1)
    continuous_lattice = (N_v >= 10.0) and (d_over_R <= 0.3)

    # 判定分類
    status = "UNKNOWN"
    notes = []

    if not small_rot:
        status = "OUT_OF_APPROX (q > 0.1: 小回転近似破綻)"
        notes.append("小回転近似破綻")
    elif not tf_valid:
        status = "FAIL (TF近似破綻: xi_c/R > 0.1)"
        notes.append("TF近似破綻")
    elif not vortex_enters:
        status = "FAIL (渦侵入不能: Omega <= Omega_c)"
        notes.append("渦侵入不能")
    elif not dilute_lattice:
        status = "FAIL (渦芯重複: xi_c/d_v > 0.2)"
        notes.append("超流動凝縮破壊")
    elif not size_filter:
        status = "FAIL (微視的縮退: xi_c < 0.1 kpc)"
        notes.append("サイズ選別未達")
    else:
        # 条件通過点
        if continuous_lattice:
            status = "PASS (連続渦格子)"
        else:
            status = "PASS_WITH_RESERVATION (離散的少数渦: N_v < 10 または d/R > 0.3)"
            notes.append("連続格子近似に留保（離散渦線）")

    return {
        "m_eV": m_eV,
        "m_kg": m_kg,
        "R_kpc": R_kpc,
        "M_msun": M_msun,
        "q": q,
        "a_s_m": a_s,
        "rho_c": rho_c,
        "n_c": n_c,
        "xi_c_kpc": xi_c / KPC,
        "Omega": Omega,
        "Omega_c": Omega_c,
        "Omega_ratio": Omega / Omega_c if (not np.isnan(Omega_c) and Omega_c > 0) else np.nan,
        "n_v_kpc2": n_v * (KPC ** 2),
        "d_v_kpc": d_v / KPC,
        "N_v": N_v,
        "xi_over_R": xi_over_R,
        "xi_over_d": xi_over_d,
        "d_over_R": d_over_R,
        "dilute_param": dilute_param,
        "non_rel_param": non_rel_param,
        "status": status,
        "notes": "; ".join(notes) if notes else "なし",
    }


def run_grid_scan():
    """代表的なパラメータ空間の系統的走査"""
    m_list = [1e-23, 3e-23, 1e-22, 3e-22, 1e-21]     # eV/c^2
    M_list = [1e9, 1e10, 1e11]                        # M_sun
    R_list = [3.0, 10.0, 20.0]                        # kpc
    q_list = [0.005, 0.02, 0.05]                      # 小回転強度 (lambda ~ 0.05 相当)

    results = []
    for m in m_list:
        for M in M_list:
            for R in R_list:
                for q in q_list:
                    res = evaluate_point(m, R, M, q)
                    results.append(res)
    return results


def main():
    print("=" * 80)
    print("自己重力TFハローにおける量子渦成立可能性評価（系統的グリッド走査）")
    print("模型: Zinner (2011) 反発2体相互作用自己重力球 (n=1ポリトロープ)")
    print("=" * 80)

    results = run_grid_scan()

    total_points = len(results)
    pass_lattice = [r for r in results if r["status"].startswith("PASS (")]
    pass_reserv = [r for r in results if r["status"].startswith("PASS_WITH_RESERVATION")]
    fail_tf = [r for r in results if "TF近似破綻" in r["status"]]
    fail_novortex = [r for r in results if "渦侵入不能" in r["status"]]
    fail_submacro = [r for r in results if "微視的縮退" in r["status"]]
    fail_core = [r for r in results if "渦芯重複" in r["status"]]

    print(f"走査総点数: {total_points}")
    print(f"  - PASS (完全格子条件成立): {len(pass_lattice)} 件")
    print(f"  - PASS_WITH_RESERVATION (条件通過だが少数離散渦・格子留保): {len(pass_reserv)} 件")
    print(f"  - FAIL (TF近似破綻 xi_c/R > 0.1): {len(fail_tf)} 件")
    print(f"  - FAIL (渦侵入不能 Omega <= Omega_c): {len(fail_novortex)} 件")
    print(f"  - FAIL (微視的縮退 xi_c < 0.1 kpc): {len(fail_submacro)} 件")
    print(f"  - FAIL (渦芯重複 xi_c/d_v > 0.2): {len(fail_core)} 件")
    print("=" * 80)

    # 通過点（留保付き含む）の詳細出力
    passed_all = pass_lattice + pass_reserv
    if passed_all:
        print("\n【条件通過パラメータ点（サイズ選別 xi_c >= 0.1 kpc を満たす点）】")
        header = f"{'m [eV]':>8} | {'M [Msun]':>8} | {'R [kpc]':>7} | {'q':>5} | {'xi_c [kpc]':>10} | {'d_v [kpc]':>9} | {'N_v':>5} | {'xi/R':>6} | {'xi/d':>6} | {'d/R':>6} | {'判定ステータス'}"
        print(header)
        print("-" * len(header))
        for r in passed_all:
            line = (
                f"{r['m_eV']:>8.0e} | {r['M_msun']:>8.0e} | {r['R_kpc']:>7.1f} | {r['q']:>5.3f} | "
                f"{r['xi_c_kpc']:>10.3f} | {r['d_v_kpc']:>9.2f} | {r['N_v']:>5.1f} | "
                f"{r['xi_over_R']:>6.3f} | {r['xi_over_d']:>6.3f} | {r['d_over_R']:>6.3f} | {r['status']}"
            )
            print(line)
    else:
        print("\n※ 条件を完全に満たすパラメータ点は存在しませんでした（許容領域は空集合）。")

    # CSVファイルへの保存（output/配下）
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "tf_vortex_feasibility_scan.csv")

    headers = [
        "m_eV", "M_msun", "R_kpc", "q", "a_s_m", "rho_c_kgm3", "n_c_m3",
        "xi_c_kpc", "Omega_rads", "Omega_c_rads", "Omega_ratio", "N_v", "d_v_kpc",
        "xi_over_R", "xi_over_d", "d_over_R", "dilute_param", "non_rel_param", "status"
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("# Feasibility Scan of Quantized Vortices in Self-Gravitating TF BEC Halos\n")
        f.write("# Model: Zinner (2011) Repulsive SI-BEC, n=1 Polytrope\n")
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                f"{r['m_eV']:.2e}", f"{r['M_msun']:.2e}", f"{r['R_kpc']:.1f}", f"{r['q']:.4f}",
                f"{r['a_s_m']:.4e}", f"{r['rho_c']:.4e}", f"{r['n_c']:.4e}", f"{r['xi_c_kpc']:.4f}",
                f"{r['Omega']:.4e}", f"{r['Omega_c']:.4e}", f"{r['Omega_ratio']:.4f}", f"{r['N_v']:.2f}",
                f"{r['d_v_kpc']:.4f}", f"{r['xi_over_R']:.4f}", f"{r['xi_over_d']:.4f}", f"{r['d_over_R']:.4f}",
                f"{r['dilute_param']:.4e}", f"{r['non_rel_param']:.4e}", f'"{r["status"]}"'
            ]
            f.write(",".join(row) + "\n")
    print(f"\n[OK] 詳細走査データを保存しました: {csv_path}")


if __name__ == "__main__":
    main()
