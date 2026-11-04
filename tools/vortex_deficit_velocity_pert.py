"""
tools/vortex_deficit_velocity_pert.py — 少数渦の非軸対称重力効果と円運動速度変化の予備計算

本スクリプトは、球対称自己重力TFハロー（n=1ポリトロープ球）を背景とし、
ハロー中心軸に平行な少数の中空円筒状渦（Nv = 1, 2, 4）が存在する場合の
重力加速度変化および円運動速度変化 δv(r, θ) を見積もる。

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


def compute_vortex_perturbation(M_msun: float, R_kpc: float, m_eV: float, config_name: str, vortex_positions: list):
    """
    指定されたハローパラメータおよび渦配置における重力加速度・軌道速度変化を計算する。
    vortex_positions: [(x_kpc, y_kpc), ...]
    """
    # 単位換算
    M_halo = M_msun * M_SUN
    R_halo = R_kpc * KPC
    m_kg = m_eV * EV / (C ** 2)

    # 1. TF背景の特性量
    v_esc = np.sqrt(2.0 * G * M_halo / R_halo)
    xi_c = HBAR / (m_kg * v_esc)
    rho_c = np.pi * M_halo / (4.0 * (R_halo ** 3))

    def rho_0(r):
        """球対称TF密度分布"""
        val = np.zeros_like(r)
        mask = (r < R_halo) & (r > 0)
        x = np.pi * r[mask] / R_halo
        val[mask] = rho_c * np.sin(x) / x
        val[r == 0] = rho_c
        return val

    def M_0(r):
        """TF累積質量"""
        val = np.zeros_like(r)
        mask = (r < R_halo) & (r > 0)
        x = np.pi * r[mask] / R_halo
        val[mask] = (M_halo / np.pi) * (np.sin(x) - x * np.cos(x))
        val[r >= R_halo] = M_halo
        return val

    def v_0(r):
        """非摂動円運動速度"""
        m_enc = M_0(r)
        val = np.zeros_like(r)
        mask = r > 0
        val[mask] = np.sqrt(G * m_enc[mask] / r[mask])
        return val

    # 2. 各渦の質量欠損の計算
    vortices_info = []
    total_M_def = 0.0
    for xv_kpc, yv_kpc in vortex_positions:
        xv = xv_kpc * KPC
        yv = yv_kpc * KPC
        d_cen = np.sqrt(xv**2 + yv**2)
        if d_cen < R_halo:
            L_z = 2.0 * np.sqrt(R_halo**2 - d_cen**2)
            z_pts = np.linspace(-L_z/2, L_z/2, 200)
            dz = z_pts[1] - z_pts[0]
            r_sph = np.sqrt(d_cen**2 + z_pts**2)
            rho_z = rho_c * np.sin(np.pi * r_sph / R_halo) / (np.pi * r_sph / R_halo)
            m_def_vortex = np.sum(rho_z * np.pi * (xi_c ** 2) * dz)
        else:
            z_pts = np.array([0.0])
            dz = 0.0
            rho_z = np.array([0.0])
            m_def_vortex = 0.0

        total_M_def += m_def_vortex
        vortices_info.append({
            "xv": xv, "yv": yv, "d_cen": d_cen, "L_z": L_z,
            "z_pts": z_pts, "dz": dz, "rho_z": rho_z, "m_def": m_def_vortex
        })

    # 3. 円盤面内（z=0）における加速度変化 δa_r(r, theta) の計算
    r_grid = np.linspace(0.05 * R_halo, 0.95 * R_halo, 120)
    theta_grid = np.linspace(0, 2 * np.pi, 72, endpoint=False)

    # 2次元メッシュ
    R_mesh, TH_mesh = np.meshgrid(r_grid, theta_grid, indexing="ij")
    X_mesh = R_mesh * np.cos(TH_mesh)
    Y_mesh = R_mesh * np.sin(TH_mesh)

    delta_ax = np.zeros_like(X_mesh)
    delta_ay = np.zeros_like(Y_mesh)

    for v in vortices_info:
        dx = X_mesh - v["xv"]
        dy = Y_mesh - v["yv"]
        d_perp_sq = dx**2 + dy**2
        z_pts = v["z_pts"]
        dz = v["dz"]
        rho_z = v["rho_z"]

        if dz > 0:
            for k in range(len(z_pts)):
                denom = (d_perp_sq + (xi_c**2) + (z_pts[k]**2)) ** 1.5
                factor = G * np.pi * (xi_c**2) * rho_z[k] * dz / denom
                delta_ax += factor * dx
                delta_ay += factor * dy

    # 動径方向加速度変化（中心向き引力の減少 $\implies$ 向心加速度の低下）
    # a_net_r = a_0_r - delta_a_r (where delta_a is directed away from center)
    delta_ar = delta_ax * np.cos(TH_mesh) + delta_ay * np.sin(TH_mesh)

    # 4. 円運動速度変化 δv(r, theta)
    v0_grid = v_0(r_grid)
    V0_mesh = np.tile(v0_grid[:, np.newaxis], (1, len(theta_grid)))

    # v = sqrt(r * (a_0 - delta_ar)) approx v_0 - r * delta_ar / (2 * v_0)
    delta_v_mesh = - R_mesh * delta_ar / (2.0 * V0_mesh)

    # 断面プロファイル
    # theta = 0（第1渦方向）
    delta_v_on_vortex = delta_v_mesh[:, 0]
    # theta = pi/4 または pi/2（渦中間方向）
    idx_mid = len(theta_grid) // 8  # 45度
    delta_v_mid = delta_v_mesh[:, idx_mid]

    # 円周平均プロファイル
    delta_v_circ_mean = np.mean(delta_v_mesh, axis=1)

    # 方位角異方性振幅（max - min）
    delta_v_azim_amp = np.max(delta_v_mesh, axis=1) - np.min(delta_v_mesh, axis=1)

    # 最大変化量と位置
    max_abs_on_vortex = np.max(np.abs(delta_v_on_vortex))
    r_peak_on_vortex = r_grid[np.argmax(np.abs(delta_v_on_vortex))]

    max_abs_mean = np.max(np.abs(delta_v_circ_mean))
    r_peak_mean = r_grid[np.argmax(np.abs(delta_v_circ_mean))]

    max_azim_amp = np.max(delta_v_azim_amp)
    r_peak_azim = r_grid[np.argmax(delta_v_azim_amp)]

    # 半値全幅 (FWHM) の推定（局在スケール）
    half_max = max_abs_on_vortex / 2.0
    indices_above_half = np.where(np.abs(delta_v_on_vortex) >= half_max)[0]
    if len(indices_above_half) > 1:
        fwhm_kpc = (r_grid[indices_above_half[-1]] - r_grid[indices_above_half[0]]) / KPC
    else:
        fwhm_kpc = 2.0 * (xi_c / KPC)

    # 判定分類
    # 基準（km/s単位で比較）:
    # - 予備的に有望: max |delta_v| >= 2.0 km/s かつ fwhm >= 0.5 kpc
    # - 予備的に弱い: max |delta_v| < 1.0 km/s または fwhm < 0.5 kpc (極端に局在)
    # - 境界的: 1.0 km/s <= max |delta_v| < 2.0 km/s
    max_kms = max_abs_on_vortex / 1e3
    status = "予備的に弱い"
    if max_kms >= 2.0 and fwhm_kpc >= 0.5:
        status = "予備的に有望 (δv >= 2km/s, 観測可能幅)"
    elif max_kms < 1.0 or fwhm_kpc < 0.5:
        status = "予備的に弱い (δv < 1km/s または極端に局在)"
    else:
        status = "境界的 (1km/s <= δv < 2km/s)"

    return {
        "config_name": config_name,
        "M_msun": M_msun,
        "R_kpc": R_kpc,
        "m_eV": m_eV,
        "xi_c_kpc": xi_c / KPC,
        "N_v": len(vortex_positions),
        "total_M_def_msun": total_M_def / M_SUN,
        "def_mass_ratio_pct": (total_M_def / M_halo) * 100.0,
        "max_delta_v_on_vortex_kms": max_abs_on_vortex / 1e3,
        "r_peak_on_vortex_kpc": r_peak_on_vortex / KPC,
        "max_delta_v_mean_kms": max_abs_mean / 1e3,
        "r_peak_mean_kpc": r_peak_mean / KPC,
        "max_azim_amp_kms": max_azim_amp / 1e3,
        "fwhm_kpc": fwhm_kpc,
        "status": status,
        "r_grid_kpc": r_grid / KPC,
        "delta_v_on_vortex_kms": delta_v_on_vortex / 1e3,
        "delta_v_circ_mean_kms": delta_v_circ_mean / 1e3,
    }


def main():
    print("=" * 85)
    print("少数渦（中空円筒近似）による円運動速度変化 δv(r) の予備計算")
    print("背景模型: 球対称・非回転TF自己重力球 (n=1ポリトロープ)")
    print("=" * 85)

    # 3つの代表ハローモデル
    models = [
        {"name": "モデル1 (標準銀河)", "M": 1.0e10, "R": 10.0, "m": 1.0e-22},
        {"name": "モデル2 (大質量銀河)", "M": 1.0e11, "R": 10.0, "m": 3.0e-23},
        {"name": "モデル3 (コンパクト銀河)", "M": 1.0e10, "R": 3.0, "m": 1.0e-22},
    ]

    configs = [
        {"id": "A", "label": "Nv=1 (中心軸 d=0)", "get_pos": lambda R: [(0.0, 0.0)]},
        {"id": "B", "label": "Nv=1 (オフセンター d=0.3R)", "get_pos": lambda R: [(0.3 * R, 0.0)]},
        {"id": "C", "label": "Nv=2 (中心対称 d=±0.3R)", "get_pos": lambda R: [(0.3 * R, 0.0), (-0.3 * R, 0.0)]},
        {"id": "D", "label": "Nv=4 (正方対称 d=0.3R)", "get_pos": lambda R: [(0.3 * R, 0.0), (-0.3 * R, 0.0), (0.0, 0.3 * R), (0.0, -0.3 * R)]},
        {"id": "E", "label": "Nv=4 (外寄り正方 d=0.5R)", "get_pos": lambda R: [(0.5 * R, 0.0), (-0.5 * R, 0.0), (0.0, 0.5 * R), (0.0, -0.5 * R)]},
    ]

    results = []

    for mod in models:
        for cfg in configs:
            pos = cfg["get_pos"](mod["R"])
            full_name = f"{mod['name']} | {cfg['label']}"
            res = compute_vortex_perturbation(mod["M"], mod["R"], mod["m"], full_name, pos)
            results.append(res)

    # 結果表示
    header = f"{'モデル・渦配置':<35} | {'Nv':>2} | {'xi_c':>6} | {'M_def [Msun]':>12} | {'M_def/M':>8} | {'max|δv|(軸上)':>12} | {'max|δv|(平均)':>12} | {'異方性':>7} | {'FWHM':>6} | {'判定'}"
    print(header)
    print("-" * 130)

    for r in results:
        line = (
            f"{r['config_name']:<35} | {r['N_v']:>2} | {r['xi_c_kpc']:>6.3f} | "
            f"{r['total_M_def_msun']:>12.2e} | {r['def_mass_ratio_pct']:>7.3f}% | "
            f"{r['max_delta_v_on_vortex_kms']:>10.3f} km/s | "
            f"{r['max_delta_v_mean_kms']:>10.3f} km/s | "
            f"{r['max_azim_amp_kms']:>5.2f}km | "
            f"{r['fwhm_kpc']:>5.2f}k | {r['status']}"
        )
        print(line)

    # CSV出力
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "vortex_deficit_perturbation_summary.csv")

    headers = [
        "config_name", "M_msun", "R_kpc", "m_eV", "xi_c_kpc", "N_v",
        "total_M_def_msun", "def_mass_ratio_pct",
        "max_delta_v_on_vortex_kms", "r_peak_on_vortex_kpc",
        "max_delta_v_mean_kms", "r_peak_mean_kpc",
        "max_azim_amp_kms", "fwhm_kpc", "status"
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("# Vortex Deficit Velocity Perturbation in Self-Gravitating TF Haloes\n")
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                f'"{r["config_name"]}"', f"{r['M_msun']:.2e}", f"{r['R_kpc']:.1f}", f"{r['m_eV']:.2e}",
                f"{r['xi_c_kpc']:.4f}", f"{r['N_v']}",
                f"{r['total_M_def_msun']:.4e}", f"{r['def_mass_ratio_pct']:.4f}",
                f"{r['max_delta_v_on_vortex_kms']:.4f}", f"{r['r_peak_on_vortex_kpc']:.4f}",
                f"{r['max_delta_v_mean_kms']:.4f}", f"{r['r_peak_mean_kpc']:.4f}",
                f"{r['max_azim_amp_kms']:.4f}", f"{r['fwhm_kpc']:.4f}", f'"{r["status"]}"'
            ]
            f.write(",".join(row) + "\n")
    print(f"\n[OK] 計算サマリーCSVを保存しました: {csv_path}")


if __name__ == "__main__":
    main()
