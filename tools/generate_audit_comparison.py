"""
tools/generate_audit_comparison.py — 修正前・修正後モデル間回転曲線比較生成スクリプト

本スクリプトは、既存グラフを上書きせず、同一の物理パラメータ・質量分布・動径範囲において
以下の4つの理論曲線を計算し、新規CSVおよび新規画像として保存する。

比較対象:
1. ニュートン模型
2. MOND (a0 = 1.200e-10 m/s^2)
3. ジャイロ内挿模型（修正前定数: a0* = 1.042e-13 m/s^2）
4. ジャイロ内挿模型（修正後定数: a0* = 1.042e-10 m/s^2）

※観測点との比較ではなく「模型間比較」であることを明記する。
※パラメータの再フィットや新しい増幅係数は一切追加しない。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# パス設定
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOL_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gyro_gravity_theory.core import (
    G,
    C,
    H0,
    KPC,
    M_SUN,
    A0_MOND,
    v_newton,
    v_mond_interp,
)


def main():
    # 共通入力パラメータ（例示的な銀河質量分布パラメータ、再フィットなし）
    galaxy_name = "例示的な銀河質量分布パラメータ"
    M_total = 5.0e10        # M_sun
    b_val = 2.0             # kpc (Plummer半径)
    R_d_val = 1.5           # kpc (円盤スケール長)
    f_disk_val = 0.8        # 円盤質量分率
    r_min = 1.0             # kpc
    r_max = 30.0            # kpc
    n_points = 200
    mu_type = "standard"    # 内挿関数: x / sqrt(1 + x^2)

    # 特性加速度の定義
    a0_mond = A0_MOND       # 1.200e-10 m/s^2
    a0_star_pre = C * (H0 * 1000.0 / (KPC * 1e6)) / (2.0 * np.pi)  # 修正前: 1.042198e-13 m/s^2
    a0_star_post = C * (H0 * 1000.0 / (KPC * 1e3)) / (2.0 * np.pi) # 修正後: 1.042198e-10 m/s^2

    # 動径グリッド
    r_grid = np.linspace(r_min, r_max, n_points)

    print("=" * 70)
    print("銀河回転曲線 模型間比較データの生成")
    print(f"対象銀河パラメータ: {galaxy_name} (M_total={M_total:.1e} M_sun, b={b_val}kpc, R_d={R_d_val}kpc)")
    print(f"内挿関数: {mu_type}")
    print(f"  1. ニュートン重力")
    print(f"  2. MOND: a0 = {a0_mond:.3e} m/s^2")
    print(f"  3. ジャイロ内挿（修正前定数）: a0* = {a0_star_pre:.3e} m/s^2")
    print(f"  4. ジャイロ内挿（修正後定数）: a0* = {a0_star_post:.3e} m/s^2")
    print("=" * 70)

    # 1. ニュートン模型
    v_N = v_newton(r_grid, M_total, b_val, R_d_val, f_disk_val)

    # 2. MOND模型 (a0 = 1.2e-10)
    v_M = v_mond_interp(r_grid, M_total, b_val, R_d_val, f_disk_val, a0=a0_mond, mu_type=mu_type)

    # 3. ジャイロ内挿模型（修正前定数: 1.042e-13）
    v_G_pre = v_mond_interp(r_grid, M_total, b_val, R_d_val, f_disk_val, a0=a0_star_pre, mu_type=mu_type)

    # 4. ジャイロ内挿模型（修正後定数: 1.042e-10）
    v_G_post = v_mond_interp(r_grid, M_total, b_val, R_d_val, f_disk_val, a0=a0_star_post, mu_type=mu_type)

    # 出力先ディレクトリの確保
    # ※ 本番・既存グラフ (images/ 配下) は上書きせず、分離された output/ に新規保存する。
    # ※ なお、本スクリプト再実行時は output/ 内の同一ファイル (model_comparison_curves.csv/png) が更新・上書きされる仕様。
    output_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "model_comparison_curves.csv")
    png_path = os.path.join(output_dir, "model_comparison_curves.png")

    # CSV出力（UTF-8エンコードで保存）
    header_comment = (
        f"# Model Comparison of Galaxy Rotation Curves (Theoretical Curves Only, No Observation Points)\n"
        f"# Galaxy Parameters: description={galaxy_name}, M_total={M_total:.2e} M_sun, b={b_val} kpc, R_d={R_d_val} kpc, f_disk={f_disk_val}\n"
        f"# Physical Constants: G={G:.5e} m^3/kg/s^2, c={C:.1f} m/s, H0={H0} km/s/Mpc\n"
        f"# Critical Acceleration a0: MOND={a0_mond:.4e} m/s^2, Gyro_pre_fix={a0_star_pre:.4e} m/s^2, Gyro_post_fix={a0_star_post:.4e} m/s^2\n"
        f"# Note: Theoretical comparison under identical input conditions without refitting or observation data fitting.\n"
        f"# Note: Example galaxy mass distribution parameters only; does not claim fit to observed data or reproduction of literature values.\n"
        f"radius_kpc,v_newton_kms,v_mond_kms,v_gyro_pre_fix_kms,v_gyro_post_fix_kms"
    )

    data_matrix = np.column_stack([r_grid, v_N, v_M, v_G_pre, v_G_post])
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header_comment + "\n")
        np.savetxt(f, data_matrix, delimiter=",", fmt="%.6e")
    print(f"[OK] CSVデータを保存しました: {csv_path}")

    # プロット作成（日本語対応フォールバック）
    font_candidates = ["Noto Sans CJK JP", "Noto Sans JP", "Yu Gothic", "MS Gothic", "DejaVu Sans"]
    available_fonts = {f.name for f in plt.matplotlib.font_manager.fontManager.ttflist}
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # 4曲線の描画
    ax.plot(r_grid, v_N, color="#1f77b4", linestyle="-", linewidth=2.0, label="ニュートン模型 (Newtonian)")
    ax.plot(r_grid, v_M, color="#d62728", linestyle="-.", linewidth=2.0, label=f"MOND (a0 = {a0_mond:.2e} m/s²)")
    ax.plot(r_grid, v_G_pre, color="#7f7f7f", linestyle=":", linewidth=2.0, label=f"ジャイロ内挿・修正前定数 (a0* = {a0_star_pre:.2e} m/s²)")
    ax.plot(r_grid, v_G_post, color="#2ca02c", linestyle="--", linewidth=2.0, label=f"ジャイロ内挿・修正後定数 (a0* = {a0_star_post:.2e} m/s²)")

    # 軸ラベル・タイトル
    ax.set_xlabel("銀河中心からの距離 $r$ [kpc]", fontsize=12)
    ax.set_ylabel("軌道速度 $v$ [km/s]", fontsize=12)
    ax.set_title(
        f"銀河回転曲線の模型間比較（観測点なし・純粋模型比較）\n"
        f"Model Comparison of Galaxy Rotation Curves — {galaxy_name}",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlim(0, 31)
    ax.set_ylim(0, max(v_N.max(), v_M.max(), v_G_post.max()) * 1.15)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    # 説明注記ボックス
    note_text = (
        "【模型間比較の注記】\n"
        "・本図は天体観測データとの照合ではなく、同一入力条件下での理論曲線比較である。\n"
        "・例示的な銀河質量分布パラメータ。特定銀河の観測適合や、文献値の再現を主張するものではない。\n"
        "・修正前定数（1000倍過小）は深-MOND効果が極めて微弱なためニュートン曲線とほぼ重複する。\n"
        "・修正後定数は深-MOND領域で MOND と約3.5%の速度差（漸近比 (a0*/a0)^0.25 ≈ 0.965）を持つ。\n"
        "・フラットな曲線形状そのものは、MOND枠組みの採用によるものであり独自機構の実証ではない。"
    )
    ax.text(
        0.03,
        0.05,
        note_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, facecolor="white")
    plt.close()
    print(f"[OK] 比較グラフを新規出力先に保存しました: {png_path}")

    # 主要半径での速度値サマリー
    r_check = [5.0, 10.0, 20.0, 30.0]
    print("\n【主要半径での軌道速度値 [km/s]】")
    print(f"{'r [kpc]':>8} | {'Newton':>10} | {'MOND':>10} | {'Gyro (修正前)':>14} | {'Gyro (修正後)':>14} | {'比率 (Gyro後/MOND)':>18}")
    print("-" * 85)
    for rc in r_check:
        vn = float(np.interp(rc, r_grid, v_N))
        vm = float(np.interp(rc, r_grid, v_M))
        vg_pre = float(np.interp(rc, r_grid, v_G_pre))
        vg_post = float(np.interp(rc, r_grid, v_G_post))
        ratio = vg_post / vm
        print(f"{rc:>8.1f} | {vn:>10.2f} | {vm:>10.2f} | {vg_pre:>14.2f} | {vg_post:>14.2f} | {ratio:>18.4f}")


if __name__ == "__main__":
    main()
