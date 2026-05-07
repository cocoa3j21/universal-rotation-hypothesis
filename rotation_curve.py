#!/usr/bin/env python3
"""
gyro_gravity_theory — 全宇宙回転ジェネレーション仮説 メイン実行スクリプト

Universal Rotation Generation Hypothesis — Main Execution Script
================================================================

このスクリプトは、3つの重力モデル（ニュートン力学、ジャイロ加算モデル、
MOND補間モデル）の銀河回転曲線を計算し、比較グラフを生成する。

Usage
-----
    python src/rotation_curve.py

または config パスを指定:
    python src/rotation_curve.py --config ../config/params.yaml

Output
------
    output/rotation_curve_comparison.png  (300dpi PNG)
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np

# パッケージインポートのためのパス設定
# src/ ディレクトリを sys.path に追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gyro_gravity_theory.core import (
    A0_MOND,
    A0_STAR,
    compute_enclosed_mass,
    compute_bec_amplification,
    compute_dirac_lnh_amplification,
    compute_total_amplification,
    compute_S_eff_over_m_amplified,
    v_gyro_additive,
    v_gyro_interpolating,
    v_mond_interp,
    v_newton,
)

# ============================================================================
# YAML パラメータ読み込み / Parameter Loading
# ============================================================================

try:
    import yaml
except ImportError:
    print(
        "[!] PyYAML がインストールされていません。"
        "`pip install pyyaml` を実行してください。"
    )
    print("[!] デフォルトパラメータで続行します。")
    yaml = None


def load_params(yaml_path: Optional[str] = None) -> dict:
    """
    YAMLパラメータファイルを読み込む。
    ファイルがない場合やエラー時はデフォルト値を返す。

    Parameters
    ----------
    yaml_path : str, optional
        YAMLファイルのパス。None の場合はデフォルトパスを試行。

    Returns
    -------
    dict
        パラメータ辞書
    """
    # デフォルトパラメータ
    default_params = {
        "galaxy": {
            "name": "M33",
            "M_total_solar": 5e10,
            "b_kpc": 2.0,
            "R_d_kpc": 1.5,
            "f_disk": 0.8,
        },
        "simulation": {
            "r_min_kpc": 1.0,
            "r_max_kpc": 30.0,
            "n_points": 200,
            "S_eff_over_m_kpc_km_s": 500,
        },
        "model": {
            "a0": A0_MOND,
            "mu_type": "standard",
        },
    }

    if yaml_path is None:
        # スクリプト位置から config を探索
        candidates = [
            os.path.join(SRC_DIR, "..", "config", "params.yaml"),
            os.path.join(os.getcwd(), "config", "params.yaml"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                yaml_path = cand
                break

    if yaml_path is None or not os.path.exists(yaml_path):
        print("[!] パラメータファイルが見つかりません。デフォルト値を使用します。")
        return default_params

    if yaml is None:
        print("[!] PyYAML が利用できないため、デフォルト値を使用します。")
        return default_params

    try:
        with open(yaml_path, "r") as f:
            params = yaml.safe_load(f)
        print(f"[✓] パラメータを読み込みました: {yaml_path}")
        return params
    except Exception as e:
        print(f"[!] パラメータ読み込みエラー: {e}")
        print("[!] デフォルト値を使用します。")
        return default_params


def extract_params(params: dict) -> dict:
    """
    パラメータ辞書から計算に必要な値を抽出・整形。

    Parameters
    ----------
    params : dict
        生のパラメータ辞書

    Returns
    -------
    dict
        整形済みパラメータ辞書
    """
    galaxy = params.get("galaxy", {})
    sim = params.get("simulation", {})
    model = params.get("model", {})

    return {
        "galaxy_name": galaxy.get("name", "M33"),
        "M_total": float(galaxy.get("M_total_solar", 5e10)),
        "b": float(galaxy.get("b_kpc", 2.0)),
        "R_d": float(galaxy.get("R_d_kpc", 1.5)),
        "f_disk": float(galaxy.get("f_disk", 0.8)),
        "r_min": float(sim.get("r_min_kpc", 1.0)),
        "r_max": float(sim.get("r_max_kpc", 30.0)),
        "n_points": int(sim.get("n_points", 200)),
        "S_eff_over_m": float(sim.get("S_eff_over_m_kpc_km_s", 500)),
        "a0": float(model.get("a0", A0_MOND)),
        "mu_type": model.get("mu_type", "standard"),
    }


# ============================================================================
# 観測データ読み込み / Observation Data Loading
# ============================================================================


def load_observation_data(
    csv_path: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    観測データ（CSV）を読み込む。

    期待されるCSV形式:
        r_kpc, v_obs_kms, v_err_kms

    Parameters
    ----------
    csv_path : str, optional
        CSVファイルのパス。None の場合は自動探索。

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        (r_obs, v_obs, v_err) のタプル。見つからない場合は None。
    """
    if csv_path is None:
        candidates = [
            os.path.join(SRC_DIR, "..", "data", "m33_rotation.csv"),
            os.path.join(SRC_DIR, "..", "data", "M33_rotation.csv"),
            os.path.join(os.getcwd(), "data", "m33_rotation.csv"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                csv_path = cand
                break

    if csv_path is None or not os.path.exists(csv_path):
        return None

    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        r_obs = data[:, 0]
        v_obs = data[:, 1]
        v_err = data[:, 2] if data.shape[1] >= 3 else np.zeros_like(v_obs)
        print(f"[✓] 観測データを読み込みました: {csv_path} ({len(r_obs)}点)")
        return (r_obs, v_obs, v_err)
    except Exception as e:
        print(f"[!] 観測データ読み込みエラー: {e}")
        return None


# ============================================================================
# プロット設定 / Plotting Setup
# ============================================================================


def setup_matplotlib_japanese() -> bool:
    """
    日本語フォントの設定。フォールバックチェーンを実装。

    Returns
    -------
    bool
        日本語フォントが設定できたかどうか
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # 優先フォントリスト（フォールバックチェーン）
    font_candidates = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "TakaoPGothic",
        "Hiragino Sans",
        "Yu Gothic",
        "MS Gothic",
    ]

    # 利用可能なフォントを取得
    available = {f.name for f in plt.matplotlib.font_manager.fontManager.ttflist}

    for font in font_candidates:
        if font in available:
            rcParams["font.family"] = font
            rcParams["axes.unicode_minus"] = False
            print(f"[✓] 日本語フォントを設定しました: {font}")
            return True

    # フォールバック: 英字のみ
    print("[!] 日本語フォントが見つかりません。英字のみで表示します。")
    print("[!] インストール: sudo apt install fonts-noto-cjk")
    return False


# ============================================================================
# プロット関数 / Plotting Function
# ============================================================================


def plot_rotation_curves(
    r: np.ndarray,
    v_N: np.ndarray,
    v_M: np.ndarray,
    v_G: np.ndarray,
    save_path: str = "../output/rotation_curve_comparison.png",
    galaxy_name: str = "M33",
    v_obs: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    show_plot: bool = False,
    amplification_info: Optional[str] = None,
) -> None:
    """
    3つの回転曲線を重ね描きし、画像として保存。

    Parameters
    ----------
    r : np.ndarray
        動径距離配列 [kpc]
    v_N : np.ndarray
        ニュートン力学の速度 [km/s]
    v_M : np.ndarray
        MOND理論の速度 [km/s]
    v_G : np.ndarray
        ジャイロモデルの速度 [km/s]
    save_path : str
        画像の保存先パス
    galaxy_name : str
        銀河名（タイトル表示用）
    v_obs : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        観測データ (r_obs, v_obs, v_err)
    show_plot : bool
        画面表示も行うか

    Graph Specifications
    --------------------
    - 横軸: r [kpc] (範囲 1–30 kpc)
    - 縦軸: v [km/s]
    - 3曲線:
      1. ニュートン力学: 青 (#1f77b4) 実線
      2. ジャイロ加算モデル: 緑 (#2ca02c) 破線
      3. MOND補間モデル: 赤 (#d62728) 一点鎖線
    - 凡例（日本語表記含む）、タイトル（日英併記）、グリッド
    - 300 dpi PNG
    """
    import matplotlib.pyplot as plt

    setup_matplotlib_japanese()

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- 3曲線のプロット ---
    # 1. ニュートン力学: 青 実線
    ax.plot(
        r,
        v_N,
        color="#1f77b4",
        linestyle="-",
        linewidth=2,
        label="ニュートン力学 (Newton)",
    )

    # 2. ジャイロ加算モデル: 緑 破線
    ax.plot(
        r,
        v_G,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label="ジャイロ加算モデル (Gyro Additive)",
    )

    # 3. MOND補間モデル: 赤 一点鎖線
    ax.plot(
        r,
        v_M,
        color="#d62728",
        linestyle="-.",
        linewidth=2,
        label="MOND補間モデル (MOND Interpolating)",
    )

    # --- 観測データ（あれば） ---
    if v_obs is not None:
        r_obs, v_obs_data, v_err = v_obs
        ax.errorbar(
            r_obs,
            v_obs_data,
            yerr=v_err,
            fmt="o",
            color="black",
            markersize=5,
            capsize=3,
            label=f"観測データ ({galaxy_name} HI)",
            alpha=0.8,
        )

    # --- 軸ラベル ---
    ax.set_xlabel("銀河中心からの距離 $r$ [kpc]", fontsize=13)
    ax.set_ylabel("軌道速度 $v$ [km/s]", fontsize=13)

    # --- タイトル（日英併記） ---
    ax.set_title(
        f"回転曲線比較 — {galaxy_name}銀河\n"
        f"Rotation Curve Comparison — Galaxy {galaxy_name}",
        fontsize=14,
        fontweight="bold",
    )

    # --- グリッド ---
    ax.grid(True, alpha=0.3, linestyle=":")

    # --- 凡例 ---
    ax.legend(fontsize=11, loc="lower right")

    # --- BEC+LNH増幅機構の情報（オプション） ---
    if amplification_info:
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.05,
            0.05,
            amplification_info,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=props,
        )

    # --- 軸範囲 ---
    ax.set_xlim(0, r.max() * 1.05)
    ax.set_ylim(0, max(v_N.max(), v_M.max(), v_G.max()) * 1.15)

    # --- 保存 ---
    # save_path が相対パスの場合、スクリプト位置からの相対として解決
    if not os.path.isabs(save_path):
        # スクリプトのある src/ からの相対 → プロジェクトルート基準に
        # 通常 save_path はプロジェクトルートからの相対
        save_abs = os.path.normpath(os.path.join(SCRIPT_DIR, "..", save_path))
    else:
        save_abs = save_path

    os.makedirs(os.path.dirname(save_abs), exist_ok=True)
    plt.savefig(
        save_abs,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print(f"[✓] グラフを保存しました: {save_abs}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ============================================================================
# メイン実行関数 / Main Execution
# ============================================================================


def main() -> None:
    """
    メイン実行関数。

    1. パラメータ読み込み
    2. 動径グリッド生成
    3. 3モデルの回転曲線計算
    4. プロット生成・保存
    """
    # --- コマンドライン引数 ---
    parser = argparse.ArgumentParser(
        description="全宇宙回転ジェネレーション仮説 — 回転曲線シミュレーション"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAMLパラメータファイルのパス",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="グラフを画面表示する",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力画像のパス（デフォルト: output/rotation_curve_comparison.png）",
    )
    args = parser.parse_args()

    # --- パラメータ読み込み ---
    raw_params = load_params(args.config)
    p = extract_params(raw_params)

    print("=" * 60)
    print(f"全宇宙回転ジェネレーション仮説 シミュレーション")
    print(f"銀河: {p['galaxy_name']}")
    print(f"質量: M_total = {p['M_total']:.2e} M_sun")
    print(f"Plummer半径: b = {p['b']:.1f} kpc")
    print(f"円盤スケール長: R_d = {p['R_d']:.1f} kpc")
    print(f"円盤質量分率: f_disk = {p['f_disk']:.2f}")
    print(f"特性加速度: a0 = {p['a0']:.2e} m/s^2")
    print(f"a0* (cH0/2π) = {A0_STAR:.2e} m/s^2")
    print(f"内挿関数: {p['mu_type']}")
    print("=" * 60)

    # === BEC + LNH 増幅機構の検算 ===
    print("\n" + "=" * 60)
    print("【BEC + ディラック大数仮説 増幅機構の検算】")
    print("=" * 60)

    eta_bec = compute_bec_amplification()
    eta_lnh = compute_dirac_lnh_amplification()
    eta_total, _, _ = compute_total_amplification()

    print(f"\n増幅率:")
    print(f"  η_BEC (f_align: 10⁻⁵ → 1) = {eta_bec:.1e}")
    print(f"  η_LNH (√N, N≈10⁸⁰)      = {eta_lnh:.1e}")
    print(f"  η_total                   = {eta_total:.1e}")

    # 必要増幅率の計算
    a_gyro_direct = 1e-50  # 代表値 (m/s²)
    a0_mond = 1.2e-10       # MOND限界加速度 (m/s²)
    eta_required = a0_mond / a_gyro_direct
    print(f"\n必要増幅率: η_required = a₀ / a_gyro ≈ {eta_required:.1e}")
    print(f"理論増幅率: η_total    = {eta_total:.1e}")

    if (
        eta_total >= eta_required * 0.1
        and eta_total <= eta_required * 10
    ):
        print(
            f"→ ✅ 桁数一致！ η_total ∈ "
            f"[η_required/10, η_required×10] を満たす"
        )
    else:
        ratio = eta_total / eta_required
        print(f"→ η_total/η_required = {ratio:.1e}")
        if ratio > 1:
            print(
                f"→ ⚠️ {ratio:.1e}倍大きい"
                f"（何らかの抑制機構が必要）"
            )
        else:
            print(
                f"→ ⚠️ {1/ratio:.1e}倍小さい"
                f"（追加の増幅機構が必要）"
            )

    # BEC+LNH補正後のS_eff/m
    S_eff_base = p["S_eff_over_m"]
    S_eff_amplified = compute_S_eff_over_m_amplified(S_eff_base, eta_total)
    print(f"\nS_eff/m (base)       = {S_eff_base} kpc·km/s")
    print(f"S_eff/m (amplified)  = {S_eff_amplified:.2e} kpc·km/s")

    # グラフ用の増幅情報テキスト
    amplification_info = (
        f"BEC+LNH増幅: η_total={eta_total:.1e}\n"
        f"  η_BEC = {eta_bec:.1e}  (f_align: 10⁻⁵→1)\n"
        f"  η_LNH = {eta_lnh:.1e}  (√N, N≈10⁸⁰)"
    )

    # --- 動径グリッド生成 ---
    r = np.linspace(p["r_min"], p["r_max"], p["n_points"])
    print(f"\n動径グリッド: {p['r_min']}–{p['r_max']} kpc, {p['n_points']}点")

    # --- 1. ニュートン力学 ---
    print("計算中: ニュートン力学...")
    v_N = v_newton(r, p["M_total"], p["b"], p["R_d"], p["f_disk"])
    print(f"  v_N (r=10kpc) = {float(np.interp(10.0, r, v_N)):.1f} km/s")

    # --- 2. ジャイロ加算モデル ---
    print("計算中: ジャイロ加算モデル...")
    v_G = v_gyro_additive(
        r, p["M_total"], p["b"], p["R_d"], p["f_disk"], p["S_eff_over_m"]
    )
    print(f"  v_G (r=10kpc) = {float(np.interp(10.0, r, v_G)):.1f} km/s")
    print(f"  S_eff/m = {p['S_eff_over_m']:.1f} kpc·km/s")

    # --- 3. MOND補間モデル ---
    print("計算中: MOND補間モデル...")
    v_M = v_mond_interp(
        r, p["M_total"], p["b"], p["R_d"], p["f_disk"], a0=p["a0"], mu_type=p["mu_type"]
    )
    print(f"  v_M (r=10kpc) = {float(np.interp(10.0, r, v_M)):.1f} km/s")

    # --- 4. 観測データ読み込み ---
    obs_data = load_observation_data()

    # --- 5. プロット ---
    print("\nプロット生成中...")
    save_path = args.output or "../output/rotation_curve_comparison.png"
    plot_rotation_curves(
        r,
        v_N,
        v_M,
        v_G,
        save_path=save_path,
        galaxy_name=p["galaxy_name"],
        v_obs=obs_data,
        show_plot=args.show,
        amplification_info=amplification_info,
    )

    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
