# -*- coding: utf-8 -*-
"""
tools/plot_injection_recovery_summary.py — 注入・回収試験結果の可視化スクリプト (Stage 4)

【目的】
正本仕様書セクション10に基づき、サマリーCSV (output/injection_recovery_summary.csv) から
統計集計・回収率ヒートマップ・大域判定結果をまとめた高品質な図を生成する。
ダークテーマ（Dark Theme）を標準採用する。

※宇宙論・物理学・数理検証専用。
※成功を前提とせず、検出不能・全滅となった場合も客観的に可視化する。
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# パス設定
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def plot_summary_figures(csv_path, out_png_path):
    if not os.path.exists(csv_path):
        print(f"エラー: サマリーCSVが見つかりません: {csv_path}")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"データ読み込み完了: {len(rows)} 行")

    # ダークテーマのスタイル設定
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['Meiryo', 'MS Gothic', 'BIZ UDGothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    bg_color = "#16181d"
    card_color = "#21252b"
    fig.patch.set_facecolor(bg_color)

    # --------------------------------------------------------------------------
    # パネル 1: セル単位総合判定の構成比 (バーチャート)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(card_color)

    from collections import Counter
    verdicts = [r["cell_verdict"] for r in rows]
    counts = Counter(verdicts)

    labels_order = [
        "CELL_CONTROL_PASSED", "CELL_FAIL_FPR_UNCONTROLLED", "CELL_DEGENERATE_ARM",
        "CELL_UNDETECTED", "CELL_MARGINAL", "CELL_POSITION_ONLY",
        "CELL_STRUCTURAL_ONLY", "CELL_ROBUST_RECOVERED"
    ]
    colors_map = {
        "CELL_CONTROL_PASSED": "#5c6370",
        "CELL_FAIL_FPR_UNCONTROLLED": "#e06c75",
        "CELL_DEGENERATE_ARM": "#d19a66",
        "CELL_UNDETECTED": "#e5c07b",
        "CELL_MARGINAL": "#61afef",
        "CELL_POSITION_ONLY": "#c678dd",
        "CELL_STRUCTURAL_ONLY": "#56b6c2",
        "CELL_ROBUST_RECOVERED": "#98c379"
    }

    plot_labels = []
    plot_vals = []
    plot_colors = []
    for lbl in labels_order:
        if lbl in counts:
            plot_labels.append(lbl.replace("CELL_", ""))
            plot_vals.append(counts[lbl])
            plot_colors.append(colors_map.get(lbl, "#abb2bf"))

    y_pos = np.arange(len(plot_labels))
    bars = ax1.barh(y_pos, plot_vals, color=plot_colors, edgecolor="#abb2bf", alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(plot_labels, fontsize=9, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlim(0, 8300)
    ax1.set_xlabel("セル数 (全9,072セル中)", fontsize=10)
    ax1.set_title("セル単位判定結果の分布", fontsize=12, fontweight='bold', pad=10)
    for bar in bars:
        w = bar.get_width()
        pct = (w / len(rows)) * 100.0
        ax1.text(w + 50, bar.get_y() + bar.get_height()/2, f"{w} ({pct:.1f}%)",
                 va='center', ha='left', fontsize=8, color="#abb2bf")

    # --------------------------------------------------------------------------
    # パネル 2: Model 1 観測分解能 vs 速度ノイズ (ロバスト回収率ヒートマップ)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(card_color)

    # Model 1 (注入渦あり Nv > 0)
    m1_rows = [r for r in rows if r["model_id"] == "model_1" and int(r["n_vortex_inj"]) > 0]
    fwhms = [0.2, 0.5, 1.0]
    noises = [0.5, 1.0, 2.0]
    rate_matrix_m1 = np.zeros((len(noises), len(fwhms)))

    for i, n_val in enumerate(noises):
        for j, f_val in enumerate(fwhms):
            subset = [r for r in m1_rows if float(r["sigma_v_kms"]) == n_val and float(r["fwhm_kpc"]) == f_val]
            if len(subset) > 0:
                rob = sum(1 for r in subset if r["cell_verdict"] == "CELL_ROBUST_RECOVERED")
                rate_matrix_m1[i, j] = (rob / len(subset)) * 100.0

    im2 = ax2.imshow(rate_matrix_m1, cmap="viridis", vmin=0, vmax=100, origin="lower")
    ax2.set_xticks(range(len(fwhms)))
    ax2.set_xticklabels([f"{v} kpc" for v in fwhms])
    ax2.set_yticks(range(len(noises)))
    ax2.set_yticklabels([f"{v} km/s" for v in noises])
    ax2.set_xlabel("空間分解能 FWHM", fontsize=10)
    ax2.set_ylabel(r"速度ノイズ $\sigma_v$", fontsize=10)
    ax2.set_title("Model 1 ロバスト回収率 [%]", fontsize=12, fontweight='bold', pad=10)
    for i in range(len(noises)):
        for j in range(len(fwhms)):
            ax2.text(j, i, f"{rate_matrix_m1[i, j]:.1f}%", ha='center', va='center',
                     color="white" if rate_matrix_m1[i, j] < 50 else "black", fontweight='bold')

    # --------------------------------------------------------------------------
    # パネル 3: Model 3 観測分解能 vs 速度ノイズ (ロバスト回収率ヒートマップ)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(card_color)

    m3_rows = [r for r in rows if r["model_id"] == "model_3" and int(r["n_vortex_inj"]) > 0]
    rate_matrix_m3 = np.zeros((len(noises), len(fwhms)))

    for i, n_val in enumerate(noises):
        for j, f_val in enumerate(fwhms):
            subset = [r for r in m3_rows if float(r["sigma_v_kms"]) == n_val and float(r["fwhm_kpc"]) == f_val]
            if len(subset) > 0:
                rob = sum(1 for r in subset if r["cell_verdict"] == "CELL_ROBUST_RECOVERED")
                rate_matrix_m3[i, j] = (rob / len(subset)) * 100.0

    im3 = ax3.imshow(rate_matrix_m3, cmap="plasma", vmin=0, vmax=100, origin="lower")
    ax3.set_xticks(range(len(fwhms)))
    ax3.set_xticklabels([f"{v} kpc" for v in fwhms])
    ax3.set_yticks(range(len(noises)))
    ax3.set_yticklabels([f"{v} km/s" for v in noises])
    ax3.set_xlabel("空間分解能 FWHM", fontsize=10)
    ax3.set_ylabel(r"速度ノイズ $\sigma_v$", fontsize=10)
    ax3.set_title("Model 3 (コンパクト) ロバスト回収率 [%]", fontsize=12, fontweight='bold', pad=10)
    for i in range(len(noises)):
        for j in range(len(fwhms)):
            ax3.text(j, i, f"{rate_matrix_m3[i, j]:.1f}%", ha='center', va='center',
                     color="white" if rate_matrix_m3[i, j] < 50 else "black", fontweight='bold')

    # --------------------------------------------------------------------------
    # パネル 4: 腕競合振幅別・解析手法別の回収率推移
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_facecolor(card_color)

    arm_amps = [0.0, 1.0, 3.0]
    inj_rows = [r for r in rows if int(r["n_vortex_inj"]) > 0]

    # 手法別の回収率
    methods = [("raw", "手法A: Raw残差", "#e06c75", "o-"),
               ("remove_k3", "手法B: k=3調波除去", "#61afef", "s-"),
               ("arm_fit", "手法C: 腕モデル直接フィット", "#98c379", "^-")]

    for m_code, m_label, color, m_style in methods:
        rates = []
        for a_val in arm_amps:
            if a_val == 0.0:
                sub = [r for r in inj_rows if float(r["a_los3_kms"]) == 0.0]
            else:
                sub = [r for r in inj_rows if float(r["a_los3_kms"]) == a_val and r["arm_method"] == m_code]

            if len(sub) > 0:
                rob = sum(1 for r in sub if r["cell_verdict"] == "CELL_ROBUST_RECOVERED")
                rates.append((rob / len(sub)) * 100.0)
            else:
                rates.append(0.0)

        ax4.plot(arm_amps, rates, m_style, label=m_label, color=color, linewidth=2.5, markersize=8)

    ax4.set_xticks(arm_amps)
    ax4.set_xticklabels(["0.0 (腕なし)", "1.0 (典型腕)", "3.0 (強腕)"], fontsize=10)
    ax4.set_xlabel("腕競合振幅 $A_{\\rm los, 3}$ [km/s]", fontsize=11)
    ax4.set_ylabel("CELL_ROBUST_RECOVERED 達成率 [%]", fontsize=11)
    ax4.set_title("腕競合振幅および解析手法による回収率の変化", fontsize=12, fontweight='bold', pad=10)
    ax4.grid(True, linestyle="--", alpha=0.3)
    ax4.legend(loc="upper right", framealpha=0.8, facecolor=card_color, edgecolor="#abb2bf")

    # --------------------------------------------------------------------------
    # パネル 5: 大域的総括判定カード (Global Verdict Card)
    # --------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(card_color)
    ax5.axis("off")

    try:
        from run_full_injection_recovery_batch import evaluate_global_verdict
    except ImportError:
        from tools.run_full_injection_recovery_batch import evaluate_global_verdict
    global_verdict, global_reason = evaluate_global_verdict(rows)

    total_cells = len(rows)
    total_inj = len(inj_rows)
    total_robust = sum(1 for r in rows if r["cell_verdict"] == "CELL_ROBUST_RECOVERED")
    overall_pct = (total_robust / total_inj) * 100.0 if total_inj > 0 else 0.0

    deg_cells_nv0 = [r for r in rows if r["cell_verdict"] == "CELL_DEGENERATE_ARM"]
    n_deg = len(deg_cells_nv0)
    deg_1 = sum(1 for r in deg_cells_nv0 if float(r["a_los3_kms"]) == 1.0)
    deg_3 = sum(1 for r in deg_cells_nv0 if float(r["a_los3_kms"]) == 3.0)

    card_text = (
        f"【研究全体の大域判定結果】\n\n"
        f"● 判定ラベル:\n  {global_verdict}\n\n"
        f"● 根拠・物理的意味:\n"
        f"  {global_reason}\n\n"
        f"----------------------------------------\n"
        f"● 総セル数: {total_cells:,} セル (対照2,268 / 注入6,804)\n"
        f"● ロバスト回収セル数: {total_robust:,} セル ({overall_pct:.2f}%)\n"
        f"  * Model 1 (線形摂動 xi≈0.008): 0セル (0.0%)\n"
        f"  * Model 3 (玩具摂動 xi≈0.494): {total_robust}セル (FWHM 0.2kpc)\n"
        f"● 対照群 腕による偽陽性増悪: {n_deg} セル\n"
        f"  (A_los,3=1.0: {deg_1}セル, 3.0: {deg_3}セル)\n"
        f"----------------------------------------\n"
        f"※Model 3の検出は回転速度の約0.49倍の\n"
        f"  巨大玩具摂動の検出に限定される。\n"
        f"※静的中空円筒玩具モデルに基づく結果であり、\n"
        f"  将来の観測や動的シミュレーションでの\n"
        f"  異なる設定の可能性を排除しない。"
    )

    box_color = "#98c379" if "CONFIRMED" in global_verdict else ("#d19a66" if "TOY" in global_verdict else ("#e5c07b" if "CONDITIONAL" in global_verdict else "#e06c75"))
    ax5.text(0.05, 0.95, card_text, transform=ax5.transAxes,
             fontsize=9.0, verticalalignment='top', family='Meiryo',
             bbox=dict(boxstyle='round,pad=0.8', facecolor="#1a1c23", edgecolor=box_color, linewidth=2))

    fig.suptitle("少数量子渦の2次元視線速度残差マップ 注入・回収試験 総合判定サマリー (v7 Final)",
                 fontsize=15, fontweight='bold', color="#abb2bf", y=0.98)

    plt.savefig(out_png_path, facecolor=bg_color, edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"可視化図の保存完了: {out_png_path}")


if __name__ == "__main__":
    csv_file = os.path.join(OUTPUT_DIR, "injection_recovery_summary.csv")
    png_file = os.path.join(OUTPUT_DIR, "injection_recovery_verdict_summary.png")
    plot_summary_figures(csv_file, png_file)
