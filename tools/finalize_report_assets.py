# -*- coding: utf-8 -*-
"""
tools/finalize_report_assets.py
最終承認用 資産最終確定スクリプト:
1. final_report_numbers.json の内容を最新の数理・検証仕様に完全同期
   - 通常格子および超サンプリングの内挿値 (1.21 および 0.91)
   - 窓打ち切り: エネルギー比 (0.878-0.892) と SNR比 (0.937-0.944) の区別
   - Model 1 テンプレート実測値 (13.8%)
   - 超サンプリング基準の全54条件サマリー
2. SHA256SUMS.txt の完全生成:
   - output/ の全成果物
   - tools/ の全解析スクリプト
   - src/ の全パッケージファイル
   - docs/INJECTION_RECOVERY_TEST_SPEC_V7.md
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import glob
import hashlib

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
DOCS_DIR = os.path.join(PROJECT_DIR, "docs")
OUT_DIR = os.path.join(PROJECT_DIR, "output")

def get_sha256(fpath):
    if not os.path.exists(fpath):
        return None
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def update_final_report_json():
    json_path = os.path.join(OUT_DIR, "final_report_numbers.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Update interpolation notes
    data["r_b_interpolation"] = {
        "standard_grid": {
            "data_points": [
                {"inc": 30.0, "r_b": 1.010, "rec_rate": 0.382},
                {"inc": 70.0, "r_b": 1.203, "rec_rate": 0.482},
                {"inc": 50.0, "r_b": 1.343, "rec_rate": 0.722}
            ],
            "interpolated_r_b_50pct": 1.21,
            "note": "通常格子基準で50%回収に必要なR_Bは約1.21"
        },
        "oversampled_grid": {
            "data_points": [
                {"inc": 30.0, "r_b": 0.795, "rec_rate": 0.324},
                {"inc": 70.0, "r_b": 0.831, "rec_rate": 0.382},
                {"inc": 50.0, "r_b": 1.023, "rec_rate": 0.688}
            ],
            "interpolated_r_b_50pct": 0.91,
            "note": "超サンプリング基準で50%回収に必要なR_Bは約0.91"
        },
        "boundary_verdict_interpretation": "境界帯 (0.85 <= R < 1.15) が50%の位置 (0.91および1.21) を挟んでおり、3点の実測と矛盾しない"
    }

    # 2. Window cutoff energy vs snr ratio
    data["window_cutoff_details"] = {
        "window_size_pixels": "41x41",
        "energy_ratio_range": [0.878, 0.892],
        "snr_ratio_range": [0.937, 0.944],
        "per_inclination": {
            "30_deg": {"energy_ratio": 0.878, "snr_ratio": 0.937},
            "50_deg": {"energy_ratio": 0.888, "snr_ratio": 0.942},
            "70_deg": {"energy_ratio": 0.892, "snr_ratio": 0.944}
        },
        "note": "窓打ち切りによる信号制限はエネルギー比で約0.88-0.89、実効SNR比（平方根）で約0.94"
    }

    # 3. Model 1 TMF confirmation
    data["model1_template_detector_result"]["verdict"] = "UNDETECTED (実測回収率13.8%で確定、50%に届かない)"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"final_report_numbers.json を最新更新完了")


def generate_full_sha256sums():
    sums_lines = []

    # 1. Output files
    output_files = [
        "v2_h0_calibration.csv",
        ".v2_snr_thresholds.json",
        "v2_cell_predictions_preregistered.csv",
        "v2_injection_recovery_summary.csv",
        "v2_exact_optimal_snr_54strata.csv",
        "v2_audited_optimal_snr_54strata.csv",
        "v2_method_b_snr_54strata.csv",
        "v2_method_b_snr_54strata_oversampled.csv",
        "v2_oversampled_template_detector_results.json",
        "v2_positive_control_oversampled_result.json",
        "final_report_numbers.json",
        "cell_predictions_preregistered.csv",
        "h0_stratum_calibration.csv",
        "injection_recovery_summary.csv"
    ]

    print("=" * 100)
    print("SHA256SUMS.txt の完全生成 (成果物 + tools/*.py + src/* + docs/*)")
    print("=" * 100)

    for fname in output_files:
        fpath = os.path.join(OUT_DIR, fname)
        sha = get_sha256(fpath)
        if sha:
            sums_lines.append(f"{sha}  output/{fname}")

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
    print(f"全 {len(sums_lines)} ファイルのチェックサムを SHA256SUMS.txt に書き出しました: {sums_path}")


if __name__ == "__main__":
    update_final_report_json()
    generate_full_sha256sums()
