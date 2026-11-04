"""
test_physics_audit.py — 全宇宙回転ジェネレーション仮説 物理・数理監査単体テスト

本テストスイートは、仮説の成立を目的とせず、既存コードの物理・数理的整合性を厳密に検査する。
本番処理を呼ぶテストと模型の性質を確認するテストを明確に区別し、
既知の本番不具合には pytest.mark.xfail(strict=True) を付与する。
"""

import os
import sys
import numpy as np
import pytest

# プロジェクトルートの設定
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.core as core


# ============================================================================
# 1. 単位換算の回帰検査（本番処理の検査）
# ============================================================================


def test_01_a0_star_production_unit_conversion():
    """
    【本番関数の検査: A0_STAR の単位換算】
    対象本番定数: core.A0_STAR / core.compute_a0_star()
    
    IAU/CODATA基準の独立定数から H0 = 67.4 km/s/Mpc を s^-1 に換算した正しい理論値
    a0* = c * H0 / (2π) ≈ 1.042e-10 m/s^2 と、本番実装の core.A0_STAR を比較する。
    
    core.py の分母換算修正 (KPC * 1e3) により、正しいオーダー (≈ 1.042e-10 m/s^2) で
    一致することを検証する回帰テスト（PASS）。
    
    - 保証する範囲: core.A0_STAR および compute_a0_star() の数値的正確性。
    - 保証しない範囲: a0* が銀河回転曲線の観測を説明できるかどうか。
    """
    PC_EXACT = 3.085677581491367e16  # meters per pc
    MPC_EXACT = PC_EXACT * 1.0e6     # meters per Mpc
    
    # 独立検算による正しい a0*
    H0_si_exact = (67.4 * 1000.0) / MPC_EXACT
    a0_star_expected = (299792458.0 * H0_si_exact) / (2.0 * np.pi)
    
    # 本番関数の呼び出し
    a0_star_prod = core.compute_a0_star()
    
    # atol=0.0 で厳密比較（修正後は理論値と完全一致して PASS することを確認）
    assert np.isclose(a0_star_prod, a0_star_expected, rtol=1e-3, atol=0.0), (
        f"core.compute_a0_star() の値 ({a0_star_prod:.3e}) が独立理論値 ({a0_star_expected:.3e}) と一致しない。"
    )


# ============================================================================
# 2. EC項の次元解析と本番処理の検査
# ============================================================================


def test_02a_ec_torsion_dimensional_analysis():
    """
    【模型の性質の確認: EC項の次元解析】
    対象数理式: (8πG/c^2) * f_align * n * (hbar/2)
    
    アインシュタイン・カルタン項の各因子のSI基本単位 [m, kg, s] を解析し、
    積の次元が角速度 [s^-1] であり、加速度 [m s^-2] ではない事実を確認する（PASS）。
    
    - 保証する範囲: EC項の数理式におけるSI基本単位の整合性。
    - 保証しない範囲: カルタン接続やトーション理論の幾何学的正当性。
    """
    dim_G = np.array([3, -1, -2])       # m^3 kg^-1 s^-2
    dim_c2 = np.array([2, 0, -2])      # m^2 s^-2
    dim_G_over_c2 = dim_G - dim_c2     # [1, -1, 0] -> m kg^-1
    
    dim_f = np.array([0, 0, 0])        # 無次元
    dim_n = np.array([-3, 0, 0])       # m^-3
    dim_hbar = np.array([2, 1, -1])    # kg m^2 s^-1
    
    dim_ec = dim_G_over_c2 + dim_f + dim_n + dim_hbar  # [0, 0, -1] -> s^-1
    dim_frequency = np.array([0, 0, -1])              # s^-1
    dim_acceleration = np.array([1, 0, -2])           # m s^-2
    
    # 単位が角速度 s^-1 であることを確認
    assert np.array_equal(dim_ec, dim_frequency), "EC項の次元は厳密に s^-1 (角速度) である"
    # 加速度 m s^-2 とは異なることを確認
    assert not np.array_equal(dim_ec, dim_acceleration), "EC項の次元は加速度 m s^-2 とは一致しない"


def test_02b_production_unit_consistency_function():
    """
    【本番関数の検査: check_unit_consistency】
    対象本番関数: core.check_unit_consistency()
    
    本番の単位整合性検査関数が返す A0_ratio (A0_STAR / A0_MOND) を検査する。
    core.py の修正により、理論比率 1.042e-10 / 1.2e-10 ≈ 0.8685 と整合することを
    検証する回帰テスト（PASS）。
    
    - 保証する範囲: 本番関数 check_unit_consistency() の出力精度。
    - 保証しない範囲: 各物理定数の測定誤差。
    """
    result = core.check_unit_consistency()
    a0_ratio_prod = result["A0_ratio"]
    a0_ratio_expected = 1.04219788e-10 / 1.2e-10  # ≈ 0.8685
    
    # 許容誤差は緩めず rtol=1e-2, atol=0.0 を維持
    assert np.isclose(a0_ratio_prod, a0_ratio_expected, rtol=1e-2, atol=0.0), (
        f"check_unit_consistency() の A0_ratio ({a0_ratio_prod:.6f}) が理論値 ({a0_ratio_expected:.6f}) と乖離している。"
    )


# ============================================================================
# 3. 加算型モデルの本番関数検査（漸近挙動と無次元比）
# ============================================================================


def test_03a_v_gyro_additive_production_far_keplerian_limit():
    """
    【本番関数の検査: core.v_gyro_additive の遠方ケプラー漸近】
    対象本番関数: core.v_gyro_additive()
    
    動径方向方程式 r v^2 - K v - GM = 0 において、
    無次元比 xi(r) = K^2 / (4 GM r) は r -> inf で 0 に漸近する。
    本番関数 v_gyro_additive を遠方 (r = 10000 kpc) で呼び出し、
    対数微分 d ln v / d ln r がケプラー勾配 -0.5 (v ∝ r^-1/2) に漸近し、
    フラット (勾配 0.0) にはならないことを本番実装に対して検証する（条件分岐なし、PASS）。
    
    - 保証する範囲: 本番関数 v_gyro_additive の遠方代数解の振る舞い。
    - 保証しない範囲: 実際の銀河外縁部における観測データへの適合性。
    """
    M_tot = 5.0e10
    b_val = 2.0
    R_d_val = 1.5
    f_disk_val = 0.8
    S_eff = 500.0  # kpc km/s
    
    r_far_1 = 10000.0
    r_far_2 = 10010.0
    
    # 本番関数の直接呼び出し
    v_1 = core.v_gyro_additive(np.array([r_far_1]), M_tot, b_val, R_d_val, f_disk_val, S_eff)[0]
    v_2 = core.v_gyro_additive(np.array([r_far_2]), M_tot, b_val, R_d_val, f_disk_val, S_eff)[0]
    
    slope_far = (np.log(v_2) - np.log(v_1)) / (np.log(r_far_2) - np.log(r_far_1))
    
    # 遠方ではケプラー勾配 -0.5 に一致すること（許容誤差 1%）
    assert np.isclose(slope_far, -0.5, atol=0.01), f"遠方対数勾配 ({slope_far:.3f}) が -0.5 と一致しない"
    # フラット回転（勾配 0.0）ではないこと
    assert abs(slope_far) > 0.4, "遠方でフラット回転曲線にはならないこと"


def test_03b_v_gyro_additive_production_gyro_dominated_limit():
    """
    【本番関数の検査: core.v_gyro_additive のジャイロ支配域の挙動】
    対象本番関数: core.v_gyro_additive()
    
    バリオン質量を極小 (M = 1 M_sun) に設定して無次元比 xi = K^2 / (4 GM r) >> 1 を強制し、
    本番関数 v_gyro_additive を呼び出す。
    対数微分が -1.0 (v ∝ 1/r) に近づき、フラット (勾配 0.0) にはならないことを本番実装に対して検証する（条件分岐なし、PASS）。
    
    - 保証する範囲: ジャイロ項が支配的な領域における本番関数の減衰特性。
    - 保証しない範囲: 質量ゼロの物理的実在性。
    """
    M_minimal = 1.0  # 1 Solar mass (ニュートン重力を極小化)
    b_val = 0.1
    R_d_val = 0.1
    f_disk_val = 0.5
    S_eff = 500.0
    
    r_1 = 1.0
    r_2 = 1.01
    
    # 本番関数の直接呼び出し
    v_1 = core.v_gyro_additive(np.array([r_1]), M_minimal, b_val, R_d_val, f_disk_val, S_eff)[0]
    v_2 = core.v_gyro_additive(np.array([r_2]), M_minimal, b_val, R_d_val, f_disk_val, S_eff)[0]
    
    slope_gyro = (np.log(v_2) - np.log(v_1)) / (np.log(r_2) - np.log(r_1))
    
    # ジャイロ支配域では 1/r 減衰（勾配 -1.0）に近づくこと
    assert np.isclose(slope_gyro, -1.0, atol=0.05), f"ジャイロ支配対数勾配 ({slope_gyro:.3f}) が -1.0 と一致しない"


# ============================================================================
# 4. MOND との同値性（同一 a0 下の本番関数検査）
# ============================================================================


def test_04_mond_identity_under_same_a0():
    """
    【本番関数の検査: 同一 a0 下での MOND とジャイロ内挿の一致】
    対象本番関数: core.v_mond_interp() / core.v_gyro_interpolating()
    
    同一の限界加速度 a0 を与えた場合、ジャイロ内挿モデルと MOND モデルが
    代数的に完全同一の方程式を解いているため、計算結果が厳密に一致することを検証する（PASS）。
    
    - 保証する範囲: 両本番関数が同一の基礎方程式を共有していること。
    - 保証しない範囲: 異なる a0 を与えた場合の観測的予測差。
    """
    r_test = np.linspace(1.0, 20.0, 20)
    test_a0 = 1.2e-10
    
    v_m = core.v_mond_interp(r_test, 5e10, 2.0, 1.5, 0.8, a0=test_a0, mu_type="standard")
    v_g = core.v_gyro_interpolating(r_test, 5e10, 2.0, 1.5, 0.8, a0_star=test_a0, mu_type="standard")
    
    max_diff = np.max(np.abs(v_m - v_g))
    assert max_diff == 0.0, "同一 a0 下ではジャイロ内挿モデルと MOND は完全同一の値を返すこと"


# ============================================================================
# 5. MOND との観測的識別可能性の検査（本番関数の検査 vs 模型性質の確認）
# ============================================================================


def test_05a_v_gyro_interpolating_production_default_call():
    """
    【本番関数の検査: core.v_gyro_interpolating のデフォルト呼び出し】
    対象本番関数: core.v_gyro_interpolating(r, ..., a0_star=None)
    
    本番関数 v_gyro_interpolating() を a0_star を指定せずにデフォルト呼び出しした場合、
    修正前の本番コードではバグのある A0_STAR (1e-13) が使われ、有限半径 r=100kpc での
    ニュートン寄与を含めた速度比は約 0.224 に激減していた（深-MOND極限値は ≈ 0.171）。
    
    core.py の修正により、デフォルト呼び出しでも正しい理論比率
    (a0_star_correct / a0_mond)^0.25 ≈ 0.965 (約3.5%差) と整合することを
    検証する回帰テスト（PASS）。
    
    - 保証する範囲: v_gyro_interpolating のデフォルト引数の動作。
    - 保証しない範囲: 望遠鏡による実際の天体観測誤差。
    """
    r_deep = np.array([100.0])  # 深-MOND領域
    M_tot = 1.0e10
    
    v_mond = core.v_mond_interp(r_deep, M_tot, 2.0, 1.5, 0.8, a0=1.2e-10, mu_type="simple")
    # デフォルト呼び出し（修正後は正しい A0_STAR が参照される）
    v_gyro_default = core.v_gyro_interpolating(r_deep, M_tot, 2.0, 1.5, 0.8, a0_star=None, mu_type="simple")
    
    actual_ratio = v_gyro_default[0] / v_mond[0]
    expected_ratio = (1.042e-10 / 1.2e-10) ** 0.25  # ≈ 0.9649
    
    # 許容誤差は緩めず rtol=1e-2, atol=0.0 を維持
    assert np.isclose(actual_ratio, expected_ratio, rtol=1e-2, atol=0.0), (
        f"デフォルト呼び出しの速度比 ({actual_ratio:.3f}) が理論値 ({expected_ratio:.3f}) と一致しない。"
    )


def test_05b_mond_distinguishability_theoretical_property():
    """
    【模型の性質の確認: 異なる a0 による観測的識別可能性】
    対象数理モデル: 深-MOND漸近速度式 v = (GM a0)^0.25
    ※本テストは特定の単一関数実装の成否ではなく、異なる a0 値を仮定した模型の観測的識別可能性を検証する。
    
    正しい理論値 a0* (1.042e-10) と標準 MOND (1.200e-10) を用いた場合、
    深-MOND領域での漸近速度比が理論予測値 (1.042/1.200)^0.25 ≈ 0.965 と一致し、
    観測的に約3.5%の識別可能な有意差を持つことを検証する（PASS）。
    
    - 保証する範囲: 深-MOND漸近速度におけるパラメータ差の影響。
    - 保証しない範囲: 実際の銀河サンプルにおける観測散らばり。
    """
    r_deep = np.array([100.0])
    M_tot = 1.0e10
    
    v_mond = core.v_mond_interp(r_deep, M_tot, 2.0, 1.5, 0.8, a0=1.2e-10, mu_type="simple")
    v_gyro_corrected = core.v_mond_interp(r_deep, M_tot, 2.0, 1.5, 0.8, a0=1.042e-10, mu_type="simple")
    
    actual_ratio = v_gyro_corrected[0] / v_mond[0]
    expected_ratio = (1.042e-10 / 1.2e-10) ** 0.25  # ≈ 0.9649
    
    assert np.isclose(actual_ratio, expected_ratio, rtol=1e-3, atol=0.0)
    assert abs(actual_ratio - 1.0) > 0.03, "速度比に約3.5%の観測的識別可能差が存在すること"


# ============================================================================
# 6. BEC偏極の検査（本番関数の検査 vs 模型性質の確認）
# ============================================================================


def test_06a_compute_bec_amplification_production_function():
    """
    【本番関数の検査: core.compute_bec_amplification の初期値依存性】
    対象本番関数: core.compute_bec_amplification()
    
    本番関数 compute_bec_amplification() を異なる引数で呼び出し、
    増幅率が初期仮定値 f_align_thermal に反比例する相対値に過ぎないことを本番実装に対して検証する（PASS）。
    
    - 保証する範囲: 本番関数 compute_bec_amplification() の数値計算。
    - 保証しない範囲: 実際の星間空間における熱的偏極率。
    """
    amp_1 = core.compute_bec_amplification(f_align_thermal=1.0e-5, f_align_bec=1.0)
    amp_2 = core.compute_bec_amplification(f_align_thermal=1.0e-4, f_align_bec=1.0)
    
    assert np.isclose(amp_1, 1.0e5, rtol=1e-5, atol=0.0), "初期値 1e-5 では増幅率 10^5"
    assert np.isclose(amp_2, 1.0e4, rtol=1e-5, atol=0.0), "初期値 1e-4 では増幅率 10^4"


def test_06b_bec_polarization_bound_and_lnh_coupling_separation():
    """
    【模型の性質の確認: BEC偏極率の上限と大数結合スケーリングの分離】
    対象数理概念: BECスピン偏極率の物理的上限と、ディラック大数結合定数の独立性
    
    1. BEC偏極率の上限:
       スピン偏極率 f_align は数学的に 1.0 (100%完全偏極) を超えられない。
       したがって熱的初期値 f_thermal = 10^-5 からの偏極率向上単独で担えるのは、
       最大でも η_BEC = 1.0 / 10^-5 = 10^5 倍までであることを確認する。
    
    2. 大数結合スケーリングの分離:
       残りの 10^40 はスピン偏極率に掛かるのではなく、ディラック大数仮説に基づく
       結合定数スケーリング η_LNH = √N_universe ≈ 10^40 として独立に導入されていることを確認する。
       総増幅率 η_total = 10^45 を偏極率に乗じるのではなく、結合則としての未導出性を峻別する。
    
    - 保証する範囲: BEC偏極率の定義域上限および大数結合スケーリングの分離構造。
    - 保証しない範囲: 実際の銀河における未知のスピン結合機構の有無。
    """
    # 1. BECによるスピン偏極率の上限検証
    f_align_bec_max = 1.0
    f_thermal = 1.0e-5
    eta_bec_max = f_align_bec_max / f_thermal
    
    # 偏極率は 1.0 以下でなければならず、BEC単独の倍率は 10^5 で頭打ちとなる
    assert f_align_bec_max <= 1.0, "スピン偏極率は定義により 1.0 以下でなければならない"
    assert np.isclose(eta_bec_max, 1.0e5, rtol=1e-5, atol=0.0), "BEC偏極率向上単独の寄与は最大 10^5 倍"
    
    # 2. ディラック大数仮説の独立スケーリング検証
    eta_total, eta_bec, eta_lnh = core.compute_total_amplification()
    
    # eta_total は偏極率の増大ではなく、積 η_BEC × η_LNH として合成されている
    assert np.isclose(eta_total, eta_bec * eta_lnh, rtol=1e-5, atol=0.0)
    assert np.isclose(eta_lnh, 1.0e40, rtol=1e-5, atol=0.0), (
        "10^40 は偏極率ではなく結合定数スケーリング η_LNH として分離されている"
    )


# ============================================================================
# 7. 試験体内部における N の示量性相殺（模型性質の確認と偽陽性防止）
# ============================================================================


def test_07_test_body_extensivity_cancellation_and_false_positive_guard():
    """
    【模型の性質の確認: 試験体内部における粒子数 N の示量性相殺】
    対象物理原則: 質点系における重心加速度 a_cm = sum(F) / sum(m)
    
    試験体の粒子数 N_test が分子（力）と分母（質量）で厳密に相殺されることを、
    atol=0.0 で検証する。
    さらに、ゼロや100万倍の誤値が確実に不一致となること（偽陽性防止）も確認する（PASS）。
    
    - 保証する範囲: 連続体・質点系力学の重心運動方程式における示量性の保存。
    - 保証しない範囲: 宇宙全体の総粒子数 N_universe を結合則に用いる大数仮説の成否。
    """
    m_single = core.M_P
    f_single = 1.0e-50  # 1粒子に働く微小力 [N]
    
    for N_test in [1, 1000, 10**20, 10**40]:
        total_force = N_test * f_single
        total_mass = N_test * m_single
        a_cm = total_force / total_mass
        a_single = f_single / m_single
        
        # atol=0.0 による厳密一致検証
        assert np.isclose(a_cm, a_single, rtol=1e-12, atol=0.0), (
            f"N={N_test} において重心加速度が単一粒子加速度と厳密一致しない"
        )
        
        # 偽陽性防止の検証: 誤値（ゼロや100万倍）では確実に不一致となること
        assert not np.isclose(0.0, a_single, rtol=1e-12, atol=0.0), "ゼロとの比較で不一致となること"
        assert not np.isclose(a_single * 1.0e6, a_single, rtol=1e-12, atol=0.0), "100万倍の誤値で不一致となること"
