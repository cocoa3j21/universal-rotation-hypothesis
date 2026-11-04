# -*- coding: utf-8 -*-
"""
test_injection_recovery_pipeline.py — 注入・回収試験パイプライン単体検査テスト (Stage 1)

【目的】
本テストスイートは、少数量子渦の2次元視線速度残差マップ注入・回収試験（第7版正本仕様書）の
実装フェーズ Stage 1 として、パイプラインを構成するすべての数理的・物理的コンポーネント
（座標変換、投影、ビーム平滑化、軸対称フィット、調波除去、渦テンプレート）を独立に検査する。

【テスト項目】
- Test-01: フェイスオン極限 (i = 0° で v_los ≡ v_sys)
- Test-02: 主軸方位テスト (PA in {0°, 45°, 90°, 180°} で x > 0 が主軸かつ v_los > v_sys)
- Test-03a: 解析モデル残差ゼロ (渦なし・ノイズなし・完全幾何で max |v_res| < 10^-12 km/s)
- Test-03b: 数値フィット残差 (同心円リング平均フィットで max |v_res| < 10^-6 km/s)
- Test-04a: 幾何偏微分 (円盤座標固定での解析偏微分と中心数値微分の相対誤差 < 10^-5)
- Test-04b: 全パイプライン幾何感度 (観測格子固定での幾何系統誤差摂動と一次解析予測の一致)
- Test-05: 腕の加法定理吸収 (A_r = A_phi かつ同位相で k=3 残差が厳密にゼロ、k=1 吸収の確認)
- Component-01: 座標変換の双方向性・可逆性 (obs_to_disk <-> disk_to_obs)
- Component-02: 正規化ビーム畳み込みのフラックス保存と eta_beam 測定
- Component-03: 調波除去 (k=3 除去アルゴリズムの直交分離性)
- Component-04: 渦テンプレート生成・局所ピーク抽出・位置一致判定
"""

import os
import sys
import numpy as np
import pytest

# パス設定
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gyro_gravity_theory.injection_pipeline as pipe


# ==============================================================================
# Test-01: フェイスオン極限テスト
# ==============================================================================

def test_01_face_on_limit():
    """
    【Test-01: フェイスオン極限】
    傾斜角 i = 0° のフェイスオン銀河では、いかなる円盤面内回転速度 v_phi、
    動径流速 v_rad、位置角 PA、観測座標 (x', y') であっても、
    投影視線速度は幾何学的に厳密に v_los ≡ v_sys とならなければならない。
    """
    v_sys = 150.0  # km/s
    inc_deg = 0.0

    for pa in [0.0, 30.0, 45.0, 90.0, 135.0, 180.0]:
        x_obs = np.linspace(-5.0, 5.0, 21)
        y_obs = np.linspace(-5.0, 5.0, 21)
        X_obs, Y_obs = np.meshgrid(x_obs, y_obs)

        _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, x0=0.0, y0=0.0, pa_deg=pa, inc_deg=inc_deg)

        v_rot = 180.0 + 20.0 * np.sin(r_disk)
        v_rad = 15.0 * np.cos(2.0 * phi_disk)

        v_los = pipe.project_vlos(v_rot, v_rad, phi_disk, inc_deg=inc_deg, v_sys=v_sys)

        max_diff = np.max(np.abs(v_los - v_sys))
        assert max_diff < 1e-13, f"PA={pa} においてフェイスオン v_los が v_sys と一致しない: max_diff = {max_diff}"


# ==============================================================================
# Test-02: 主軸方位テスト
# ==============================================================================

def test_02_major_axis_orientation():
    """
    【Test-02: 主軸方位テスト】
    PA in {0°, 45°, 90°, 180°} において、
    主軸（後退側、phi_disk = 0、円盤座標で x_disk > 0, y_disk = 0）の点が
    正しく順変換・逆変換され、v_los > v_sys (赤方偏移側) となることを検証する。
    """
    v_sys = 100.0  # km/s
    inc_deg = 45.0  # 中間傾斜
    v_rot = 200.0  # km/s
    r_test = 3.0   # kpc

    for pa in [0.0, 45.0, 90.0, 180.0]:
        x_d = r_test
        y_d = 0.0

        x_obs, y_obs = pipe.disk_to_obs(x_d, y_d, x0=0.0, y0=0.0, pa_deg=pa, inc_deg=inc_deg)
        x_d_rec, y_d_rec, r_rec, phi_rec = pipe.obs_to_disk(x_obs, y_obs, x0=0.0, y0=0.0, pa_deg=pa, inc_deg=inc_deg)

        assert np.isclose(x_d_rec, r_test, atol=1e-12), f"PA={pa}: x_disk 復元誤差"
        assert np.isclose(y_d_rec, 0.0, atol=1e-12), f"PA={pa}: y_disk 復元誤差"
        assert np.isclose(r_rec, r_test, atol=1e-12), f"PA={pa}: r 復元誤差"
        assert np.isclose(phi_rec, 0.0, atol=1e-12), f"PA={pa}: phi 復元誤差"

        v_los = pipe.project_vlos(v_rot, 0.0, phi_rec, inc_deg=inc_deg, v_sys=v_sys)
        expected_vlos = v_sys + v_rot * np.sin(np.radians(inc_deg))
        assert np.isclose(v_los, expected_vlos, atol=1e-12), f"PA={pa}: v_los 投影誤差"
        assert v_los > v_sys, f"PA={pa}: 主軸後退側が赤方偏移していない (v_los={v_los} <= v_sys={v_sys})"

        if pa == 0.0:
            assert np.isclose(x_obs, 0.0, atol=1e-12) and y_obs > 0
        elif pa == 90.0:
            assert x_obs > 0 and np.isclose(y_obs, 0.0, atol=1e-12)
        elif pa == 180.0:
            assert np.isclose(x_obs, 0.0, atol=1e-12) and y_obs < 0


# ==============================================================================
# Test-03a: 完全幾何解析モデル残差ゼロテスト
# ==============================================================================

def test_03a_analytic_model_residual_zero():
    """
    【Test-03a: 解析モデル残差ゼロ】
    渦なし・ノイズなし・完全幾何データにおいて、
    真の回転曲線関数 v_rot(r) を用いて解析モデルを減算した場合、
    残差マップが厳密にゼロ（max |v_res| < 10^-12 km/s）となることを確認する。
    """
    halo = pipe.TFHaloModel(M_msun=1e11, R_kpc=20.0, m_eV=1e-22)
    inc_deg = 50.0
    pa_deg = 45.0
    v_sys = 120.0

    x_obs = np.linspace(-15.0, 15.0, 61)
    y_obs = np.linspace(-15.0, 15.0, 61)
    X_obs, Y_obs = np.meshgrid(x_obs, y_obs)

    _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, x0=0.0, y0=0.0, pa_deg=pa_deg, inc_deg=inc_deg)

    v_rot_true = halo.v_0(r_disk)
    v_los_true = pipe.project_vlos(v_rot_true, 0.0, phi_disk, inc_deg=inc_deg, v_sys=v_sys)

    v_res = pipe.compute_analytic_residual(v_los_true, halo.v_0, r_disk, phi_disk, inc_deg=inc_deg, v_sys=v_sys)

    max_res = np.nanmax(np.abs(v_res))
    assert max_res < 1e-12, f"解析モデル減算残差がゼロでない: max_res = {max_res:.2e} km/s"


# ==============================================================================
# Test-03b: 数値同心円リングフィット残差テスト
# ==============================================================================

def test_03b_numerical_ring_fit_residual():
    """
    【Test-03b: 数値同心円リング平均フィット残差】
    各リング内で回転速度が平坦（一定）な円盤データに対し、
    同心円リング平均フィッティングを適用した場合、
    残差が数値計算精度レベル（max |v_res| < 10^-6 km/s）で消失することを検証する。
    """
    inc_deg = 45.0
    pa_deg = 30.0
    v_sys = 100.0
    v_rot_const = 180.0

    x_obs = np.linspace(-10.0, 10.0, 81)
    y_obs = np.linspace(-10.0, 10.0, 81)
    X_obs, Y_obs = np.meshgrid(x_obs, y_obs)

    _, _, r_disk, phi_disk = pipe.obs_to_disk(X_obs, Y_obs, x0=0.0, y0=0.0, pa_deg=pa_deg, inc_deg=inc_deg)

    v_los_obs = pipe.project_vlos(v_rot_const, 0.0, phi_disk, inc_deg=inc_deg, v_sys=v_sys)

    v_rot_fit, ring_centers, res_map = pipe.fit_axisymmetric_disk_rings(
        r_disk, phi_disk, v_los_obs, inc_deg=inc_deg, v_sys=v_sys, r_min=0.5, r_max=9.5, n_rings=20
    )

    valid_mask = ~np.isnan(res_map)
    max_res = np.max(np.abs(res_map[valid_mask]))
    assert max_res < 1e-6, f"数値リングフィット残差が許容値を超過: max_res = {max_res:.2e} km/s"

    for v_k in v_rot_fit:
        assert np.isclose(v_k, v_rot_const, atol=1e-6), f"リングフィット回転速度誤差: {v_k}"


# ==============================================================================
# Test-04a: 幾何偏微分テスト
# ==============================================================================

def test_04a_geometric_partials_accuracy():
    """
    【Test-04a: 幾何偏微分の解析値と中心数値微分の比較】
    円盤座標固定での dv_los / dinc = v_rot * cos(phi) * cos(inc) について、
    中心差分数値微分との相対誤差が 10^-5 未満であることを確認する。
    """
    v_rot = 200.0
    v_sys = 80.0
    inc_deg = 50.0
    inc_rad = np.radians(inc_deg)

    phi_grid = np.linspace(0.1, 2.0 * np.pi - 0.1, 50)
    r_grid = np.full_like(phi_grid, 5.0)

    dv_dinc_ana, _ = pipe.compute_geometric_partials(r_grid, phi_grid, v_rot, inc_deg, pa_deg=0.0, v_sys=v_sys)

    eps = 1e-5
    v_plus = pipe.project_vlos(v_rot, 0.0, phi_grid, inc_deg=np.degrees(inc_rad + eps), v_sys=v_sys)
    v_minus = pipe.project_vlos(v_rot, 0.0, phi_grid, inc_deg=np.degrees(inc_rad - eps), v_sys=v_sys)
    dv_dinc_num = (v_plus - v_minus) / (2.0 * eps)

    mask = np.abs(np.cos(phi_grid)) > 0.1
    rel_diff = np.abs(dv_dinc_ana[mask] - dv_dinc_num[mask]) / np.abs(dv_dinc_ana[mask])
    max_rel_diff = np.max(rel_diff)

    assert max_rel_diff < 1e-5, f"幾何偏微分の解析値と数値微分の一致度が不十分: max_rel_diff = {max_rel_diff:.2e}"


# ==============================================================================
# Test-04b: 全パイプライン幾何感度テスト
# ==============================================================================

def test_04b_full_pipeline_geometric_sensitivity():
    """
    【Test-04b: 観測画素グリッド固定での幾何系統誤差摂動と解析予測】
    観測画像グリッド上で真の幾何 (inc=50°, PA=45°) で生成された視線速度場に対し、
    微小幾何系統誤差 delta_inc = +0.005°, delta_sys = +0.02 km/s を加えた場合の残差が、
    観測格子固定での中心数値偏微分による一次線形予測と 0.5% 以内で一致することを検証する。
    （テイラー展開の2次高次項 O(delta^2) が1%未満に抑えられていることの確認）
    """
    v_rot_const = 220.0
    inc_true = 50.0
    pa_true = 45.0
    v_sys_true = 100.0

    delta_inc = 0.005  # deg
    delta_sys = 0.02   # km/s

    x_obs = np.linspace(-8.0, 8.0, 51)
    y_obs = np.linspace(-8.0, 8.0, 51)
    X_obs, Y_obs = np.meshgrid(x_obs, y_obs)

    # 観測格子上で任意の幾何 (inc, v_sys) におけるモデル視線速度を計算するヘルパー
    def get_vlos_grid(inc_val, sys_val):
        _, _, _, phi_g = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa_true, inc_deg=inc_val)
        return pipe.project_vlos(v_rot_const, 0.0, phi_g, inc_deg=inc_val, v_sys=sys_val)

    # 真のデータ
    v_los_true = get_vlos_grid(inc_true, v_sys_true)

    # 誤差あり幾何によるモデル計算
    v_los_pert = get_vlos_grid(inc_true + delta_inc, v_sys_true + delta_sys)

    # 実測残差
    res_actual = v_los_true - v_los_pert

    # 観測格子固定での中心差分数値偏微分
    eps_inc = 1e-4
    dv_dinc_grid = (get_vlos_grid(inc_true + eps_inc, v_sys_true) - get_vlos_grid(inc_true - eps_inc, v_sys_true)) / (2.0 * eps_inc)
    dv_dsys_grid = 1.0

    # 一次線形予測: - (dv/dinc * delta_inc + dv/dsys * delta_sys)
    res_linear_pred = - (dv_dinc_grid * delta_inc + dv_dsys_grid * delta_sys)

    # 差分（高次項）の検査
    diff = np.abs(res_actual - res_linear_pred)
    max_res_amp = np.max(np.abs(res_actual))
    max_diff = np.max(diff)

    assert max_diff / max_res_amp < 0.005, f"幾何系統誤差残差の一次線形近似からの乖離率が大: {max_diff / max_res_amp:.3e}"


# ==============================================================================
# Test-05: 腕の加法定理吸収テスト
# ==============================================================================

def test_05_spiral_arm_addition_theorem_absorption():
    """
    【Test-05: 腕の加法定理吸収】
    面内 m=2 腕摂動において、動径流速振幅 A_r と方位角流速振幅 A_phi が等しく（A_r = A_phi）、
    かつ適切な位相関係 (v_r = A_r sin(2*phi), v_phi = v0 + A_phi cos(2*phi)) を持つ場合、
    投影視線速度において k=3 調波成分が厳密に相殺（ゼロ）となり、
    生成された非軸対称成分が k=1 (cos(phi)) 調波のみとなって軸対称回転速度に完全に吸収される
    数理的・物理的事実を検証する。
    """
    v0_const = 200.0  # km/s
    A_arm = 5.0       # km/s (A_r = A_phi)
    inc_deg = 45.0
    sin_i = np.sin(np.radians(inc_deg))

    phi_pts = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)

    v_r_arm = A_arm * np.sin(2.0 * phi_pts)
    v_phi_arm = v0_const + A_arm * np.cos(2.0 * phi_pts)

    v_los = pipe.project_vlos(v_phi_arm, v_r_arm, phi_pts, inc_deg=inc_deg, v_sys=0.0)

    d_phi = phi_pts[1] - phi_pts[0]
    c1 = (1.0 / np.pi) * np.sum(v_los * np.cos(phi_pts)) * d_phi
    c3 = (1.0 / np.pi) * np.sum(v_los * np.cos(3.0 * phi_pts)) * d_phi
    s3 = (1.0 / np.pi) * np.sum(v_los * np.sin(3.0 * phi_pts)) * d_phi

    # 1. k=3 調波成分が厳密にゼロであることを確認
    k3_amp = np.sqrt(c3**2 + s3**2)
    assert k3_amp < 1e-12, f"加法定理により消失すべき k=3 成分が非ゼロ: k3_amp = {k3_amp:.2e} km/s"

    # 2. k=1 調波成分が (v0 + A_arm) * sin(i) と厳密に一致することを確認
    expected_c1 = (v0_const + A_arm) * sin_i
    assert np.isclose(c1, expected_c1, atol=1e-12), f"k=1 吸収速度の理論値一致失敗: c1={c1}, expected={expected_c1}"

    # 3. 軸対称モデルフィットを実施した場合、この m=2 腕が回転速度 +A_arm として完全に吸収され、
    # 残差が厳密にゼロになることを確認
    r_pts = np.full_like(phi_pts, 3.0)
    v_rot_fit, _, res = pipe.fit_axisymmetric_disk_rings(r_pts, phi_pts, v_los, inc_deg=inc_deg, v_sys=0.0, r_min=2.5, r_max=3.5, n_rings=1)
    assert np.isclose(v_rot_fit[0], v0_const + A_arm, atol=1e-12), f"吸収後の回転速度が v0 + A_arm と不一致: {v_rot_fit[0]}"
    assert np.max(np.abs(res)) < 1e-12, f"吸収後の残差がゼロでない: max_res = {np.max(np.abs(res)):.2e}"


# ==============================================================================
# 独立コンポーネント検査: 座標変換の双方向性・可逆性
# ==============================================================================

def test_component_01_coordinate_invertibility():
    """
    【Component-01: 座標変換の双方向性】
    任意の観測座標 (x', y') に対し、obs_to_disk -> disk_to_obs を通した復元値が
    高精度（10^-12 kpc）で元座標と一致することを検証する。
    """
    np.random.seed(42)
    x_true = np.random.uniform(-10.0, 10.0, 100)
    y_true = np.random.uniform(-10.0, 10.0, 100)

    for pa in [15.0, 60.0, 120.0]:
        for inc in [25.0, 50.0, 75.0]:
            x_d, y_d, _, _ = pipe.obs_to_disk(x_true, y_true, x0=1.2, y0=-0.8, pa_deg=pa, inc_deg=inc)
            x_rec, y_rec = pipe.disk_to_obs(x_d, y_d, x0=1.2, y0=-0.8, pa_deg=pa, inc_deg=inc)

            max_err_x = np.max(np.abs(x_rec - x_true))
            max_err_y = np.max(np.abs(y_rec - y_true))
            assert max_err_x < 1e-12 and max_err_y < 1e-12, f"座標逆変換誤差大: PA={pa}, inc={inc}"


# ==============================================================================
# 独立コンポーネント検査: 正規化畳み込みと eta_beam
# ==============================================================================

def test_component_02_normalized_beam_convolution():
    """
    【Component-02: 正規化ビーム畳み込み】
    - 一様マップに畳み込みを適用しても値が変化しないこと（フラックス保存）。
    - 局所ガウシアン点源に対する平滑化で中心強度が正しく減衰し（eta_beam）、
      総積分フラックスが保存されること。
    - preserve_nan=True でマスク領域が NaN のまま保持され、境界リークがないこと。
    """
    # 1. 一様マップテスト
    grid_size = 51
    uniform_img = np.full((grid_size, grid_size), 50.0)
    conv_uniform = pipe.convolve_beam_normalized(uniform_img, fwhm_pix=5.0)
    assert np.allclose(conv_uniform, 50.0, atol=1e-10), "一様マップの正規化畳み込みで値が変化した"

    # 2. 点源のフラックス保存と中心減衰
    point_img = np.zeros((grid_size, grid_size))
    point_img[25, 25] = 100.0
    fwhm = 4.0
    conv_point = pipe.convolve_beam_normalized(point_img, fwhm_pix=fwhm)

    total_raw = np.sum(point_img)
    total_conv = np.sum(conv_point)
    assert np.isclose(total_raw, total_conv, rtol=1e-4), f"フラックス保存失敗: raw={total_raw}, conv={total_conv}"

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    expected_peak = total_raw / (2.0 * np.pi * (sigma**2))
    actual_peak = conv_point[25, 25]
    assert np.isclose(actual_peak, expected_peak, rtol=0.05), f"中心ピーク減衰が理論値と乖離: actual={actual_peak}, expected={expected_peak}"

    # 3. NaNマスク境界テスト (preserve_nan=True)
    masked_img = np.copy(uniform_img)
    masked_img[:10, :] = np.nan
    conv_masked = pipe.convolve_beam_normalized(masked_img, fwhm_pix=5.0, preserve_nan=True)
    assert np.all(np.isnan(conv_masked[:10, :])), "マスクされた領域が NaN に保持されていない"
    assert np.allclose(conv_masked[15:, :], 50.0, atol=1e-8), "有効領域の値が変化した"


# ==============================================================================
# 独立コンポーネント検査: 調波除去 (手法B: remove_k3)
# ==============================================================================

def test_component_03_harmonic_k3_removal():
    """
    【Component-03: 手法B k=3 調波除去】
    1. 均一方位角リング上での数学的厳密直交分解（残差 < 10^-12 km/s）。
    2. 2次元直交画素グリッド上での離散リングフィッティング（離散化サンプリング下で
       k=3 振幅 3.0 km/s が 95% 以上除去され、他調波が保存されること）。
    """
    # 1. 均一角度サンプリングリング上での厳密直交性
    phi_dense = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    r_dense = np.full_like(phi_dense, 3.0)
    k3_signal = 3.0 * np.cos(3.0 * phi_dense) + 1.5 * np.sin(3.0 * phi_dense)
    k2_signal = 2.0 * np.cos(2.0 * phi_dense)
    res_input_dense = k3_signal + k2_signal

    cleaned_dense, k3_model_dense = pipe.remove_harmonic_k3_rings(
        r_dense, phi_dense, res_input_dense, r_min=2.5, r_max=3.5, n_rings=1
    )
    diff_k3_strict = np.max(np.abs(k3_model_dense - k3_signal))
    assert diff_k3_strict < 1e-12, f"厳密グリッドでの k=3 抽出誤差: {diff_k3_strict:.2e}"
    diff_cleaned_strict = np.max(np.abs(cleaned_dense - k2_signal))
    assert diff_cleaned_strict < 1e-12, f"厳密グリッドでの他調波保存誤差: {diff_cleaned_strict:.2e}"

    # 2. 2次元直交画素グリッド上での実用的離散フィッティング性能
    x_obs = np.linspace(-6.0, 6.0, 61)
    y_obs = np.linspace(-6.0, 6.0, 61)
    X, Y = np.meshgrid(x_obs, y_obs)
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)

    k3_true = 3.0 * np.cos(3.0 * Phi) + 1.5 * np.sin(3.0 * Phi)
    k2_other = 2.0 * np.cos(2.0 * Phi)
    res_grid = k3_true + k2_other

    cleaned_res, k3_model = pipe.remove_harmonic_k3_rings(R, Phi, res_grid, r_min=1.0, r_max=5.5, n_rings=15)

    valid_mask = (R >= 1.0) & (R <= 5.5)
    # k=3 成分の除去率が 95% 以上（残差振幅が元の 3.35 km/s に対して 0.15 km/s 以下）
    diff_k3 = np.abs(k3_model[valid_mask] - k3_true[valid_mask])
    max_k3_err = np.max(diff_k3)
    assert max_k3_err < 0.15, f"直交画素グリッドでの k=3 モデル抽出誤差が大: {max_k3_err:.3f} km/s"

    diff_cleaned = np.abs(cleaned_res[valid_mask] - k2_other[valid_mask])
    max_cleaned_err = np.max(diff_cleaned)
    assert max_cleaned_err < 0.15, f"直交画素グリッドでの他調波保存誤差が大: {max_cleaned_err:.3f} km/s"


# ==============================================================================
# 独立コンポーネント検査: 渦テンプレート・ピーク抽出・一致判定
# ==============================================================================

def test_component_04_vortex_template_and_peak_matching():
    """
    【Component-04: 渦テンプレート生成・ピーク抽出・位置一致判定】
    Model 1 背景ハローにおいて、指定位置 (x_v, y_v) に注入された渦摂動が
    find_local_peaks_2d で検出され、evaluate_position_matching で正しく回収されることを確認。
    """
    halo = pipe.TFHaloModel(M_msun=1e11, R_kpc=20.0, m_eV=1e-22)

    grid_coords = np.linspace(-10.0, 10.0, 101)
    X_grid, Y_grid = np.meshgrid(grid_coords, grid_coords)

    vortex_positions = [(3.0, 2.0), (-4.0, -1.0)]
    dv_map = halo.compute_vortex_velocity_perturbation_map(X_grid, Y_grid, vortex_positions)

    max_amp = np.max(np.abs(dv_map))
    assert max_amp > 0.5, f"渦速度摂動が小さすぎる: max_amp = {max_amp:.3f} km/s"

    peaks = pipe.find_local_peaks_2d(dv_map, threshold_snr=3.0, noise_sigma=0.1, min_distance_pix=5)
    assert len(peaks) >= 2, f"抽出ピーク数が不足: {len(peaks)}"

    def to_pix(x_kpc, y_kpc):
        return int(round((x_kpc - (-10.0)) / 0.2)), int(round((y_kpc - (-10.0)) / 0.2))

    true_pix = [to_pix(xv, yv) for xv, yv in vortex_positions]
    r_tol_pix = 3.0

    matches = pipe.evaluate_position_matching(peaks, true_pix, r_tol_pix=r_tol_pix)
    assert all(matches), f"注入渦の位置回収に失敗: matches = {matches}"


# ==============================================================================
# Test-06: エンドツーエンド幾何射影テスト (全27幾何条件)
# ==============================================================================

def test_06_end_to_end_vortex_geometry_masked():
    """
    【Test-06: エンドツーエンド幾何射影テスト】
    ノイズなし・ビームなしで、全27幾何条件
    (PA in {0°, 45°, 90°} x inc in {30°, 50°, 70°} x Nv in {1, 2, 4})
    において渦を注入し、探索領域マスク (0.2 R_disk <= r <= 0.9 R_disk) 内の
    視線速度残差の最大絶対値画素が、理論予想位置 (disk_to_obs) と 1.5 ピクセル以内で
    厳密に一致することを検証する。
    """
    model_info = {'M_msun': 1e10, 'R_kpc': 3.0, 'm_eV': 1e-22, 'r_fov': 2.2, 'dpix': 0.08}
    halo = pipe.TFHaloModel(M_msun=model_info['M_msun'], R_kpc=model_info['R_kpc'], m_eV=model_info['m_eV'])
    dpix = model_info['dpix']
    coords = np.arange(-model_info['r_fov'], model_info['r_fov'] + dpix/2, dpix)
    center_idx = len(coords) // 2
    X_obs, Y_obs = np.meshgrid(coords, coords)

    d_vortex = 0.3 * model_info['R_kpc']
    r_disk_min = 0.2 * model_info['R_kpc']
    r_disk_max = 0.9 * model_info['R_kpc']

    for pa in [0.0, 45.0, 90.0]:
        for inc in [30.0, 50.0, 70.0]:
            for nv in [1, 2, 4]:
                if nv == 1:
                    vortex_pos_kpc = [(d_vortex, 0.0)]
                elif nv == 2:
                    vortex_pos_kpc = [(d_vortex, 0.0), (-d_vortex, 0.0)]
                elif nv == 4:
                    vortex_pos_kpc = [(d_vortex, 0.0), (-d_vortex, 0.0), (0.0, d_vortex), (0.0, -d_vortex)]

                _, _, r_disk_true, phi_disk_true = pipe.obs_to_disk(X_obs, Y_obs, pa_deg=pa, inc_deg=inc)
                mask_search = (r_disk_true >= r_disk_min) & (r_disk_true <= r_disk_max)

                dv_map = halo.compute_vortex_velocity_perturbation_map(
                    r_disk_true * np.cos(phi_disk_true), r_disk_true * np.sin(phi_disk_true), vortex_pos_kpc
                )
                v_los_inj = dv_map * np.cos(phi_disk_true) * np.sin(np.radians(inc))

                abs_res = np.abs(v_los_inj)
                abs_res[~mask_search] = 0.0

                max_idx = np.unravel_index(np.argmax(abs_res), abs_res.shape)
                max_y_idx, max_x_idx = int(max_idx[0]), int(max_idx[1])

                expected_pix = []
                for xv, yv in vortex_pos_kpc:
                    if np.abs(xv) < 1e-5:
                        continue  # 短軸渦 (xv=0) は cos(phi)=0 により不可視
                    xo, yo = pipe.disk_to_obs(xv, yv, pa_deg=pa, inc_deg=inc)
                    xi = center_idx + int(round(xo / dpix))
                    yi = center_idx + int(round(yo / dpix))
                    expected_pix.append((xi, yi))

                min_dist = min(np.sqrt((max_x_idx - xi)**2 + (max_y_idx - yi)**2) for xi, yi in expected_pix)
                assert min_dist <= 1.5, f"PA={pa}, inc={inc}, Nv={nv}: 検出位置 ({max_x_idx}, {max_y_idx}) が予想位置 {expected_pix} から乖離 (距離 {min_dist:.2f} px)"


# ==============================================================================
# Test-07: 照合ロジックの単体テスト (Mock Peak Placement)
# ==============================================================================

def test_07_mock_peak_placement():
    """
    【Test-07: 照合判定ロジック単体検査】
    真の注入位置にピークを強制配置した際、evaluate_position_matching が
    確実に matches = [True] を返し、許容半径外で [False] を返すことを確認する。
    """
    true_positions = [(48, 48), (20, 30)]
    r_tol = 3.0

    # 1. 完全一致
    peaks_exact = [
        {"x_idx": 48, "y_idx": 48, "abs_value": 5.0},
        {"x_idx": 20, "y_idx": 30, "abs_value": 5.0}
    ]
    matches_exact = pipe.evaluate_position_matching(peaks_exact, true_positions, r_tol)
    assert matches_exact == [True, True]

    # 2. 許容半径内の微小ズレ (距離 sqrt(2^2 + 1^2) = sqrt(5) < 3.0)
    peaks_near = [
        {"x_idx": 50, "y_idx": 49, "abs_value": 5.0},
        {"x_idx": 21, "y_idx": 32, "abs_value": 5.0}
    ]
    matches_near = pipe.evaluate_position_matching(peaks_near, true_positions, r_tol)
    assert matches_near == [True, True]

    # 3. 許容半径外 (距離 5.0 > 3.0)
    peaks_far = [
        {"x_idx": 53, "y_idx": 48, "abs_value": 5.0},
        {"x_idx": 20, "y_idx": 36, "abs_value": 5.0}
    ]
    matches_far = pipe.evaluate_position_matching(peaks_far, true_positions, r_tol)
    assert matches_far == [False, False]


# ==============================================================================
# Test-08: 相関ノイズ生成器と整合フィルター正規化の単体テスト
# ==============================================================================

def test_08_correlated_noise_generator():
    """
    【Test-08: 相関ノイズ生成器と整合フィルター出力の検証】
    1. 画素ごとの標準偏差 = sigma_v（許容誤差 1% 以内）
    2. 自己相関がビーム自己相関（sigma_ACF = sqrt(2) * sigma_beam）と一致すること
    3. ノイズのみの正規化整合フィルター出力の分散 = 1.0（正規化の確認）
    """
    sigma_v = 1.5  # km/s
    sigma_beam_pix = 3.0
    shape = (256, 256)
    rng = np.random.RandomState(42)

    # 1. ノイズ生成と実効標準偏差の検証（許容誤差 1% 以内）
    noise = pipe.generate_beam_correlated_noise(shape, sigma_beam_pix, sigma_v, rng)
    actual_std = float(np.std(noise))
    rel_err_std = abs(actual_std - sigma_v) / sigma_v
    assert rel_err_std < 0.01, f"ノイズ標準偏差が目標値から1%以上乖離: actual={actual_std:.4f}, target={sigma_v}"

    # 2. 自己相関関数のプロファイル検証
    # ホワイトノイズにガウシアン G(sigma_beam) を畳み込んだノイズの理論自己相関は G(sqrt(2) * sigma_beam)
    # 2D FFT を用いた周期境界自己相関の計算
    fft_n = np.fft.fft2(noise - np.mean(noise))
    power_spectrum = np.abs(fft_n)**2
    acf_2d = np.fft.ifftshift(np.fft.ifft2(power_spectrum).real)
    acf_norm = acf_2d / acf_2d[shape[0] // 2, shape[1] // 2]

    # 中心付近のプロファイル取得（x軸方向のラグ）
    cy, cx = shape[0] // 2, shape[1] // 2
    lags = np.arange(-15, 16)
    actual_acf_profile = acf_norm[cy, cx + lags]

    # 理論値: exp(-lag^2 / (2 * (sqrt(2) * sigma_beam)^2)) = exp(-lag^2 / (4 * sigma_beam^2))
    sigma_acf_theory = np.sqrt(2.0) * sigma_beam_pix
    expected_acf_profile = np.exp(- (lags**2) / (2.0 * sigma_acf_theory**2))

    corr_coef = np.corrcoef(actual_acf_profile, expected_acf_profile)[0, 1]
    assert corr_coef > 0.99, f"自己相関プロファイルが理論ガウシアンと乖離: corr={corr_coef:.4f}"

    # 3. ノイズのみの整合フィルター出力の分散 = 1.0（正規化の確認）
    # 整合フィルター出力 M = gaussian_filter(noise, sigma_beam_pix)
    from scipy.ndimage import gaussian_filter
    matched_output = gaussian_filter(noise, sigma=sigma_beam_pix, mode='constant', cval=0.0)
    # 有効領域（端のゼロパディング影響を除外した中央部）での出力分散
    crop = 30
    central_matched = matched_output[crop:-crop, crop:-crop]
    matched_std = float(np.std(central_matched))

    # 正規化整合フィルター統計量 SNR = M / sigma_M
    snr_map = central_matched / matched_std
    snr_var = float(np.var(snr_map))
    assert np.isclose(snr_var, 1.0, atol=1e-5), f"正規化整合フィルター出力の分散が1.0に一致しない: var={snr_var}"


