"""
gyro_gravity_theory — 全宇宙回転ジェネレーション仮説 コア物理モジュール

Universal Rotation Generation Hypothesis — Core Physics Module
================================================================

このモジュールは、銀河回転曲線の3モデル（ニュートン力学、ジャイロ加算モデル、
MOND補間モデル）を計算するための全物理関数を提供する。

References
----------
- Milgrom, M. (1983). ApJ, 270, 365–370.
- Mashhoon, B. (1988). Phys. Rev. Lett., 61, 2639.
- Mashhoon, B. (1993). Phys. Lett. A, 173, 347–352.
- Hehl, F. W., et al. (1976). Rev. Mod. Phys., 48, 393.
- Planck Collaboration (2020). A&A, 641, A6.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import fsolve

# ============================================================================
# 物理定数 / Physical Constants (CODATA 2018 / Planck 2018)
# すべて SI 単位系 (m, kg, s)
# ============================================================================

# 重力定数 / Gravitational constant
G: float = 6.67430e-11            # m^3 kg^-1 s^-2

# 換算プランク定数 / Reduced Planck constant
HBAR: float = 1.054571817e-34      # J·s

# 光速度 / Speed of light
C: float = 299792458.0             # m/s

# ハッブル定数 / Hubble constant (Planck 2018, TT+lowE+lensing)
H0: float = 67.4                   # km/s/Mpc

# 太陽質量 / Solar mass
M_SUN: float = 1.98847e30          # kg

# 陽子質量 / Proton mass
M_P: float = 1.67262192369e-27     # kg

# パーセク / Parsec
PC: float = 3.085677581e16         # m/pc
KPC: float = PC * 1000             # m/kpc

# MOND限界加速度 / MOND critical acceleration (Milgrom 1983)
A0_MOND: float = 1.2e-10           # m/s^2

# ジャイロモデル特性加速度 / Gyro model characteristic acceleration
# a0* = c * H0 / (2π)  （H0 を s^-1 に変換して計算）
A0_STAR: float = C * (H0 * 1000.0 / (KPC * 1e3)) / (2.0 * np.pi)  # m/s^2

# 宇宙論単位系の G（参考値） / G in astro units (kpc, km/s, M_sun)
# v = sqrt(G_astro * M_Msun / r_kpc) で直接 km/s
G_ASTRO: float = G * M_SUN / (KPC * 1000.0**2)  # kpc (km/s)^2 / M_sun


# ============================================================================
# 質量分布関数 / Mass Distribution Functions
# ============================================================================


def mass_plummer(
    r: np.ndarray,
    M_total: float,
    a: float,
) -> np.ndarray:
    """
    Plummer球の累積質量分布。

    M(r) = M_total * r^3 / (r^2 + a^2)^(3/2)

    Parameters
    ----------
    r : np.ndarray
        動径距離配列 [kpc]
    M_total : float
        全質量 [M_sun]
    a : float
        Plummer スケール半径 [kpc]

    Returns
    -------
    np.ndarray
        各 r における累積質量 [M_sun]

    Notes
    -----
    - r → ∞ で M(r) → M_total
    - r = a で M(r) = M_total / (2√2) ≈ 0.354 * M_total（半質量半径）
    """
    return M_total * r**3 / (r**2 + a**2) ** 1.5


def mass_disk(
    r: np.ndarray,
    M_disk: float,
    R_d: float,
) -> np.ndarray:
    """
    指数関数的減衰円盤の累積質量分布（2D表面密度を3Dに射影）。

    M(r) = M_disk * [1 - (1 + r/R_d) * exp(-r/R_d)]

    Parameters
    ----------
    r : np.ndarray
        動径距離配列 [kpc]
    M_disk : float
        円盤全質量 [M_sun]
    R_d : float
        スケール長 [kpc]

    Returns
    -------
    np.ndarray
        各 r における累積質量 [M_sun]

    Notes
    -----
    - Σ(r) = Σ0 * exp(-r/R_d) の表面密度分布を積分
    - r → ∞ で M(r) → M_disk
    - 円盤は薄いため、3D射影の補正は無視（thin-disk近似）
    """
    x = r / R_d
    return M_disk * (1.0 - (1.0 + x) * np.exp(-x))


def mass_bulge_plus_disk(
    r: np.ndarray,
    M_bulge: float,
    a_bulge: float,
    M_disk: float,
    R_d: float,
) -> np.ndarray:
    """
    バルジ（Plummer）＋円盤（指数）の合成質量分布。

    Parameters
    ----------
    r : np.ndarray
        動径距離配列 [kpc]
    M_bulge : float
        バルジ質量 [M_sun]
    a_bulge : float
        バルジ Plummer 半径 [kpc]
    M_disk : float
        円盤質量 [M_sun]
    R_d : float
        円盤スケール長 [kpc]

    Returns
    -------
    np.ndarray
        各 r における合成累積質量 [M_sun]
    """
    return mass_plummer(r, M_bulge, a_bulge) + mass_disk(r, M_disk, R_d)


def mass_nfw(
    r: np.ndarray,
    rho0: float,
    r_s: float,
) -> np.ndarray:
    """
    Navarro-Frenk-White (NFW) ダークマターハローの累積質量分布。

    M(r) = 4π * rho0 * r_s^3 * [ln(1 + r/r_s) - r/(r + r_s)]

    Parameters
    ----------
    r : np.ndarray
        動径距離配列 [kpc]
    rho0 : float
        特性密度 [M_sun / kpc^3]
    r_s : float
        スケール半径 [kpc]

    Returns
    -------
    np.ndarray
        各 r における累積質量 [M_sun]

    Notes
    -----
    標準的な ΛCDM モデルのハロープロファイル。
    本シミュレーションでは比較対象としてのみ実装。
    """
    x = r / r_s
    return 4.0 * np.pi * rho0 * r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))


# ============================================================================
# 内挿関数 / Interpolation Functions
# ============================================================================


def mu_function(
    x: np.ndarray,
    mu_type: str = "standard",
) -> np.ndarray:
    """
    MOND/ジャイロモデルの内挿関数 μ(x)。

    Parameters
    ----------
    x : np.ndarray
        a/a0（加速度比）
    mu_type : str
        関数の種類:
        - 'simple'    : μ(x) = x / (1 + x)
        - 'standard'  : μ(x) = x / sqrt(1 + x^2)
        - 'exponential': μ(x) = 1 - exp(-x)

    Returns
    -------
    np.ndarray
        μ(x) の値

    Notes
    -----
    いずれの関数も以下を満たす:
    - x ≪ 1 (深-MOND領域): μ(x) → x
    - x ≫ 1 (ニュートン領域): μ(x) → 1
    """
    if mu_type == "simple":
        return x / (1.0 + x)
    elif mu_type == "standard":
        return x / np.sqrt(1.0 + x**2)
    elif mu_type == "exponential":
        return 1.0 - np.exp(-x)
    else:
        raise ValueError(
            f"Unknown mu_type: '{mu_type}'. "
            f"Choose from 'simple', 'standard', 'exponential'."
        )


# ============================================================================
# 補助関数 / Helper Functions
# ============================================================================


def _mond_equation(
    a: float,
    a_N: float,
    a0: float,
    mu_type: str,
) -> float:
    """
    MOND/ジャイロ内挿方程式の残差関数。

    F(a) = a * μ(a / a0) - a_N

    fsolve でこの残差のゼロ点を探索する。

    Parameters
    ----------
    a : float
        未知の加速度 [m/s^2]（fsolve が更新する変数）
    a_N : float
        ニュートン重力加速度 [m/s^2]
    a0 : float
        特性加速度 [m/s^2]
    mu_type : str
        内挿関数の種類

    Returns
    -------
    float
        残差（ゼロに近いほど良い解）
    """
    return a * mu_function(np.array([a / a0]), mu_type)[0] - a_N


def compute_a0_star(
    c_val: Optional[float] = None,
    h0_val: Optional[float] = None,
) -> float:
    """
    ジャイロモデルの特性加速度 a0* を宇宙論定数から計算。

    a0* = c * H0 / (2π)

    Parameters
    ----------
    c_val : float, optional
        光速度 [m/s]。None の場合は標準値 C を使用。
    h0_val : float, optional
        ハッブル定数 [km/s/Mpc]。None の場合は標準値 H0 を使用。

    Returns
    -------
    float
        特性加速度 [m/s^2]
    """
    c_used = C if c_val is None else c_val
    h0_used = H0 if h0_val is None else h0_val
    # H0 を km/s/Mpc → s^-1 に変換
    h0_s = h0_used * 1000.0 / (KPC * 1e3)
    return c_used * h0_s / (2.0 * np.pi)


def compute_enclosed_mass(
    r: np.ndarray,
    M_total: float,
    b: float,
    R_d: float,
    f_disk: float,
) -> np.ndarray:
    """
    指定された半径 r 内の累積バリオン質量を計算。

    M(r) = M_bulge(r) + M_disk(r)

    Parameters
    ----------
    r : np.ndarray
        動径距離配列 [kpc]
    M_total : float
        全質量 [M_sun]
    b : float
        Plummer 半径（バルジ）[kpc]
    R_d : float
        円盤スケール長 [kpc]
    f_disk : float
        円盤質量分率（0〜1）

    Returns
    -------
    np.ndarray
        各 r における累積質量 [M_sun]
    """
    M_bulge = (1.0 - f_disk) * M_total
    M_disk_val = f_disk * M_total
    return mass_bulge_plus_disk(r, M_bulge, b, M_disk_val, R_d)


def compute_acceleration_newton(
    r_kpc: np.ndarray,
    M_kpc: np.ndarray,
) -> np.ndarray:
    """
    ニュートン重力加速度 a_N = G * M / r^2 を計算。

    Parameters
    ----------
    r_kpc : np.ndarray
        動径距離 [kpc]
    M_kpc : np.ndarray
        累積質量 [M_sun]

    Returns
    -------
    np.ndarray
        ニュートン重力加速度 [m/s^2]
    """
    r_m = r_kpc * KPC
    M_kg = M_kpc * M_SUN
    return G * M_kg / r_m**2


def convert_S_eff_over_m_to_SI(
    S_eff_over_m_kpc_kms: float,
) -> float:
    """
    実効スピンパラメータを宇宙論単位 (kpc·km/s) から SI (m^2/s) に変換。

    Parameters
    ----------
    S_eff_over_m_kpc_kms : float
        実効スピン角運動量密度パラメータ [kpc·km/s]

    Returns
    -------
    float
        SI単位系での値 [m^2/s]
    """
    return S_eff_over_m_kpc_kms * KPC * 1000.0


def convert_velocity_to_kms(
    v_ms: np.ndarray,
) -> np.ndarray:
    """
    速度を m/s から km/s に変換。

    Parameters
    ----------
    v_ms : np.ndarray
        速度 [m/s]

    Returns
    -------
    np.ndarray
        速度 [km/s]
    """
    return v_ms / 1000.0


# ============================================================================
# 回転曲線計算関数 / Rotation Curve Functions
# ============================================================================


def v_newton(
    r: np.ndarray,
    M_total: float,
    b: float,
    R_d: float,
    f_disk: float,
) -> np.ndarray:
    """
    ニュートン力学（ケプラー回転）による軌道速度。

    v(r) = sqrt(G * M(r) / r)

    Parameters
    ----------
    r : np.ndarray
        動径距離配列 [kpc]
    M_total : float
        銀河全質量 [M_sun]
    b : float
        Plummer 半径 [kpc]
    R_d : float
        円盤スケール長 [kpc]
    f_disk : float
        円盤質量分率

    Returns
    -------
    np.ndarray
        軌道速度 [km/s]
    """
    M_r = compute_enclosed_mass(r, M_total, b, R_d, f_disk)
    r_m = r * KPC
    M_kg = M_r * M_SUN
    v_ms = np.sqrt(G * M_kg / r_m)
    return convert_velocity_to_kms(v_ms)


def v_gyro_additive(
    r: np.ndarray,
    M_total: float,
    b: float,
    R_d: float,
    f_disk: float,
    S_eff_over_m: float,
) -> np.ndarray:
    """
    ジャイロ加算モデル（Additive Model）の解析解による軌道速度。

    a = a_N + a_gyro  ただし  a_gyro = (S_eff/m) * v / r^2

    解析解:
    v = [S_eff/(mr) + sqrt((S_eff/(mr))^2 + 4GM/r)] / 2

    Parameters
    ----------
    r : np.ndarray
        動径距離配列 [kpc]
    M_total : float
        銀河全質量 [M_sun]
    b : float
        Plummer 半径 [kpc]
    R_d : float
        円盤スケール長 [kpc]
    f_disk : float
        円盤質量分率
    S_eff_over_m : float
        実効スピン角運動量密度パラメータ [kpc·km/s]

    Returns
    -------
    np.ndarray
        軌道速度 [km/s]

    Notes
    -----
    この解析解は2次方程式 v^2 - (S_eff/(mr))v - GM/r = 0 の正の根。
    加算型モデルは自己無撞着に解かれており、反復は不要。
    """
    M_r = compute_enclosed_mass(r, M_total, b, R_d, f_disk)

    # SI単位系に変換
    r_m = r * KPC
    M_kg = M_r * M_SUN
    S_eff_over_m_SI = convert_S_eff_over_m_to_SI(S_eff_over_m)

    # 解析解: v = [S/(mr) + sqrt((S/(mr))^2 + 4GM/r)] / 2
    term = S_eff_over_m_SI / r_m
    v_ms = 0.5 * (term + np.sqrt(term**2 + 4.0 * G * M_kg / r_m))
    return convert_velocity_to_kms(v_ms)


def v_mond_interp(
    r: np.ndarray,
    M_total: float,
    b: float,
    R_d: float,
    f_disk: float,
    a0: float = A0_MOND,
    mu_type: str = "standard",
    fsolve_xtol: float = 1e-12,
) -> np.ndarray:
    """
    MOND補間モデル（Interpolating Model）による軌道速度。

    a · μ(a / a0) = a_N  を fsolve で数値求解。
    v = sqrt(a * r)

    Parameters
    ----------
    r : np.ndarray
        動径距離配列 [kpc]
    M_total : float
        銀河全質量 [M_sun]
    b : float
        Plummer 半径 [kpc]
    R_d : float
        円盤スケール長 [kpc]
    f_disk : float
        円盤質量分率
    a0 : float
        MOND限界加速度 [m/s^2]。デフォルト: 1.2e-10
    mu_type : str
        内挿関数の種類 ('simple', 'standard', 'exponential')
    fsolve_xtol : float
        fsolve の許容誤差。デフォルト: 1e-12

    Returns
    -------
    np.ndarray
        軌道速度 [km/s]

    Notes
    -----
    各半径点で独立に fsolve を実行。
    深-MOND領域 (a ≪ a0) では v^4 = G * M * a0 が成り立つ。
    """
    M_r = compute_enclosed_mass(r, M_total, b, R_d, f_disk)
    a_N = compute_acceleration_newton(r, M_r)

    r_m = r * KPC
    a_solutions = np.zeros_like(r, dtype=np.float64)

    for i in range(len(r)):
        # 初期値はニュートン加速度（これで十分収束する）
        a_guess = a_N[i]
        # 深-MOND領域では解析的近似 √(a_N * a0) を初期値に
        if a_N[i] < a0 * 0.1:
            a_guess = np.sqrt(a_N[i] * a0)

        sol = fsolve(
            _mond_equation,
            x0=a_guess,
            args=(a_N[i], a0, mu_type),
            xtol=fsolve_xtol,
            maxfev=500,
        )
        a_solutions[i] = sol[0]

    v_ms = np.sqrt(a_solutions * r_m)
    return convert_velocity_to_kms(v_ms)


def v_gyro_interpolating(
    r: np.ndarray,
    M_total: float,
    b: float,
    R_d: float,
    f_disk: float,
    a0_star: Optional[float] = None,
    mu_type: str = "standard",
) -> np.ndarray:
    """
    ジャイロ内挿モデル（Gyro Interpolating Model）による軌道速度。

    MONDと数学的同型で、a0* を特性加速度として使用。
    a · μ(a / a0*) = a_N

    Parameters
    ----------
    r : np.ndarray
        動径距離配列 [kpc]
    M_total : float
        銀河全質量 [M_sun]
    b : float
        Plummer 半径 [kpc]
    R_d : float
        円盤スケール長 [kpc]
    f_disk : float
        円盤質量分率
    a0_star : float, optional
        ジャイロモデル特性加速度 [m/s^2]。
        None の場合は c*H0/(2π) ≈ 1.05e-10 m/s^2 が使用される。
    mu_type : str
        内挿関数の種類

    Returns
    -------
    np.ndarray
        軌道速度 [km/s]

    Notes
    -----
    - a0_star = c*H0/(2π) の場合、MOND (a0=1.2e-10) と近い結果を与える
    - 差異は a0 と a0* の値の違い（約12%）のみ
    """
    if a0_star is None:
        a0_star = A0_STAR
    return v_mond_interp(r, M_total, b, R_d, f_disk, a0=a0_star, mu_type=mu_type)


# ============================================================================
# ユーティリティ / Utility Functions
# ============================================================================


def check_unit_consistency() -> dict:
    """
    物理定数と単位変換の整合性を検証。

    Returns
    -------
    dict
        検証結果を含む辞書:
        - G_astro_expected: 理論上の G_astro 値
        - G_astro_actual:   計算された G_astro 値
        - G_error_percent:  誤差率
        - A0_STAR:          c*H0/(2π) の値
        - KPC_in_m:         kpc のメートル換算
    """
    # G の宇宙論単位系への変換を検証
    # G_astro = G_SI * M_sun / (kpc * (km/s)^2 換算)
    # = G_SI * M_sun / (kpc * 1000^2)
    G_expected = G * M_SUN / (KPC * 1e6)
    error_pct = abs(G_expected - G_ASTRO) / G_expected * 100.0

    return {
        "G_astro_expected": G_expected,
        "G_astro_actual": G_ASTRO,
        "G_error_percent": error_pct,
        "A0_STAR": A0_STAR,
        "A0_MOND": A0_MOND,
        "A0_ratio": A0_STAR / A0_MOND,
        "KPC_in_m": KPC,
    }


def compute_kepler_velocity(
    r_kpc: np.ndarray,
    M_kpc: np.ndarray,
) -> np.ndarray:
    """
    任意の質量分布に対するケプラー速度。低レベルAPI。

    Parameters
    ----------
    r_kpc : np.ndarray
        動径距離 [kpc]
    M_kpc : np.ndarray
        累積質量 [M_sun]

    Returns
    -------
    np.ndarray
        軌道速度 [km/s]
    """
    r_m = r_kpc * KPC
    M_kg = M_kpc * M_SUN
    v_ms = np.sqrt(G * M_kg / r_m)
    return convert_velocity_to_kms(v_ms)


def compute_deep_mond_velocity(
    M_total: float,
    a0: float = A0_MOND,
) -> float:
    """
    深-MOND領域における漸近速度（Tully-Fisher関係）。

    v^4 = G * M * a0

    Parameters
    ----------
    M_total : float
        銀河全質量 [M_sun]
    a0 : float
        特性加速度 [m/s^2]

    Returns
    -------
    float
        漸近速度 [km/s]
    """
    M_kg = M_total * M_SUN
    v_ms = (G * M_kg * a0) ** 0.25
    return v_ms / 1000.0


# ============================================================================
# BEC + ディラック大数仮説 増幅機構 / BEC + Dirac LNH Amplification
# ============================================================================


def compute_bec_amplification(
    f_align_thermal: float = 1e-5,
    f_align_bec: float = 1.0,
) -> float:
    """
    BECによるスピン同期増幅率を計算。

    η_BEC = f_align_bec / f_align_thermal

    BEC暗黒物質が銀河中のスピンを完全同期させることで、
    熱的揺らぎ下のアラインメント分率 (f_align ≈ 10⁻⁵) が
    完全同期 (f_align ≈ 1) に増幅される。

    Parameters
    ----------
    f_align_thermal : float
        熱的平衡下のスピンアラインメント分率。デフォルト: 1e-5
    f_align_bec : float
        BEC完全同期下のスピンアラインメント分率。デフォルト: 1.0

    Returns
    -------
    float
        BEC増幅率 η_BEC (理論値: 10⁵)
    """
    return f_align_bec / f_align_thermal


def compute_dirac_lnh_amplification(
    N_universe: float = 1e80,
) -> float:
    """
    ディラック大数仮説 (LNH) による結合スケーリング増幅率。

    η_LNH = √N

    ディラックの大数仮説に基づき、電磁力と重力の巨大な比
    (≈ 10⁴⁰) が宇宙の陽子数 N ≈ 10⁸⁰ の平方根に由来する
    という関係から、重力相互作用の有効強度をスケーリングする。

    Parameters
    ----------
    N_universe : float
        宇宙の陽子数。デフォルト: 1e80

    Returns
    -------
    float
        ディラックLNH増幅率 η_LNH (理論値: √(10⁸⁰) = 10⁴⁰)
    """
    return np.sqrt(N_universe)


def compute_total_amplification() -> tuple:
    """
    総増幅率 η_total = η_BEC × η_LNH を計算。

    2つの独立した増幅機構:
    1. BECスピン同期: η_BEC = 10⁵ (f_align: 10⁻⁵ → 1)
    2. ディラック大数仮説: η_LNH = 10⁴⁰ (√N, N≈10⁸⁰)

    Returns
    -------
    tuple
        (eta_total, eta_bec, eta_lnh)
        - eta_total : float  総増幅率 (理論値: 10⁴⁵)
        - eta_bec   : float  BEC増幅率 (理論値: 10⁵)
        - eta_lnh   : float  ディラックLNH増幅率 (理論値: 10⁴⁰)
    """
    eta_bec = compute_bec_amplification()
    eta_lnh = compute_dirac_lnh_amplification()
    return eta_bec * eta_lnh, eta_bec, eta_lnh


def compute_S_eff_over_m_amplified(
    S_eff_over_m_base: float,
    eta_total: float | None = None,
) -> float:
    """
    BEC+LNH増幅後の実効スピンパラメータ。

    S_eff/m (amplified) = S_eff/m (base) × η_total

    増幅後の実効スピン角運動量密度パラメータを計算する。
    これにより、10⁻⁴⁴〜10⁻⁵⁹ m/s² の直接計算値が
    MOND限界加速度 a₀ ≈ 1.2×10⁻¹⁰ m/s² のオーダーに到達する。

    Parameters
    ----------
    S_eff_over_m_base : float
        増幅前の実効スピンパラメータ [kpc·km/s]
    eta_total : float, optional
        総増幅率。None の場合は compute_total_amplification() から計算。

    Returns
    -------
    float
        増幅後の実効スピンパラメータ [kpc·km/s]
    """
    if eta_total is None:
        eta_total, _, _ = compute_total_amplification()
    return S_eff_over_m_base * eta_total
