"""
gyro_gravity_theory — 全宇宙回転ジェネレーション仮説 シミュレーションパッケージ

Universal Rotation Generation Hypothesis — Simulation Package
=============================================================

このパッケージは、銀河回転曲線問題に対する3モデル（ニュートン力学、
ジャイロ加算モデル、MOND補間モデル）の比較シミュレーションを提供する。

Submodules
----------
core : 物理定数、質量分布関数、回転曲線計算関数
"""

from .core import (
    # 物理定数
    G,
    HBAR,
    C,
    H0,
    M_SUN,
    M_P,
    KPC,
    A0_MOND,
    A0_STAR,
    G_ASTRO,
    # 質量分布
    mass_plummer,
    mass_disk,
    mass_bulge_plus_disk,
    mass_nfw,
    # 内挿関数
    mu_function,
    # 回転曲線
    v_newton,
    v_gyro_additive,
    v_mond_interp,
    v_gyro_interpolating,
    # 補助関数
    compute_enclosed_mass,
    compute_acceleration_newton,
    compute_a0_star,
    compute_kepler_velocity,
    compute_deep_mond_velocity,
    # ユーティリティ
    check_unit_consistency,
    convert_S_eff_over_m_to_SI,
    convert_velocity_to_kms,
)

__all__ = [
    "G",
    "HBAR",
    "C",
    "H0",
    "M_SUN",
    "M_P",
    "KPC",
    "A0_MOND",
    "A0_STAR",
    "G_ASTRO",
    "mass_plummer",
    "mass_disk",
    "mass_bulge_plus_disk",
    "mass_nfw",
    "mu_function",
    "v_newton",
    "v_gyro_additive",
    "v_mond_interp",
    "v_gyro_interpolating",
    "compute_enclosed_mass",
    "compute_acceleration_newton",
    "compute_a0_star",
    "compute_kepler_velocity",
    "compute_deep_mond_velocity",
    "check_unit_consistency",
    "convert_S_eff_over_m_to_SI",
    "convert_velocity_to_kms",
]
