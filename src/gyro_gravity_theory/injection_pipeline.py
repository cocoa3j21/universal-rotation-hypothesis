# -*- coding: utf-8 -*-
"""
injection_pipeline.py — 少数量子渦の2次元視線速度残差マップ 注入・回収試験パイプラインコアモジュール

本モジュールは、BEC/超流動暗黒物質ハローにおける少数量子渦の2次元視線速度残差マップにおける
注入・回収試験（Injection-Recovery Test）に必要な数理計算・物理処理を提供する。

【主要機能】
1. 座標変換（観測座標 (x', y') ⇔ 円盤面座標 (x, y)）
2. 投影視線速度場 v_los(x', y') の計算
3. 正規化ビーム畳み込み（Normalized Beam Convolution）
4. 軸対称円盤フィッティング（解析・数値リング平均）および幾何偏微分
5. 面内腕摂動モデルと調波分解・除去（k=3調波除去、腕加法定理検証）
6. 自己重力TFハロー背景における少数量子渦局所速度摂動テンプレート
7. 局所ピーク抽出・位置一致判定・複数構造同定

※宇宙論・物理学・数理検証専用モジュール。
※成功を前提とせず、検出限界・縮退・偽陽性を客観的に評価する。
"""

import numpy as np
from scipy.ndimage import gaussian_filter

# ==============================================================================
# 物理定数（SI単位系）
# ==============================================================================
HBAR = 1.054571817e-34    # J s
C = 299792458.0           # m/s
EV = 1.602176634e-19      # J
KPC = 3.085677581e19      # m
M_SUN = 1.98847e30        # kg
G = 6.67430e-11           # m^3 kg^-1 s^-2


# ==============================================================================
# 1. 座標系変換・幾何投影
# ==============================================================================

def obs_to_disk(x_obs, y_obs, x0=0.0, y0=0.0, pa_deg=0.0, inc_deg=0.0):
    """
    観測座標 (x', y') から円盤面座標 (x, y), 動径 r, 方位角 phi を計算する。

    【定義】
    - 観測座標: x' は西（West, +x'）、y' は北（North, +y'）。原点 (x0, y0) は銀河中心。
    - 位置角 PA: 北（+y'軸）から反時計回り（東向き、-x'方向... 天文学の標準PAは北から東へ反時計回り）。
      本設計書（セクション8.1）の定義:
        x =  (x' - x0) * sin(PA) + (y' - y0) * cos(PA)
        y = [-(x' - x0) * cos(PA) + (y' - y0) * sin(PA)] / cos(inc)
        r = sqrt(x^2 + y^2)
        phi = atan2(y, x)  (主軸・後退側が phi = 0, x > 0, y = 0)
    """
    pa_rad = np.radians(pa_deg)
    inc_rad = np.radians(inc_deg)
    cos_i = np.cos(inc_rad)
    if np.abs(cos_i) < 1e-7:
        raise ValueError("Inclination cannot be 90 degrees (edge-on singularity).")

    dx = x_obs - x0
    dy = y_obs - y0

    sin_pa = np.sin(pa_rad)
    cos_pa = np.cos(pa_rad)

    x_disk = dx * sin_pa + dy * cos_pa
    y_disk = (-dx * cos_pa + dy * sin_pa) / cos_i

    r_disk = np.sqrt(x_disk**2 + y_disk**2)
    phi_disk = np.arctan2(y_disk, x_disk)

    return x_disk, y_disk, r_disk, phi_disk


def disk_to_obs(x_disk, y_disk, x0=0.0, y0=0.0, pa_deg=0.0, inc_deg=0.0):
    """
    円盤面座標 (x, y) から観測座標 (x', y') を計算する逆変換。
    """
    pa_rad = np.radians(pa_deg)
    inc_rad = np.radians(inc_deg)
    cos_i = np.cos(inc_rad)

    sin_pa = np.sin(pa_rad)
    cos_pa = np.cos(pa_rad)

    dx = x_disk * sin_pa - (y_disk * cos_i) * cos_pa
    dy = x_disk * cos_pa + (y_disk * cos_i) * sin_pa

    x_obs = dx + x0
    y_obs = dy + y0

    return x_obs, y_obs


def project_vlos(v_rot, v_rad, phi_disk, inc_deg=0.0, v_sys=0.0):
    """
    円盤面内の回転速度 v_rot(r, phi) および動径速度 v_rad(r, phi) を観測視線速度 v_los に投影する。

    式: v_los = v_sys + (v_rot * cos(phi) + v_rad * sin(phi)) * sin(inc)
    """
    inc_rad = np.radians(inc_deg)
    sin_i = np.sin(inc_rad)
    return v_sys + (v_rot * np.cos(phi_disk) + v_rad * np.sin(phi_disk)) * sin_i


# ==============================================================================
# 2. 正規化ビーム平滑化（Normalized Beam Convolution）
# ==============================================================================

def convolve_beam_normalized(image, fwhm_pix, mask=None, preserve_nan=True):
    """
    正規化畳み込み（Normalized Convolution）による2次元ビームPSF平滑化。

    式: I_conv = ((image * weight) * G) / (weight * G)
    NaNやマスク領域の漏れ込みを完全に防止し、フラックス保存を担保する。
    preserve_nan: True の場合、入力で NaN またはマスクされていた領域は出力でも NaN に保つ。
    """
    if fwhm_pix <= 0.0:
        return np.copy(image)

    sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    is_nan_orig = np.isnan(image)
    if mask is None:
        valid_weight = (~is_nan_orig).astype(float)
    else:
        valid_weight = ((~is_nan_orig) & mask).astype(float)

    clean_image = np.nan_to_num(image, nan=0.0) * valid_weight

    # ガウシアンフィルタ畳み込み (truncate=4.0 で十分な裾野)
    num = gaussian_filter(clean_image, sigma=sigma_pix, mode='constant', cval=0.0, truncate=4.0)
    denom = gaussian_filter(valid_weight, sigma=sigma_pix, mode='constant', cval=0.0, truncate=4.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denom > 1e-6, num / denom, np.nan)

    if preserve_nan:
        result[valid_weight == 0.0] = np.nan

    return result


# ==============================================================================
# 3. 軸対称円盤フィット・解析モデル残差・幾何偏微分
# ==============================================================================

def compute_analytic_residual(v_los_obs, v_rot_true_func, r_disk, phi_disk, inc_deg, v_sys=0.0):
    """
    真の回転曲線関数 v_rot_true_func(r) を用いた完全解析モデルの減算残差。
    v_res = v_los_obs - [v_sys + v_rot_true(r) * cos(phi) * sin(inc)]
    """
    v_model = project_vlos(v_rot_true_func(r_disk), 0.0, phi_disk, inc_deg=inc_deg, v_sys=v_sys)
    return v_los_obs - v_model


def fit_axisymmetric_disk_rings(r_disk, phi_disk, v_los_obs, inc_deg, v_sys=0.0,
                                r_min=0.0, r_max=None, n_rings=30):
    """
    動径リング平均による同心円リング軸対称回転曲線フィッティング。
    各リング k において:
      v_los - v_sys = v_rot(r_k) * cos(phi) * sin(inc)
    から重み付き線形回帰により v_rot(r_k) を求める。
    """
    inc_rad = np.radians(inc_deg)
    sin_i = np.sin(inc_rad)
    if np.abs(sin_i) < 1e-5:
        return np.zeros(n_rings), np.linspace(r_min, r_max if r_max is not None else 1.0, n_rings), np.zeros_like(v_los_obs)

    if r_max is None:
        r_max = np.nanmax(r_disk)

    ring_edges = np.linspace(r_min, r_max, n_rings + 1)
    ring_centers = 0.5 * (ring_edges[:-1] + ring_edges[1:])
    v_rot_fitted = np.zeros(n_rings)
    v_model_map = np.full_like(v_los_obs, np.nan)

    cos_phi = np.cos(phi_disk)
    v_proj_factor = cos_phi * sin_i

    for k in range(n_rings):
        if k == n_rings - 1:
            mask = (r_disk >= ring_edges[k]) & (r_disk <= ring_edges[k+1]) & (~np.isnan(v_los_obs))
        else:
            mask = (r_disk >= ring_edges[k]) & (r_disk < ring_edges[k+1]) & (~np.isnan(v_los_obs))

        if np.sum(mask) >= 1:
            x_k = v_proj_factor[mask]
            y_k = v_los_obs[mask] - v_sys
            denom = np.sum(x_k**2)
            if denom > 1e-10:
                v_rot_k = np.sum(x_k * y_k) / denom
                v_rot_fitted[k] = v_rot_k
                v_model_map[mask] = v_sys + v_rot_k * x_k

    residual_map = v_los_obs - v_model_map
    return v_rot_fitted, ring_centers, residual_map


def compute_geometric_partials(r_disk, phi_disk, v_rot, inc_deg, pa_deg, v_sys=0.0):
    """
    円盤座標固定での視線速度の幾何偏微分を計算する。
      ∂v_los / ∂inc = v_rot * cos(phi) * cos(inc)
      ∂v_los / ∂v_sys = 1
    """
    inc_rad = np.radians(inc_deg)
    cos_i = np.cos(inc_rad)
    cos_phi = np.cos(phi_disk)

    dv_dinc = v_rot * cos_phi * cos_i  # per radian
    dv_dvsys = np.ones_like(r_disk)

    return dv_dinc, dv_dvsys


# ==============================================================================
# 4. 面内腕摂動モデル・調波分解・腕加法定理
# ==============================================================================

def generate_spiral_arm_velocity_perturbation(r_disk, phi_disk, A_r_func, A_phi_func,
                                              pitch_angle_deg=15.0, r_ref=1.0,
                                              phi_arm_0=0.0, phase_diff_deg=0.0):
    """
    面内 m=2 渦状腕による動径速度・方位角速度摂動 (v_r_arm, v_phi_arm) を生成する。
    """
    tan_pitch = np.tan(np.radians(pitch_angle_deg))
    with np.errstate(divide='ignore', invalid='ignore'):
        log_r = np.where(r_disk > 0, np.log(r_disk / r_ref), 0.0)
    cot_pitch = 1.0 / tan_pitch if np.abs(tan_pitch) > 1e-6 else 0.0
    psi = 2.0 * (phi_disk - np.radians(phi_arm_0) - cot_pitch * log_r)

    phase_diff_rad = np.radians(phase_diff_deg)
    v_r_arm = A_r_func(r_disk) * np.sin(psi)
    v_phi_arm = A_phi_func(r_disk) * np.cos(psi + phase_diff_rad)

    return v_r_arm, v_phi_arm, psi


def remove_harmonic_k3_rings(r_disk, phi_disk, residual_map, r_min=0.0, r_max=None, n_rings=30):
    """
    手法B: 各動径リングにおいて k=3 調波成分 (cos(3*phi), sin(3*phi)) を回帰減算する。
    """
    if r_max is None:
        r_max = np.nanmax(r_disk)

    ring_edges = np.linspace(r_min, r_max, n_rings + 1)
    cleaned_residual = np.copy(residual_map)
    k3_model_map = np.zeros_like(residual_map)

    cos_3phi = np.cos(3.0 * phi_disk)
    sin_3phi = np.sin(3.0 * phi_disk)

    for k in range(n_rings):
        if k == n_rings - 1:
            mask = (r_disk >= ring_edges[k]) & (r_disk <= ring_edges[k+1]) & (~np.isnan(residual_map))
        else:
            mask = (r_disk >= ring_edges[k]) & (r_disk < ring_edges[k+1]) & (~np.isnan(residual_map))

        if np.sum(mask) >= 6:
            c3 = cos_3phi[mask]
            s3 = sin_3phi[mask]
            y = residual_map[mask]

            X = np.column_stack([c3, s3])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                k3_fitted = beta[0] * c3 + beta[1] * s3
                cleaned_residual[mask] = y - k3_fitted
                k3_model_map[mask] = k3_fitted
            except Exception:
                pass

    return cleaned_residual, k3_model_map


def fit_spiral_arm_direct_c(r_disk, phi_disk, residual_map, inc_deg,
                            pitch_candidates_deg=(10.0, 15.0, 20.0), r_ref=1.0):
    """
    手法C: 面内対数螺旋腕モデルの直接フィッティング。
    各ピッチ角候補に対して、三角関数の線形展開基底（cos(psi), sin(psi)）による
    重回帰を解き、最良適合（最小RSS）の腕モデルを減算する。
    """
    inc_rad = np.radians(inc_deg)
    sin_i = np.sin(inc_rad)

    valid_mask = (~np.isnan(residual_map)) & (r_disk > 0.1)
    if np.sum(valid_mask) < 20:
        return np.copy(residual_map), np.zeros_like(residual_map), 0.0, 15.0

    r_val = r_disk[valid_mask]
    phi_val = phi_disk[valid_mask]
    y_val = residual_map[valid_mask]
    log_r = np.log(r_val / r_ref)

    best_rss = np.inf
    best_arm_model = np.zeros_like(residual_map)
    best_pitch = 15.0
    best_amp = 0.0

    for pitch_deg in pitch_candidates_deg:
        tan_p = np.tan(np.radians(pitch_deg))
        cot_p = 1.0 / tan_p if np.abs(tan_p) > 1e-4 else 0.0

        # 対数螺旋位相: psi = 2 * (phi - cot_p * ln(r/r_ref))
        psi_val = 2.0 * (phi_val - cot_p * log_r)

        # 視線速度射影基底:
        # v_los,arm ≈ [A_c * cos(psi) + A_s * sin(psi)] * cos(phi) * sin(i)
        #             + [B_c * cos(psi) + B_s * sin(psi)] * sin(phi) * sin(i)
        # 主に k=3 成分: cos(psi)*cos(phi)*sin(i) 等
        basis_c = np.cos(psi_val) * np.cos(phi_val) * sin_i
        basis_s = np.sin(psi_val) * np.cos(phi_val) * sin_i
        basis_rc = np.sin(psi_val) * np.sin(phi_val) * sin_i
        basis_rs = np.cos(psi_val) * np.sin(phi_val) * sin_i

        X = np.column_stack([basis_c, basis_s, basis_rc, basis_rs])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y_val, rcond=None)
            pred = X @ beta
            rss = np.sum((y_val - pred) ** 2)
            if rss < best_rss:
                best_rss = rss
                best_pitch = pitch_deg
                best_amp = np.sqrt(np.sum(beta**2))

                # 全グリッド展開
                psi_all = 2.0 * (phi_disk - cot_p * np.where(r_disk > 0, np.log(np.maximum(r_disk, 1e-3) / r_ref), 0.0))
                bc_all = np.cos(psi_all) * np.cos(phi_disk) * sin_i
                bs_all = np.sin(psi_all) * np.cos(phi_disk) * sin_i
                brc_all = np.sin(psi_all) * np.sin(phi_disk) * sin_i
                brs_all = np.cos(psi_all) * np.sin(phi_disk) * sin_i
                best_arm_model = beta[0] * bc_all + beta[1] * bs_all + beta[2] * brc_all + beta[3] * brs_all
        except Exception:
            continue

    cleaned_res = np.copy(residual_map)
    cleaned_res[valid_mask] = y_val - best_arm_model[valid_mask]
    return cleaned_res, best_arm_model, best_amp, best_pitch


def compute_bic(residual_map, k_params, noise_sigma=1.0):
    """
    残差マップに対するベイズ情報量基準 (BIC)。
    BIC = chi^2 + k * ln(N_pix)
    chi^2 = sum( (v_res / noise_sigma)^2 )
    """
    valid = (~np.isnan(residual_map))
    n_pix = np.sum(valid)
    if n_pix == 0:
        return np.nan

    chi2 = np.sum((residual_map[valid] / noise_sigma) ** 2)
    return chi2 + k_params * np.log(float(n_pix))


# ==============================================================================
# 5. 自己重力TFハロー背景における少数量子渦面内摂動テンプレート
# ==============================================================================

class TFHaloModel:
    """球対称自己重力TFハロー（n=1ポリトロープ球）"""
    def __init__(self, M_msun: float, R_kpc: float, m_eV: float):
        self.M_halo = M_msun * M_SUN
        self.R_halo = R_kpc * KPC
        self.m_kg = m_eV * EV / (C ** 2)

        self.v_esc = np.sqrt(2.0 * G * self.M_halo / self.R_halo)
        self.xi_c = HBAR / (self.m_kg * self.v_esc)  # meters
        self.xi_c_kpc = self.xi_c / KPC
        self.rho_c = np.pi * self.M_halo / (4.0 * (self.R_halo ** 3))

    def v_0(self, r_kpc):
        """非摂動円運動速度 v_0(r) [km/s]"""
        r_m = np.asarray(r_kpc) * KPC
        m_enc = np.zeros_like(r_m, dtype=float)

        mask = (r_m < self.R_halo) & (r_m > 0)
        x = np.pi * r_m[mask] / self.R_halo
        m_enc[mask] = (self.M_halo / np.pi) * (np.sin(x) - x * np.cos(x))
        m_enc[r_m >= self.R_halo] = self.M_halo

        val = np.zeros_like(r_m, dtype=float)
        m_pos = r_m > 0
        val[m_pos] = np.sqrt(G * m_enc[m_pos] / r_m[m_pos])
        return val / 1000.0  # m/s -> km/s

    def compute_vortex_mass_deficit(self, d_cen_kpc):
        """ハロー中心からの距離 d_cen における中空円筒状渦の質量欠損 M_def [kg]"""
        d_cen_m = d_cen_kpc * KPC
        if d_cen_m >= self.R_halo:
            return 0.0

        L_z = 2.0 * np.sqrt(self.R_halo**2 - d_cen_m**2)
        z_pts = np.linspace(-L_z/2.0, L_z/2.0, 200)
        dz = z_pts[1] - z_pts[0]
        r_sph = np.sqrt(d_cen_m**2 + z_pts**2)
        x = np.pi * r_sph / self.R_halo
        rho_z = self.rho_c * np.sin(x) / x
        return np.sum(rho_z * np.pi * (self.xi_c ** 2) * dz)

    def compute_vortex_velocity_perturbation_map(self, x_grid_kpc, y_grid_kpc, vortex_positions_kpc):
        """
        円盤面内の格子点 (x_grid, y_grid) における複数渦による面内円運動速度摂動 delta_v_phi [km/s] を計算。
        """
        r_grid_kpc = np.sqrt(x_grid_kpc**2 + y_grid_kpc**2)
        v0_grid = self.v_0(r_grid_kpc) * 1000.0  # m/s

        delta_v_phi = np.zeros_like(x_grid_kpc, dtype=float)
        eps_smooth_m = 0.5 * self.xi_c  # 特異点正則化コア

        for xv_kpc, yv_kpc in vortex_positions_kpc:
            d_cen_kpc = np.sqrt(xv_kpc**2 + yv_kpc**2)
            M_def = self.compute_vortex_mass_deficit(d_cen_kpc)
            if M_def <= 0.0:
                continue

            dx_m = (x_grid_kpc - xv_kpc) * KPC
            dy_m = (y_grid_kpc - yv_kpc) * KPC
            dist_sq_m = dx_m**2 + dy_m**2 + eps_smooth_m**2
            dist_m = np.sqrt(dist_sq_m)

            r_m = np.maximum(r_grid_kpc * KPC, 1e-5)
            e_phi_x = - (y_grid_kpc * KPC) / r_m
            e_phi_y = (x_grid_kpc * KPC) / r_m

            with np.errstate(divide='ignore', invalid='ignore'):
                dv_local = np.where(v0_grid > 1e-2, (G * M_def / (dist_m * v0_grid)) * (dx_m * e_phi_x + dy_m * e_phi_y) / dist_m, 0.0)
            delta_v_phi += dv_local

        return delta_v_phi / 1000.0  # m/s -> km/s


# ==============================================================================
# 6. ピーク抽出・位置一致判定・複数構造同定
# ==============================================================================

def find_local_peaks_2d(image, threshold_snr, noise_sigma=1.0, min_distance_pix=3):
    """
    2次元残差マップから指定SNR閾値以上の極値（局所ピーク）を抽出する。
    """
    from scipy.ndimage import maximum_filter

    abs_image = np.abs(image)
    thresh_val = threshold_snr * noise_sigma

    footprint = np.ones((2 * min_distance_pix + 1, 2 * min_distance_pix + 1), dtype=bool)
    max_filtered = maximum_filter(abs_image, footprint=footprint, mode='constant', cval=0.0)

    is_peak = (abs_image == max_filtered) & (abs_image >= thresh_val)
    peak_y, peak_x = np.nonzero(is_peak)

    peaks = []
    for y_idx, x_idx in zip(peak_y, peak_x):
        peaks.append({
            "x_idx": x_idx,
            "y_idx": y_idx,
            "value": image[y_idx, x_idx],
            "abs_value": abs_image[y_idx, x_idx],
            "snr": abs_image[y_idx, x_idx] / noise_sigma
        })
    return peaks


def evaluate_position_matching(peaks, true_positions_pix, r_tol_pix):
    """
    複数渦における注入位置ごとの独立一致判定（Per-Injection-Site Matching）。
    C_j = 1 if min_m ||x_peak,m - x_true,j|| <= r_tol else 0
    """
    if len(true_positions_pix) == 0:
        return []

    matches = []
    for xj_true, yj_true in true_positions_pix:
        matched = False
        min_dist = np.inf
        for p in peaks:
            dist = np.sqrt((p["x_idx"] - xj_true)**2 + (p["y_idx"] - yj_true)**2)
            if dist < min_dist:
                min_dist = dist
            if dist <= r_tol_pix:
                matched = True
        matches.append(matched)
    return matches


def compute_tolerance_radius(model_id, fwhm_kpc, r_disk_kpc=None):
    """
    正本仕様書セクション1.1, 1.2に基づき、許容半径 r_tol [kpc] を一元的に算出する（DRY原則）。
    r_tol = max(xi_c, FWHM/2)
    Model 3 の場合は min(r_tol, 0.15 * R_disk) で上限制限。
    """
    if model_id == "model_1":
        xi_c_kpc = 0.092442  # M=1e11, R=20, m=1e-22 eV
        r_tol = max(xi_c_kpc, fwhm_kpc / 2.0)
    elif model_id == "model_3":
        xi_c_kpc = 0.113218  # M=1e10, R=3, m=1e-22 eV
        r_tol = max(xi_c_kpc, fwhm_kpc / 2.0)
        r_disk = r_disk_kpc if r_disk_kpc is not None else 3.0
        r_tol = min(r_tol, 0.15 * r_disk)
    else:
        r_tol = fwhm_kpc / 2.0
    return r_tol


def find_matched_filter_candidate(residual_map, sigma_beam_pix, mask_search):
    """
    正本仕様書セクション5.2に基づき、ビーム幅ガウシアン整合フィルターを用いて
    探索領域内の最良候補点 (px, py) を特定する。
    単一ピクセルのノイズスパイクではなく、ビーム幅スケールの空間コヒーレントな構造を抽出する。
    """
    clean_res = np.nan_to_num(residual_map, nan=0.0) * mask_search.astype(float)
    matched_image = gaussian_filter(clean_res, sigma=sigma_beam_pix, mode='constant', cval=0.0)
    matched_image[~mask_search] = 0.0

    abs_matched = np.abs(matched_image)
    max_idx = np.unravel_index(np.argmax(abs_matched), abs_matched.shape)
    py, px = int(max_idx[0]), int(max_idx[1])
    max_val = matched_image[py, px]
    return px, py, max_val


def compute_empirical_matched_filter_stat(residual_map, sigma_beam_pix, mask_search):
    """
    正本仕様書セクション5.2に基づき、空間相関ノイズ場における
    経験的整合フィルター優位性統計量 Delta_TS_MF を算出する。
    Delta_TS_MF = S_max^2 - 1.0 * ln(N_beam)
    ここで S_max = max |M(x, y)| / sigma_M は正規化整合フィルターSNR。
    """
    clean_res = np.nan_to_num(residual_map, nan=0.0) * mask_search.astype(float)
    matched_image = gaussian_filter(clean_res, sigma=sigma_beam_pix, mode='constant', cval=0.0)

    valid_vals = matched_image[mask_search]
    if len(valid_vals) < 10:
        return 0, 0, 0.0, 0.0

    sigma_M = float(np.std(valid_vals))
    if sigma_M < 1e-10:
        return 0, 0, 0.0, 0.0

    snr_map = np.abs(matched_image) / sigma_M
    snr_map[~mask_search] = 0.0

    max_idx = np.unravel_index(np.argmax(snr_map), snr_map.shape)
    py, px = int(max_idx[0]), int(max_idx[1])
    s_max = float(snr_map[py, px])
    s_max_sq = s_max ** 2

    n_pix_valid = np.sum(mask_search)
    n_beam = compute_n_beam(n_pix_valid, sigma_beam_pix)
    penalty = float(np.log(n_beam))

    delta_ts_mf = s_max_sq - penalty
    return px, py, s_max, delta_ts_mf



def generate_beam_correlated_noise(shape, sigma_beam_pix, sigma_v, rng):
    """
    熱雑音のビーム平滑化に伴う空間相関ノイズを生成する。
    各ピクセルの実効標準偏差は指定値 sigma_v [km/s] を厳密に保つ。
    """
    white = rng.normal(0.0, 1.0, size=shape)
    if sigma_beam_pix <= 0.5:
        return white * sigma_v
    smooth = gaussian_filter(white, sigma=sigma_beam_pix, mode='constant', cval=0.0)
    std_val = np.std(smooth)
    if std_val < 1e-10:
        return white * sigma_v
    return smooth * (sigma_v / std_val)


def compute_n_beam(n_pix_valid, sigma_beam_pix):
    """
    空間相関を考慮した有効独立ビーム数 (分解能要素数) N_beam を算出する。
    Omega_beam = 2 * pi * sigma_beam_pix^2
    N_beam = N_pix / Omega_beam
    """
    beam_area = 2.0 * np.pi * (sigma_beam_pix ** 2)
    if beam_area < 1.0:
        return float(n_pix_valid)
    return max(1.0, float(n_pix_valid) / beam_area)


def predict_a50(snr_th, sigma_v, eta_beam, eta_fit, inc_deg, phi_disk_rad=0.0):
    """
    50%回収閾値振幅 A_50 [km/s] の理論近似式。
    A_50 ≈ (SNR_th * sigma_v) / (eta_beam * eta_fit * |cos(phi)| * sin(i))
    """
    sin_i = np.sin(np.radians(inc_deg))
    cos_phi = np.abs(np.cos(phi_disk_rad))
    denom = eta_beam * eta_fit * cos_phi * sin_i
    if denom < 1e-6:
        return np.inf
    return (snr_th * sigma_v) / denom


