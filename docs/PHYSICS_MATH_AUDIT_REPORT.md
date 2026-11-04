# 全宇宙回転ジェネレーション仮説 物理・数理再監査報告書（第4版・修正後確定版）
**Physical and Mathematical Re-Audit Report on the Universal Rotation Generation Hypothesis (v4)**

- **対象プロジェクト**: [`universal-rotation-hypothesis`](file:///D:/Workspace/Labs/universal-rotation-hypothesis)（宇宙論・物理学プロジェクトのみ）
- **再監査実施日**: 2026-10-26
- **監査基本方針**:
  - 本報告は仮説の擁護や成立を目的とせず、既存コードおよび数理モデルの厳密な物理・数学的整合性を客観的に点検・記録する。
  - 既知の本番不具合の修正前実行結果（8 PASS・3 XFAIL）は履歴として完全に保持し、修正後の回帰テスト結果（11 PASS）を分離して記録する。
  - **既存グラフ画像・公開記事の変更は行わない**。
  - **未検証の物理機構や天体観測との適合性を「監査完了」と記載せず、「未検証事項」として明記する**。

---

## 1. 全モデルの加速度式とコード行番号の対応

対象ファイル：
- [`src/gyro_gravity_theory/core.py`](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py)
- [`rotation_curve.py`](file:///D:/Workspace/Labs/universal-rotation-hypothesis/rotation_curve.py)

| モデル名 | 加速度式 / 方程式 | 実装関数 | ファイル名・行番号 | 監査ステータス |
| :--- | :--- | :--- | :--- | :--- |
| **ニュートン重力** | $a_N(r) = \frac{G M(r)}{r^2}$ | `compute_acceleration_newton`<br>`v_newton` | [`core.py` L344-366](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L344-L366)<br>[`core.py` L411-446](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L411-L446) | 正常 |
| **ジャイロ加算モデル**<br>(Gyro Additive) | $a(r) = a_N(r) + \frac{K v(r)}{r^2}$<br>$K \equiv \frac{S_{\text{eff}}}{m}$ [$\text{m}^2/\text{s}$]<br>解: $v(r) = \frac{1}{2} \left[ \frac{K}{r} + \sqrt{\left(\frac{K}{r}\right)^2 + \frac{4GM(r)}{r}} \right]$ | `v_gyro_additive` | [`core.py` L448-500](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L448-L500) | 循環論法および漸近極限の物理的破綻あり（§3 参照） |
| **MOND補間モデル**<br>(MOND Interpolating) | $a(r) \cdot \mu\left(\frac{a(r)}{a_0}\right) = a_N(r)$<br>$a_0 = 1.200 \times 10^{-10}\text{ m/s}^2$ | `mu_function`<br>`_mond_equation`<br>`v_mond_interp` | [`core.py` L203-242](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L203-L242)<br>[`core.py` L249-279](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L249-L279)<br>[`core.py` L502-571](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L502-L571) | 正常（Milgrom 1983 の忠実な実装） |
| **ジャイロ内挿モデル**<br>(Gyro Interpolating) | $a(r) \cdot \mu\left(\frac{a(r)}{a_0^*}\right) = a_N(r)$<br>$a_0^* = \frac{c H_0}{2\pi} \approx 1.042 \times 10^{-10}\text{ m/s}^2$ | `v_gyro_interpolating`<br>`compute_a0_star` | [`core.py` L56, L281-307](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L56)<br>[`core.py` L573-619](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L573-L619) | **単位換算バグ修正済み**（分母を `KPC * 1e3` に修正、§7 参照） |
| **アインシュタイン・カルタン項**<br>(EC Torsion) | $\left(\frac{8\pi G}{c^2}\right) f_{\text{align}} n \left(\frac{\hbar}{2}\right)$ | 文書式(24)-(25) | [`mathematical_model.md` L257-268](file:///D:/Workspace/Labs/universal-rotation-hypothesis/docs/mathematical_model.md#L257-L268) | **次元不整合**（単位が $\text{s}^{-1}$、加速度ではない、§2 参照） |
| **BEC+LNH増幅機構** | $\eta_{\text{BEC}} = \frac{f_{\text{bec}}}{f_{\text{thermal}}}$<br>$\eta_{\text{LNH}} = \sqrt{N_{\text{universe}}}$<br>$\eta_{\text{total}} = \eta_{\text{BEC}} \times \eta_{\text{LNH}}$ | `compute_bec_amplification`<br>`compute_dirac_lnh_amplification`<br>`compute_total_amplification` | [`core.py` L713-785](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L713-L785) | BEC偏極率上限 $1.0$ と大数結合スケーリングの分離（§5, §6 参照） |

---

## 2. アインシュタイン・カルタン項の次元検算と加速度表示の撤回

### 2.1 厳密な次元解析
対象数理式：
$$T \sim \left(\frac{8\pi G}{c^2}\right) f_{\text{align}} n \left(\frac{\hbar}{2}\right)$$
各因子のSI基本単位（$\text{m, kg, s}$）：
- $\frac{G}{c^2} = \frac{\text{m}^3\text{kg}^{-1}\text{s}^{-2}}{\text{m}^2\text{s}^{-2}} = \text{m}\cdot\text{kg}^{-1}$
- $f_{\text{align}} = [1]$（無次元）
- $n = [\text{m}^{-3}]$（数密度）
- $\frac{\hbar}{2} = [\text{J}\cdot\text{s}] = [\text{kg}\cdot\text{m}^2\cdot\text{s}^{-1}]$（スピン角運動量）

合成次元：
$$[T] = (\text{m}\cdot\text{kg}^{-1}) \times [1] \times (\text{m}^{-3}) \times (\text{kg}\cdot\text{m}^2\cdot\text{s}^{-1}) = \mathbf{s^{-1}} \quad (\text{角速度・逆時間の次元})$$

### 2.2 監査所見と加速度表示・比率の撤回
1. **次元不整合**:
   右辺の単位は $\text{s}^{-1}$ であり、加速度の次元（$\text{m}\cdot\text{s}^{-2}$）と一致しない。
2. **加速度表示・比率の撤回**:
   過去の報告で記載されていた「トーション加速度 $9.84 \times 10^{-60}\text{ m/s}^2$」「49.1桁不足」という記述は、**角速度（$\text{s}^{-1}$）と加速度（$\text{m/s}^2$）という異なる物理次元の量を直接比較した誤謬**であったため、**加速度としての数値表示および $a_0$ との比率比較を完全に撤回**する。
3. **結論**:
   トーションテンソルから測地線偏差方程式を通じて試験粒子に働く有効力を導出する力学モデルが提示されていない限り、本式を加速度として扱うことは物理的・数学的に無効（次元不整合）である。

---

## 3. 加算型モデルの漸近挙動と無次元比による領域定義

### 3.1 基礎方程式と無次元比 $\xi(r)$ の導入
動径方向の向心加速度釣り合い式：
$$\frac{v^2}{r} = \frac{GM(r)}{r^2} + \frac{K v}{r^2} \iff r v^2 - K v - GM(r) = 0 \quad \left(K \equiv \frac{S_{\text{eff}}}{m}\right)$$
正の根：
$$v(r) = \frac{1}{2} \left[ \frac{K}{r} + \sqrt{\left(\frac{K}{r}\right)^2 + \frac{4GM(r)}{r}} \right]$$
平方根内の第1項（ジャイロ寄与）と第2項（ニュートン寄与）の比率を表す無次元比 $\xi(r)$ を定義する：
$$\xi(r) \equiv \frac{(K/r)^2}{4GM(r)/r} = \frac{K^2}{4GM(r) r}$$

### 3.2 領域別の漸近挙動
$M > 0$ かつ $K = \text{const}$ の場合の挙動：
1. **真の無限遠・ケプラー支配域 ($\xi(r) \ll 1$、すなわち $r \gg \frac{K^2}{4GM}$)**:
   $r \to \infty$ では $\xi(r) \propto 1/r \to 0$ となる。
   ルート内はニュートン項 $4GM/r$ が支配的となり、$\sqrt{4GM/r} \propto r^{-1/2}$ は第1項 $K/r \propto r^{-1}$ よりも遅く減衰する。
   したがって：
   $$v(r) \approx \frac{1}{2} \left[ 0 + \sqrt{\frac{4GM}{r}} \right] = \sqrt{\frac{GM}{r}} \propto r^{-1/2} \quad (r \to \infty)$$
   **真の無限遠では、ニュートン的なケプラー減衰に漸近する。**
2. **ジャイロ項支配域 ($\xi(r) \gg 1$、すなわち $r \ll \frac{K^2}{4GM}$)**:
   ルート内で $(K/r)^2$ が支配的となる場合：
   $$v(r) \approx \frac{1}{2} \left[ \frac{K}{r} + \frac{K}{r} \right] = \frac{K}{r} \propto r^{-1}$$
   この領域では速度は $1/r$ で減衰し、フラット（$v = \text{const}$）にはならない。
3. **結論**:
   加算型モデルは近距離で $1/r$ 減衰、遠距離で $r^{-1/2}$ 減衰を示し、**広域にわたってフラットな回転曲線（$d\ln v / d\ln r = 0$）を維持することは原理的に不可能**である。また、外縁部ではケプラー則（$v^2 \propto GM/r$）に帰着するため、深-MONDのバリオン Tully-Fisher 関係（$v^4 \propto GMa_0$）を満たさない。

---

## 4. MONDとの同値性と観測的識別可能性の分離

### 4.1 数理構造の完全一致（同一パラメータ下での同値性）
- 同一の限界加速度 $a_0$ および同一の内挿関数 $\mu(x)$ を与えた場合、ジャイロ内挿モデルの方程式 $a \mu(a/a_0) = a_N$ は MOND と**数学的に完全同一（差が厳密にゼロ）**である。
- したがって、「スピン機構から独自の非線形内挿関数が導出された」という事実はなく、数学的には MOND の内挿枠組みをそのまま利用している。

### 4.2 約3.5%の理論予測差と実観測での識別能力の峻別
標準 MOND の限界加速度 $a_0 = 1.200 \times 10^{-10}\text{ m/s}^2$ に対し、宇宙論定数から導かれるジャイロ理論値は $a_0^* = \frac{c H_0}{2\pi} \approx 1.042 \times 10^{-10}\text{ m/s}^2$（約13.2%差）である。

1. **深-MOND領域 ($a_N \ll a_0$) における理論予測差**:
   - 加速度比: $\frac{a^*}{a} = \sqrt{\frac{a_0^*}{a_0}} \approx 0.932$（約 $-6.8\%$ 差）
   - 漸近速度比（Tully-Fisher速度）: $\frac{v^*}{v} = \left(\frac{a_0^*}{a_0}\right)^{1/4} \approx 0.965$（約 $-3.5\%$ 差）
2. **実観測データにおける識別能力との峻別**:
   - **「理論的な予測差（約3.5%）が存在すること」と「実観測データにおいて統計的に有意に識別できる能力があるかどうか」は全く別の問題である。**
   - 約3.5%の予測差を実観測で識別できるかは、銀河までの距離・円盤傾斜角・恒星質量光度比 $\Upsilon_*$ などの不確かさを含めて評価する必要があり、現時点では**未検証**である。
   - したがって、観測的に両理論を峻別するには、これら観測的不確定性を精密にモデル化した統計的アンサンブル解析が不可欠である。

---

## 5. 粒子数増幅の厳密な物理的区分（3つの $N$ の分離）

3つの異なる粒子数 $N$ を明確に分離して評価する。

1. **試験体（プローブ）の粒子数 $N_{\text{test}}$**:
   試験体を構成する粒子数。ニュートンの運動方程式 $a_{\text{cm}} = \sum F / \sum m = (N_{\text{test}} F_1) / (N_{\text{test}} m_1) = F_1/m_1$ により、試験体自身の粒子数 $N_{\text{test}}$ は分子と分母で厳密に相殺される（示量性の保存）。
2. **重力源（ソース）の粒子数 $N_{\text{source}}$**:
   重力場を生成する銀河内のバリオン粒子数。質量 $M = N_{\text{source}} m$ に比例して重力場は強くなる（正当な物理的加算）。
3. **宇宙全体の総粒子数 $N_{\text{universe}} \approx 10^{80}$**:
   本仮説が導入したのは、試験体でも局所重力源でもない「宇宙全体の総粒子数 $N_{\text{universe}}$」を、スピン-トーション結合定数のスケーリング則 $\kappa = \kappa_0 \sqrt{N_{\text{universe}}} \approx 10^{40}$ として仮定した点である。
- **所見**:
   本仮説の理論的課題は、「因果関係のない宇宙全体の総粒子数 $\sqrt{N_{\text{universe}}}$ が、なぜ局所銀河のスピン結合定数に乗算されるのか、その場の理論・ラグランジアンからの正当化が一切存在しない（未定義）」という点にある。

---

## 6. BEC偏極率の上限と大数結合スケーリングの分離

機構全体を単一の「偏極率」に集約して否定するのではなく、2つの独立した仮定を明確に分離して評価する。

### 6.1 BECスピン偏極率の物理的上限
- スピン-1/2粒子系における偏極率 $f_{\text{align}}$ は定義により **$f_{\text{align}} \le 1.0$ (100%完全偏極)** に制限される。
- 熱的初期偏極率を $f_{\text{thermal}} = 10^{-5}$ と置いた場合、BEC完全同期によって得られる偏極率向上比率は最大でも **$\eta_{\text{BEC}} = 1.0 / 10^{-5} = 10^5$ 倍** で頭打ちとなる（初期値が $10^{-4}$ なら $10^4$ 倍）。
- したがって、**スピン偏極率の向上「単独」で担えるのは最大 $10^5$ 倍まで** である。

### 6.2 ディラック大数結合スケーリングの分離（未導出性）
- 45桁のギャップのうち残りの $10^{40}$ は、スピン偏極率の増大ではなく、ディラック大数仮説に基づく結合定数のスケーリング則 **$\eta_{\text{LNH}} = \sqrt{N_{\text{universe}}} \approx 10^{40}$** として独立に導入されたものである。
- したがって、「総増幅率 $10^{45}$ を偏極率に乗じると偏極率が $10^{40} > 1$ になって破綻する」という批判は不正確であり、正しくは **「BEC偏極率は上限 $1.0$（$10^5$倍）で正当に制限されているが、独立に導入された $10^{40}$ 倍の大数結合スケーリング則 $\kappa = \kappa_0 \sqrt{N}$ に場の理論的・第一原理的導出が存在しない（未定義）」** と整理される。

---

## 7. 単位換算バグの独立検算と修正内容

### 7.1 分子・分母の独立検算と修正
- **修正前の問題**:
  旧コード（[`core.py`](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py) L56, L305）では、ハッブル定数 $H_0\text{ [km/s/Mpc]}$ を $\text{s}^{-1}$ に換算する分母に `(KPC * 1e6)` を代入していた。
  `KPC` は $1\text{ kpc}$ であるため、`(KPC * 1e6)` は $10^6\text{ kpc} = \mathbf{1\text{ Gpc}}$ となり、分母が本来の $1\text{ Mpc} = 10^3\text{ kpc}$ より 1000 倍過大であった。
  その結果、算出される $a_0^* = \frac{c H_0}{2\pi}$ は本来の $1.042 \times 10^{-10}\text{ m/s}^2$ より **正確に 1000 倍過小（$1.042 \times 10^{-13}\text{ m/s}^2$）** となっていた。
- **実施した限定修正（2か所のみ）**:
  1. [`src/gyro_gravity_theory/core.py` L56](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L56):
     ```python
     # 修正前: A0_STAR: float = C * (H0 * 1000.0 / (KPC * 1e6)) / (2.0 * np.pi)
     A0_STAR: float = C * (H0 * 1000.0 / (KPC * 1e3)) / (2.0 * np.pi)  # m/s^2
     ```
  2. [`src/gyro_gravity_theory/core.py` L305](file:///D:/Workspace/Labs/universal-rotation-hypothesis/src/gyro_gravity_theory/core.py#L305) (`compute_a0_star()` 内):
     ```python
     # 修正前: h0_s = h0_used * 1000.0 / (KPC * 1e6)
     h0_s = h0_used * 1000.0 / (KPC * 1e3)
     ```
  ※ `G_ASTRO` などにある `KPC * 1000.0**2` や `KPC * 1e6` は速度単位換算（$\text{km/s}$）に由来するため変更せず維持。

### 7.2 テスト05aにおける修正前速度比（実測値 約0.224）
- 深-MOND極限における理論速度比は $(1.042 \times 10^{-13} / 1.200 \times 10^{-10})^{0.25} \approx 0.171$ である。
- しかし、テスト05aの設定（有限半径 $r=100\text{ kpc}$、バリオン質量 $M=10^{10} M_\odot$、内挿関数 `simple`）では、ニュートン重力加速度 $a_N$ が完全にゼロではない有限値を持つ。
- このため、**修正前の本番デフォルト呼び出し `core.v_gyro_interpolating(a0_star=None)` による実測速度比は約 0.224 であった**（修正後は理論値 $\approx 0.965$ と厳密に一致）。

---

## 8. 監査単体テスト実行結果（修正前履歴 vs 修正後回帰結果）

テストスイート [`tests/test_physics_audit.py`](file:///D:/Workspace/Labs/universal-rotation-hypothesis/tests/test_physics_audit.py) における実行結果。

### 8.1 修正前履歴（Phase 1 監査結果: 8 passed, 3 xfailed）
`core.py` の単位換算修正前に実行された公式監査記録。3件の本番不具合が厳密に検出されていた。

| テスト関数名 | 監査分類 | 判定結果 | 検出理由（修正前） |
| :--- | :--- | :---: | :--- |
| `test_01_a0_star_production_unit_conversion` | 本番関数の検査 | **XFAIL** | `core.py L56`: 1 Mpc を 1 Gpc と誤認し $a_0^*$ が 1000 倍過小（1.042e-13） |
| `test_02a_ec_torsion_dimensional_analysis` | 模型性質の確認 | **PASSED** | EC項の次元が逆時間 [$\text{s}^{-1}$] であり加速度ではない事実を検証 |
| `test_02b_production_unit_consistency_function` | 本番関数の検査 | **XFAIL** | `core.check_unit_consistency()` の出力 `A0_ratio` が 1000 倍過小（0.000868） |
| `test_03a_v_gyro_additive_production_far_keplerian_limit` | 本番関数の検査 | **PASSED** | 遠方（$r=10000\text{ kpc}$）で対数勾配がケプラー勾配 $-0.5$（$v \propto r^{-1/2}$）に漸近 |
| `test_03b_v_gyro_additive_production_gyro_dominated_limit` | 本番関数の検査 | **PASSED** | 質量極小域で対数勾配が $-1.0$（$v \propto 1/r$）に漸近 |
| `test_04_mond_identity_under_same_a0` | 本番関数の検査 | **PASSED** | 同一 $a_0$ 下で MOND との差が厳密に $0.0$ |
| `test_05a_v_gyro_interpolating_production_default_bug` | 本番関数の検査 | **XFAIL** | デフォルト呼び出しがバグ値 `A0_STAR` を参照し、速度比が約 0.224 に激減 |
| `test_05b_mond_distinguishability_theoretical_property` | 模型性質の確認 | **PASSED** | 深-MOND領域で速度比が $(1.042/1.200)^{0.25} \approx 0.965$ となり識別可能 |
| `test_06a_compute_bec_amplification_production_function` | 本番関数の検査 | **PASSED** | 引数 `f_align_thermal` を変更すると増幅率が反比例する初期値依存性を確認 |
| `test_06b_bec_polarization_bound_and_lnh_coupling_separation` | 模型性質の確認 | **PASSED** | BEC偏極率上限 $1.0$ と大数結合スケーリング $10^{40}$ の分離構造を確認 |
| `test_07_test_body_extensivity_cancellation_and_false_positive_guard` | 模型性質の確認 | **PASSED** | `atol=0.0` で試験体の $N_{\text{test}}$ 相殺を確認し、偽陽性防止ガードを確認 |

- **修正前サマリー**: **8 passed, 3 xfailed**

---

### 8.2 修正後回帰結果（Phase 2 回帰テスト結果: 11 passed）
`core.py` の単位換算2か所修正後、xfail マーカーを解除して実行した公式回帰テスト結果。許容誤差や期待値は一切緩めず、厳密比較（`atol=0.0`）を維持した。

| テスト関数名 | 監査分類 | 検査対象 | 判定結果 | 保証する範囲 | 保証しない範囲 |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `test_01_a0_star_production_unit_conversion` | 本番関数の検査 | `core.compute_a0_star()` / `core.A0_STAR` | **PASSED** | $a_0^* \approx 1.042 \times 10^{-10}\text{ m/s}^2$ の数値的正確性 | $a_0^*$ が実際の銀河観測を説明できるかどうか |
| `test_02a_ec_torsion_dimensional_analysis` | 模型性質の確認 | 数理式 `(8πG/c^2) f n (ħ/2)` のSI単位解析 | **PASSED** | EC項の次元が厳密に角速度 [$\text{s}^{-1}$] であること | カルタン接続やトーション理論の幾何学的正当性 |
| `test_02b_production_unit_consistency_function` | 本番関数の検査 | `core.check_unit_consistency()` の出力 | **PASSED** | 本番関数の `A0_ratio` が理論値 $\approx 0.8685$ と整合すること | 各物理定数の実験測定誤差 |
| `test_03a_v_gyro_additive_production_far_keplerian_limit` | 本番関数の検査 | `core.v_gyro_additive()` の遠方代数解 | **PASSED** | 遠方でケプラー勾配 $-0.5$ に漸近し、フラットにならないこと | 実際の銀河外縁部における観測データへの適合性 |
| `test_03b_v_gyro_additive_production_gyro_dominated_limit` | 本番関数の検査 | `core.v_gyro_additive()` のジャイロ支配域 | **PASSED** | 質量極小域で対数勾配が $-1.0$ に漸近すること | 質量ゼロの物理的実在性 |
| `test_04_mond_identity_under_same_a0` | 本番関数の検査 | `core.v_mond_interp()` vs `core.v_gyro_interpolating()` | **PASSED** | 同一 $a_0$ 下で両者が完全同一の方程式を解いていること | 異なる $a_0$ を与えた場合の観測的予測差 |
| `test_05a_v_gyro_interpolating_production_default_call` | 本番関数の検査 | `core.v_gyro_interpolating(a0_star=None)` | **PASSED** | デフォルト呼び出しで正しい理論速度比 $\approx 0.965$ を返すこと | 望遠鏡による実際の天体観測誤差 |
| `test_05b_mond_distinguishability_theoretical_property` | 模型性質の確認 | 深-MOND漸近解における $a_0$ 差の影響 | **PASSED** | 速度比が $(1.042/1.200)^{0.25} \approx 0.965$ となる理論的性質 | 実際の銀河サンプルにおける観測散らばり |
| `test_06a_compute_bec_amplification_production_function` | 本番関数の検査 | `core.compute_bec_amplification()` | **PASSED** | 本番関数の増幅率計算が初期値に完全に依存した相対値であること | 実際の星間空間における熱的偏極率 |
| `test_06b_bec_polarization_bound_and_lnh_coupling_separation` | 模型性質の確認 | 偏極率上限と大数結合スケーリングの分離 | **PASSED** | 偏極率上限 $1.0$ と $\eta_{\text{LNH}} = 10^{40}$ の独立な分離構造 | 未知の長距離スピン結合機構の有無 |
| `test_07_test_body_extensivity_cancellation_and_false_positive_guard` | 模型性質の確認 | 重心運動方程式における示量性相殺 | **PASSED** | `atol=0.0` で試験体の $N_{\text{test}}$ 相殺および偽陽性防止ガード | 宇宙全体の総粒子数 $N_{\text{universe}}$ を用いる大数仮説の成否 |

- **修正後サマリー**: **11 passed in 0.71s**
- **FAILED**: 0 件, **XFAIL**: 0 件, **XPASS**: 0 件, **未実行**: 0 件

---

## 9. 未検証事項（本監査で未検証・未解決として残された項目）

本単体テストスイートで保証された範囲は、上記テスト表の「保証する範囲」に記載された数理的・実装的性質に限定される。以下の宇宙物理学・観測天文学上の重要項目は**一切検証されておらず、「未検証事項」として残されている**（これを「監査完了」や「検証済み」として扱ってはならない）。

1. **実銀河観測データ（SPARCデータセット 175銀河）との残差検証（未検証）**:
   本提案モデルと標準MOND、NFWハローについて、実際の銀河回転曲線データとの $\chi^2$ フィッティングおよび動径加速度関係（RAR）の残差比較は未実施。
2. **弾丸銀河団（Bullet Cluster 1E 0657-558）の重力レンズ分離（未検証）**:
   バリオンガス（X線放射）と質量中心（弱い重力レンズ）が空間的に分離している観測事実に対し、本モデルが重力場をどのように説明できるかは未検証。
3. **宇宙マイクロ波背景放射（CMB）非等方性パワースペクトル（未検証）**:
   Planck 2018 の音響ピーク（第1〜第3ピーク）の再現性、および宇宙全体の角速度制限（$\omega < 10^{-29}\text{ rad/s}$）との詳細な整合性は未検証。
4. **トーションテンソルからの有効加速度導出（未定義・未検証）**:
   接続のねじれテンソルから測地線偏差方程式を通じて試験粒子に働く有効加速度を第一原理的に導出する数理モデルは未定義。
5. **ディラック大数 $\sqrt{N_{\text{universe}}}$ の局所結合則の理論的導出（未定義・未検証）**:
   因果関係のない宇宙全体の総粒子数が局所力学の結合定数に作用する場の理論・作用積分（ラグランジアン）からの導出は未定義。
6. **真空スピン角運動量密度 $S_{\text{vac}}$ の量子場としての定義（未定義・未検証）**:
   $a_0^* = \frac{cH_0}{2\pi} \left(\frac{S_{\text{vac}}}{\hbar/2}\right)$ と置かれているが、$S_{\text{vac}}$ の物理的実体および測定方法は未定義。
7. **既存の画像・グラフ生成物（`output/` 等）の再検証（未検証）**:
   本番コードの修正が完了したため、過去に出力された比較画像（`rotation_curve_comparison.png` 等）は修正前のバグ値に基づいている可能性が高く、新定数による再描画および数値検証が完了するまではすべて「未検証状態」として扱う。
