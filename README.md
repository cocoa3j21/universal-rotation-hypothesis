# 🌌 全宇宙回転ジェネレーション仮説

**銀河回転曲線を、ジャイロ比喩モデルで遊ぶ Python シミュレーション（示唆・解釈実験）**

Zenn 記事「【物理学終了】全宇宙回転ジェネレーション仮説」で紹介したコードの開発正本です。
記事タイトルのノリはそのままですが、**数値結果は玩具モデルの対照**として読んでください（宇宙論の「結論」とは書きません）。

| リンク | URL |
| :--- | :--- |
| **GitHub（旧デモ・公開既定）** | [universal-rotation-hypothesis `main`](https://github.com/cocoa3j21/universal-rotation-hypothesis/tree/main) — Zenn 初出向けの短いデモ履歴 |
| **GitHub（Labs 正本・v7）** | [universal-rotation-hypothesis `v7`](https://github.com/cocoa3j21/universal-rotation-hypothesis/tree/v7) — 注入回復パイプライン・`output/`・`tools/` 一式 |
| **Labs 索引（親）** | [`../README.md`](../README.md) |
| Buy Me a Coffee | https://www.buymeacoffee.com/cocoa3j21 |

## 開発正本と公開リポジトリ

| 役割 | 場所 |
| :--- | :--- |
| **開発正本（ローカル）** | `D:\Workspace\Labs\universal-rotation-hypothesis\`（`ai-budget-meter` には未収録） |
| **リモート `origin/main`** | [旧デモ用](https://github.com/cocoa3j21/universal-rotation-hypothesis/tree/main) — **上書きしない**（force push 禁止） |
| **リモート `origin/v7`** | [Labs 正本のミラー](https://github.com/cocoa3j21/universal-rotation-hypothesis/tree/v7) — ローカル `main` から push |
| **巨大成果物** | `output/` の CSV / JSON / PNG（再現用。ハッシュは `output/SHA256SUMS.txt`） |
| **Zenn 原稿** | [`articles/zenn_article_final.md`](articles/zenn_article_final.md)（frontmatter `published: true` — 公開状態は Zenn 側で要確認） |

**トーン:** 示唆・玩具モデル・解釈実験。観測データとの一致は**数値遊びの対照**であり、標準宇宙論の代替宣言とはしません。

---

## 代表スモーク（エントリ）

ルートの [`rotation_curve.py`](rotation_curve.py) で、ニュートン / ジャイロ加算 / MOND 補間の 3 曲線を比較します（数十秒程度）。

```bash
cd Labs/universal-rotation-hypothesis   # パスは環境に合わせて調整
pip install -r requirements.txt
python rotation_curve.py
```

**確認（2026-11-25）:** 上記コマンドで終了コード 0。出力例: `v_N (r=10kpc) ≈ 145 km/s`、`v_G ≈ 172`、`v_M ≈ 179`。図は `output/rotation_curve_comparison.png`（`--output` で変更可）。

> `config/params.yaml` の読み込みで Windows ロケールにより警告が出る場合があります。そのときはスクリプト内デフォルト値で続行します（スモーク目的なら許容）。

---

## フルパイプライン（参考）

| 層 | パス | 内容 |
| :--- | :--- | :--- |
| ライブラリ | `src/gyro_gravity_theory/` | コア計算・注入回復パイプライン |
| 一括ツール | `tools/` | キャリブレーション・バッチ・監査スクリプト |
| 仕様・監査 | `docs/` | 実装計画・監査メモ |
| テスト | `tests/` | `pytest`（パイプライン系） |

フル再現は時間・ディスクを要します。まずは代表スモーク、必要に応じて `tools/run_v2_full_pipeline.py` 等を README や `docs/` から選んでください。

```bash
python -m pytest -q tests/
```

---

## 出力例（README 用・ルート）

ニュートン力学（青線）では速度が落ち込むが、ジャイロモデル（赤線）はフラットな回転曲線を**示すプロット**になります。

![rotation_curve.png](rotation_curve.png)

---

## 今後の予定

| # | 項目 | 状態 |
| :---: | :--- | :--- |
| 1 | GitHub 同期方針 | **済（選択肢2）** — 旧デモは `origin/main` のまま、v7 系は **`origin/v7`** に push |
| 2 | 代表スモーク | **確認済み**（2026-11-25） |
| 3 | Labs 索引 | 親 [`../README.md`](../README.md) で main / v7 を併記 |
| 4 | Zenn | 原稿は `articles/`。公開・更新は人手（次フェーズ） |
| 5 | `main` へのマージ | **未実施**（unrelated 履歴のため）。将来マージする場合も force は使わない |

---

## ☕ Support

この手の妄想検証を続けるためのコーヒー代をいただけると嬉しいです。  
👉 [Buy Me a Coffee](https://www.buymeacoffee.com/cocoa3j21)