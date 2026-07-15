# Game_Phase

Game_Phase(GP) とは Ryfamate における終盤の NNUE ハンドオーバーに用いられているであろう、局の進行度を示す関数をアピール文章の追試と著者の妄想により再現したファンアートである。

## GP 算出

### 特徴量

| 特徴量 | 説明 |
|---|---|
| WHAND | 駒価値加重手駒合計 (歩=1, 香桂=3, 銀金=5, 角=8, 飛=10) |
| PROMO | 盤上の成駒数 (両者合計) |
| CAMP | 敵陣3段以内の玉以外の駒数 |
| KING | 玉前進シグナル (0〜12) |
| EXPOSURE | 玉露出度 = 16 − 両玉周囲の味方駒合計 |
| PLY | 手数 (ply-mode で切替可) |

### 算出式

```
S = W_Hand·WHAND/Ref_Hand + W_Promo·PROMO/Ref_Promo + W_Camp·CAMP/Ref_Camp
  + W_King·KING/Ref_King   + W_Expo·EXPOSURE/Ref_Expo + W_Ply·PLY/Ref_Ply

GP_core = 1 − exp(−S / Scale)                                    … [0, 1)
bonus   = 0.5 · clamp((KING − Onset) / (Full − Onset), 0, 1)    … [0, 0.5]
GP      = clamp(GP_core + bonus, 0.0, 1.5)
```

### 現行パラメータ

| | Weight | Ref |
|---|---|---|
| Hand | 0.1253 | 9.2786 |
| Promo | 0.0000 | 5.6865 |
| Camp | 0.1078 | 7.3143 |
| King | 0.0979 | 1.5329 |
| Ply | 0.5243 | 139.16 |
| Exposure | 0.0000 | 12.245 |

Scale = 1.2853, Nyugyoku Onset = 0.5, Full = 2.5

パラメータは `gp_core.py torch-fit` で棋譜データから再最適化できる。

### GP 値の目安

| GP 範囲 | 局面 |
|---|---|
| 0.0 – 0.2 | 序盤 |
| 0.2 – 0.4 | 中盤 |
| 0.4 – 0.8 | 終盤 |
| 0.8 – 1.5 | 入玉 |

## gp_group_psv.py

PackedSfenValue (.bin) 教師データを GP 値でバケット分割するツール。
[rshogi](https://github.com/SH11235/rshogi) / [tatara](https://github.com/SH11235/tatara) / [bullet-shogi](https://github.com/SH11235/bullet-shogi) の PSV フォーマットと互換。

```bash
# 単一ファイル (0.1 刻み 16 バケット)
python gp_group_psv.py /path/to/teacher.bin

# フォルダ内の全 .bin を一括処理
python gp_group_psv.py /path/to/chunks/

# torch レーン (GPU 自動検出)
python gp_core.py torch-group /path/to/teacher.bin
```

出力は `./teacher_gp/` のようなサブディレクトリを自動生成し、その中にバケットファイルを出力する。`--out-dir` で変更可。

```
./teacher_gp/
  ├── teacher_gp_0.00-0.10.bin
  ├── teacher_gp_0.10-0.20.bin
  │   ...
  ├── teacher_gp_1.40-1.50.bin
  └── teacher_gp_report.txt
```

### その他のオプション

```bash
# ply-mode 指定 (dedup 教師は state 推奨)
python gp_group_psv.py /path/to/teacher.bin --ply-mode state

# レポートのみ (ファイル出力なし)
python gp_group_psv.py /path/to/teacher.bin --report-only
```

## 依存

- Python 3.8+
- オプション: `numba`, `torch`

## パイプライン例

```
rshogi gensfen → teacher.bin
                    ├→ Game_Phase GP 分割 → gp_*.bin → rshogi shuffle → tatara nnue-train
                    └→ tatara progress-kpabs-train (⚠ GP 分割前のデータを使うこと)
```

## ライセンス

[LICENSE.txt](LICENSE.txt)

## 参考文献

水無瀬香澄ほか. "Ryfamate アピール文書." *第31回世界コンピュータ将棋選手権*, コンピュータ将棋協会, 2021, www.apply.computer-shogi.org/wcsc31/appeal/Ryfamate/wcsc31_ryfamate_20210517.pdf.
