# tools/calib — 校正を調べ直すためのスクリプト

校正（カメラの位置、関節の原点、机の高さ）を確かめたり、調べ直したりするときに、手で実行するスクリプトです。`fd` のコマンドからは呼ばれません。ふだんの校正は、`fd` のコマンド（`ref`、`calib-at`、`calib-fit`、`calib-roll`。中身は `tools/cli/fd.py`）で行い、結果は `config/` のファイルに入ります。`table-fit` は `fd probe` の値（実際より 3〜18 mm 深い）を使うので、机の高さには使いません。`config/` の値は、作者の装置 1 組で測ったものです。別のアームや別の配置では、測り直すまで、作業のコマンドを実機で使わないでください。測り直しの手順は [docs/SETUP_SO101.md](../../docs/SETUP_SO101.md) にあります。

| ファイル | 役割 |
|---|---|
| `visual_touch.py` | 高さの点検。閉じた指先を 2 mm ずつ下ろしては撮影し、写真の中のグリッパーの動きから、指先が机に最初に触れた高さを求める。**実機が動く**（`fd tip`、`fd snap` などを順に呼ぶ。1 か所で約 30 秒） |
| `kinfit.py` | カメラの位置と、関節の原点のずれ（shoulder_lift、elbow_flex、wrist_flex）を、校正点と机の高さのデータから同時に当てはめる解析。実機は使わない |
| `sheet.py` | 何枚かの画像を 1 枚に並べた、一覧の画像を作る |

## 入口

- `python tools/calib/visual_touch.py X,Y [X,Y ...] [--above M] [--below M] [--no-park] [--out FILE]`: 机の上の、物がない場所を m で指定します。待機姿勢から始め、終わると収納します（`--no-park` を付けると収納しません）。
- `python tools/calib/kinfit.py [焦点距離]`: `config/` の `calib_points.json` と `probes.json` を読みます（焦点距離は画素の数で、省略すると 1400）。
- `python tools/calib/sheet.py out.jpg img1 img2 ...`
- `fd` コマンド: このフォルダを使うコマンドはありません。ほかのモジュールからも import されません。

## 必要なもの

- 読み込むフォルダ: `vision`（`camera`、`vision`）。`kinfit.py` は、`ik`、`motion`（`robot.load_config()`）、`cli/fd.py`（定数 `AXIS_TOOL`）も使います。`sheet.py` は OpenCV と numpy だけです。
- ハードウェア: `visual_touch.py` だけが、アームとカメラを使います。`kinfit.py` と `sheet.py` は、保存したデータと画像だけで動きます。

## テスト

テストはありません。

## 実機での状況

`visual_touch.py` は、実機を動かすスクリプトです。`config/config.json` の `calibration_note` によると、2026-11-11 に shoulder_lift の原点を直したときは、カメラで見つけた「最初に触れた高さ」6 か所に合わせました。[docs/MANUAL.md](../../docs/MANUAL.md) の 9 章は、高さを点検し直す方法として、このスクリプトを挙げています。`fd probe` で触って測った高さは、実際より 3〜18 mm 深く出るので、高さの確認には使いません（同じ 9 章）。
