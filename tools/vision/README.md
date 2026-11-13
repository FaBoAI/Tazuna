# tools/vision — カメラと画像認識

ロボットの正面に固定したカメラ 1 台（Logitech C920）の写真から、机の上の物（箱型の物、円柱、容器、容器の中の物）の位置と寸法を求めます。カメラは、焦点・露出・ホワイトバランスを固定して撮ります。画素とロボットの座標は、カメラモデル（`config/camera_model.json`）で対応させます。認識は、机の色と違う画素を物として切り出します。明るさの基準は、同じ写真の机の明るさから取ります。使うのは numpy と OpenCV だけです。

| ファイル | 役割 |
|---|---|
| `camera.py` | 撮影（`grab()`）、カメラの設定の固定と解除（`lock_settings()`、`restore_auto()`）、露出の切り替え（机用の `desk`、白い紙用の `paper`）、カメラモデル（`CameraModel`、`fit_camera()`） |
| `fastcam.py` | `FastCamera`: カメラを開いたままにして、いつも最新の 1 枚を持つスレッド（毎回開くと、1 枚に約 1.2 秒かかるため） |
| `scene.py` | 認識の本体。`perceive()` が物の一覧（`Scene`）を返す。物がまだそこにあるかの確認、容器の中に見える色、爪の位置の検出（`detect_jaws()`）もここ |
| `vision.py` | 最初の版（v1）からある画像処理の部品（色の切り出し、指先の検出 `find_tip()`、画像への描き込みと保存 `save_view()`）。低レベルのコマンドと校正のコマンドが使う |
| `blockfit.py`、`capfit.py` | 写真 1 枚から箱型の物を測る `measure_block()` と、立っている円柱を測る `fit_cap()`。中身は `scene.perceive()` を呼ぶ、互換のための入口 |

## 入口

- `scene.perceive(img_bgr, cam, cfg, arm_q, jaw_q_deg)`: 写真 1 枚から `fdtypes.Scene` を返します。`scene.annotate()` は、人や AI が見るための絵を作ります。
- `scene.object_present()`、`scene.object_presence()`、`scene.objects_in_container()`、`scene.detect_jaws()`: スキル（`pick/skills.py`）が、動作の途中と最後の確認に使います。
- `fastcam.FastCamera`（`latest()`、`wait_fresh()`）、`camera.grab()`、`camera.CameraModel.load()`。
- `fd` コマンド: `scene`、`pick`、`take`、`do`、`repl`（認識と確認）、`snap`、`selftest`、`camera lock|auto`、校正の `ref`、`calib-at`、`calib-fit`、`calib-roll`。`fd tool calibrate` と `fd draw` は、実機では `camera.grab(exposure="paper")` で撮ります。
- 単体の実行: `python tools/vision/scene.py frame.jpg [annotated.jpg]`（保存した写真に何が見えるかを JSON で出す）、`python tools/vision/blockfit.py frame.jpg view.jpg [color]`、`python tools/vision/capfit.py frame.jpg view.jpg [x0,y0,x1,y1]`。

## 必要なもの

- 読み込むフォルダ: `ik`（写真からアームを除くためと、爪の形）、`cli/fdtypes.py`（`Scene`、`SceneObject`）。カメラモデルの場所は `tazuna_paths` が教えます。
- ハードウェア: カメラ（定数 `camera.DEV`。値は [docs/MANUAL.md](../../docs/MANUAL.md)「はじめに」）と `v4l2-ctl`。`scene.py`、`vision.py`、`blockfit.py`、`capfit.py` は計算だけなので、保存した写真でも動きます。

## テスト

`tests/test_scene.py`（50 件。保存した写真と、手で測った正解値。`--live` を付けたときだけ、実カメラを 1 回開きます）、`tests/test_ink.py`（露出の切り替え）。`tests/test_skills.py` も、保存した写真で認識を通します。

## 実機での状況

2026-11-11 の実機のピックで、この認識と最後の確認を使いました。爪の位置の検出は、実機のピックでは記録しただけで、位置の補正にはまだ使っていません。紙用の露出での撮影は、`fd` からは実機でまだ行っていません。認識の限界（同じ色の物が接していると 1 個に見える、机に近い色は見えない、など）は、MANUAL 9 章にあります。
