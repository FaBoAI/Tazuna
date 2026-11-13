# tools/draw — 道具（ペン）と線画の描画

グリッパーにテープで留めた道具（ペンなど）をハーネスに登録し、紙の上で校正して、線画を描く機能です。道具は、1 個のファイル（`tool_active.json`。リポジトリには入れません）で表します。登録されている間は、「先端」が道具の先端になります。グリッパーには指令を出さず、`park` や `pick` などは、動く前に断ります。ペンを紙に下ろすのは、`penskill.py` の校正と描画だけです。線があるかどうかは、写真で確かめます。答えは「ある」「ない」「分からない」の 3 通りで、「分からない」ときは、それ以上深く下ろさずに止まります。

| ファイル | 役割 |
|---|---|
| `tooling.py` | 道具のファイル（`Tool`、`active()`、`save()`、`clear()`）。道具を付けたままの待機・収納・空中の移動（`go_ready()`、`rest()`、`free_move()`、`lift()`）。コマンドを断るときの結果（`refusal()`） |
| `penskill.py` | 校正の `calibrate()`（先端 → 接触高さ → 位置補正 → 線の向き）と、描画の `draw()`。下ろし方の決まりは、ここにある |
| `ink.py` | 写真から紙の上の線を読む（`can_show_ink()`、`ladder_readout()`、`marks_readout()`、`stroke_coverage()`）。カメラもロボットも使わない計算 |
| `strokes.py` | 線画のファイルの読み込み（`load()`）、紙の上への配置（`layout()`、`fit_box()`）、線の切り分け、位置補正の式（`fit_map()`、`invert_map()`）、指令への変換（`plan_pieces()`） |
| `tracking.py` | 動いている間に、先端が計画にどれだけ付いていくかを記録して測る（`record()`、`Trace.analyse()`）。ファインモードを使ってよいかの判定（`fine_accepted()`） |
| `drawings/` | 付属の線画。`cat.json`（2026-11-11 の猫、18 画）だけ |

## 入口

- `penskill.calibrate(sess, stages, opts)` と `penskill.draw(sess, drawing, opts)`: `sess` は `PenSession`、`opts` は `PenOptions` です。
- `tooling.active()`: 登録されている道具を返します（なければ `None`）。`robot.Arm` が作られるときに読みます。`tooling.go_ready()` と `tooling.rest()` は、道具を付けたままの待機と片付けです。
- `fd` コマンド: `tool attach|status|set|rest|detach|calibrate`、`draw`、`track-test`。`snap` の結果の `clipped` と `ink_visible` は、`ink.py` が出します。道具が登録されている間の `ready` は、`tooling.go_ready()` が行います。

## 必要なもの

- 読み込むフォルダ: `motion`、`ik`、`pick/skills.py`（時間の測り方などの補助）。`--sim` のときは `sim/simpaper.py`。`ink.py` と `strokes.py` は、ほかのモジュールを import しません。
- ハードウェア: サーボ（`Arm` を通します）とカメラ（撮影の関数は `fd.py` が渡します）。`--sim` を付けると、どちらも使いません。

## テスト

`tests/test_tooling.py`（33 件）、`test_ink.py`（28 件。2026-11-11 の保存写真と正解値）、`test_strokes.py`（30 件）、`test_tracking.py`（13 件）、`test_penskill.py`（23 件）、`test_integration_pen.py`（15 件）。どれも、シミュレーションのアームと紙、保存した写真だけで動きます。

## 実機での状況

**このフォルダの機能（`fd tool …`、`fd draw`、`fd track-test`）は、実機では一度も動かしていません。** 確かめたのは、シミュレーションのアームと紙、2026-11-11 の保存写真だけです。2026-11-11 に実機で猫を 1 枚描いたのは、試作スクリプト（[`../prototypes/pen_draw/`](../prototypes/pen_draw/README.md)）の方です。使い方は [docs/MANUAL.md](../../docs/MANUAL.md) の 4 章に、初回の実機確認の手順は 5 章にあります。
