# tools/sim — 実機の代わりになるシミュレーション

実機に触れずにプログラムを動かすための代役です。`fd --sim <コマンド>` とテストが使います。`sim_bus.SimBus` は、本物の `servo_bus.Bus` と同じ呼び方ができる、サーボの模型です。`robot.Arm(bus=SimBus())` のように渡すと、動作と安全検査のプログラムがそのまま動きます。`simpaper.SimPaper` は、紙・ペン・カメラの模型です。ハーネスには教えない「本当の値」（ペン先のずれ、紙に触れる高さ、位置の誤差など）を隠し持ち、ハーネスには、関節の角度と、あとで撮った写真だけを見せます。**シミュレーションの成功は、実機の成功ではありません。** 接触の力、インクの出方、紙が動くことなどは、再現していません。

| ファイル | 役割 |
|---|---|
| `sim_bus.py` | `SimBus`: 運動学だけのサーボの模型（速度の上限つきで目標へ近づく。障害物 `blocker` を置ける）。`fd --sim` のアームが使う |
| `phys_bus.py` | `PhysBus`: トルクまで扱う模型（重力、摩擦、接触のばね）。既定の値は、実機のログに合わせてある。`STICK_SLIP_20261111` は、2026-11-11 のペン先の記録に合わせた「引っかかっては飛ぶ」摩擦 |
| `fakebus.py` | `FakeBus`: 疑似端末（pty）の向こうで Feetech のパケットに答える、疑似サーボ 6 台。本物の `servo_bus.Bus` を、そのまま試すためのもの |
| `simpaper.py` | `SimPaper`: 紙・ペン・カメラの模型。`night_truths()` は、2026-11-11 の夜に似せた誤差の一式 |
| `sim_fuzz.py` | 単体で実行するスクリプト。ランダムな配置で「計画 → シミュレーションで実行」をくり返し、計画できた割合と所要時間を出す |

## 入口

- `sim_bus.SimBus(realtime=False)`: `fd.make_arm(True)`（`fd --sim`）が作ります。
- `phys_bus.PhysBus(cfg, ...)`: `fd --sim track-test --phys` と `sim_fuzz.py --phys` が使います。
- `simpaper.SimPaper(arm, cam, ...)`: `penskill.sim_world()` が作ります（`fd --sim tool calibrate`、`fd --sim draw`）。
- `fakebus.FakeBus()` と `FakeBus.bus()`: テスト専用です。
- `python tools/sim/sim_fuzz.py [-n 60] [--tiers slow,normal,fast] [--seed 1] [--phys] [--json out.json]`
- `fd` コマンド: `--sim` を付けた全部のコマンド。`--sim` はサーボに触れません。ただし、カメラを使うコマンド（`scene`、`pick`、`take`、`do`、`snap` など）は、保存した写真を `--frame` で渡さないと、実カメラを開きます。`tool …`、`track-test`、`draw` は、実カメラも開きません（[docs/MANUAL.md](../../docs/MANUAL.md) 1 章の表と、4 章「シミュレーションでの予行」）。

## 必要なもの

- 読み込むフォルダ: `motion/servo_bus.py`（同じレジスタの名前と例外を使う）、`ik`、`vision/camera.py`（`simpaper.py` がカメラモデルを使う）。`sim_fuzz.py` は、`pick`、`motion`、`cli/fdtypes.py` も使います。
- ハードウェア: なし。`fakebus.py` が開くのは、自分で作った疑似端末だけです。`servo_bus.py` を import するので、Python の `scservo_sdk` は要ります。

## テスト

`tests/test_sim.py`（`SimBus` の上で、動作と安全検査を 97 件）、`tests/test_motion_phys.py`（`PhysBus`、16 件）、`tests/test_bus_fake.py`（`FakeBus`、77 件）、`tests/test_strokes.py`（`SimPaper`）、`tests/test_tracking.py`（`PhysBus` の引っかかりの摩擦）。ほかのテストも、多くが `SimBus` のアームを使います。

## 実機での状況

実機は使いません。`PhysBus` の既定の値は、実機のログ（静止と、25 °/s 以下の動き）に合わせたものです。それより速い動きについてこの模型が言うことは、測定ではなく予測です（`phys_bus.py` の冒頭の説明）。
