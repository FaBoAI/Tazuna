# tools/motion — アームの動作と安全検査、サーボとの通信、実機のロック

`robot.py` の `Arm` が、SO-101 を動かす入口です。ほかのフォルダは、サーボに直接は書き込まず、`Arm` を通してアームを動かします。動く前に、経路の全体を 1.5° ごとに検査します。見るのは、関節の限界、机より下へ行かないこと、立入禁止の範囲、速度、サーボの状態です。通らなければ、何も動かさずに `SafetyError` で断ります。動いている間は、約 50 Hz で目標を送り、周期ごとに全サーボの位置・負荷・電圧・温度・異常ビットを読みます。異常があるときと、目標に届かないときは、押し付けを解放して止まり、`MotionAborted` を返します。位置はサーボの生の目盛りで扱い、サーボに保存された校正（EEPROM）は書き換えません。

| ファイル | 役割 |
|---|---|
| `robot.py` | `Arm`: 関節の角度と目盛りの換算、安全検査、なめらかな動作、グリッパー、速度段、実機ベンチ。例外の `SafetyError` と `MotionAborted`。設定を読む `load_config()` |
| `servo_bus.py` | `Bus`: Feetech STS3215 サーボ 6 個との通信（生のレジスタ）。ポートを 1 つのプロセスだけが使うためのロックと、応答の検証。失敗は `BusError`。実機のロック（下）が掛かっている間は、実機のポートを開く前に断る |
| `hwlock.py` | 実機のロック（`fd hw lock\|unlock\|status`。2026-11-13 に追加）。ユーザーごとに 1 個のファイル（`~/.config/tazuna/hardware.lock`。`$TAZUNA_HW_LOCK` で別のファイル）があれば「ロック中」で、このフォルダのハーネスは、サーボのポートもカメラも開かない。`info()`、`message()`、`lock()`、`unlock()`、ロックを知らないコピーを探す `copies_without_lock()`（隣のフォルダと、2026-11-14 からは、このフォルダの中の控えも）。ほかのモジュールを import しない |

## 入口

- `robot.Arm`: `torque_on()` と `torque_off()`、`state()`、`solve()`（逆運動学を、このアームの関節の限界つきで解く）、`check_q()` と `check_path()`（動かさずに検査）、`run_path()`（関節の経路を 1 本の動作として実行）、`plan_tip_path()`、`move_tip()`、`move_joints()`、`ready_q()` と `rest_q()`（待機姿勢と収納姿勢）。
- グリッパー: `set_gripper()`、`grasp()`、`grasp_fast()`、`pinch()`。面に触れるまで下ろす `probe_down()` が返す高さは、最初に触れた高さより 3〜18 mm 深い値です（MANUAL 9 章）。
- 速度段（名前の付いた速度の設定）: `speeds()`、`apply_profile()`、`bench_motion()`。実機のベンチに合格していない段は、`SafetyError` で断ります。
- `fd` コマンド: 低レベルの `state`、`torque`、`gains`、`joints`、`tip`、`check`、`grip`、`grasp`、`probe` と、`bench` が、`Arm` を直接呼びます。`pick`、`take`、`do`、`ready`、`park`、`draw`、`paint`、`calib-air` などは、`pick/`、`draw/`、`paint/`、`calib/` を通して同じ `Arm` を使います。
- 実機のロック: `hwlock.info()` が、ロック中ならその内容（理由、いつから）を、そうでなければ `None` を返します。`servo_bus.Bus` と `vision/camera.py` が、実機に触れる前に聞き、ロック中なら `hwlock.HardwareLocked` で断ります（`fd` の結果では `stage: hardware`）。`--sim` のアームとテストは、ロック中でも動きます。**ロックが効くのは、このロックを知っているコピーだけです**（`fd hw status` の `copies_without_lock`。MANUAL 11 章）。道具ごとの関節の制限（`Tool.joint_limits_deg`。2026-11-13 に追加）は、`Arm.q_limits()` が適用します。

## 必要なもの

- 読み込むフォルダ: `ik`（`kinematics`、`gripper_model`）。設定は `config/config.json` から読みます（場所は `tazuna_paths` が教えます）。道具のファイルを読むために、関数の中で `draw/tooling.py` を import します。
- ハードウェア: サーボ（ポートは定数 `servo_bus.PORT`。値は [docs/MANUAL.md](../../docs/MANUAL.md)「はじめに」）。Python の `scservo_sdk`（feetech-servo-sdk）が要ります。`Arm(bus=SimBus())` のように別の `bus` を渡すと、実機なしで動きます（[`../sim/`](../sim/README.md)）。

## テスト

`tests/test_sim.py`（動作と安全検査、起動まわりの安全確認。シミュレーションのサーボの上で 97 件）、`tests/test_motion_phys.py`（トルクまで扱う模型の上で 16 件）、`tests/test_bus_fake.py`（本物の `servo_bus.Bus` を疑似サーボ 6 台につないで 77 件）。`test_tracking.py` はファインモード（`run_path(fine=True)`）を、`test_tooling.py` は道具が登録されているときの断り方と、道具ごとの関節の制限も確かめます。実機のロックは `tests/test_hwlock.py`（16 件）です。

## 実機での状況

2026-11-11 に実機で動かしました。速度段の `slow`、`slow_cruise`、`normal` が実機のベンチに合格していて、既定は `slow_cruise` です（MANUAL 3 章）。ファインモードは、実機ではまだ動かしていません（MANUAL 9 章）。道具を付けたままの移動は、2026-11-13 に、筆を付けた状態で実機で使いました（道具ごとの関節の制限つきの `fd tool rest` と、その日の版の `fd paint` の筆の動き。MANUAL 11 章）。実機のロックは、2026-11-13 の夜から掛けたままです（その間の作業は、シミュレーションとテストだけ）。その夜と 2026-11-14 未明に加えたもの（爪を記録の位置と照らす、ロックを知らないコピーをこのフォルダの中でも探す、ほか）は、シミュレーションでしか確かめていません。安全のための設計は、MANUAL 7 章にまとめてあります。
