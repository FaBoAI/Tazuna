# tools/draw — 道具（ペン、筆）と線画の描画

グリッパーにテープで留めた道具（ペン、筆など）をハーネスに登録し、ペンなら紙の上で校正して、線画を描く機能です。道具は、1 個のファイル（`tool_active.json`。リポジトリには入れません）で表します。登録されている間は、「先端」が道具の先端になります。グリッパーには指令を出さず、`park` や `pick` などは、動く前に断ります。ペンを紙に下ろすのは、`penskill.py` の校正と描画だけです。線があるかどうかは、写真で確かめます。答えは「ある」「ない」「分からない」の 3 通りで、「分からない」ときは、それ以上深く下ろさずに止まります。

2026-11-13 に、筆を留めて絵の具で描いたときのことが加わりました（筆で描く機能そのものは [`../paint/`](../paint/README.md)）。道具は、**関節の制限**を持てます（長い柄が腕に当たる筆）。**可動爪の外側に留めた道具**は、爪を固定爪に押し当てて保持できます（`fd tool jaw`。グリッパーに指令を出す、ただ 1 つの例外です）。道具のファイルには、どの爪に付いているか、爪を保持する目標値、先端を空中校正で測った記録が入るようになりました。先端の値は、記録した爪の位置でだけ成り立つので、爪がその位置にないときは、待機姿勢への移動と筆の作業を、トルクを入れる前に断ります（2026-11-14 からは、ペンの校正と描画も）。

2026-11-14 未明には、**ペンの校正（`fd tool calibrate`）が、絵の具用の道具を断る**ようになりました。筆先を空中で測った記録のある道具、可動爪の外側か爪の間に留めたという記録・爪を保持した記録のある道具と、絵のセットアップ（`paint_setup.json`）がある間のすべての道具は、`--pen`（「付いているのはペンで、ペンの校正をする」という申告）がなければ、動く前に `stage: tool` で断り、`fd paint status` を案内します。固定爪に留めたという記録（`fd tool jaw mount fixed`）だけでは、絵の具用になりません（2026-11-14 朝に修正）。判定は `cli/fd.py` の `_painting_tool()` にあります。

2026-11-14 朝（3 回目の修正）には、次のものが変わりました。`fd park`・`fd grip` などを断る文、`fd torque off` の断り、断った `fd ready` の `next` は、絵の具用の道具で絵のセットアップがあるとき、`fd tool rest` ではなく `fd paint rest` を挙げます（`tooling.next_locked()`、`refusal()`、`torque_off_block()`）。爪のことで断った `fd tool calibrate` と `fd draw` は、`next` に「nothing moved.」とその留め方の直し方を、`say` に爪のことを入れます（`penskill._jaw_refused()`）。`fd tool jaw status` と、成功した `fd tool jaw mount moving_outer` は、筆先が未測定で絵のセットアップがある間、`fd paint status` を先に挙げます。`fd tool jaw` のオプションは、省略形では通りません。`tooling.py` と `tooljaw.py` の文は、逆引用符の中に仮の名前を出さなくなりました（`penskill.py` には、`fd tool attach --name pen --along-fingers LEN`、`fd draw NAME` などが残っています）。

| ファイル | 役割 |
|---|---|
| `tooling.py` | 道具のファイル（`Tool`、`active()`、`save()`、`clear()`）。道具を付けたままの待機・収納・空中の移動（`go_ready()`、`rest()`、`free_move()`、`lift()`）。コマンドを断るときの結果（`refusal()`）。道具ごとの関節の制限（`Tool.joint_limits_deg`、`check_joint_limits()`）。可動爪に留めた道具の爪（`move_jaw()`、`hold_jaw()`、`jaw_in_place()`、`jaw_tolerance()`、`jaw_lost()`）。測った先端の保存 `set_measured_tip()`。シミュレーションの道具の写し（`sim_tool_state()`） |
| `tooljaw.py` | コマンド `fd tool jaw status\|mount\|close\|open` の本体。動かす前の断り（どの爪に付いているかの記録、筆先が机に近づかないか）、道具のファイルの更新、結果の `next` と `say` |
| `penskill.py` | 校正の `calibrate()`（先端 → 接触高さ → 位置補正 → 線の向き）と、描画の `draw()`。下ろし方の決まりは、ここにある |
| `ink.py` | 写真から紙の上の線を読む（`can_show_ink()`、`ladder_readout()`、`marks_readout()`、`stroke_coverage()`）。カメラもロボットも使わない計算 |
| `strokes.py` | 線画のファイルの読み込み（`load()`）、紙の上への配置（`layout()`、`fit_box()`）、線の切り分け、位置補正の式（`fit_map()`、`invert_map()`）、指令への変換（`plan_pieces()`） |
| `tracking.py` | 動いている間に、先端が計画にどれだけ付いていくかを記録して測る（`record()`、`Trace.analyse()`）。ファインモードを使ってよいかの判定（`fine_accepted()`） |
| `drawings/` | 付属の線画。`cat.json`（2026-11-11 の猫、18 画）だけ |

## 入口

- `penskill.calibrate(sess, stages, opts)` と `penskill.draw(sess, drawing, opts)`: `sess` は `PenSession`、`opts` は `PenOptions` です。
- `tooling.active()`: 登録されている道具を返します（なければ `None`）。`robot.Arm` が作られるときに読みます。`tooling.go_ready()` と `tooling.rest()` は、道具を付けたままの待機と片付けです。
- `tooling.move_jaw(arm, "close" | "open", to=…)`: 可動爪の外側に留めた道具の爪を、15 ticks ずつ動かし、付いて来なくなった所で止めます。固定爪に当たれば、押し付けて保持します。`tooling.hold_jaw(arm)` は、トルクを切って消えた押し付けを戻します。`tooling.jaw_in_place(arm, tool)` は、爪が道具のファイルの位置にあるかを読むだけで確かめ、なければ `SafetyError` です（押し付けで保持している爪は 40 ticks、保持していない爪は 8 ticks まで。固定爪に付いた道具と、爪の位置の記録がない道具は比べません）。`brushwork.BrushRun.power()`、`go_ready()`、`fd calib-air` と、`penskill.py` の `power()`（`fd draw`、`fd tool calibrate`。2026-11-14 から）が、動く前に呼びます。断るときの文は、その留め方の直し方を、そのまま打てるコマンドで言います。
- `tooljaw.main(argv)`: `fd tool jaw` の後ろの語を受け取り、結果の辞書を返します（`fd --sim` のときは先頭に `--sim`）。`fd tool jaw --help`（`-h`、`help`、サブコマンドの後ろの `--help` も）は、使い方とオプションの説明を、結果の `usage` と `help` に入れて返します。途中でサーボが応答しなくなったときは、`stage` が `hardware` の結果を返し、爪の位置が分からなければ、道具のファイルを「爪の位置は不明、保持なし、先端は未測定」に書き換えます。
- シミュレーションの道具: `fd --sim` は、シミュレーション用の道具（`fd --sim tool attach` で登録したもの）がなければ、実機の道具のファイルを読むだけで使います。その状態で `--sim` のコマンドが道具を書き換えると（`fd --sim tool jaw mount`、`fd --sim tool set` など）、実機の記録の**写し**を、木の外のシミュレーション用のファイルに作ります（印 `sim_copy_of` 付き）。写しを読むのは、実機の道具のファイルが写したときのままの間だけで、実機の側が変わると、写しは使わずに実機の記録へ戻ります。`tooling.sim_tool_state()` が、いまどちらを読んでいるかを答えます。
- `fd` コマンド: `tool attach|status|set|rest|detach|calibrate|jaw`、`draw`、`track-test`。`snap` の結果の `clipped` と `ink_visible` は、`ink.py` が出します。道具が登録されている間の `ready` は、`tooling.go_ready()` が行います。`fd tool attach|set --joint-limit 関節=下:上` が関節の制限を書き、`fd tool rest` は、絵のセットアップのキャンバスの上へは畳みません（`--over-canvas` を付けたときを除く。判定は `paint/paintskill.py` の `rest_report()`）。

## 必要なもの

- 読み込むフォルダ: `motion`、`ik`、`pick/skills.py`（時間の測り方などの補助）。`--sim` のときは `sim/simpaper.py`（`tooljaw.py` は `sim/sim_bus.py`）。`ink.py` と `strokes.py` は、ほかのモジュールを import しません。
- ハードウェア: サーボ（`Arm` を通します）とカメラ（撮影の関数は `fd.py` が渡します）。`--sim` を付けると、どちらも使いません。`fd tool jaw close / open` は、グリッパーのサーボだけにトルクを入れます（トルク OFF で畳んだ腕は、そのままです）。実機のロック（`fd hw lock`）が掛かっている間は、サーボを開く前に断ります（`fd tool status` と `fd tool jaw status` は、ファイルの記録だけを答えます）。

## テスト

`tests/test_tooling.py`（49 件。うち 16 件は 2026-11-13〜14 の追加: 道具ごとの関節の制限、シミュレーションの道具の写し、爪が動いていたときの `fd ready`、`fd draw` と `fd tool calibrate` が動く前に爪の位置を確かめ、断るときに `next` と `say` で直し方を言うこと）、`test_tooljaw.py`（35 件。爪の動かし方と保持、爪の位置の照合、途中でサーボが応答しなくなったとき、`fd tool jaw`、シミュレーションの道具の写し、結果に出てくるコマンドの綴り、筆先が未測定の間の `next` の順番、オプションの省略形を断ること）、`test_ink.py`（29 件。2026-11-11 の保存写真と正解値）、`test_strokes.py`（30 件）、`test_tracking.py`（13 件）、`test_penskill.py`（23 件）、`test_integration_pen.py`（15 件）。`fd tool set --joint-limit` の書き方と、`fd tool rest` のキャンバスの判定は、`test_cli_words.py` が確かめます。絵の具用の道具と `fd tool calibrate --pen` も、`test_cli_words.py` です。どれも、シミュレーションのアームと紙、保存した写真だけで動きます。2026-11-14 の 0 時台に失敗していた `test_tooling.py` の 1 件（`--joint-limit` の古い期待値）と、実機がロックされている間の `test_ink.py` の 4 件は、同じ日の未明に直しました（[tests/README.md](../../tests/README.md)）。

## 実機での状況

- **ペンの校正と描画（`fd tool calibrate`、`fd draw`、`fd track-test`）は、実機では一度も動かしていません。** 確かめたのは、シミュレーションのアームと紙、2026-11-11 の保存写真だけです。2026-11-11 に実機で猫を 1 枚描いたのは、試作スクリプト（[`../prototypes/pen_draw/`](../prototypes/pen_draw/README.md)）の方です。使い方は [docs/MANUAL.md](../../docs/MANUAL.md) の 4 章に、初回の実機確認の手順は 5 章にあります。
- **道具の登録と片付け（`fd tool attach / set / status / rest`）と、道具ごとの関節の制限は、2026-11-13 に、筆を付けた状態で実機で使いました。** `fd tool rest --pan -75 --clearance 0.04` と `--pan -100 --clearance 0.04` で畳んでトルクを切り、沈みませんでした（筆先は机の 41 mm 上）。関節の制限は `wrist_flex` 20〜97°、`wrist_roll` ±5°（97 は、103 で終わる動きが 3.7° 手前で止まったための値）です。
- **`fd tool jaw` は、実機では一度も動かしていません。** その日、爪を閉じて押さえる操作は、スクリプトで行いました（固定爪に当たった位置 raw 2041、目標値 2016、負荷 約 110、90 分後 30 ℃）。コマンドの歩幅と判定は、そのスクリプトの値です。爪の位置の照合（8 ticks / 40 ticks）、途中で通信が切れたときの記録、`fd tool rest` のキャンバスの判定、道具を付けたままの持ち上げ方の追加（制限の外から始まる姿勢、関節 1 つでの持ち上げ）も、シミュレーションだけです。使い方は MANUAL の 11 章にあります。
- **2026-11-14 未明に加えたものも、シミュレーションだけです**: `fd tool calibrate` が絵の具用の道具を断ることと `--pen`、ペンの校正と描画の前の爪の照合、`fd ready` と爪のコマンドが言う「次の一手」、`fd --sim tool …` がシミュレーションの道具の写しについて言うこと。`--pen` を付けたペンの校正は、試し書きを、写真に写った紙の上に置きます（絵のセットアップのキャンバスかどうかは見ません）。2026-11-14 朝に変えたもの（上）も、シミュレーションだけです。
