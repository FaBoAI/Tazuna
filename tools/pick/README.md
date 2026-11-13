# tools/pick — つかんで置くスキルと、つかみ方の計画

`skills.py` は、1 回の呼び出しで「見る → 計画 → 近づく → つかむ → 運ぶ → 放す → 確かめる」までを行います。机の上の物を箱に入れる `pick_place()` と、箱の中の物を取り出して机に置く `take_out()` があります。動く前に、全部の経路を計画して検査します。各段階のあとには、数値で確かめます。確認に落ちたら、どの段階で、なぜ止まったか、アームがどの状態かを、結果に入れて返します。何も持っていなければ待機姿勢へ戻ります。物を持っているときと、動作に異常があったときは、その場で止まって保持します。`grasp.py` は、どの向きから、どれだけ開いてつかむかを、式で決めます。グリッパーの形を点の集まりで表し、まわりの物や机との空きを調べます。

| ファイル | 役割 |
|---|---|
| `skills.py` | `pick_place()`（机 → 箱）、`take_out()`（箱 → 机）、`go_ready()`（待機姿勢へ）、`park()`（収納してトルク OFF）。アームとカメラをまとめて持つ `Session`。オプションの `PickOptions` と `TakeOptions` |
| `grasp.py` | `plan_grasp()`（つかみ方の候補を、良い順に返す）と `plan_release()`（箱の上で放す位置）。動作も通信もしない、計画だけの計算 |

## 入口

- `skills.pick_place(sess, target_query, container_query, opts)` と `skills.take_out(sess, target_query, container_query, opts)`: 結果は dict です（`ok`、`stage`、`reason`、`arm_state`、`checks`、`timings` など）。
- `skills.go_ready(arm, tier)`、`skills.park(arm, sp)`、`skills.Session(arm, camera, cam_model, image_dir)`、`skills.options_from_config(cfg)`、`skills.take_options_from_config(cfg)`。
- `grasp.plan_grasp(target, scene, arm, cfg)` は `GraspCandidate` のリストを、`grasp.plan_release(target, container, cand, arm, cfg, scene)` は `ReleasePlan` を返します。
- `fd` コマンド: `pick`、`take`、`do`、`repl`、`ready`、`park`。`grasp-plan` は `plan_grasp()` だけを呼びます。`bench` と `track-test` も、待機姿勢への移動に `go_ready()` を使います。

## 必要なもの

- 読み込むフォルダ: `motion`、`ik`、`cli/fdtypes.py`。関数の中で、`vision/scene.py`、`takeout/takeout.py`（`take_out()` のとき）、`draw/tooling.py`（道具が登録されているとき）、`sim/sim_bus.py`（シミュレーションかどうかの判定）を import します。
- ハードウェア: 直接は触れません。サーボは `motion` の `Arm` を、カメラは `Session` に渡されたものを通します。

## テスト

`tests/test_skills.py`（117 件。代役を使った制御の流れと失敗の経路、取り出しの計画、シミュレーションのアームと保存した写真での通し）、`tests/test_grasp.py`（37 件。公式のメッシュと、実際につかんだ 3 個の記録に照らす）。`tools/sim/sim_fuzz.py` は、ランダムな配置で計画と実行をくり返す、調査用のスクリプトです。

## 実機での状況

`pick_place()`（`fd do "…を箱に入れて"`、`fd pick`）は、2026-11-11 に実機で動かしました。午前に 2 個を箱に入れ、午後にも 2 回成功しています。午前の 1 個目は、物は箱に入ったものの、最後の確認が「未確認」と判定して、コマンドは失敗を返しました（判定は同じ日に直しています）。`take_out()`（`fd take`）は、実機では一度も動かしていません。確かめたのは、シミュレーションと保存した写真だけです。細目は [PROGRESS.md](../../PROGRESS.md) と、[docs/MANUAL.md](../../docs/MANUAL.md) の冒頭・5 章・9 章にあります。
