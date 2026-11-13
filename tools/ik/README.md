# tools/ik — 運動学（順運動学と逆運動学）とグリッパーの形

関節の角度から手先の位置を求める計算（順運動学）と、手先の位置から関節の角度を求める計算（逆運動学、IK）を、このフォルダにまとめています。寸法は、SO-101 の公式の URDF（`so101_new_calib.urdf`）から写した値です。計算だけのモジュールで、サーボにもカメラにも触れません。逆運動学は式で解きます。台座の向きを決めると、肩・ひじ・手首は 1 つの鉛直な面の中の 3 本のリンクになるので、「ひじが上」の解を求めます。手先の向きは「水平から何度下を向くか」（`pitch_down_deg`。90 が真下）で指定し、手首の回転（ロール）は呼ぶ側が決めます。関節の限界の外になる解は、丸めずに `ok: False` で返します。

| ファイル | 役割 |
|---|---|
| `kinematics.py` | 順運動学（`fk`、`link_frames`、`tool_offset`）、逆運動学（`ik`）、その位置で取れる手先の傾き（`feasible_pitches`、`pitch_range`、`best_pitch`）、爪の向きと位置（`jaw_open_axis`、`jaw_tilt_deg`、`jaw_azimuth_deg`、`roll_for_azimuth`、`jaw_points`）。単位は m とラジアン |
| `kinfast.py` | 同じ式の、速い版。つかみ方の計画は候補を何千通りも試し、動作の記録には何千もの姿勢があるためです。同じ式を、行列を使わずに数（float）だけで計算する逆運動学（`ik_fast`。`kinematics.ik` の約 10 倍速い）、爪を開く向きから手首のロールを求める `roll_for_azimuth`、取れる傾きをまとめて調べる `pitch_feasible`、多数の姿勢の先端位置を一度に求める順運動学（`fk_points`） |
| `gripper_model.py` | グリッパーの形を式にしたもの（公式のメッシュから測った値。実行時にメッシュは要らない）。爪の輪郭、物の幅と爪の角度（`q_for_width`、`width_for_q`）、サーボの目盛りと爪の角度（`q_from_raw`、`raw_from_q`）、つかんだときに爪が止まる位置の見込み（`stall_raw_for_width`）。単位は mm と度 |

## 入口

- `kinematics.ik()`: 手先の位置 `target_xyz`（m）、傾き `pitch_down_deg`（度）、手首のロール `wrist_roll`（ラジアン）から、`(q5, info)` を返します。`info["ok"]` が `True` のときだけ、`q5`（関節 5 個の角度）を使います。
- `kinematics.fk(q, tool=None)`: 手先の姿勢（4×4 の行列）を返します。`kinematics.best_pitch()` は、希望の傾きが取れないとき、いちばん近い取れる傾きを返します。
- ふつうは、`motion/robot.py` の `Arm.solve(xyz, pitch_down_deg, roll_deg)` を通して呼びます。`Arm.solve` は、このアームの関節の限界と、先端の位置（指先、または登録した道具の先端）を渡し、解けないときは `SafetyError` にします。まっすぐな経路は、`Arm.plan_tip_path()` が `Arm.solve` をくり返して作ります。
- `kinfast.ik_fast()` と `kinfast.fk_points()`: つかみ方の計画（`pick/grasp.py`）と、動作の記録の解析（`draw/tracking.py`）が使います。2026-11-13 に、この 2 つのファイルから中身を変えずに移しました（どちらも、元の名前のまま呼べます）。`ik_fast` で計画した姿勢は、最後に `Arm.solve` で確かめます。
- 運動学を「使う側」の関数は、それぞれの機能フォルダにあります。たとえば `Arm.solve`（このアームの限界と先端を当てはめる）や、`pick/grasp.py` の `_solve_pair`（つかむ点とその真上の点を同じ姿勢で結ぶ）です。
- `fd` コマンド: `fd check`（届くかどうかと、取れる傾きの範囲）と、`fd tip` / `fd check` の `--jaw-azimuth`（爪の向きから手首のロールを決める）が、直接使います。アームを動かすコマンドは、どれも `Arm` を通してここを使います。

## 必要なもの

- 読み込むフォルダ: なし。`kinematics.py` は `gripper_model.py` と numpy だけを、`kinfast.py` は `kinematics.py`、`gripper_model.py`、numpy だけを、`gripper_model.py` は標準ライブラリだけを使います。逆に、機能フォルダのうち `publish` を除く全部が、ここを import します。
- ハードウェア: なし。

## テスト

`tests/test_kinematics.py`（30 件）。順運動学は、`tests/refs/urdf/so101_new_calib.urdf` を読む別の実装と比べます。逆運動学は、往復の一致、関節の限界、2026-11-10 に実機を動かした版との一致を確かめます。グリッパーの形は、公式のメッシュ（`tests/refs/urdf/assets/`）と比べます。`kinfast.py` は、`tests/test_grasp.py`（`ik_fast` などを `kinematics.ik` と比べる）と、`tests/test_tracking.py`・`tests/test_integration_pen.py`（`fk_points` を `kinematics.fk` と比べる）が確かめます。`gripper_model.py` は、単体で実行すると自己テストを行います（`python tools/ik/gripper_model.py`）。

## 実機での状況

2026-11-11 の実機のピックは、このモジュールで関節の角度を求めて動かしました。関節の原点（`zero_ticks`）や指先の位置（`tip_offset`）のような装置ごとの値は `config/config.json` にあり、`motion/robot.py` が当てはめます。shoulder_lift の原点は、2026-11-11 に直しました。ひじと手首の原点のずれは分けて測っていないので、机の高さで真上から作業する姿勢のほかは、数 mm ずれる可能性があります（[docs/MANUAL.md](../../docs/MANUAL.md) 9 章）。`gripper_model.py` の「つかんだときに爪が止まる位置」は、実機の 3 回の把持から決めた値です。
