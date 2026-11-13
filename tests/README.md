# tests — 実機を使わないテスト

TAZUNA のテストは、シミュレーションのアーム、疑似サーボ、保存した写真だけで動きます。実機と実カメラには触れません（実カメラを開くのは `python tests/test_scene.py --live` と指定したときだけです）。シミュレーションでの合格は、実機で動くことの確認ではありません。道具（グリッパーに留めたペン）と描画のテストの合格は、「手順と止まり方が決めたとおり」という意味です（[docs/MANUAL.md](../docs/MANUAL.md) 8 章）。

## 実行のしかた

```bash
python tests/run_all.py            # 全部。14 本を 1 本ずつ別のプロセスで実行します（Jetson Orin Nano で約 20 分）
python tests/run_all.py ink tool   # ファイル名にその語を含むものだけ（この例は test_ink.py と test_tooling.py）
python tests/test_ink.py           # 1 本だけ
```

- テストは、`fd` と同じ Python で動きます（環境変数 `TAZUNA_PYTHON`、なければ `config/python` に書いたパス、どちらもなければ `run_all.py` を起動した Python）。その Python に numpy、OpenCV、feetech-servo-sdk、pyserial が必要です。1 本だけ実行するときは、その Python で起動してください。確かめた組み合わせは、Python 3.13 + numpy 2.5 + OpenCV 4.13（全部）と、Python 3.12 + numpy 1.26 + OpenCV 4.8（短い 4 本: nlq、ink、sim、skills）です。
- `run_all.py` は、最初に使う Python を 1 行表示し、そのあと 1 本ごとに 1 行（`ok` か `FAIL`、スイートの名前、秒数、最後の出力行）を表示します。全部合格すると、最後に `all suites passed` と出ます。語で絞ったときは `the N selected suite(s) passed`、どのスイートにも当たらないときは何も実行せずにそう表示します。

## スイートの一覧

| スイート | 確かめること | テスト数 | 時間 |
|---|---|---|---|
| `tools/cli/nlq.py`（自己テスト） | 依頼文をルールで読む。素直な肯定の命令文でない 89 文は断る | 157 文 | 1 秒未満 |
| `test_ink.py` | 写真から線を読む（2026-11-11 の保存写真と正解値）。白飛びした写真には「分からない」と答える | 28 | 約 15 秒 |
| `test_tooling.py` | 道具の登録、断るコマンド、道具を付けたままの待機姿勢・収納姿勢・空中の移動 | 33 | 約 30 秒 |
| `test_strokes.py` | 線画の配置、線の切り分け、位置補正の式、シミュレーションの紙 | 30 | 約 1 分 |
| `test_tracking.py` | 追従の測り方（2026-11-11 の記録の数値を再現）、ファインモード、`fd track-test` | 13 | 約 2 分 |
| `test_integration_pen.py` | 道具・写真・線画・追従をつないだ通し。道具が登録されたままでも、ほかのテストが通ること | 15 | 約 30 秒 |
| `test_penskill.py` | 校正と描画（`fd tool calibrate` / `fd draw`）: 下ろし方の決まり、登録 → 校正 → 猫の描画、中断と古い校正の扱い | 23 | 約 5 分 |
| `test_skills.py` | 箱に入れる・箱から取り出す作業の制御の流れと失敗の経路、取り出しの把持計画、保存画像での通し | 117 | 約 40 秒 |
| `test_sim.py` | 動作制御と安全検査（シミュレーションのアーム）。起動まわりの安全確認（使う Python の選び方、設定ファイルが無いときに実機を開かないこと、`--sim` で装置のファイルを書き換えないこと、実行記録を書けないときも結果を返すこと） | 97 | 約 15 秒 |
| `test_grasp.py` | 把持とリリースの計画 | 37 | 約 2 分 |
| `test_kinematics.py` | 順運動学・逆運動学とグリッパー形状（`tools/ik/`）。参照用の URDF とメッシュに照らす | 30 | 約 1.5 分 |
| `test_scene.py` | 画像認識（保存画像 24 枚と正解値） | 50 | 約 45 秒 |
| `test_bus_fake.py` | 実機で使う通信コード（`servo_bus.py`）を、疑似サーボ 6 台につないで確かめる | 77 | 約 1 分 |
| `test_motion_phys.py` | トルクまで扱う物理モデルでの動作と、速度段の検討 | 16 | 約 3.5 分 |

- テスト数は、各ファイルの行頭（モジュール直下）にある `def test_…` を数えたものです。Python の `ast` で構文を読んで数えました。合計 566 件です。nlq は、自己テストが判定する文の数です。
- 時間は、Jetson Orin Nano Super で 2026-11-13 に全部を通して実行したときの実測です（合計 約 20 分。全部合格）。
- `test_bus_fake.py` の 1 件は任意のテストで、ふつうの実行では `SKIP` になります。動かすには `python tests/test_bus_fake.py --with-arm` とします。疑似サーボ越しに `robot.Arm` を動かすもので、実機は使いません。この 1 件は、まだ一度も実行していません（PROGRESS.md）。

## `testdata/` — テストが読むデータ

- `*_raw.jpg`（24 枚）と `frames.json`: 机の上の場面の写真と、手で測った正解値（物の位置と寸法）。
- `takeout_20261111/`: 箱からキャップを手動操作で取り出した日の写真 16 枚。有無・種類・数・寸法・位置関係の確認にだけ使います。
- `pen_20261111/`: 2026-11-11 夜のペンの写真 25 枚（うち 6 枚は白飛びした写真）、正解値 `manifest.json`、追従の記録 `tracks/`、ペンのテストが使う設定とカメラモデル。
- `calib_20261110/`: 2026-11-10 の校正。机の上の場面の写真はこの校正で測ったものなので、`test_grasp.py`・`test_scene.py`・`test_skills.py` はこれに固定しています。実機を校正し直しても、テストは変わりません。
- 写真は全部で 65 枚です。うち 42 枚は、人が写っていた部分を灰色に塗りつぶしてあります（[`tools/publish/privacy_wipe.py`](../tools/publish/privacy_wipe.py)）。確認した写真の一覧は `reviewed_photos.json` です。

## `refs/` — 照らし合わせる資料

`refs/urdf/` は SO-101 の URDF・MJCF・メッシュ（STL）で、[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) から取ったものです。出どころ（フォルダ、コミット、ライセンス）は、リポジトリのルートの [NOTICE](../NOTICE) に書いてあります。`refs/gripper/` は、同じメッシュから測ったグリッパー形状の参照実装です。テストはこの 2 つを、運動学とグリッパー形状の独立した基準として使います。
