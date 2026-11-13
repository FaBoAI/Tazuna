# tools/cli — コマンド fd と、依頼文の解釈

`fd` は、AI（Claude Code）や人が呼ぶコマンドです。1 回呼ぶと 1 つの作業を最後まで行い、結果を JSON 1 個で出力して、実行ログ（`runs/<日付>/events.jsonl`）に追記します。`fd do "<依頼文>"` は、`nlq.py` が文をルールで読み、箱に入れる依頼なら `fd pick`、箱から取り出す依頼なら `fd take` と同じ処理を行います。読めない文と、素直な肯定の命令文でない文（「〜しないで」、質問、条件つきなど）は、何も動かさずに `need_llm` を返します。つかむ、取り出す、描くといった作業そのものは、各機能フォルダの関数に任せます。校正（`calib-at`、`calib-fit` など）、設定（`config`）、ログの集計（`report`）のコマンドは、`fd.py` の中に書いてあります。

| ファイル | 役割 |
|---|---|
| `fd.py` | 全部のコマンド。`COMMANDS`（名前と関数の表）と `build_parser()`（引数）。`main()` がカメラとアームを開き、関数を呼び、結果を出力してログに書く |
| `nlq.py` | 依頼文（日本語と英語）をルールで読む。`parse(text)` が、動作（`put_in` か `take_out`）と、対象と箱の条件を返す。単体で実行すると自己テスト |
| `fdtypes.py` | 認識 → 計画 → スキルの間で受け渡すデータ型（`SceneObject`、`Scene`、`GraspCandidate`、`ReleasePlan`） |

## 入口

- コマンドライン: リポジトリ直下の `./fd <コマンド>`。一覧は `./fd --help`、使い方は [docs/MANUAL.md](../../docs/MANUAL.md) の 1 章（道具と描画は 4 章）にあります。
- コマンドは 4 種類です。1 回で作業を終えるもの（`do`、`pick`、`take`、`scene`、`ready`、`park`、`repl`）。1 動作ずつの低レベル（`state`、`torque`、`joints`、`tip`、`check`、`grip`、`grasp`、`probe`、`snap`）。道具と描画（`tool …`、`draw`、`track-test`）。校正と保守（`ref`、`calib-at`、`calib-fit`、`calib-roll`、`table-fit`、`bench`、`selftest`、`config`、`camera`、`report`）。このほかに `gains`、`grasp-plan`、`servo-tip`（古い方式）があり、`home` は `ready` の別名です。
- `fd.main(argv)`: コマンドラインと同じ処理です。テストはこれを呼びます。`fd.make_arm(sim)` は、実機のアームか、`--sim` のアームを作ります。
- `nlq.parse(text)`: `fd do` が呼びます（`fd repl` も `fd do` を通ります）。
- `fdtypes` の型: `vision/scene.py` が `Scene` と `SceneObject` を作り、`pick/grasp.py` が `GraspCandidate` と `ReleasePlan` を返します。

## 必要なもの

- 読み込むフォルダ: 起動時に `motion` と `ik`。`pick`、`vision`、`draw`、`sim` は、そのコマンドが使うときだけ import します。`nlq.py` と `fdtypes.py` は、標準ライブラリだけで動きます。
- ハードウェア: サーボとカメラ。`--sim` を付けるとサーボを使いません。設定や校正のファイルを書き換えるコマンド（`config --set`、`calib-fit`、`calib-roll`、`table-fit`、`ref`）と `camera` は、`--sim` を付けると断ります。`scene`、`pick`、`take`、`do` などは、保存した写真を `--frame` で渡すとカメラを開きません（アームを動かす `pick`、`take`、`do` では、`--frame` は `--sim` か `--dry` と一緒のときだけ使えます）。Python には numpy、OpenCV、feetech-servo-sdk、pyserial が要ります。

## テスト

専用のテストファイルはありません。`nlq.py` は自己テストを持っていて（依頼文 157 例。MANUAL 8 章）、`tests/run_all.py` が最初に実行します。`fd` のコマンドは、`tests/test_skills.py`（`pick`、`take`、`do`）、`test_tooling.py`、`test_penskill.py`、`test_ink.py`（`snap`）、`test_tracking.py`（`track-test`）、`test_integration_pen.py` が、`fd.main()` などを呼んで確かめます。

## 実機での状況

`fd do "…を箱に入れて"` と `fd pick` は、2026-11-11 に実機で動かしました。`fd take`、`fd tool …`、`fd draw` は、シミュレーションと保存した写真でしか確かめていません（[PROGRESS.md](../../PROGRESS.md)）。実機を動かさずに試すには、`--sim`（シミュレーションのアーム）か `--dry`（計画だけ）を使います。
