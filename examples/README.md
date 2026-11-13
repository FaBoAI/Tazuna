# examples — プロンプトの例

ここでの「プロンプト」は、TAZUNA でアームを操作する AI（Claude Code）に、人が言う言葉のことです。AI はその言葉を受けて `fd` コマンドを呼び、返ってきた JSON を読んで人に答えます。

| ファイル | プロンプト | 起こること | 実機で実行したか |
|---|---|---|---|
| [01-put-one-in-box.md](01-put-one-in-box.md) | 青い円柱を箱に入れて | `fd do` 1 回で、見る → つかむ → 箱に入れる → 確認 | 済み（2026-11-11 11:01） |
| [02-put-all-in-box.md](02-put-all-in-box.md) | テーブルの上のものを全部箱に入れて | `fd do` か `fd pick --all` 1 回で、条件に合う物がなくなるまで運ぶ | 一部だけ（2026-11-11 16:50 に `fd pick --kind box --all`。運んだのは 1 個。`fd do` のこの文と、2 個以上の連続はまだ） |
| [03-need-llm.md](03-need-llm.md) | 青いキューブを右に動かして | ルールで読めず、何も動かさずに `need_llm` が返る。AI が文を読み、`fd pick --id N` などで続ける | まだ（記録はシミュレーション） |
| [04-take-out-of-box.md](04-take-out-of-box.md) | 箱の中の白いキャップを取り出して | `fd do` か `fd take` 1 回で、箱から取り出して机に置く | まだ（シミュレーションだけ） |
| [05-draw-cat-with-pen.md](05-draw-cat-with-pen.md) | SO-101の先にマジックペンをつけました。猫の絵を描いてください | `fd tool attach` → `fd tool calibrate` → `fd draw cat` → `fd tool rest` | まだ（シミュレーションだけ。2026-11-11 の猫は試作スクリプトで描いた） |
| [06-park.md](06-park.md) | 終わりです。アームを収納してください | `fd park` で収納して、トルクを切る | 済み（2026-11-11 11:01） |

## 読み方

どの例も「プロンプト → AI がすること → 返ってくるもの → AI が人に伝えること → 状況 → うまくいかないとき」の順に書いてあります。「実機の記録」と書いた結果は、実行記録（`runs/<日付>/events.jsonl`。公開リポジトリには入っていません）から取ったもので、「シミュレーション」や「CLAUDE.md の手順」と書いたものは、実機の結果ではありません。

- コマンドは、リポジトリのフォルダで実行する形（`./fd …`）で書いています。AI 向けの手順書は [CLAUDE.md](../CLAUDE.md)、詳しい説明は [docs/MANUAL.md](../docs/MANUAL.md)（1 章と 2 章）です。
- 記録の時刻は、コマンドが終わった時刻です。
- 実機が動くコマンドは、人が実機の操作を頼んだときだけ実行します。試すだけなら `./fd --sim …`（シミュレーション）か `--dry`（計画だけ）を使います。ロボットとカメラのない PC では、保存した写真を渡して `./fd --sim … --frame tests/testdata/0148_cube_scene_raw.jpg` のようにします（`--frame` がないと、`--sim` でも本物のカメラを開いて焦点と露出を固定します）。
- 設定と校正は、作者の装置 1 組のものです。別の装置では、測り直すまで、作業のコマンド（`fd do` / `pick` / `take` / `draw`）を実機で実行しないでください（測り直しの手順は docs/SETUP_SO101.md）。

## 動画

YouTube の再生リスト「Tazuna」にあります。

- 物を箱に入れる作業の説明: [[Tazuna開発記] Day1 Fable5.2 Direct用のRoboハーネスを開発](https://www.youtube.com/watch?v=vImrWLpaMcs&list=PLS2xQbZhQGBk)
- ペンで絵を描く作業: [[Tazuna開発記] マジックペンで絵を描いてみる](https://www.youtube.com/watch?v=Fakua6h1_IE&list=PLS2xQbZhQGBk)
