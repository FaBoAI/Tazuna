# 01 物を 1 個、箱に入れる

## プロンプト

> 青い円柱を箱に入れて

## AI がすること

文をそのまま `fd do` に渡します。ルールで読める文なので、AI が判断することはありません（CLAUDE.md「依頼を受けたら」の 1）。事前の `fd scene` や撮影、事後の確認撮影もしません。1 回の呼び出しが、見る → 計画 → つかむ → 箱に入れる → 確認まで行います。

```bash
./fd do "青い円柱を箱に入れて"
```

## 返ってくるもの

実機の記録です（2026-11-11 11:01:17。抜粋）。コマンドの開始から終了までは 23.08 秒でした。

```json
{"ok": true, "stage": "done",
 "say": "青い円柱を箱に入れました（22秒）。",
 "timings": {"perceive": 0.118, "plan": 0.161, "approach": 7.254, "visual": 0.246, "descend": 2.912,
             "grasp": 0.991, "transfer": 3.641, "release": 2.334, "verify": 4.0, "total": 21.659}}
```

## AI が人に伝えること

`say` をそのまま伝えます:「青い円柱を箱に入れました（22秒）。」（CLAUDE.md の手順です。そのときの AI の返事そのものは、記録に残っていません。）

## 状況

- **実機で実行済み**です。記録のコマンドには `--tier slow_cruise --verbose` が付いていました。`slow_cruise` は、このあと既定の速度段になりました（docs/MANUAL.md 3 章）。
- このときの物は、高さ 7.7 mm の青い円盤です。低い物（高さ約 10 mm 未満）は既定ではつかまず、計画の段階で断ります。この回は設定 `grasp.min_engage=0.0035` で許可しています（docs/MANUAL.md 9 章）。
- 動画: [[Tazuna開発記] Day1 Fable5.2 Direct用のRoboハーネスを開発](https://www.youtube.com/watch?v=vImrWLpaMcs&list=PLS2xQbZhQGBk)

## うまくいかないとき

- **`ok: false`、`stage: "verify"`**。実例は同じ日の 10:11:42、`./fd do "アイボリーの円柱を箱に入れて" --verbose` です。`say` は「完了できませんでした（verify: the object is not confirmed inside the container）。アームは待機姿勢です。」でした。物は箱に入っていて、判定は同じ日に直しています（PROGRESS.md）。こういうとき AI は `next` を読み、`images` の写真を 1 枚だけ見て、人に伝えます。
- **`need_llm: true`**。文が読めず、何も動いていません。→ [03-need-llm.md](03-need-llm.md)
