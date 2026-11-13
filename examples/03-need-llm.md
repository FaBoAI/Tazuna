# 03 ルールで読めない依頼（`need_llm`）

## プロンプト

> 青いキューブを右に動かして

## AI がすること

1. まず 1 コマンドです。`fd do` が実行するのは素直な肯定の命令文だけです。読めない文は推測せず、何も動かさずに `need_llm` を返します。
2. 返ってきたら、AI が文を自分で読み、`objects` から対象を選びます（CLAUDE.md「依頼を受けたら」の 1 と 2）。動かす依頼でなければ（「〜しないで」、質問、報告など）、何も動かさずに答えます。

```bash
./fd do "青いキューブを右に動かして"
./fd pick --id N     # 箱に入れる依頼だったとき（容器が複数なら --into-id M も）
./fd take --id N     # 箱から取り出す依頼だったとき（キャップ・コップなら --hollow も）
```

## 返ってくるもの

シミュレーションの記録です（2026-11-11 02:24:02、0.46 秒。抜粋）。コマンドは `./fd --sim do "青いキューブを右に動かして" --frame tests/testdata/0148_cube_scene_raw.jpg` で、保存画像を使い、実機もカメラも使っていません（記録の画像のパスは、当時の配置のものです）。`objects` はカメラに見えている物の一覧です。`say` は付きません。禁止・条件・質問などの文では、`not_plain` に種類が入ります。

```
{"ok": false, "need_llm": true,
 "reason": "not a put-into request (no verb such as 入れて / put)",
 "next": "choose the object yourself from \"objects\" and call `fd pick --id N` (add `--into-id M` if there are several containers)",
 "objects": [{"id": …, "kind": …, "color": …, "center": …, "size_mm": …, "in_container": …}, …]}
```

## AI が人に伝えること

この文は「右に動かす」依頼で、箱に入れる依頼ではないので、`fd pick` では済みません。TAZUNA が 1 コマンドでできる作業は「箱に入れる」「箱から取り出す」「線画を描く」の 3 つです。それ以外は低レベルコマンドを AI が組み立てることになり、遅くなります（docs/MANUAL.md 9 章、2 章「定型外の作業」）。`say` がないので、AI は自分の言葉で伝えます。

## 状況

記録があるのは、`need_llm` が返ったところまでです（**シミュレーション**）。その先の `fd pick --id N` は、CLAUDE.md と `tools/cli/nlq.py` から書いたもので、この文についての記録はありません。実機では、`ok: false` のあとに `./fd pick --id 4` で続けた記録があります（[02-put-all-in-box.md](02-put-all-in-box.md)）。

## うまくいかないとき

- **物と箱の名前が読めても、実行しない文があります。** 「箱の中の白いキャップを取り出さないで」は `not_plain: "prohibition"` で返ります。名前だけを見て `fd take` を呼んではいけません。この文は動かす依頼ではないので、何も動かさずに答えます。
- **「全部箱に入れて」は読めません**（「を」がないため）。`./fd do "テーブルの上のものを全部箱に入れて"` の形にします。
