# 02 机の上の物を全部、箱に入れる

## プロンプト

> テーブルの上のものを全部箱に入れて

## AI がすること

次のどちらかを 1 回だけ呼びます。物ごとに何度も呼びません（CLAUDE.md「依頼を受けたら」の 3）。条件に合う物がなくなるまで、1 回の呼び出しで最大 6 個（`--max-objects`）運びます。「全部箱に入れて」だけでは「を」がなく、ルールでは読めません。

```bash
./fd do "テーブルの上のものを全部箱に入れて"
./fd pick --color blue --all     # 色や形（--kind box|cylinder）で絞るとき
```

## 返ってくるもの

実機の記録は `./fd pick --kind box --all`（形がブロックの物を全部）です。このときの人の言葉は、記録に残っていません。続きの 3 行は、その直後に別の 1 個を運んだ記録です（どれも 2026-11-11）。

| 時刻 | コマンド | 結果 |
|---|---|---|
| 16:50:07 | `./fd pick --kind box --all` | `ok: true`、`stage: "done"`、`timings.total` 19.183 秒（開始から終了まで 24.67 秒） |
| 16:50:28 | `./fd pick --color red` | `ok: false`、`stage: "perceive"`、`reason`: `target not found: only uncertain detections match {'color': 'red'}: id 4 (…). If one of them is right, pick it by id` |
| 16:50:48 | `./fd pick --id 4` | `ok: false`、`stage: "visual"`、`reason`: `could not confirm the jaw position in the image and the grasp is tight` |
| 16:51:20 | `./fd pick --id 4 --visual off` | `ok: true`、`stage: "done"`、`timings.total` 17.828 秒 |

1 行目で運んだのは 1 個です。2 個以上運ぶと、結果に `picked`（1 個ごとの `ok` と秒数）が付きます。

## AI が人に伝えること

`fd do` なら `say` をそのまま伝えます（2 個以上運んだときは「N個を箱に入れました。」の形）。`fd pick` の結果に `say` はないので、`ok` と `timings`、`picked` を見て伝えます。

## 状況

シミュレーション（`--sim`）では、`--all` を付けても運ぶのは 1 個です（写真が変わらないためで、結果の `notes` にそう出ます）。

**実機では一部だけ実行済み**です（上の表）。`fd do` のこの文と、1 回の呼び出しで 2 個以上を続けて運ぶ動作は、実機ではまだです。docs/MANUAL.md の冒頭も、`--all` での連続動作を「まだ実機で試していないもの」に挙げています。

## うまくいかないとき

- **不確かに見えた物は、色や形の指定では運ばれません**（表の 2 行目）。`--all` も、条件に合う物が見つからなくなると成功で終わるので、そういう物は残ります。結果の `scene` で番号を見て、`./fd pick --id N` で運びます。
- **`stage: "visual"`**（表の 3 行目）。爪の位置をカメラで確かめられず、降りずに待機姿勢へ戻っています。`next` のとおり、隣の物を離してもらうか、`--visual off` でやり直します（隣の物に触れることがあります）。
