# tools/paint — 筆と絵の具で描く（`fd paint`）

グリッパーにテープで留めた筆（登録した道具。[`../draw/`](../draw/README.md) の `tooling.py`）で、パレットの絵の具と水入れを使い、紙の上に、用意したプラン（線の集まりをレイヤーに分けた JSON）を描く機能です。筆はペンと違って、紙と絵の具の山と水入れの底に、わざと押し付けます。どこまで押してよいかは `brushwork.py` に上限として入っていて、移動・補充・筆洗い・線のすべてが、動く前に検査されます。紙の高さ（筆が最初に紙に触れる高さの平面）は、机の高さからは決まらないので、捨て紙の上のはしごと写真 2 枚から測り、**測ったときの筆の状態と一緒に**絵のセットアップに保存します。筆先を空中校正（[`../calib/`](../calib/README.md)）で測った記録のない筆、紙の高さを測ったときと状態の違う筆では、描きません。どの筆の状態で測ったかの記録がない平面でも、描きません（2026-11-14 から、ファイルの日時に関係なく）。使い方、2026-11-13 に実機で分かった数値、止まったときの読み方は、[docs/MANUAL.md](../../docs/MANUAL.md) の 11 章にあります。

| ファイル | 役割 |
|---|---|
| `paintcli.py` | コマンド `fd paint …` の本体（`status`、`brush`、`check`、`look`、`locate`、`test`、`contact`、`preview`、`run`、`rinse`、`rest`）。引数の読み取り、動く前の断り、動くコマンドの共通の終わり方（どう終わっても筆を上げて撮影姿勢へ行き、状態を保存し、筆をどこに残したかを答える）、結果の `next` と `say` |
| `paintskill.py` | 絵のセットアップ（`paint_setup.json`）と状態（`paint_state.json`）の読み書きと検査。`Painter`: 水、筆洗い、絵の具のレシピによる補充、線、はしご、撮影、レイヤーの実行（補充、時間の区切り、続きから）。紙の高さの測定（`read_contact()`、`sheet_check()`、記録の照合 `contact_record_state()`）、届くかどうかの確認 `check_reach()`、片付けの向き（`rest_choice()`、`rest_report()`、筆先が未測定の筆のための `highest_rest()`、すでに畳んである腕をそのままにする `rest_as_it_lies()`）、パレットの位置合わせ `relocate_wells()`（ずれと回転を当てはめる。写真に絵の具のない場所へは位置を移さず、「位置未確認」にする）と、位置未確認の絵の具からの補充を断る `require_wells()`、所要時間の式（`fit_times()`、`times_of()`）、プランの検査 `plan_problems()` |
| `brushwork.py` | `BrushRun`: 筆の検査つきの動き（遠くへの移動 `hop`、直線 `line`、持ち上げ `raise_to`、押し付け `dab`、線 `stroke`、退避 `retreat`）。深さの上限（線の筆圧 6 mm、机の 6 mm 下）と、移動の高さ・速さの実測値（`BrushOptions`）。トルクを入れる前に爪の位置を確かめ、押し付けを戻す。爪が道具のファイルの位置にないときは `JawRefusal`（コマンドは `stage: tool`、`next` は `fd tool jaw status` で答える） |
| `roundbrush.py` | 細い丸筆の設計キット（`Plan`、`Layer` の `line` / `band` / `fill_…`）、仕上がりの絵 `render()`（補充のたびに濃くなり、かすれていく様子を描く）、統計 `stats()`（補充の回数、見積もりの分数、実行側が断る線）。アームもカメラも使わない |
| `render.py` | プランの形式と、平筆の模型（ロボットの方向の線は幅が広く、横向きは細い）、プランの絵と統計、写真との照合 `fit_to_photo()`。丸筆のプランは `roundbrush.py` に渡す |
| `shapes.py` | 平筆用: 塗りつぶした形から、筆が実際に引ける線への変換（`Canvas`、`valid_centres()`、`contour_strokes()`、`fill_strokes()`）と、塗り残しの白がどれだけ残るかの見込み `design_report()` |
| `paintings/` | 描いた絵。`kaminarimon.py`（雷門。`build()` は 2026-11-13 に描いたプランそのもの、`build(variant="next")` は描き直し用で、レイヤーごとのコマンドを `run` に持つ）、`kaminarimon_plan.json` と `kaminarimon_letters_touchup.json`（その夜に実行したファイル）、`akabeko.py`（赤べこ。平筆） |

## 入口

- `fd` コマンド: `fd paint …`。`tools/cli/fd.py` が、`paint` の後ろの語をそのまま `paintcli.main(argv)` に渡し（`fd --sim` のときは先頭に `--sim`）、返ってきた辞書を JSON で出します。`paintcli` のパーサー自身は、`--sim` というオプションを持ちません（2026-11-14 から。後ろに書いた `--sim` や、`--si` のような省略形は、`stage: options` で断ります）。サブコマンドのオプションも、最後まで書きます（2026-11-14 朝から。`--scrap`、`--tip` のような省略形は、アームを作る前に `stage: options` で断り、`next` が最後まで書いた綴りを言います）。結果の文は、コマンドが値を知らない所（プランのファイル、写真のフォルダ）を、逆引用符の中の仮の名前ではなく、文で言います（`paintskill.FROM_PHOTOS`、`paintcli.PLAN_NEXT`、`paintcli.READ_NEXT`）。使い方は `fd paint --help` と `fd paint <サブコマンド> --help`。`fd paint test --help` の末尾に、新しい筆の測り方があります。
- `paintskill.Painter(arm, setup, photo=…, out_dir=…)`: `load(paint)`、`rinse()`、`blot()`、`stroke_xy()`、`ladder()`、`look()`、`run_layer(plan, li, …)`。動くものは、すべて `brushwork.BrushRun` を通ります。断るときは `paintskill.Refusal`（`SafetyError` の一種で、`stage` と `next` を持つ）です。
- `paintskill.load_setup()` / `save_setup()`、`load_state()` / `save_state()`: ファイルの場所は `setup_path()` と `state_path(sim)` が教えます（`$FD_PAINT_SETUP`、`$FD_PAINT_STATE` で別の場所）。
- `paintskill.rest_choice(arm, setup)` と `rest_report(arm, tool, setup, pan_deg, clearance_m)`: 筆がキャンバスにかからない畳み方。`fd paint rest` のほか、`fd calib-air` の終わりと `fd tool rest` が使います。`fd paint rest` は、筆先が未測定の筆には `highest_rest()`（畳める一番高い姿勢）を、すでにトルク OFF で畳んであってキャンバスの外にある腕には `rest_as_it_lies()`（動かさない）を使います。
- 絵を作る側: `roundbrush.Plan(pan_axis_mm=…)` でプランを組み、`p.save()`、`p.render()`、`p.stats(setup=…)`。`python tools/paint/paintings/kaminarimon.py 出力フォルダ [painted|next]` は、プランと絵を書き出し、統計（`next` では `open` と `run` も）を表示します。`python tools/paint/paintings/akabeko.py 出力フォルダ` も同じです。

## 必要なもの

- 読み込むフォルダ: `draw`（`tooling.py`、`tracking.py`）、`motion`（`robot.py`、`servo_bus.py`、`hwlock.py`）、`ik`。`paintcli.py` は、必要なときに `vision/camera.py`（紙用の露出での撮影、カメラモデル）を、`--sim` のときに `sim/sim_bus.py` を読み込みます。`paintskill.py` は、シミュレーションの状態を置くフォルダの名前を `cli/fd.py` に聞きます。`roundbrush.py`、`render.py`、`shapes.py` は、numpy と OpenCV だけで動きます。
- ファイル: 絵のセットアップ `paint_setup.json`（キャンバス、紙の高さ、絵の具と水入れの場所、レシピ、筆。その机のもので、手で書きます。リポジトリには入れません）と、状態 `paint_state.json`（筆に付いている絵の具、レイヤーの進み具合、所要時間の係数）。写真は `runs/<日付>_paint/`（`$FD_PAINT_OUT`）。`--sim` のコマンドは、状態と絵を木の外（`/tmp/tazuna_sim_<ユーザー番号>/…/`）に置き、装置のセットアップは書きません。
- ハードウェア: サーボ（`Arm` を通します）とカメラ。`--sim` を付けると、どちらも使いません（カメラがないので、`locate`、`contact` の読み取り、`preview` の写真への重ね描きはできません）。実機のロック（`fd hw lock`）が掛かっている間、動くコマンドは `stage: hardware` で断り、`status`、`check`、`preview`、`brush` は答えます。

## テスト

`tests/test_paint.py`（58 件、約 4.5 分。コマンド、深さの決まり、補充、レイヤーの中断と続き、紙の高さの測定、片付け、2026-11-13 のレビューと 2026-11-14 の 2 回の再検証で加えた断り方。テスト 54 は、パレットを回した写真で、`locate` が絵の具の写っていない場所に位置を書かないことを確かめます。テスト 49 は、装置のファイルの写しで、`fd paint status` の `next` を出たとおりに順に打って、準備が終わることを確かめます。最後から 2 番目のテストは、結果に出てきた `fd …` のコマンドを、パーサーに通します）と、`tests/test_paint_model.py`（33 件、約 40 秒。丸筆と平筆の模型、見積もりの式、雷門と赤べこのプランが実行側の検査を通ること）。`fd paint` の語の渡し方と `fd tool rest` のキャンバスの判定は `tests/test_cli_words.py`、`fd calib-air` が絵のセットアップに合わせて畳むことは `tests/test_aircal.py` が確かめます。どれも、シミュレーションのアーム、描いて作った写真、2026-11-13 の保存写真だけで動き、道具・セットアップ・状態・写真のフォルダは、テスト用の場所（`$FD_TOOL`、`$FD_PAINT_SETUP`、`$FD_PAINT_STATE`、`$FD_PAINT_OUT`）を使います。その日の写真とプラン（`runs/20261113_paint/`）はリポジトリに入れていないので、それを使うテストは、ない場所では `SKIP` になります。見積もりの式、はしごの読み取り、白い紙かどうかの判定のテストには、合わせたデータそのもので確かめているものがあり、テストの説明にそう書いてあります（in-sample）。

## 実機での状況

- **2026-11-13 に、実機で水彩画を 2 枚描きました**（午前: 平筆で赤べこ、赤 26 本と黒 8 本。夕方: 細い丸筆で雷門、8 レイヤー 70 本と、文字の書き直し 13 本）。使ったのは、その日の時点の `brushwork.py`、`paintskill.py`、`paintcli.py`（`fd paint look / locate / test / run / rinse / status`）です。`BrushOptions` の数値（運ぶ高さ 65 mm、移動 20°/s など）と、筆圧の上限は、その日の実測です。
- **その夜の更新で加えたものは、実機では一度も動かしていません**: `fd paint brush`、`check`、`contact`（はしごを写真から読む、`--scrap-laid`、`--read`、`--adopt`）、`preview`、`rest`、`--lower-mm`、筆先が未測定の筆を断ること、紙の高さの記録と筆の状態の照合、爪の位置の確認、`locate` の保存の条件、試し線とパレットの距離の検査、状態ファイルの検査。確かめたのは、シミュレーションのアームと、その日の保存写真（はしごの読み取りは、夕方の写真 1 組）だけです。
- 更新の前からあった `run`、`test`、`rinse` は、同じ指令に対してサーボへ送る目標値の列が変わっていないことを、シミュレーションで比べてあります（夕方に実機で動いた版と更新後の版を 4 通り、レビュー後の修正の前後を 4 通り）。この比較は、2026-11-14 未明の修正の前のものです。
- **2026-11-14 未明の再検証のあとに変えたものも、実機では一度も動かしていません**: 記録のない平面を断ること、`locate` のずれと回転の当てはめと「位置未確認」、位置未確認の絵の具からの補充を断ること（`stage: wells`）、キャンバスの上の試し線の `--scrap-laid` と写真の確認、中身の分からない筆を最初の補充の前に洗うこと、`fd paint rest` の「すでに畳んである腕は動かさない」と「未測定の筆先は一番高い姿勢で畳む」（その姿勢でトルクを切ったことはありません）、爪のことで断るときの `stage: tool`、`run` の `next`。`locate` は、その日の保存写真 14 枚で、担当者が確かめています。
- **2026-11-14 朝の 3 回目の修正で変えたものも、同じです**（シミュレーションと保存写真だけ）: `relocate_wells()` が、色の塊で見つけた場所を含めて、書く場所を全部「絵の具が写っているか」で検査すること（まん中が絵の具でない塊と、2 つの絵の具から 8 mm 以内の塊は、どの絵の具にも渡さない。夕方の写真 8 枚 × 回転 11 通り × ずれ 11 通りで、絵の具の外に書いた位置は 33 個から 0 個に）、どの向きにも畳めない道具の `fd paint rest` が道具を外す道を言うこと（`cannot_fold`）、`run` の続きが `--stop` を残すこと、申告で記録した平面が今の筆の状態と違うときに `fd paint contact --adopt` を挙げること、成功した `fd paint brush` の `next` が `fd paint status` であること、オプションの省略形を断ること。高さを指定した `fd paint rest --clearance 0.03` が、収納姿勢にならない道具で、動いてから `ok: false` で終わることは、残っています（`../draw/tooling.py` の判定）。
- 模型（`roundbrush.py`、`render.py`、`shapes.py`）の数値は、その日の写真から読んだものです。仕上がりの絵と実際の紙は、まだ違います（屋根のたわみ、濃い赤の広がり、文字の太さなど）。薄い色（濃い青と黒の `_pale`）は、2 回目の補充からほぼ原色で出るので、絵の計画では当てにできません。
- 解決していないことの一覧は、MANUAL 11 章の「残っていること・実機で確かめること」にあります。
