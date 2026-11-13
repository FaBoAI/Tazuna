# tools/publish — 公開する写真から人を消す

カメラはロボットの正面からロボットの方を向いているので、机に座っている人が写ります。公開リポジトリに入れる写真は、決めた範囲を灰色に塗りつぶし、確認済みの一覧（`tests/testdata/reviewed_photos.json`。パスと SHA-256）に載せます。塗りつぶしは、`jpegtran -wipe` で JPEG のブロックを書き換えるので、再圧縮はしません。範囲の外で変わるのは、範囲に接する画素だけです。

| ファイル | 役割 |
|---|---|
| `privacy_wipe.py` | 塗る範囲の表（写真ごとの四角形）と、`check`、`apply`、`accept` の 3 つの操作 |

## 入口

- `python tools/publish/privacy_wipe.py check`: 次の 3 つを調べます。表にある範囲が全部灰色か。確認済みの一覧の写真が変わっていないか。git が追跡している写真が、全部一覧にあるか。コミットの前に、毎回実行します。
- `python tools/publish/privacy_wipe.py apply`: 表にある範囲を塗りつぶします。元の写真は、先に `backups/` へ写します（`backups/` はリポジトリに入りません）。
- `python tools/publish/privacy_wipe.py accept`: いま git が追跡している写真から、確認済みの一覧を書き直します。新しい写真と変わった写真を、全部、等倍で見たあとにだけ実行します。
- 新しい写真を入れる順序: 写真を見る → 塗る範囲を表（`EXTRA`）に足す → `apply` → もう一度見る → `git add -f` → `accept`。
- `fd` コマンドからは使いません。

## 必要なもの

- 読み込むフォルダ: なし。OpenCV と numpy を使います。`apply` には `jpegtran`（libjpeg-turbo 2.1 以上。PATH になければ、環境変数 `JPEGTRAN` で指定）が、`check` の 3 つ目と `accept` には git が要ります。
- ハードウェア: なし。

## テスト

テストはありません。

## 実機での状況

ロボットは動かしません。公開リポジトリの写真 65 枚のうち 42 枚を、この道具で塗りつぶしてあります。確認済みの一覧には、アイコンの画像 2 枚（`docs/images/`）も載せています。写真の確認は AI が行ったもので、人の目による全数の確認はしていません（[PROGRESS.md](../../PROGRESS.md)、[docs/MANUAL.md](../../docs/MANUAL.md) 10 章）。
