"""Keep people out of the photos that go into the public repository.

The camera stands in front of the robot and looks back at it, so whoever sits at the desk is in the frame.
Before a photo is published, the areas listed below are greyed out, and every published photo is on a list of
checked photos (tests/testdata/reviewed_photos.json: path and SHA-256; docs/MANUAL.md chapter 10 says how they were
checked).

  python tools/publish/privacy_wipe.py check    1. every listed area is grey in the photos of this folder,
                                        2. every photo on the reviewed list is unchanged,
                                        3. every photo that git tracks is on the reviewed list.
                                        Needs only OpenCV (and git for 3). Run it before every commit.
  python tools/publish/privacy_wipe.py apply    wipe the listed areas in place (needs `jpegtran` of libjpeg-turbo 2.1 or
                                        newer; set $JPEGTRAN if it is not on the PATH). Each original is first copied
                                        to backups/photos_before_privacy_wipe_20261112/ (never in the repository),
                                        and a photo is always wiped starting from that original.
  python tools/publish/privacy_wipe.py accept   write the reviewed list from the photos that git tracks now. Only after you
                                        have LOOKED at every new or changed photo at full size.

How: `jpegtran -wipe` works on the stored JPEG blocks, so nothing is re-compressed. Outside the rectangles only the
pixels that touch a rectangle change (the decoder smooths chroma across the edge; measured on the 42 published
photos: about 0.06 % of the outside pixels, by at most 22 of 255). Every other pixel decodes exactly as before. `apply`
checks that for every photo (it tolerates changes up to 16 pixels away from a rectangle). Inside, one pixel in from
the edge, everything is grey 128.

Rectangles are (x, y, w, h) in pixels of the 1920 x 1080 photo, all multiples of 16.
A new photo: look at it, add its rectangles to EXTRA (or to a default), `apply`, look again, `git add -f`, `accept`.
Photos that exist only on the robot's computer (not in a clone) are skipped.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKUP = os.path.join(ROOT, "backups", "photos_before_privacy_wipe_20261112")
REVIEWED = os.path.join(ROOT, "tests", "testdata", "reviewed_photos.json")
MEDIA = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".ppm",
         ".mp4", ".mov", ".avi", ".mkv", ".webm")

PEN = "tests/testdata/pen_20261111/photos"
OVER = "tests/testdata/pen_20261111/overexposed"
TAKE = "tests/testdata/takeout_20261111"
# every photo of a set gets the set's default (where the person usually was), some get more
PEN_DEFAULT = [(1456, 0, 464, 400)]
TAKE_DEFAULT = [(1344, 0, 576, 432), (1856, 432, 64, 32)]
OVER_DEFAULT = [(1120, 0, 800, 464), (1552, 464, 368, 144)]
EXTRA = {
    PEN + "/220504_probe_ex110.jpg": [(528, 0, 48, 64), (576, 0, 128, 240), (704, 0, 208, 336), (912, 0, 448, 384), (1360, 0, 96, 400)],
    PEN + "/220506_probe_ex60.jpg": [(528, 0, 48, 64), (576, 0, 128, 240), (704, 0, 208, 336), (912, 0, 448, 384), (1360, 0, 96, 400)],
    PEN + "/221534_st_a1.jpg": [(0, 0, 704, 480), (704, 0, 32, 128)],
    PEN + "/222115_newpen0.jpg": [(1728, 400, 192, 32)],
    PEN + "/225720_cat1.jpg": [(1392, 0, 528, 480)],
    PEN + "/230006_lad_P80.jpg": [(1776, 400, 144, 96)],
    PEN + "/230159_grid_before.jpg": [(1776, 400, 144, 96)],
    PEN + "/230347_after_cal80.jpg": [(1776, 400, 144, 96)],
    PEN + "/231913_clean0.jpg": [(1696, 400, 224, 176)],
    PEN + "/232115_lad_LA.jpg": [(1776, 400, 144, 96)],
    PEN + "/232151_lad_LB.jpg": [(1776, 400, 144, 96)],
    TAKE + "/0008_scene_raw.jpg": [(1200, 0, 144, 512), (1344, 432, 576, 80), (1776, 512, 144, 48)],
    TAKE + "/0030_scene_raw.jpg": [(1696, 432, 224, 48), (1776, 480, 144, 64)],
    TAKE + "/0034_hover1_raw.jpg": [(1856, 432, 64, 48)],
    TAKE + "/0048_clean2_raw.jpg": [(1504, 432, 416, 80), (1856, 512, 64, 32)],
    TAKE + "/0084_carry_raw.jpg": [(1552, 432, 368, 64), (1712, 496, 112, 32)],
    TAKE + "/0088_result_raw.jpg": [(1376, 432, 544, 160)],
    "tests/testdata/0124_lifted_raw.jpg": [(992, 0, 720, 384), (1712, 0, 208, 352)],
}


def table() -> dict:
    """photo (path relative to the project folder) -> list of rectangles; only photos that exist here."""
    t = {}
    for d, default in ((PEN, PEN_DEFAULT), (TAKE, TAKE_DEFAULT), (OVER, OVER_DEFAULT)):
        for f in sorted(glob.glob(os.path.join(ROOT, d, "*.jpg"))):
            t[os.path.relpath(f, ROOT)] = list(default)
    for rel, rects in EXTRA.items():
        if os.path.isfile(os.path.join(ROOT, rel)):
            t[rel] = t.get(rel, []) + rects
    return t


def _mask(shape, rects):
    inside = np.zeros(shape[:2], np.uint8)
    for x, y, w, h in rects:
        inside[y:y + h, x:x + w] = 1
    return inside


def is_wiped(path: str, rects) -> bool:
    """True when every listed area of the photo is grey 128 (the outermost pixel of an area may carry a tint)."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return False
    core = cv2.erode(_mask(im.shape, rects), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
    px = im[core]
    return px.size > 0 and int(px.min()) == 128 and int(px.max()) == 128


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def tracked_media():
    """Photos and videos that git tracks here (staged ones included); None when this is not a git working tree."""
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    names = [n for n in out.decode("utf-8", "replace").split("\0") if n]
    return sorted(n for n in names if n.lower().endswith(MEDIA))


def reviewed() -> dict:
    if not os.path.isfile(REVIEWED):
        return {}
    with open(REVIEWED, encoding="utf-8") as f:
        return json.load(f)["photos"]


def check() -> int:
    t = table()
    not_wiped = [rel for rel, rects in sorted(t.items()) if not is_wiped(os.path.join(ROOT, rel), rects)]
    rev = reviewed()
    changed = [rel for rel, h in sorted(rev.items())
               if not os.path.isfile(os.path.join(ROOT, rel)) or _sha256(os.path.join(ROOT, rel)) != h]
    tracked = tracked_media()
    unreviewed = [] if tracked is None else [rel for rel in tracked if rel not in rev]
    print(f"photos with listed areas: {len(t)}; not wiped: {not_wiped}")
    print(f"photos on the reviewed list: {len(rev)}; missing or changed since they were looked at: {changed}")
    print("tracked photos that are not on the reviewed list: "
          + ("(not a git working tree: not checked)" if tracked is None else str(unreviewed)))
    return 1 if (not_wiped or changed or unreviewed) else 0


def accept() -> int:
    tracked = tracked_media()
    if tracked is None:
        print("not a git working tree: nothing to list")
        return 2
    t = table()
    bad = [rel for rel in tracked if rel in t and not is_wiped(os.path.join(ROOT, rel), t[rel])]
    if bad:
        print("refused: listed areas are not wiped in", bad)
        return 1
    data = {"note": "The photos in the repository as they were when they were checked for people and private "
                    "content (path -> SHA-256); docs/MANUAL.md chapter 10 says how. Written by "
                    "`python tools/publish/privacy_wipe.py accept`; `check` compares against it.",
            "photos": {rel: _sha256(os.path.join(ROOT, rel)) for rel in tracked}}
    tmp = REVIEWED + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, REVIEWED)
    print(f"reviewed list written: {len(tracked)} photos")
    return 0


def _wipe(jpegtran: str, src: str, rects, dst: str, work: str) -> None:
    cur = src
    for i, (x, y, w, h) in enumerate(rects):
        if x % 16 or y % 16 or w % 16 or h % 16:
            raise ValueError(f"rectangle {(x, y, w, h)} is not on the 16-pixel block grid")
        nxt = os.path.join(work, f"step{i}.jpg")
        # -copy none: no EXIF, comment or thumbnail of the source survives (today's photos have none)
        subprocess.run([jpegtran, "-copy", "none", "-wipe", f"{w}x{h}+{x}+{y}", "-outfile", nxt, cur], check=True)
        cur = nxt
    shutil.copyfile(cur, dst)


def _unchanged_outside(src: str, dst: str, rects) -> bool:
    a, b = cv2.imread(src, cv2.IMREAD_COLOR), cv2.imread(dst, cv2.IMREAD_COLOR)
    if a is None or b is None or a.shape != b.shape:
        return False
    near = cv2.dilate(_mask(a.shape, rects), np.ones((33, 33), np.uint8)).astype(bool)      # the areas and 16 px around them
    return not np.any(a[~near] != b[~near])


def apply() -> int:
    jpegtran = os.environ.get("JPEGTRAN") or shutil.which("jpegtran")
    if not jpegtran:
        print("jpegtran not found (libjpeg-turbo-progs); set $JPEGTRAN to its path")
        return 2
    t = table()
    failed = []
    with tempfile.TemporaryDirectory(prefix="tazuna_wipe_") as work:      # half-wiped steps never lie in the project
        for rel, rects in sorted(t.items()):
            live, bak = os.path.join(ROOT, rel), os.path.join(BACKUP, rel)
            if not os.path.isfile(bak):
                if is_wiped(live, rects):
                    continue                                   # wiped already and no original here (a fresh clone)
                os.makedirs(os.path.dirname(bak), exist_ok=True)
                shutil.copy2(live, bak)
            out = os.path.join(work, "out.jpg")
            _wipe(jpegtran, bak, rects, out, work)
            if not (is_wiped(out, rects) and _unchanged_outside(bak, out, rects)):
                failed.append(rel)
                continue
            tmp = live + ".tmp_wipe"
            shutil.copyfile(out, tmp)
            os.chmod(tmp, os.stat(live).st_mode)
            os.replace(tmp, live)
    print(f"photos with listed areas: {len(t)}; failed: {failed}; originals: {BACKUP}")
    return 1 if failed else 0


def main(argv) -> int:
    mode = argv[1] if len(argv) > 1 else "check"
    if mode == "check":
        return check()
    if mode == "apply":
        return apply()
    if mode == "accept":
        return accept()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
