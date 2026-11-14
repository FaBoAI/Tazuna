"""Camera thread: keeps the C920 open and always holds the newest frame.

Opening the device and warming it up costs about 1.2 s per `camera.grab()`; a pick-and-place
needs several looks, so the camera is opened ONCE and a background thread keeps reading.

    with FastCamera() as cam:               # locks focus / exposure / white balance like camera.lock_settings()
        img = cam.latest()                   # newest frame, a few ms once warm
        ...move the arm...
        img = cam.wait_fresh(time.time())    # first frame EXPOSED after this instant
    # leaving the block stops the thread, releases the device and (restore=True) puts the camera
    # back to automatic exposure / focus / white balance

A frame is never handed out silently stale: `latest()` raises RuntimeError when the newest frame
is older than `max_age_s`, `wait_fresh()` when no new frame arrives in time.
"""

from __future__ import annotations

if __name__ == "__main__":  # run as a script: put tools/ and its feature folders on the import path first
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import tazuna_paths  # noqa: F401

import threading
import time
from collections import deque

import cv2
import numpy as np

import camera as C


class FastCamera:
    """Background capture on camera.DEV (MJPG, camera.W x camera.H).

    lock:    apply camera.lock_settings() once when starting (fixed focus, exposure, gain, white balance);
             RuntimeError when the camera then still reports an automatic mode (the lock did not take)
    restore: call camera.restore_auto() when closing
    settle_s: when the settings were not locked before, frames of the first settle_s seconds are
              dropped as well (the lens is still travelling to its fixed focus position)
    latency_s: time between the start of a frame's exposure and the moment the driver hands it over
               (exposure 33 ms + readout + USB); used by wait_fresh() so that a frame taken while
               the arm was still moving is never returned as fresh.
    """

    def __init__(self, lock: bool = True, restore: bool = True, dev: str | None = None, width: int | None = None,
                 height: int | None = None, warm_frames: int = 5, latency_s: float = 0.08, start: bool = True,
                 settle_s: float = 0.6):
        self.lock, self.restore = bool(lock), bool(restore)
        self.dev = dev or C.DEV
        self.width, self.height = int(width or C.W), int(height or C.H)
        self.warm_frames = int(warm_frames)
        self.latency_s = float(latency_s)
        self.settle_s = float(settle_s)     # extra warm-up when the settings had to be changed (focus travels)
        self._warm_until = 0.0
        self._cap = None
        self._thread = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._frames = deque(maxlen=8)      # (grab_time, sequence, frame)
        self._seq = 0
        self._error = None
        self._min_time = 0.0                # frames exposed before this instant are not handed out
        self._locked = False
        self.exposure = "desk"              # the preset lock_settings() leaves; see set_exposure()
        self.stats = {"frames": 0, "read_failures": 0, "started": None, "fps": 0.0, "settings_changed": None}
        if start:
            self.start()

    # ------------------------------------------------------------------ life cycle
    def start(self) -> "FastCamera":
        if self._thread is not None:
            return self
        C._locked("The camera")             # hardware lock (tools/motion/hwlock.py): raises before anything is touched
        if self.lock:
            already = C.settings_locked()       # None = cannot tell: treat as changed
            C.lock_settings()
            self._locked = True
            still_auto = C.auto_modes_on()      # [] = locked; None = cannot tell (carry on)
            if still_auto:
                self._restore_if_needed()
                raise RuntimeError(f"camera {self.dev}: the settings could not be locked (still automatic: "
                                   f"{', '.join(still_auto)}); the camera model and the colour thresholds assume "
                                   f"the locked settings. Check with: v4l2-ctl -d {self.dev} -l")
            self.stats["settings_changed"] = not already
            self._warm_until = 0.0 if already else time.time() + self.settle_s
        cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            self._restore_if_needed()
            raise RuntimeError(f"cannot open camera {self.dev}")
        self._cap = cap
        self._stop.clear()
        self._error = None
        self.stats.update(frames=0, read_failures=0, started=time.time(), fps=0.0)
        self._thread = threading.Thread(target=self._run, name="fastcam", daemon=True)
        self._thread.start()
        # wait for the first usable frame so that a dead camera is reported here, not later
        deadline = time.time() + 5.0
        with self._cond:
            while not self._frames and self._error is None and time.time() < deadline:
                self._cond.wait(0.1)
            ok = bool(self._frames)
            err = self._error
        if not ok:
            self.close()
            raise RuntimeError(f"camera {self.dev} delivered no frame within 5 s" + (f" ({err})" if err else ""))
        return self

    def _run(self):
        cap = self._cap
        n_bad = 0
        warm = self.warm_frames
        t_prev = None
        while not self._stop.is_set():
            ok = cap.grab()
            t = time.time()
            frame = None
            if ok:
                ok, frame = cap.retrieve()
            if not ok or frame is None or frame.shape[0] != self.height or frame.shape[1] != self.width:
                n_bad += 1
                self.stats["read_failures"] += 1
                if n_bad >= 30:
                    with self._cond:
                        self._error = "30 consecutive failed reads"
                        self._cond.notify_all()
                    return
                time.sleep(0.01)
                continue
            n_bad = 0
            if warm > 0 or t < self._warm_until:      # the first frames after opening are still adapting
                warm -= 1
                continue
            with self._cond:
                self._seq += 1
                self._frames.append((t, self._seq, frame))
                self.stats["frames"] += 1
                if t_prev is not None and t > t_prev:
                    self.stats["fps"] = 0.9 * self.stats["fps"] + 0.1 / (t - t_prev) if self.stats["fps"] else 1.0 / (t - t_prev)
                t_prev = t
                self._cond.notify_all()

    def _restore_if_needed(self):
        if self.restore and self._locked:
            C.restore_auto()
        self._locked = False

    def close(self):
        """Stop the thread, release the device and (restore=True) return the camera to automatic mode."""
        self._stop.set()
        th, self._thread = self._thread, None
        if th is not None:
            th.join(timeout=2.0)
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
        with self._cond:
            self._frames.clear()
            self._cond.notify_all()
        self._restore_if_needed()

    # ------------------------------------------------------------------ exposure by scene
    def set_exposure(self, name: str, settle_s: float = 0.5) -> None:
        """Switch the exposure preset (camera.EXPOSURES: "desk" | "paper") while the thread keeps running.
        Frames exposed before the change, and during settle_s after it, are never handed out afterwards
        (latest() waits for new ones). The caller puts "desk" back: use paper_frame() for one photo."""
        C.set_exposure(name)
        self.exposure = name
        with self._cond:
            self._min_time = max(self._min_time, time.time() + float(settle_s))

    def paper_frame(self, n_avg: int = 4, exposure: str = "paper", settle_s: float = 0.5) -> np.ndarray:
        """One photo with another exposure preset (default "paper": white paper not clipped, ink visible), then
        back to "desk", also when no frame comes. For a command that already holds the camera open, where
        camera.grab(exposure=...) could not open the device a second time."""
        self.set_exposure(exposure, settle_s)
        try:
            self.wait_fresh(self._min_time, timeout_s=float(settle_s) + 2.0)   # the first frame with the new exposure
            return self.latest(n_avg=n_avg, max_age_s=0.2)
        finally:
            if exposure != "desk":
                self.set_exposure("desk", settle_s)

    def __enter__(self) -> "FastCamera":
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ frames
    def _check_alive(self):
        if self._thread is None:
            raise RuntimeError("camera is not running (start() it or use it as a context manager)")
        if self._error is not None:
            raise RuntimeError(f"camera stopped delivering frames: {self._error}")

    def _usable(self, now: float, max_age_s: float) -> list:
        """Buffered frames (oldest first) that are young enough and not older than the last wait_fresh mark."""
        return [f for f in self._frames
                if now - f[0] <= max_age_s and f[0] - self.latency_s >= self._min_time]

    def latest(self, n_avg: int = 1, max_age_s: float = 0.2) -> np.ndarray:
        """Newest frame (BGR, uint8). n_avg > 1 returns the mean of the newest n_avg frames, which
        must all be newer than max_age_s + (n_avg - 1) frame periods (and newer than the last
        wait_fresh() mark); waits briefly for them if needed. RuntimeError when the camera has
        stalled; never returns an old frame silently."""
        n_avg = max(1, int(n_avg))
        self._check_alive()
        span = max_age_s + (n_avg - 1) * 0.045
        deadline = time.time() + max(1.0, 2.0 * span)
        with self._cond:
            while True:
                self._check_alive()
                now = time.time()
                good = self._usable(now, span)
                if len(good) >= n_avg:
                    frames = [f[2] for f in good[-n_avg:]]
                    break
                if now >= deadline:
                    newest = self._frames[-1][0] if self._frames else None
                    age = "none" if newest is None else f"{now - newest:.2f} s old"
                    raise RuntimeError(f"camera: no fresh frame (wanted {n_avg} newer than {span:.2f} s; newest is {age})")
                self._cond.wait(0.05)
        if n_avg == 1:
            return frames[0].copy()
        acc = frames[0].astype(np.float32)
        for f in frames[1:]:
            acc += f
        return np.clip(acc / n_avg + 0.5, 0, 255).astype(np.uint8)

    def wait_fresh(self, after_time: float, timeout_s: float = 2.0) -> np.ndarray:
        """First frame whose exposure started after `after_time` (a time.time() value, e.g. the
        moment the arm came to rest). Later latest() calls also ignore anything older."""
        self._check_alive()
        deadline = time.time() + timeout_s
        with self._cond:
            self._min_time = max(self._min_time, float(after_time))
            while True:
                self._check_alive()
                for t, _seq, frame in self._frames:
                    if t - self.latency_s >= after_time:
                        return frame.copy()
                if time.time() >= deadline:
                    raise RuntimeError(f"camera: no frame exposed after the requested time within {timeout_s:.1f} s")
                self._cond.wait(0.05)

    def age(self) -> float:
        """Seconds since the newest frame was handed over by the driver (inf when there is none)."""
        with self._cond:
            return time.time() - self._frames[-1][0] if self._frames else float("inf")


if __name__ == "__main__":
    # quick self-check on the live camera: frame rate, latency of latest(), always back to auto mode
    with FastCamera() as cam:
        t0 = time.time()
        img = cam.latest()
        print("first latest():", img.shape, f"{(time.time() - t0) * 1000:.1f} ms")
        time.sleep(1.0)
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            cam.latest()
            ts.append(time.perf_counter() - t)
            time.sleep(0.02)
        t = time.time()
        cam.wait_fresh(t)
        print(f"latest(): median {np.median(ts) * 1000:.2f} ms; wait_fresh: {(time.time() - t) * 1000:.0f} ms; "
              f"stats {cam.stats}")
    print("closed; camera restored to auto")
