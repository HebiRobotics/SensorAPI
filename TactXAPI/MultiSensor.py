# multisensor_rate.py
from __future__ import annotations
import threading, time
from typing import Dict, Optional, Tuple, Callable
import numpy as np
from TactXAPI.ForceSensor import ForceSensor

class MultiSensor:
    def __init__(
        self,
        sensors: Dict[str, ForceSensor],
        read_hz: Optional[float] = None,   # Throttle for device reading, e.g. 100.0
        emit_hz: Optional[float] = None,   # External publishing frequency, e.g. 100.0
    ):
        self.sensors = sensors
        self.read_hz = read_hz
        self.emit_hz = emit_hz

        # Threads and control
        self._reader_threads: Dict[str, threading.Thread] = {}
        self._reader_stops: Dict[str, threading.Event] = {k: threading.Event() for k in sensors.keys()}
        self._emit_thread: Optional[threading.Thread] = None
        self._emit_stop = threading.Event()

        # Cache and locks
        self._latest: Dict[str, np.ndarray] = {k: np.zeros_like(v.data) for k, v in sensors.items()}
        self._ts: Dict[str, float] = {k: 0.0 for k in sensors.keys()}
        self._locks: Dict[str, threading.RLock] = {k: threading.RLock() for k in sensors.keys()}

        # Callback: when each sensor frame arrives
        #   cb(frame, ts, name) -> None
        self.on_frame: Dict[str, Optional[Callable[[np.ndarray, float, str], None]]] = {k: None for k in sensors.keys()}

        # Callback: at fixed sampling/publishing frequency (aggregated)
        #   cb(dict{name: frame}, ts) -> None
        self.on_emit: Optional[Callable[[Dict[str, np.ndarray], float], None]] = None

    # ---------- lifecycle ----------
    def start(self) -> None:
        # Reading threads
        for name in list(self.sensors.keys()):
            if name in self._reader_threads and self._reader_threads[name].is_alive():
                continue
            self._reader_stops[name].clear()
            t = threading.Thread(target=self._reader_loop, args=(name,), name=f"MultiSensor-Reader-{name}", daemon=True)
            self._reader_threads[name] = t
            t.start()

        # Emission thread (optional)
        if self.emit_hz and (self._emit_thread is None or not self._emit_thread.is_alive()):
            self._emit_stop.clear()
            self._emit_thread = threading.Thread(target=self._emit_loop, name="MultiSensor-Emit", daemon=True)
            self._emit_thread.start()

    def stop(self, join: bool = True) -> None:
        for ev in self._reader_stops.values():
            ev.set()
        self._emit_stop.set()
        if join:
            for t in self._reader_threads.values():
                if t.is_alive():
                    t.join(timeout=1.5)
            if self._emit_thread and self._emit_thread.is_alive():
                self._emit_thread.join(timeout=1.5)

    def close(self) -> None:
        self.stop(join=True)
        for s in self.sensors.values():
            try:
                if s.ser and s.ser.is_open:
                    s.ser.close()
            except Exception:
                pass

    # ---------- getters ----------
    def get_latest(self, name: str, copy: bool = True) -> Optional[np.ndarray]:
        if name not in self._latest:
            return None
        with self._locks[name]:
            return self._latest[name].copy() if copy else self._latest[name]

    def get_all_latest(self, copy: bool = True) -> Dict[str, np.ndarray]:
        out = {}
        for name in self.sensors.keys():
            with self._locks[name]:
                out[name] = self._latest[name].copy() if copy else self._latest[name]
        return out

    def get_max(self, name: str) -> Optional[Tuple[int, int, int]]:
        if name not in self._latest:
            return None
        with self._locks[name]:
            frame = self._latest[name]
            idx = int(np.argmax(frame))
            row, col = divmod(idx, self.sensors[name].num_cols)
            return int(frame[row, col]), row, col

    def get_all_max(self) -> Dict[str, Tuple[int, int, int]]:
        out = {}
        for name in self.sensors.keys():
            val = self.get_max(name)
            if val is not None:
                out[name] = val
        return out

    # ---------- internal loops ----------
    def _reader_loop(self, name: str) -> None:
        sensor = self.sensors[name]
        stop_evt = self._reader_stops[name]

        period = (1.0 / self.read_hz) if self.read_hz and self.read_hz > 0 else None
        next_deadline = time.perf_counter()

        while not stop_evt.is_set():
            try:
                # If throttling frequency is set, wait until the next cycle before calling find_frame
                if period is not None:
                    now = time.perf_counter()
                    if now < next_deadline:
                        time.sleep(next_deadline - now)
                    next_deadline += period

                frame = sensor.find_frame()  # Block until one frame is received (limited by device frame rate)
                ts = time.perf_counter()

                with self._locks[name]:
                    if frame.shape == self._latest[name].shape and frame.dtype == self._latest[name].dtype:
                        self._latest[name][:] = frame
                    else:
                        self._latest[name] = frame.copy()
                    self._ts[name] = ts

                cb = self.on_frame.get(name)
                if cb:
                    try:
                        cb(self._latest[name], ts, name)
                    except Exception:
                        pass

            except Exception:
                time.sleep(0.001)

    def _emit_loop(self) -> None:
        """Fixed-frequency aggregated publishing (unified external callback on_emit)."""
        assert self.emit_hz and self.emit_hz > 0
        period = 1.0 / self.emit_hz
        next_deadline = time.perf_counter()

        while not self._emit_stop.is_set():
            now = time.perf_counter()
            if now < next_deadline:
                time.sleep(next_deadline - now)
            next_deadline += period

            ts = time.perf_counter()
            if self.on_emit:
                try:
                    self.on_emit(self.get_all_latest(copy=True), ts)
                except Exception:
                    pass
