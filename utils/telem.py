import serial
import json
import math
import numpy as np
import threading
import queue
import time
import enum

def _json_default(o):
    if isinstance(o, enum.Enum):
        return o.name
    return str(o)

class TelemetrySender:
    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.ser = None
        self.enabled = False
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.enabled = True
            print(f"[TELEM_UTILS] Connected {self.port} @ {self.baud}")
        except Exception as e:
            print(f"[TELEM_UTILS] Serial open failed: {e}. Telemetry disabled.")

    @staticmethod
    def _clean_num(x):
        try:
            if x is None:
                return None
            xf = float(x)
            return xf if math.isfinite(xf) else None
        except Exception:
            return None

    def send(self, payload: dict):
        if not self.enabled:
            return
        try:
            clean = {k: (self._clean_num(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                     for k, v in payload.items()}
            line = json.dumps(clean, ensure_ascii=False, default=_json_default) + "\n"
            self.ser.write(line.encode("utf-8"))
        except Exception as e:
            print(f"[TELEM_UTILS] Send failed: {e}")

    def close(self):
        if self.ser:
            self.ser.close()

class CommandReceiver(threading.Thread):
    def __init__(self, telemetry, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.telemetry = telemetry
        self.ser = getattr(telemetry, "ser", None)
        self.out_queue = out_queue
        self._stop = threading.Event()

    def run(self):
        if not self.ser:
            return
        while not self._stop.is_set():
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    time.sleep(0.01)
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "cmd" in obj:
                        self.out_queue.put(obj)
                except Exception:
                    pass
            except Exception:
                time.sleep(0.05)

    def stop(self):
        self._stop.set()