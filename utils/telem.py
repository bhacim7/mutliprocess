import queue
import time
import datetime
import serial
import json

class TelemetrySender:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            print(f"[TelemetrySender] Initialized on {port} @ {baud}")
        except Exception as e:
            print(f"[TelemetrySender] Error opening {port}: {e}")

    def send(self, payload):
        if self.ser and self.ser.is_open:
            try:
                data = json.dumps(payload) + "\n"
                self.ser.write(data.encode('utf-8'))
            except Exception as e:
                print(f"[TelemetrySender] Transmit error: {e}")
        else:
            # Fallback if no serial
            pass

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("[TelemetrySender] Closed.")

class CommandReceiver:
    def __init__(self, telemetry, cmd_queue):
        self.telemetry = telemetry
        self.cmd_queue = cmd_queue
        self.running = False
        import threading
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True

    def start(self):
        self.running = True
        self.thread.start()
        print("[CommandReceiver] Started listening for commands.")

    def stop(self):
        self.running = False

    def _listen(self):
        while self.running:
            if self.telemetry.ser and self.telemetry.ser.is_open:
                try:
                    if self.telemetry.ser.in_waiting > 0:
                        line = self.telemetry.ser.readline().decode('utf-8').strip()
                        if line:
                            cmd = json.loads(line)
                            self.cmd_queue.put(cmd)
                except Exception:
                    pass
            time.sleep(0.1)

class TelemetryTx:
    def __init__(self, telemetry, max_hz=10):
        self.telemetry = telemetry
        self.max_hz = max_hz

    def send(self, payload):
        self.telemetry.send(payload)

def handle_command(cmd, controller, cfg, manual_mode, mission_started):
    cmd_str = cmd.get("cmd")
    if cmd_str == "manual_override":
        manual_mode = True
        print("[Telem Utils] Switched to MANUAL mode.")
    elif cmd_str == "auto_mode":
        manual_mode = False
        print("[Telem Utils] Switched to AUTO mode.")
    return manual_mode, mission_started
