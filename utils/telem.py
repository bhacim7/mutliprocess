import queue
import time
import datetime
import serial
import json
import threading

import config as cfg


class TelemetrySender:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self._tx_lock = threading.Lock()
        self._oversize_warned = False
        try:
            # write_timeout bounds ser.write() too - without it, a link glitch blocks forever
            # even though send() below already wraps the call in try/except.
            self.ser = serial.Serial(port, baud, timeout=1, write_timeout=0.5)
            print(f"[TelemetrySender] Initialized on {port} @ {baud}")
        except Exception as e:
            print(f"[TelemetrySender] Error opening {port}: {e}")

    def _write_budget_bytes(self):
        """
        How many bytes a single write() can physically clear before write_timeout fires.

        At 57600 baud 8N1 that is 5760 B/s, so a 0.5 s timeout caps one write at 2880 bytes.
        The old telemetry packet was 2960 B with 10 tracked objects and 4960 B with 20 - i.e.
        it could not complete, ever, once the Task 2 buoy field came into view.
        """
        try:
            timeout = self.ser.write_timeout or 0.5
        except Exception:
            timeout = 0.5
        return int((self.baud / 10.0) * timeout)

    def send(self, payload):
        """
        Serialise and transmit one telemetry line. Returns True if the whole line went out.

        Two things changed here, and together they are what actually fixed the GCS link:

        1. Size guard. A write that exceeds the byte budget above raises
           SerialTimeoutException *after* the driver has already shipped part of the line.
           The remainder is dropped, so the line reaches the GCS with no trailing '\\n'. The
           GCS then glues the fragment onto the next packet, json.loads() throws, and BOTH
           packets are discarded silently (GCSv1000.SerialWorker catches JSONDecodeError and
           passes). That is why the green RX led never blinked while commands - which are
           tiny - always got through. Refusing to start an oversized write is the fix.

        2. Resynchronisation. If a write does time out anyway, flush whatever is still queued
           and emit a lone newline. That terminates the corrupt fragment as its own bad line
           so the NEXT packet starts on a clean boundary instead of being poisoned too.

        Compact separators are used because json.dumps defaults to ', ' / ': ' and those
        spaces are pure airtime.
        """
        if not (self.ser and self.ser.is_open):
            return False

        try:
            data = (json.dumps(payload, separators=(',', ':')) + "\n").encode('utf-8')
        except Exception as e:
            print(f"[TelemetrySender] Serialize error: {e}")
            return False

        budget = self._write_budget_bytes()
        max_payload = int(getattr(cfg, 'TELEM_MAX_PAYLOAD_B', 1500))
        limit = min(budget, max_payload) if budget > 0 else max_payload

        if len(data) > limit:
            if not self._oversize_warned:
                print(f"[TelemetrySender] Payload {len(data)} B exceeds the {limit} B "
                      f"link budget - dropping it instead of sending a truncated line. "
                      f"Trim the payload built in telem_process.telem_worker().")
                self._oversize_warned = True
            return False
        self._oversize_warned = False

        with self._tx_lock:
            try:
                self.ser.write(data)
                return True
            except Exception as e:
                print(f"[TelemetrySender] Transmit error: {e}")
                # Drop the un-transmitted remainder and close off the partial line so the
                # receiver's line buffer resynchronises on the next packet.
                try:
                    self.ser.reset_output_buffer()
                    self.ser.write(b"\n")
                except Exception:
                    pass
                return False

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("[TelemetrySender] Closed.")

class CommandReceiver:
    """
    Reads command lines from the radio and forwards them to nav_process.

    Rewritten from readline() to a buffered read, for three reasons measured on the water:

      * readline() blocks up to the port's 1 s timeout when a line arrives in two radio
        frames - in_waiting sees the first fragment, readline() then sits waiting for the
        newline. Every command behind it (the emergency stop included) waited too.
      * One line per 100 ms iteration capped intake at 10 lines/s, and the boat's radio
        hears EVERYTHING on the channel - the drone's 3 Hz packets included - so the real
        inbound rate sat uncomfortably close to that ceiling and bursts backed up.
      * Those drone packets were queued for nav_process, which shuttled them over a
        Manager queue only to ignore them. Anything without a "cmd" key is dropped here.

    read(in_waiting) never blocks; all complete lines in the buffer are processed in one
    pass; a partial line simply stays in the buffer for the next pass.
    """
    def __init__(self, telemetry, cmd_queue):
        self.telemetry = telemetry
        self.cmd_queue = cmd_queue
        self.running = False
        self._buf = b""
        import threading
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True

    def start(self):
        self.running = True
        self.thread.start()
        print("[CommandReceiver] Started listening for commands.")

    def stop(self):
        self.running = False

    def _process_bytes(self, data):
        """Append raw bytes; queue every complete command line found."""
        self._buf += data
        if len(self._buf) > 8192:      # newline lost in a corrupt stretch: resync
            self._buf = b""
            return
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line.decode('utf-8'))
            except Exception:
                continue
            # Commands all carry "cmd". The drone's telemetry (and our own, if the radio
            # ever echoes) does not, and forwarding it was pure IPC noise.
            if isinstance(cmd, dict) and cmd.get("cmd"):
                self.cmd_queue.put(cmd)

    def _listen(self):
        while self.running:
            try:
                ser = self.telemetry.ser
                if ser and ser.is_open and ser.in_waiting > 0:
                    self._process_bytes(ser.read(ser.in_waiting))
            except Exception:
                pass
            time.sleep(0.05)

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