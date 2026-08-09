from pymavlink import mavutil
import math
import time
import threading

import config as cfg


def _output_channels():
    """
    The servo channels this vehicle actually drives, taken from config.
    Hardcoding 1..5 here left the real channels (SOL_MOTOR=6 and FRONT_STEER_SERVO=7)
    out of the disarm sweep, so they held their last commanded PWM on shutdown.
    """
    names = ('SOL_MOTOR', 'SAG_MOTOR', 'FRONT_SOL_MOTOR', 'FRONT_SAG_MOTOR', 'FRONT_STEER_SERVO')
    channels = []
    for n in names:
        ch = getattr(cfg, n, None)
        if ch is not None and ch not in channels:
            channels.append(int(ch))
    return channels or [1, 2, 3, 4, 5]


class USVController:
    """
    Interface to communicate with the Flight Controller (e.g., OrangeCube) via PyMavlink.
    """

    def __init__(self, port="/dev/ttyACM0", baud=57600):
        self.port = port
        self.baud = baud
        # Channel assignment lives in config.py (SOL_MOTOR / SAG_MOTOR / FRONT_* / STEER).
        self.channels = _output_channels()
        self.pwms = {ch: 1500 for ch in self.channels}

        print(f"[USVController] Initializing on {port} at {baud} baud. Waiting for connection...")
        try:
            self.master = mavutil.mavlink_connection(port, baud=baud)

            # Without this, pyserial's underlying write() blocks forever if the link glitches
            # (buffer full / device unresponsive), which freezes the entire single-threaded
            # nav loop - stuck heading/distance/PWM, no watchdog recovery since the process
            # is still technically alive. Cap it so a bad write raises instead of hanging.
            try:
                self.master.port.write_timeout = 0.2
                self.master.port.timeout = 0.2
            except Exception:
                pass

            print("[USVController] Connected to MAVLINK. Waiting for heartbeat...")
            # Bounded wait. An unbounded wait_heartbeat() blocks the nav process forever
            # if the FC link is down - and because the orchestrator watchdog restarts nav
            # by re-running this constructor, a dead link turned into an endless
            # kill/restart/hang cycle with no telemetry explaining why.
            if self.master.wait_heartbeat(timeout=10) is None:
                raise RuntimeError("No MAVLink heartbeat within 10s")
            print("[USVController] Heartbeat found!")

            # Arka planda dinlenecek mesajlar için veri yapısı
            self.msg_dict = {}
            self.running = True

            # MAVLink mesaj stream'lerini başlat
            self._request_data_streams()

            # Dinleyici thread başlat (Eski sistemindeki gibi veri akışını kitlememek için)
            self.listener_thread = threading.Thread(target=self._listen_messages)
            self.listener_thread.daemon = True
            self.listener_thread.start()
        except Exception as e:
            print(f"[USVController] Connection failed: {e}")
            self.master = None

    def _request_data_streams(self):
        """Tüm gerekli mesajların gelmesini sağlar."""
        if self.master:
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                5,  # 5 Hz hızında iste
                1  # 1 = Başlat
            )
            print("[USVController] Data stream requested (ALL @ 5Hz).")

    def _listen_messages(self):
        """Arka planda MAVLink mesajlarını dinler ve msg_dict'te saklar."""
        while self.running and self.master:
            # Drain the entire queue to prevent buffer bloat
            while True:
                msg = self.master.recv_match(blocking=False)
                if not msg:
                    break
                self.msg_dict[msg.get_type()] = msg
            time.sleep(0.01)  # CPU'yu boğmamak için ufak gecikme

    def stop_listener(self):
        """Thread'i güvenli şekilde durdurur."""
        self.running = False
        if hasattr(self, 'listener_thread'):
            self.listener_thread.join()

    def get_current_position(self):
        """Returns latitude and longitude."""
        msg = self.msg_dict.get('GPS_RAW_INT')
        if msg:
            return msg.lat / 1e7, msg.lon / 1e7
        return 0.0, 0.0

    def get_horizontal_speed(self):
        """Returns ground speed in m/s."""
        msg = self.msg_dict.get('LOCAL_POSITION_NED')
        if msg:
            vx = msg.vx
            vy = msg.vy
            return math.sqrt(vx ** 2 + vy ** 2)
        return 0.0

    def get_heading(self):
        """
        Returns compass heading in degrees, or None if the FC has not reported one.

        Returning 0.0 for "no data" was indistinguishable from a genuine due-North
        heading: with HEADING_SOURCE='FC' the nav process fed that straight into
        magnetic_heading and steered as if the boat were pointing North.
        """
        msg = self.msg_dict.get('GLOBAL_POSITION_INT')
        if msg and msg.hdg != 65535:  # 65535 = UINT16_MAX = "unknown" per the MAVLink spec
            return msg.hdg / 100.0  # cdeg to degrees

        # Fallback
        msg_vfr = self.msg_dict.get('VFR_HUD')
        if msg_vfr:
            return float(msg_vfr.heading)
        return None

    def get_servo_pwm(self, channel):
        """Returns the last commanded PWM for a given channel."""
        # Seri portu meşgul etmemek için doğrudan önbellekteki (cache) PWM'i döndürüyoruz
        return self.pwms.get(channel, 1500)

    def set_servo(self, channel, pwm):
        """Commands the hardware to set a specific PWM on a servo channel."""
        self.pwms[channel] = pwm
        if self.master:
            try:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                    0,
                    channel,
                    pwm,
                    0, 0, 0, 0, 0
                )
            except Exception as e:
                # A bounded write_timeout means we land here instead of hanging the nav loop.
                # Don't re-raise: the caller (nav_worker) needs to keep looping so its
                # heartbeat stays fresh and the watchdog doesn't have to kill+restart it
                # just because one servo write glitched.
                print(f"[USVController][WARNING] set_servo({channel}) write failed: {e}")

    def set_mode(self, mode_name):
        """Changes the vehicle flight mode."""
        print(f"[USVController] Mode set to {mode_name}")
        if self.master and mode_name in self.master.mode_mapping():
            mode_id = self.master.mode_mapping()[mode_name]
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )

    def arm_vehicle(self):
        """Arms the thrusters."""
        print("[USVController] Vehicle arming...")
        if self.master:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1, 0, 0, 0, 0, 0, 0
            )

    def disarm_vehicle(self):
        """Disarms the thrusters for safety."""
        print("[USVController] Vehicle disarming...")
        for channel in self.channels:
            self.set_servo(channel, 1500)

        if self.master:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0, 0, 0, 0, 0, 0, 0
            )

    def get_gps_fix_type_verbose(self):
        """Returns a human-readable GPS fix type."""
        msg = self.msg_dict.get('GPS_RAW_INT')
        fix_type = msg.fix_type if msg else None

        fix_map = {
            0: "No GPS", 1: "No FIX", 2: "2D Fix", 3: "3D Fix",
            4: "DGPS", 5: "RTK Float", 6: "RTK Fixed", 7: "STATIC", 8: "PPP",
        }
        return fix_map.get(fix_type, "Unknown") if fix_type is not None else "No Data"
