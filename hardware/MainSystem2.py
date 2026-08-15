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
        # Last time each channel was actually transmitted (see set_servo's rate limiting).
        self._last_servo_tx = {ch: 0.0 for ch in self.channels}

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

            # RELAY_STATUS is not part of the legacy stream groups, so ask for it by id.
            # Without this the relay state can only be inferred from what we last sent,
            # which goes stale the moment the transmitter switch is used.
            try:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    376,        # RELAY_STATUS
                    500000,     # microseconds -> 2 Hz
                    0, 0, 0, 0, 0
                )
                print("[USVController] RELAY_STATUS requested @ 2Hz.")
            except Exception as e:
                print(f"[USVController][WARNING] Could not request RELAY_STATUS: {e}")

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
        """
        Returns ground speed in m/s, or None if no source reported one.

        This used to read LOCAL_POSITION_NED only. ArduRover does not stream that message
        in the default set, so the call returned 0.0 on every single cycle - the HUD showed
        "HIZ: 0.0" for the whole mission and, worse, Pure Pursuit's speed-adaptive lookahead
        collapsed to its PURE_PURSUIT_MIN_LOOKAHEAD floor (0.8 m) permanently. At ~1.8 m/s
        that is a 0.45 s horizon, which is what made the path follower so twitchy.

        Returning None rather than 0.0 for "no data" lets the caller tell a genuinely
        stationary boat apart from a missing message.
        """
        # 1. VFR_HUD carries ground speed directly and is in the default stream set.
        msg = self.msg_dict.get('VFR_HUD')
        if msg is not None:
            try:
                spd = float(msg.groundspeed)
                if spd == spd and abs(spd) < 100.0:  # NaN guard + sanity
                    return abs(spd)
            except (AttributeError, TypeError, ValueError):
                pass

        # 2. GPS_RAW_INT.vel is ground speed in cm/s (65535 = unknown per the MAVLink spec).
        msg = self.msg_dict.get('GPS_RAW_INT')
        if msg is not None:
            try:
                if msg.vel != 65535:
                    return msg.vel / 100.0
            except (AttributeError, TypeError):
                pass

        # 3. GLOBAL_POSITION_INT vx/vy are cm/s in the NED frame.
        msg = self.msg_dict.get('GLOBAL_POSITION_INT')
        if msg is not None:
            try:
                return math.hypot(msg.vx, msg.vy) / 100.0
            except (AttributeError, TypeError):
                pass

        # 4. Last resort - the original source, in case a different FC does stream it.
        msg = self.msg_dict.get('LOCAL_POSITION_NED')
        if msg is not None:
            try:
                return math.hypot(msg.vx, msg.vy)
            except (AttributeError, TypeError):
                pass

        return None

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

    def set_servo(self, channel, pwm, force=False):
        """
        Commands the hardware to set a specific PWM on a servo channel.

        `force=True` bypasses the rate limiting below - use it for anything safety
        critical (disarm, emergency neutralise) where the command MUST go out even if
        the cached value already matches.

        Rate limiting: the nav loop drives 5 channels every cycle. At the old 50 Hz that
        was 250 COMMAND_LONG/s, and ArduPilot answers every one of them with a
        COMMAND_ACK - enough to starve its scheduler and delay the telemetry streams.
        Measured effect in flight: GPS_RAW_INT stalled for ~1.5 s (the HUD position and
        HEDEFE_MESAFE froze) while the 50 Hz control loop kept integrating on stale data.

        DO_SET_SERVO latches on the FC, so re-sending an unchanged value is pure waste.
        Skip writes smaller than SERVO_MIN_PWM_DELTA, but always refresh at least every
        SERVO_REFRESH_S so a dropped command cannot leave a channel stuck.
        """
        pwm = int(pwm)
        now = time.time()

        prev = self.pwms.get(channel)
        last_tx = self._last_servo_tx.get(channel, 0.0)
        min_delta = getattr(cfg, 'SERVO_MIN_PWM_DELTA', 3)
        refresh_s = getattr(cfg, 'SERVO_REFRESH_S', 0.5)

        self.pwms[channel] = pwm

        if (not force
                and prev is not None
                and abs(pwm - prev) < min_delta
                and (now - last_tx) < refresh_s):
            return

        self._last_servo_tx[channel] = now

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

    def set_relay(self, state, relay=None):
        """
        Switch the motor-power relay via MAVLink.

        This reaches the SAME relay state inside ArduPilot that the transmitter's
        RC7_OPTION=28 switch writes to, so the two are not rivals - whichever wrote last
        wins. The RC path is edge triggered (it acts when the switch MOVES, not
        continuously), which is why turning the transmitter off leaves the relay as it was.

        param1 of MAV_CMD_DO_SET_RELAY is the relay INSTANCE and it is ZERO based:
        instance 0 is the one configured by RELAY1_PIN / RELAY1_FUNCTION. Sending 1 here
        would silently address a second relay that does not exist.
        """
        if not self.master:
            return False
        if relay is None:
            relay = getattr(cfg, 'RELAY_INSTANCE', 0)
        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_RELAY,
                0,
                int(relay),          # instance, 0 based
                1 if state else 0,   # 1 = energised, 0 = off
                0, 0, 0, 0, 0
            )
            return True
        except Exception as e:
            print(f"[USVController][WARNING] set_relay({state}) failed: {e}")
            return False

    def get_relay_state(self, relay=None):
        """
        True/False from the vehicle's own RELAY_STATUS, or None if it never arrives.

        Reading the real state matters because the transmitter can change it behind our
        back; echoing back "whatever the GCS last sent" would be wrong exactly when it
        matters most. RELAY_STATUS carries two bitmasks - `present` says the relay exists,
        `on` says it is energised.
        """
        msg = self.msg_dict.get('RELAY_STATUS')
        if msg is None:
            return None
        if relay is None:
            relay = getattr(cfg, 'RELAY_INSTANCE', 0)
        try:
            bit = 1 << int(relay)
            if not (int(msg.present) & bit):
                return None
            return bool(int(msg.on) & bit)
        except (AttributeError, TypeError, ValueError):
            return None

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
            # force: never let set_servo's rate limiting swallow a neutralise command.
            self.set_servo(channel, 1500, force=True)

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