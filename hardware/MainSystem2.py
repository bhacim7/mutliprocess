from pymavlink import mavutil
import math
import time
import threading

class USVController:
    """
    Interface to communicate with the Flight Controller (e.g., OrangeCube) via PyMavlink.
    """
    def __init__(self, port="/dev/ttyACM0", baud=57600):
        self.port = port
        self.baud = baud
        self.pwms = {1: 1500, 3: 1500} # left=1, right=3
        
        print(f"[USVController] Initializing on {port} at {baud} baud. Waiting for connection...")
        try:
            self.master = mavutil.mavlink_connection(port, baud=baud)
            print("[USVController] Connected to MAVLINK. Waiting for heartbeat...")
            self.master.wait_heartbeat()
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
                5, # 5 Hz hızında iste
                1  # 1 = Başlat
            )
            print("[USVController] Data stream requested (ALL @ 5Hz).")

    def _listen_messages(self):
        """Arka planda MAVLink mesajlarını dinler ve msg_dict'te saklar."""
        while self.running and self.master:
            msg = self.master.recv_match(blocking=False)
            if msg:
                self.msg_dict[msg.get_type()] = msg
            time.sleep(0.01) # CPU'yu boğmamak için ufak gecikme

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
        """Returns compass heading in degrees."""
        msg = self.msg_dict.get('GLOBAL_POSITION_INT')
        if msg:
            return msg.hdg / 100.0 # cdeg to degrees
        
        # Fallback
        msg_vfr = self.msg_dict.get('VFR_HUD')
        if msg_vfr:
            return float(msg_vfr.heading)
        return 0.0

    def get_servo_pwm(self, channel):
        """Returns the last commanded PWM for a given channel."""
        # Seri portu meşgul etmemek için doğrudan önbellekteki (cache) PWM'i döndürüyoruz
        return self.pwms.get(channel, 1500)

    def set_servo(self, channel, pwm):
        """Commands the hardware to set a specific PWM on a servo channel."""
        self.pwms[channel] = pwm
        if self.master:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                channel,
                pwm,
                0, 0, 0, 0, 0
            )

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
