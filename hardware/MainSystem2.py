from pymavlink import mavutil
import math
import time
import threading

class USVController:
    def __init__(self, connection_string="/dev/ttyACM0", baud=57600):
        print("[HARDWARE] OrangeCube'a bağlanılıyor...")
        self.master = mavutil.mavlink_connection(connection_string, baud=baud)
        print("[HARDWARE] MAVLINK Bağlandı. Heartbeat bekleniyor...")
        self.master.wait_heartbeat()
        print("[HARDWARE] Heartbeat alındı!")

        self.msg_dict = {}  # Gelen MAVLink mesajlarının son hallerini tutar
        self.running = True

        self._request_data_streams()

        # Mesajları asenkron okuyan arka plan thread'i
        self.listener_thread = threading.Thread(target=self._listen_messages)
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def _request_data_streams(self):
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            5,  # 5 Hz okuma hızı
            1   # Başlat
        )
        print("[HARDWARE] Veri akışı (Data stream) istendi (5Hz).")

    def _listen_messages(self):
        while self.running:
            msg = self.master.recv_match(blocking=False)
            if msg:
                self.msg_dict[msg.get_type()] = msg
            time.sleep(0.01)

    def stop_listener(self):
        self.running = False
        if self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)

    def get_current_position(self):
        msg = self.msg_dict.get('GPS_RAW_INT')
        if msg:
            return msg.lat / 1e7, msg.lon / 1e7
        return None, None

    def get_gps_fix_type(self):
        msg = self.msg_dict.get('GPS_RAW_INT')
        if msg:
            return msg.fix_type
        return None

    def arm_vehicle(self):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        self.master.motors_armed_wait()
        print('[HARDWARE] Armed!')

    def disarm_vehicle(self):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        self.master.motors_disarmed_wait()
        print('[HARDWARE] Disarmed!')

    def get_horizontal_speed(self):
        msg = self.msg_dict.get('LOCAL_POSITION_NED')
        if msg:
            return math.sqrt(msg.vx ** 2 + msg.vy ** 2)
        return None

    def set_servo(self, servo_pin, pwm_value):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, servo_pin, pwm_value, 0, 0, 0, 0, 0
        )

    def set_mode(self, mode_name):
        mode_id = self.master.mode_mapping()[mode_name]
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        print(f"[HARDWARE] Mod değiştirildi: {mode_name}")