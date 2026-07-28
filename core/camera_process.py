import time
import math
import numpy as np
import cv2

import socket
import struct
import pickle

# Hardware/Utils
import config as cfg
import utils.kamera as kamera
from utils.kamera import TimestampHandler
from utils.navigasyon import calculate_obj_gps
import utils.navigasyon as nav
from utils.headingFilter import KalmanFilter

# YOLO / Supervision
from ultralytics import YOLO
import supervision as sv

# ZED Camera
import pyzed.sl as sl

import threading
import queue
import socket
import struct
import cv2


class AsyncStreamer(threading.Thread):
    def __init__(self, ip, port=5000, max_queue=5):
        super().__init__()
        self.ip = ip
        self.port = port
        self.q = queue.Queue(maxsize=max_queue)
        self.client_socket = None
        self.running = True

        # Soket bağlantısını başlat
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Zaman aşımı ekleyelim ki bağlantı koptuğunda sistemi kitlemesin
            self.client_socket.settimeout(2.0)
            self.client_socket.connect((self.ip, self.port))
            # Bağlandıktan sonra timeout'u kaldır ki büyük verilerde kopmasın
            self.client_socket.settimeout(None)
            print(f"[STREAMER] Connected to {self.ip}:{self.port}")
        except Exception as e:
            print(f"[STREAMER][WARNING] Could not connect: {e}")
            self.client_socket = None

    def enqueue(self, frame):
        """Kameradan gelen kareyi kuyruğa atar. Kuyruk doluysa eski kareyi çöpe atar (Canlı yayın mantığı)."""
        if self.running and self.client_socket:
            if self.q.full():
                try:
                    self.q.get_nowait()  # En eski kareyi çöpe at (Drop frame)
                except queue.Empty:
                    pass
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass

    def run(self):
        """Arka planda sürekli kuyruktan frame alıp Wi-Fi üzerinden yollar."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]  # Kalite %40

        while self.running:
            try:
                # 0.1 saniye timeout ile kuyruktan kare bekle
                frame = self.q.get(timeout=0.1)

                if self.client_socket:
                    try:
                        # Görüntüyü küçült ve sıkıştır (Bu işlem artık ana kamerayı yavaşlatmaz!)
                        stream_frame = cv2.resize(frame, (860, 540))
                        ret, buffer = cv2.imencode('.jpg', stream_frame, encode_param)

                        if ret:
                            data = buffer.tobytes()
                            size_pack = struct.pack("!Q", len(data))
                            self.client_socket.sendall(size_pack + data)
                    except Exception as e:
                        print(f"[STREAMER] Connection lost during send: {e}")
                        self.client_socket = None  # Koptuysa bir daha göndermeyi deneme

                self.q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                pass

    def stop(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        self.join()


class ObjectMemoryManager:
    def __init__(self):
        # Format: [{'id': 1, 'lat': 0.0, 'lon': 0.0, 'type': 0, 'color': 0, 'last_seen': 0.0, 'v_lat': 0.0, 'v_lon': 0.0}]
        self.memory = []
        self.id_counter = 1
        self.MERGE_DISTANCE = 2.5  # 2.5 meters merge distance

    def update_and_get_id(self, lat, lon, obj_type, color):
        current_time = time.time()
        best_match = None
        min_dist = float('inf')

        # Clean up old objects (not seen in 5 seconds)
        self.memory = [obj for obj in self.memory if (current_time - obj['last_seen']) < 5.0]

        for obj in self.memory:
            # Predict current position based on velocity
            dt = current_time - obj['last_seen']
            pred_lat = obj['lat'] + (obj['v_lat'] * dt)
            pred_lon = obj['lon'] + (obj['v_lon'] * dt)

            # Calculate distance to predicted position
            dy = (pred_lat - lat) * 111139
            dx = (pred_lon - lon) * 85000
            dist = math.sqrt(dx * dx + dy * dy)

            # Also check color and type for stricter matching
            if dist < self.MERGE_DISTANCE and obj['type'] == obj_type and obj['color'] == color:
                if dist < min_dist:
                    min_dist = dist
                    best_match = obj

        if best_match:
            # Update velocity
            dt = current_time - best_match['last_seen']
            if dt > 0:
                best_match['v_lat'] = best_match['v_lat'] * 0.8 + ((lat - best_match['lat']) / dt) * 0.2
                best_match['v_lon'] = best_match['v_lon'] * 0.8 + ((lon - best_match['lon']) / dt) * 0.2

            # Smooth position (Alpha filter)
            best_match['lat'] = best_match['lat'] * 0.8 + lat * 0.2
            best_match['lon'] = best_match['lon'] * 0.8 + lon * 0.2
            best_match['last_seen'] = current_time
            return best_match['id'], best_match['lat'], best_match['lon']
        else:
            new_id = self.id_counter
            self.id_counter += 1
            self.memory.append({
                'id': new_id,
                'lat': lat,
                'lon': lon,
                'type': obj_type,
                'color': color,
                'last_seen': current_time,
                'v_lat': 0.0,
                'v_lon': 0.0
            })
            return new_id, lat, lon


class ProtoEnum:
    OBJECT_UNKNOWN = 0
    OBJECT_BOAT = 1
    OBJECT_LIGHT_BEACON = 2
    OBJECT_BUOY = 3
    COLOR_UNKNOWN = 0
    COLOR_YELLOW = 1
    COLOR_BLACK = 2
    COLOR_RED = 3
    COLOR_GREEN = 4
    COLOR_ORANGE = 5
    TASK_UNKNOWN = 0
    TASK_NONE = 1
    TASK_ENTRY_EXIT = 2
    TASK_NAV_CHANNEL = 3
    TASK_SPEED_CHALLENGE = 4
    TASK_OBJECT_DELIVERY = 5
    TASK_DOCKING = 6
    TASK_SOUND_SIGNAL = 7


TASK_CONTEXT_MAP = {
    "TASK1_STATE_ENTER": ProtoEnum.TASK_NONE,
    "TASK1_STATE_MID": ProtoEnum.TASK_ENTRY_EXIT,
    "TASK1_STATE_EXIT": ProtoEnum.TASK_ENTRY_EXIT,
    "FINISHED": ProtoEnum.TASK_NONE,
    "TASK2_START": ProtoEnum.TASK_NONE,
    "TASK2_GO_TO_MID": ProtoEnum.TASK_NAV_CHANNEL,
    "TASK2_GO_TO_END": ProtoEnum.TASK_NAV_CHANNEL,
    "T3_START": ProtoEnum.TASK_NONE,
    "T3_MID": ProtoEnum.TASK_SPEED_CHALLENGE,
    "TASK3_SEARCH_KAMIKAZE": ProtoEnum.TASK_SPEED_CHALLENGE,
    "GPS1": ProtoEnum.TASK_NONE,
    "GPS2": ProtoEnum.TASK_ENTRY_EXIT,
    "GPS3": ProtoEnum.TASK_ENTRY_EXIT,
    "GPS4": ProtoEnum.TASK_NONE
}


def camera_worker(shared_state, hf_data):
    """
    Independent process handling ZED Camera operations, YOLO Inference,
    and Object tracking. Updates the shared_state dictionary with lightweight metadata.
    """
    print("[CAM_PROCESS] Starting Camera Worker...")

    streamer = None
    if getattr(cfg, 'STREAM', False):
        gcs_ip = getattr(cfg, 'GCS_IP', "192.168.1.25")
        streamer = AsyncStreamer(gcs_ip, 5000)
        streamer.start()

    # Initialize Object Manager
    obj_manager = ObjectMemoryManager()

    # Load YOLO Model
    model_path = getattr(cfg, 'MODEL_PATH', "/home/yarkin/roboboatIDA/roboboat/weights/TNA.engine")
    import os
    if not os.path.exists(model_path):
        print(f"[CAM_PROCESS][WARNING] {model_path} not found, using yolov11n.pt")
        model = YOLO("yolov11n.pt")
    else:
        model = YOLO(model_path)

    try:
        # model.to('cuda').half()
        print("[CAM_PROCESS] Model loaded in FP16 mode.")
    except Exception as e:
        print(f"[CAM_PROCESS] Model GPU error: {e}")

    # Initialize ZED Camera
    print("[CAM_PROCESS] Initializing Camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = getattr(cfg, 'CAM_FPS', 30)
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
    init_params.coordinate_units = sl.UNIT.METER

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[CAM_PROCESS][ERROR] Failed to open ZED Camera: {err}")
        return

    # Enable Positional Tracking
    tracking_parameters = sl.PositionalTrackingParameters()
    tracking_parameters.enable_imu_fusion = True
    tracking_parameters.set_floor_as_origin = False
    tracking_parameters.enable_area_memory = False
    tracking_parameters.enable_pose_smoothing = True

    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[CAM_PROCESS][ERROR] Positional tracking failed: {err}")

    camera_info = zed.get_camera_information()
    width = camera_info.camera_configuration.resolution.width
    height = camera_info.camera_configuration.resolution.height
    hfov_rad = math.radians(getattr(cfg, 'CAM_HFOV', 110.0))

    image = sl.Mat()
    depth = sl.Mat()
    sensors_data = sl.SensorsData()
    ts_handler = TimestampHandler()
    zed_pose = sl.Pose()

    magnetic_filter = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)

    print("[CAM_PROCESS] Camera initialized and ready.")
    last_printed_task = None

    # Video Writer logic from legacy
    writer = None
    if getattr(cfg, 'RECORD_VIDEO', True):
        import datetime
        from utils.utilities import AsyncVideoWriter
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"kayit_{ts}.mp4"
        fps_val = float(getattr(cfg, 'CAM_FPS', 30.0))
        writer = AsyncVideoWriter(video_path, fps=fps_val, max_queue=130)
        writer.start()
        print(f"[CAM_PROCESS] Video recording started: {video_path} at {fps_val} FPS")

    try:
        while not shared_state['shutdown']:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # 1. READ SENSORS (Heading)
                if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT):
                    if ts_handler.is_new(sensors_data.get_imu_data()):
                        raw_heading = sensors_data.get_magnetometer_data().magnetic_heading
                        zed_heading = magnetic_filter.update(raw_heading)
                        # Magnetic declination for Istanbul is approx +6 degrees.
                        zed_heading = (zed_heading + 6) % 360

                        # Push raw zed heading to shared state.
                        # nav_process will determine final magnetic_heading
                        shared_state['zed_heading'] = zed_heading

                # 1.5 READ POSE (Pitch/Roll Stability & Odometry)
                state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
                if state == sl.POSITIONAL_TRACKING_STATE.OK:
                    t_vec = zed_pose.get_translation().get()
                    # Axis Swap for Costmap
                    shared_state['zed_x'] = float(t_vec[1])
                    shared_state['zed_y'] = float(-t_vec[0])

                    ox = zed_pose.get_orientation().get()[0]
                    oy = zed_pose.get_orientation().get()[1]
                    oz = zed_pose.get_orientation().get()[2]
                    ow = zed_pose.get_orientation().get()[3]

                    # 1. Roll
                    sinr_cosp = 2 * (ow * ox + oy * oz)
                    cosr_cosp = 1 - 2 * (ox * ox + oy * oy)
                    robot_roll = math.atan2(sinr_cosp, cosr_cosp)
                    robot_roll_deg = math.degrees(robot_roll)

                    # 2. Pitch
                    sinp = 2 * (ow * oy - oz * ox)
                    if abs(sinp) >= 1:
                        robot_pitch = math.copysign(math.pi / 2, sinp)
                    else:
                        robot_pitch = math.asin(sinp)
                    robot_pitch_deg = math.degrees(robot_pitch)

                    # 3. Stability Check
                    max_tilt = getattr(cfg, 'MAX_TILT_ANGLE', 5.0)
                    if abs(robot_roll_deg) > max_tilt or abs(robot_pitch_deg) > max_tilt:
                        shared_state['lidar_wave_stable'] = False
                    else:
                        shared_state['lidar_wave_stable'] = True

                # 2. READ FRAME & DEPTH
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)

                # Update frame ready flag
                shared_state['vision_frame_ready'] = True

                # --- THROTTLED TERMINAL HEARTBEAT (FPS & STATUS) ---
                mevcut_gorev = shared_state.get('current_task', 'UNKNOWN')
                if mevcut_gorev != last_printed_task:
                    print(f"[CAM_PROCESS] Task State Changed to: {mevcut_gorev}")
                    last_printed_task = mevcut_gorev

                # 3. YOLO INFERENCE
                conf_val = getattr(cfg, 'YOLO_CONFIDENCE', 0.50)
                results = model(frame, conf=conf_val, imgsz=[544, 960], verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)

                # Fetch necessary shared state vars for calculation
                ida_enlem = hf_data['gps_lat'].value
                ida_boylam = hf_data['gps_lon'].value
                magnetic_heading = hf_data['magnetic_heading'].value
                mevcut_gorev = shared_state.get('current_task', 'TASK_UNKNOWN')

                current_frame_objects = []

                if detections and magnetic_heading is not None and ida_enlem != 0:
                    coords = detections.xyxy.tolist()
                    cids = detections.class_id.tolist()

                    for i, cid in enumerate(cids):
                        # 1. Filter and assign properties
                        r_type = ProtoEnum.OBJECT_UNKNOWN
                        r_color = ProtoEnum.COLOR_UNKNOWN

                        if cid in [0, 1, 2, 3, 4]:
                            r_type = ProtoEnum.OBJECT_BUOY
                        else:
                            continue

                        if cid == 0:
                            r_color = ProtoEnum.COLOR_RED
                        elif cid == 1:
                            r_color = ProtoEnum.COLOR_YELLOW
                        elif cid == 2:
                            r_color = ProtoEnum.COLOR_BLACK
                        elif cid == 3:
                            r_color = ProtoEnum.COLOR_ORANGE
                        elif cid == 4:
                            r_color = ProtoEnum.COLOR_GREEN

                        # 2. Calculate Position
                        x1, y1, x2, y2 = map(int, coords[i])
                        cx = int((x1 + x2) / 2)

                        # --- FIX: Measure depth at bottom 15% with ROI Median Filter ---
                        box_h = y2 - y1
                        target_cy = int(y2 - (box_h * 0.15))
                        cy = max(0, min(target_cy, height - 1))

                        # Extract a 5x5 ROI around the target pixel to filter out water reflection noise
                        roi_size = 2
                        depth_values = []
                        for dy in range(-roi_size, roi_size + 1):
                            for dx in range(-roi_size, roi_size + 1):
                                sample_x = max(0, min(cx + dx, width - 1))
                                sample_y = max(0, min(cy + dy, height - 1))
                                err, d_val = depth.get_value(sample_x, sample_y)
                                if not np.isnan(d_val) and not np.isinf(d_val) and d_val > 0.1:
                                    depth_values.append(d_val)

                        if depth_values:
                            # Use the median depth to reject extreme outliers (like NaN or Inf points)
                            dist_m = float(np.median(depth_values))
                        else:
                            dist_m = float('inf')  # Invalid depth

                        if not np.isnan(dist_m) and not np.isinf(dist_m) and 0.5 < dist_m < 15.0:
                            pixel_offset = (cx - (width / 2)) / width
                            angle_offset_rad = -pixel_offset * hfov_rad
                            angle_offset_deg = math.degrees(angle_offset_rad)
                            obj_bearing = (magnetic_heading + angle_offset_deg) % 360

                            obj_lat, obj_lon = calculate_obj_gps(ida_enlem, ida_boylam, dist_m, obj_bearing)

                            final_id, f_lat, f_lon = obj_manager.update_and_get_id(obj_lat, obj_lon, r_type, r_color)
                            t_ctx = TASK_CONTEXT_MAP.get(mevcut_gorev, ProtoEnum.TASK_NONE)

                            if t_ctx != ProtoEnum.TASK_NONE:
                                current_frame_objects.append({
                                    "type": r_type,
                                    "color": r_color,
                                    "lat": f_lat,
                                    "lon": f_lon,
                                    "dist": dist_m,
                                    "id": final_id,
                                    "ctx": t_ctx,
                                    "cid": cid,
                                    "cx": cx,
                                    "cy": cy,
                                    "area": (x2 - x1) * (y2 - y1)
                                })

                                # Draw Bounding Box and Label
                                color_map = {
                                    0: (0, 0, 255),  # Red
                                    1: (0, 255, 255),  # Yellow
                                    2: (0, 0, 0),  # Black
                                    3: (0, 165, 255),  # Orange
                                    4: (0, 255, 0)  # Green
                                }
                                label_map = {
                                    0: "Red",
                                    1: "Yellow",
                                    2: "Black",
                                    3: "Orange",
                                    4: "Green"
                                }

                                box_color = color_map.get(cid, (255, 255, 255))
                                label_text = f"{label_map.get(cid, 'Unknown')} {dist_m:.1f}m"

                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                cv2.putText(frame, label_text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                # Push lightweight metadata to shared state
                shared_state['vision_detected_objects'] = current_frame_objects
                shared_state['vision_detected_objects'] = current_frame_objects

                import datetime

                # 1. Verileri Paylaşımlı Bellekten Çek
                fps_val = round(zed.get_current_fps()) if 'zed' in locals() else 0
                task = shared_state.get('current_task', 'UNKNOWN')
                pwm_l = shared_state.get('motor_pwm_left', 1500)
                pwm_r = shared_state.get('motor_pwm_right', 1500)
                pwm_fl = shared_state.get('motor_pwm_front_left', 1500)
                pwm_fr = shared_state.get('motor_pwm_front_right', 1500)
                pwm_steer = shared_state.get('motor_pwm_steer', 1500)
                shared_state['camera_fps'] = fps_val

                # Navigasyon Verileri
                dist = shared_state.get('target_dist', 0.0)
                adv_crs = shared_state.get('adviced_course', 0.0)
                err_ang = shared_state.get('angle_error', 0.0)
                ctrl_err = shared_state.get('control_error', 0.0)
                hdg = hf_data['magnetic_heading'].value
                lat = hf_data['gps_lat'].value
                lon = hf_data['gps_lon'].value
                t_lat = shared_state.get('target_lat', 0.0)
                t_lon = shared_state.get('target_lon', 0.0)

                # 2. Yazı Tipleri ve Renkler (BGR formatında)
                font = cv2.FONT_HERSHEY_DUPLEX
                scale = 0.7
                thick = 1
                c_red = (0, 0, 255)
                c_yellow = (255, 255, 0)  # Cyan for better contrast
                c_green = (0, 255, 0)
                c_orange = (255, 255, 255)  # Pure White for better contrast
                c_magenta = (255, 0, 255)

                # 3. YAZI ÇİZİMİ (GÖLGELİ METİN - MAKSİMUM FPS)
                def draw_text(text, x, y, text_color):
                    # Draw black shadow slightly offset
                    #cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0), thick + 1)
                    # Draw main text
                    cv2.putText(frame, text, (x, y), font, scale, text_color, thick)

                # SOL SÜTUN ÇİZİMİ
                y = 30
                draw_text(f"ZAMAN: {datetime.datetime.now().strftime('%H:%M:%S')}", 10, y, c_red);
                y += 30
                draw_text(f"FPS: {fps_val}", 10, y, c_yellow);
                y += 30
                draw_text(f"MANUEL: {shared_state.get('manual_mode', False)}", 10, y, c_orange);
                y += 30
                draw_text(f"MEVCUT_GOREV: {task}", 10, y, c_red);
                y += 30

                y += 20
                draw_text(f"ARKA_SOL_PWM: {int(pwm_l)}", 10, y, c_red);
                y += 30
                draw_text(f"ARKA_SAG_PWM: {int(pwm_r)}", 10, y, c_red);
                y += 30
                draw_text(f"ON_SOL_PWM: {int(pwm_fl)}", 10, y, c_red);
                y += 30
                draw_text(f"ON_SAG_PWM: {int(pwm_fr)}", 10, y, c_red);
                y += 30
                draw_text(f"DUMEN_PWM: {int(pwm_steer)}", 10, y, c_red);
                y += 30

                # SAĞ SÜTUN ÇİZİMİ
                x_r = 900
                y_r = 30
                draw_text(f"HIZ: {shared_state.get('horizontal_speed', 0.0):.1f}", x_r, y_r, c_red);
                y_r += 30
                draw_text(f"HDG: {hdg:.0f}", x_r, y_r, c_yellow);
                y_r += 30
                draw_text(f"HEDEF_HDG: {adv_crs:.0f}", x_r, y_r, c_orange);
                y_r += 30
                draw_text(f"ACI_FARKI: {err_ang:.0f}", x_r, y_r, c_yellow);
                y_r += 30
                draw_text(f"KONTROL_HATA: {ctrl_err:.0f}", x_r, y_r, c_yellow);
                y_r += 30
                draw_text(f"HEDEFE_MESAFE: {dist:.1f}", x_r, y_r, c_red);
                y_r += 30

                y_r += 20
                draw_text(f"IDA_KONUM:", x_r, y_r, c_orange);
                y_r += 30
                draw_text(f" {lat:.6f}, {lon:.6f}", x_r, y_r, c_orange);
                y_r += 30
                draw_text(f"HEDEF_KONUM:", x_r, y_r, c_orange);
                y_r += 30
                draw_text(f" {t_lat:.6f}, {t_lon:.6f}", x_r, y_r, c_orange);
                y_r += 30
                draw_text(f"SENSOR_SAGLIK: GOOD", x_r, y_r, c_orange);
                y_r += 30
                # -----------------------------------------------

                # 4. VIDEO RECORDING
                if writer:
                    writer.enqueue(frame)

                if streamer:
                    streamer.enqueue(frame)

    except Exception as e:
        print(f"[CAM_PROCESS][ERROR] Loop crashed: {e}")
    finally:
        print("[CAM_PROCESS] Shutting down...")
        if writer:
            writer.stop()
        if 'streamer' in locals() and streamer is not None:
            streamer.stop()
        zed.close()
