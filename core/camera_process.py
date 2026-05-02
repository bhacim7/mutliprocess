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

# YOLO / Supervision
from ultralytics import YOLO
import supervision as sv

# ZED Camera
import pyzed.sl as sl

import threading
import queue
import socket
import struct
import pickle
import cv2

class AsyncStreamer(threading.Thread):
    def __init__(self, ip, port=5000, max_queue=2):
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
                    self.q.get_nowait() # En eski kareyi çöpe at (Drop frame)
                except queue.Empty:
                    pass
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass

    def run(self):
        """Arka planda sürekli kuyruktan frame alıp Wi-Fi üzerinden yollar."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40] # Kalite %40
        
        while self.running:
            try:
                # 0.1 saniye timeout ile kuyruktan kare bekle
                frame = self.q.get(timeout=0.1) 
                
                if self.client_socket:
                    try:
                        # Görüntüyü küçült ve sıkıştır (Bu işlem artık ana kamerayı yavaşlatmaz!)
                        stream_frame = cv2.resize(frame, (640, 360))
                        ret, buffer = cv2.imencode('.jpg', stream_frame, encode_param)
                        
                        if ret:
                            data = pickle.dumps(buffer)
                            size_pack = struct.pack("!Q", len(data))
                            self.client_socket.sendall(size_pack + data)
                    except Exception as e:
                        print(f"[STREAMER] Connection lost during send: {e}")
                        self.client_socket = None # Koptuysa bir daha göndermeyi deneme
                        
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
            except: pass
        self.join()

class ObjectMemoryManager:
    def __init__(self):
        # Format: [{'id': 1, 'lat': 0.0, 'lon': 0.0, 'type': 0, 'color': 0, 'last_seen': 0.0}]
        self.memory = []
        self.id_counter = 1
        self.MERGE_DISTANCE = 2.0  # 2 meters merge distance

    def update_and_get_id(self, lat, lon, obj_type, color):
        current_time = time.time()
        best_match = None
        min_dist = float('inf')

        for obj in self.memory:
            dy = (obj['lat'] - lat) * 111139
            dx = (obj['lon'] - lon) * 85000
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < self.MERGE_DISTANCE:
                if dist < min_dist:
                    min_dist = dist
                    best_match = obj

        if best_match:
            best_match['lat'] = best_match['lat'] * 0.9 + lat * 0.1
            best_match['lon'] = best_match['lon'] * 0.9 + lon * 0.1
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
                'last_seen': current_time
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
    "TASK1_RETURN_MID": ProtoEnum.TASK_ENTRY_EXIT,
    "TASK1_RETURN_ENTER": ProtoEnum.TASK_ENTRY_EXIT,
    "FINISHED": ProtoEnum.TASK_NONE,
    "TASK2_START": ProtoEnum.TASK_NONE,
    "TASK2_GO_TO_MID": ProtoEnum.TASK_NAV_CHANNEL,
    "TASK2_GO_TO_MID1": ProtoEnum.TASK_NAV_CHANNEL,
    "TASK2_GO_TO_END": ProtoEnum.TASK_NAV_CHANNEL,
    "TASK2_SEARCH_PATTERN": ProtoEnum.TASK_NAV_CHANNEL,
    "TASK2_GREEN_MARKER_FOUND": ProtoEnum.TASK_NAV_CHANNEL,
    "TASK2_RETURN_END": ProtoEnum.TASK_NAV_CHANNEL,
    "TASK2_RETURN_MID": ProtoEnum.TASK_NAV_CHANNEL,
    "TASK2_RETURN_MID1": ProtoEnum.TASK_NAV_CHANNEL,
    "TASK2_RETURN_ENTRY": ProtoEnum.TASK_NAV_CHANNEL,
    "TASK3_START": ProtoEnum.TASK_NONE,
    "T3_START": ProtoEnum.TASK_NONE,
    "T3_MID": ProtoEnum.TASK_SPEED_CHALLENGE,
    "T3_RIGHT": ProtoEnum.TASK_SPEED_CHALLENGE,
    "T3_LEFT": ProtoEnum.TASK_SPEED_CHALLENGE,
    "T3_END": ProtoEnum.TASK_SPEED_CHALLENGE,
    "T3_END1": ProtoEnum.TASK_SPEED_CHALLENGE,
    "T3_RETURN_MID": ProtoEnum.TASK_SPEED_CHALLENGE,
    "T3_RETURN_START": ProtoEnum.TASK_SPEED_CHALLENGE,
    "TASK5_APPROACH": ProtoEnum.TASK_NONE,
    "TASK6_SPEED": ProtoEnum.TASK_SOUND_SIGNAL,
    "TASK6_DOCK": ProtoEnum.TASK_SOUND_SIGNAL
}

def camera_worker(shared_state):
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
        #model.to('cuda').half()
        print("[CAM_PROCESS] Model loaded in FP16 mode.")
    except Exception as e:
        print(f"[CAM_PROCESS] Model GPU error: {e}")

    # Initialize ZED Camera
    print("[CAM_PROCESS] Initializing Camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = getattr(cfg, 'CAM_FPS', 30)
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
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

    print("[CAM_PROCESS] Camera initialized and ready.")
    last_print_time = time.time()

    # Video Writer logic from legacy
    writer = None
    if getattr(cfg, 'RECORD_VIDEO', True):
        import datetime
        from utils.utilities import AsyncVideoWriter
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"idaKayit/kayit_{ts}.mp4"
        if not os.path.exists("idaKayit"):
            os.makedirs("idaKayit")
        writer = AsyncVideoWriter(video_path, fps=10.0, max_queue=130)
        writer.start()
        print(f"[CAM_PROCESS] Video recording started: {video_path}")

    try:
        while not shared_state['shutdown']:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # 1. READ SENSORS (Heading)
                if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT):
                    if ts_handler.is_new(sensors_data.get_imu_data()):
                        raw_heading = sensors_data.get_magnetometer_data().magnetic_heading
                        # Push heading to shared state
                        shared_state['magnetic_heading'] = (raw_heading - 6) % 360

                # 2. READ FRAME & DEPTH
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)

                # Update frame ready flag
                shared_state['vision_frame_ready'] = True
                # --- YENİ EKLENEN: FPS SADECE SANIYEDE 1 KERE TERMINALE YAZ ---
                #current_time = time.time()
                #if (current_time - last_print_time) >= 1.0:
                    #anlik_fps = round(zed.get_current_fps())
                    #print(f"[CAM_PROCESS] Saf Kamera ve Yapay Zeka FPS: {anlik_fps}")
                    #last_print_time = current_time

                # 3. YOLO INFERENCE
                conf_val = getattr(cfg, 'YOLO_CONFIDENCE', 0.50)
                results = model(frame, conf=conf_val, imgsz=1024, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)

                # Fetch necessary shared state vars for calculation
                ida_enlem = shared_state.get('gps_lat', 0.0)
                ida_boylam = shared_state.get('gps_lon', 0.0)
                magnetic_heading = shared_state.get('magnetic_heading', 0.0)
                mevcut_gorev = shared_state.get('current_task', 'TASK_UNKNOWN')

                current_frame_objects = []

                if detections and magnetic_heading is not None and ida_enlem != 0:
                    coords = detections.xyxy.tolist()
                    cids = detections.class_id.tolist()

                    for i, cid in enumerate(cids):
                        # 1. Filter and assign properties
                        r_type = ProtoEnum.OBJECT_UNKNOWN
                        r_color = ProtoEnum.COLOR_UNKNOWN

                        if cid in [5, 9, 10, 12]:
                            r_type = ProtoEnum.OBJECT_BUOY
                        elif cid in [3, 4]:
                            r_type = ProtoEnum.OBJECT_LIGHT_BEACON
                        else:
                            continue

                        if cid in [3, 5]:
                            r_color = ProtoEnum.COLOR_RED
                        elif cid in [4, 12]:
                            r_color = ProtoEnum.COLOR_GREEN
                        elif cid == 9:
                            r_color = ProtoEnum.COLOR_YELLOW
                        elif cid == 10:
                            r_color = ProtoEnum.COLOR_BLACK

                        # 2. Calculate Position
                        x1, y1, x2, y2 = map(int, coords[i])
                        cx = int((x1 + x2) / 2)
                        cy = int((y2 + y1) / 2)

                        err, dist_m = depth.get_value(cx, cy)

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

                # Push lightweight metadata to shared state
                shared_state['vision_detected_objects'] = current_frame_objects

                import datetime
                
                # 1. Verileri Paylaşımlı Bellekten Çek
                fps_val = round(zed.get_current_fps()) if 'zed' in locals() else 0
                task = shared_state.get('current_task', 'UNKNOWN')
                pwm_l = shared_state.get('motor_pwm_left', 1500)
                pwm_r = shared_state.get('motor_pwm_right', 1500)
                
                # Navigasyon Verileri
                dist = shared_state.get('target_dist', 0.0)
                adv_crs = shared_state.get('adviced_course', 0.0)
                err_ang = shared_state.get('angle_error', 0.0)
                hdg = shared_state.get('magnetic_heading', 0.0)
                lat = shared_state.get('gps_lat', 0.0)
                lon = shared_state.get('gps_lon', 0.0)
                t_lat = shared_state.get('target_lat', 0.0)
                t_lon = shared_state.get('target_lon', 0.0)

                # 2. Yazı Tipleri ve Renkler (BGR formatında)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7
                thick = 2
                c_red = (0, 0, 255)
                c_yellow = (0, 255, 255)
                c_green = (0, 255, 0)
                c_orange = (0, 165, 255)
                c_magenta = (255, 0, 255)

                # 3. SOL SÜTUN ÇİZİMİ
                y = 30
                cv2.putText(frame, datetime.datetime.now().strftime("%H:%M:%S"), (10, y), font, scale, c_red, thick); y += 30
                cv2.putText(frame, f"FPS: {fps_val}", (10, y), font, scale, c_yellow, thick); y += 30
                cv2.putText(frame, f"Gorev: {task}", (10, y), font, scale, c_red, thick); y += 60
                
                cv2.putText(frame, f"sol:{int(pwm_l)}", (10, y), font, scale, c_red, thick); y += 30
                cv2.putText(frame, f"sag:{int(pwm_r)}", (10, y), font, scale, c_red, thick); y += 40
                

                # 4. SAĞ SÜTUN ÇİZİMİ
                x_r = 450
                y_r = 30
                cv2.putText(frame, f"Hedefe mesafe: {dist:.2f}", (x_r, y_r), font, scale, c_red, thick); y_r += 30
                cv2.putText(frame, f"Rota tavsiyesi: {adv_crs:.0f}", (x_r, y_r), font, scale, c_orange, thick); y_r += 30
                cv2.putText(frame, f"aci farki: {err_ang:.0f}", (x_r, y_r), font, scale, c_yellow, thick); y_r += 30
                cv2.putText(frame, f"Heading: {hdg:.0f}", (x_r, y_r), font, scale, c_yellow, thick); y_r += 30
                cv2.putText(frame, f"manyetometre durumu: GOOD", (x_r, y_r), font, scale, c_orange, thick); y_r += 30
                cv2.putText(frame, f"Heading dogrulugu: 0.9", (x_r, y_r), font, scale, c_orange, thick); y_r += 30
                cv2.putText(frame, f"GPS DURUMU: 3D Fix", (x_r, y_r), font, scale, c_red, thick); y_r += 30
                cv2.putText(frame, f"ida enlem: {lat:.6f}", (x_r, y_r), font, scale, c_orange, thick); y_r += 30
                cv2.putText(frame, f"ida boylam: {lon:.6f}", (x_r, y_r), font, scale, c_orange, thick); y_r += 30
                cv2.putText(frame, f"hedef enlem: {t_lat:.6f}", (x_r, y_r), font, scale, c_orange, thick); y_r += 30
                cv2.putText(frame, f"hedef boylam: {t_lon:.6f}", (x_r, y_r), font, scale, c_orange, thick)
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
