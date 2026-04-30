import time
import sys
import os

import cv2
import numpy as np
import math
import datetime
try:
    from ultralytics import YOLO
    import supervision as sv
    import torch
except ImportError:
    YOLO = None
    sv = None
    torch = None
    print("[CAMERA_PROCESS] YOLO/Supervision not found, running in mock inference mode.")

import config as cfg
from utils import utilities as utils
from utils.navigasyon import haversine
try:
    import pyzed.sl as sl
except ImportError:
    sl = None
    print("[CAMERA_PROCESS] ZED SDK not found, running in mock mode.")

# Ana dizinden utils klasörüne erişim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import kamera
from utils.utilities import KalmanFilter
from utils.kamera import TimestampHandler

try:
    import pyzed.sl as sl
except ImportError:
    sl = None



def camera_worker(shared_state):
    """
    Kamera ve Sensör (Göz/Denge) Prosesi.
    ZED2i kamerasını başlatır, IMU/Manyetometre verilerini okur,
    Kalman filtresinden geçirir ve paylaşımlı belleğe yazar.
    """
    print("[CAMERA_PROCESS] Başlatılıyor...")

    # 1. ZED KAMERA VE TAKİP SİSTEMİNİ BAŞLAT
    if sl is not None:
        try:
            zed = kamera.initialize_camera()

            # Orijinal kodundaki konumsal takip başlatma fonksiyonunu çağırıyoruz
            kamera.initialize_positional_tracking(zed)
            print("[CAMERA_PROCESS] ZED Kamera ve Konumsal Takip (Positional Tracking) başlatıldı!")

        except Exception as e:
            print(f"[CAMERA_PROCESS] KRİTİK HATA! ZED Kamera açılamadı: {e}")
            shared_state['shutdown'] = True
            return
    else:
        print("[CAMERA_PROCESS] MOCK MODE: Skipping ZED Camera initialization.")

    # 2. YARDIMCI SINIFLAR (Filtreler ve Zamanlayıcılar)
    if sl is not None:
        ts_handler = TimestampHandler()
        sensors_data = sl.SensorsData()
        image = sl.Mat()
        depth = sl.Mat()

    # Orijinal kodundaki Kalman Filtresi nesnesi (process ve measurement varyansları ile)
    magnetic_filter = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)

    # YOLO Model initialization
    model_path = "/home/yarkin/roboboatIDA/roboboat/weights/TNA.engine"
    if YOLO is not None:
        if not os.path.exists(model_path):
            print(f"[UYARI] {model_path} bulunamadı, YOLOv8n (default) yükleniyor.")
            model = YOLO("yolov8n.pt")  # Use yolov8n as fallback since it downloads automatically
        else:
            model = YOLO(model_path)

        try:
            model.to('cuda').half()
            print("[INFO] Model FP16 modunda yüklendi.")
        except Exception as e:
            print(f"Model GPU hatası: {e}")

        bounding_box_annotator = sv.RoundBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
    else:
        model = None

    print("[CAMERA_PROCESS] Sensör okuma döngüsü başlıyor...")

    # 3. SENSÖR OKUMA DÖNGÜSÜ
    # ZED IMU verileri çok yüksek hızlarda (örneğin 400Hz) akabilir.
    while not shared_state['shutdown']:
        start_time = time.time()

        if sl is not None:
            # Sensör verilerini anlık (CURRENT) olarak çek
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:

                # Veri yeniyse işle (TimestampHandler ile kontrol)
                if ts_handler.is_new(sensors_data.get_imu_data()):
                    # ZED'den ham manyetik pusula verisini al
                    raw_heading = sensors_data.get_magnetometer_data().magnetic_heading

                    # Kalman filtresinden geçirerek yumuşat
                    filtered_heading = magnetic_filter.update(raw_heading)

                    # Manyetik sapma vb. için ufak düzeltme (Orijinal kodundaki +6 derece vb. ayarlar)
                    final_heading = (filtered_heading - 6) % 360

                    # Paylaşımlı belleğe anında yaz (nav_process buradan okuyacak)
                    shared_state['magnetic_heading'] = final_heading

            # YENİ: GÖRÜNTÜ ALMA VE YOLO INFERENCE
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)

                camera_info = zed.get_camera_information()
                width = camera_info.camera_configuration.resolution.width
                height = camera_info.camera_configuration.resolution.height
                hfov_rad = math.radians(getattr(cfg, 'CAM_HFOV', 110.0))

                current_robot_lat = shared_state.get('gps_lat', 0.0)
                current_robot_lon = shared_state.get('gps_lon', 0.0)
                current_yaw = shared_state.get('magnetic_heading', 0.0)

                detected_objects_payload = []
                virtual_obstacles = [] # Sanal harita engelleri A* için

                if model is not None:
                    conf_val = getattr(cfg, 'YOLO_CONFIDENCE', 0.50)
                    results = model(frame, conf=conf_val, imgsz=1024, verbose=False)[0]
                    detections = sv.Detections.from_ultralytics(results)

                    if detections:
                        coords = detections.xyxy.tolist()
                        cids = detections.class_id.tolist()

                        for i, cid in enumerate(cids):
                            x1, y1, x2, y2 = map(int, coords[i])
                            cx = int((x1 + x2) / 2)
                            box_h = y2 - y1
                            target_cy = int(y2 - (box_h * 0.15))
                            cy = max(0, min(target_cy, height - 1))

                            err, dist_m = depth.get_value(cx, cy)

                            if not np.isnan(dist_m) and not np.isinf(dist_m) and 0.5 < dist_m < 15.0:
                                pixel_offset = (cx - (width / 2)) / width
                                angle_offset_rad = -pixel_offset * hfov_rad
                                obj_global_angle = math.radians(current_yaw) + angle_offset_rad

                                # Sanal engel koordinatlarını (offset) hesaplayıp NavProcess'e ilet (Local Frame)
                                obj_dx = dist_m * math.cos(obj_global_angle)
                                obj_dy = dist_m * math.sin(obj_global_angle)
                                virtual_obstacles.append((obj_dx, obj_dy))

                                # Juri Payload'ı için Global Obje Konumu
                                angle_offset_deg = math.degrees(angle_offset_rad)
                                obj_bearing = (current_yaw + angle_offset_deg) % 360

                                detected_objects_payload.append({
                                    "class_id": cid,
                                    "dist_m": dist_m,
                                    "bearing": obj_bearing,
                                    "dx": obj_dx,
                                    "dy": obj_dy
                                })

                # Update shared state with meaningful objects
                shared_state['camera_objects'] = detected_objects_payload
                shared_state['camera_virtual_obstacles'] = virtual_obstacles
        else:
            shared_state['magnetic_heading'] = 0.0

        # İşlemciyi (Orin'i) %100 kullanmamak için çok ufak bir uyku
        # (Sensör polleme hızını ~100Hz civarında tutmak idealdir)
        elapsed = time.time() - start_time
        sleep_time = max(0.001, 0.01 - elapsed)
        time.sleep(sleep_time)

    # 4. GÜVENLİ KAPANIŞ
    print("[CAMERA_PROCESS] Kapanış sinyali alındı. ZED SDK güvenlice kapatılıyor...")
    if sl is not None:
        try:
            kamera.temiz_kapat(zed)
        except Exception as e:
            print(f"[CAMERA_PROCESS] ZED Kapatma Hatası: {e}")

    print("[CAMERA_PROCESS] Kapandı.")