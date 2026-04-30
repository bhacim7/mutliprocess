import time
import math
import sys

# Proje ana dizininden modülleri çekebilmek için (Eğer script kendi klasöründen çalıştırılırsa diye)
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
try:
    from hardware.MainSystem2 import USVController
except ImportError:
    pass

# Fallback for dev environments
try:
    import sys, os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from mock_hardware import MockUSVController
    USVController = MockUSVController
except ImportError:
    pass
from utils import navigasyon as nav
import numpy as np
import cv2
from utils import planner
from utils import utilities as utils

# Harita Ayarları (IDA1.py'dan taşındı)
COSTMAP_SIZE_PX = (800, 800)
COSTMAP_RES_M_PER_PX = 0.10
ROBOT_RADIUS_M = 0.25

import numpy as np
from utils import planner
from utils import utilities as utils



def nav_worker(shared_state, command_queue):
    """
    Otonom Navigasyon ve Beyin Prosesi.
    OrangeCube ile haberleşir, GPS okur, PID hesaplar ve motorlara PWM basar.
    """
    print("[NAV_PROCESS] Başlatılıyor...")

    # 1. ORANGECUBE (MAVLINK) BAĞLANTISI
    try:
        if 'MockUSVController' in globals() and USVController == MockUSVController:
             controller = USVController()
        else:
             controller = USVController(connection_string=cfg.ORANGE_PORT, baud=cfg.ORANGE_BAUD)
        print("[NAV_PROCESS] OrangeCube bağlantısı başarılı!")

        # Otonom sürüş için Pixhawk'ı MANUAL moda alıp motorları arm etmeliyiz
        # (Gerçek testlerde Arm komutunu manuel vermek isteyebilirsin, şimdilik otomatik yapıyoruz)
        controller.set_mode("MANUAL")
        print("[NAV_PROCESS] Mod MANUAL olarak ayarlandı.")
    except Exception as e:
        print(f"[NAV_PROCESS] KRİTİK HATA! OrangeCube'a bağlanılamadı: {e}")
        try:
             import sys, os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from mock_hardware import MockUSVController
             controller = MockUSVController()
             print("[NAV_PROCESS] Falling back to MOCK OrangeCube.")
        except Exception as fallback_e:
             print(f"[NAV_PROCESS] Fallback failed: {fallback_e}")
             shared_state['shutdown'] = True
             return

    # Görev geçiş haritası (Hangi görev bitince hangisine geçilecek?)
    NEXT_TASK_MAP = {
        'TASK_1': 'TASK_2',
        'TASK_2': 'TASK_3',
        'TASK_3': 'TASK_4',
        'TASK_4': 'TASK_5',
        'TASK_5': 'TASK_6',
        'TASK_6': 'FINISHED'
    }

    # Motorlara gidecek PWM değerlerini sınırlayan yardımcı fonksiyon
    def clamp_pwm(pwm_val):
        return max(cfg.MIN_PWM_LIMIT, min(cfg.MAX_PWM_LIMIT, int(pwm_val)))

    print("[NAV_PROCESS] Ana navigasyon döngüsü (Control Loop) başlıyor...")

    # 2. ANA KONTROL DÖNGÜSÜ (Control Loop)
    # Saniyede yaklaşık 20 defa (20Hz) dönecek şekilde ayarlanmıştır.
    while not shared_state['shutdown']:
        start_time = time.time()

        # A. ORANGECUBE'DAN GPS VERİSİNİ OKU VE PAYLAŞIMLI BELLEĞE YAZ
        lat, lon = controller.get_current_position()
        if lat is not None and lon is not None:
            shared_state['gps_lat'] = lat
            shared_state['gps_lon'] = lon
        else:
            # GPS verisi yoksa veya koptuysa motorları durdur ve bekle
            controller.set_servo(cfg.SOL_MOTOR, cfg.BASE_PWM)
            controller.set_servo(cfg.SAG_MOTOR, cfg.BASE_PWM)
            time.sleep(0.1)
            continue

        # B. PAYLAŞIMLI BELLEKTEN ZED PUSULASINI VE DURUMLARI OKU
        current_heading = shared_state['magnetic_heading']
        current_task = shared_state['current_task']
        manual_mode = shared_state['manual_mode']
        lidar_emergency = shared_state.get('lidar_emergency', False)
        acoustic_interrupt = shared_state.get('acoustic_interrupt', False)

        # Lidar'dan acil durum geldiyse dur!
        if lidar_emergency or acoustic_interrupt:
            if acoustic_interrupt:
                print("[NAV_PROCESS] 🚨 AKUSTİK KESME ALGILANDI! GÖREV BEKLETİLİYOR 🚨")
            controller.set_servo(cfg.SOL_MOTOR, cfg.BASE_PWM)
            controller.set_servo(cfg.SAG_MOTOR, cfg.BASE_PWM)
            shared_state['motor_pwm_left'] = cfg.BASE_PWM
            shared_state['motor_pwm_right'] = cfg.BASE_PWM
            time.sleep(0.1)
            continue

        # C. MANUEL MOD KONTROLÜ (Yer İstasyonu Devraldıysa)
        if manual_mode:
            # Motor PWM'leri telem_process tarafından shared_state'e yazılacaktır.
            # Biz sadece o değerleri OrangeCube'a iletiyoruz.
            controller.set_servo(cfg.SOL_MOTOR, shared_state['motor_pwm_left'])
            controller.set_servo(cfg.SAG_MOTOR, shared_state['motor_pwm_right'])
            time.sleep(0.05)
            continue

        # D. GÖREV BİTTİYSE DUR
        if current_task == 'FINISHED':
            controller.set_servo(cfg.SOL_MOTOR, cfg.BASE_PWM)
            controller.set_servo(cfg.SAG_MOTOR, cfg.BASE_PWM)
            shared_state['motor_pwm_left'] = cfg.BASE_PWM
            shared_state['motor_pwm_right'] = cfg.BASE_PWM
            time.sleep(0.5)
            continue

        # E. OTONOM SÜRÜŞ VE STATE MACHINE (Durum Makinesi)
        # Mevcut görevin hedef koordinatlarını al
        if current_task in cfg.TASK_WAYPOINTS:
            target_lat, target_lon = cfg.TASK_WAYPOINTS[current_task]
        else:
            target_lat, target_lon = lat, lon

        # 1. Mesafe Hesapla
        distance_m = nav.haversine(lat, lon, target_lat, target_lon)

        # 2. Hedefe Vardık Mı? (Transition)
        if distance_m < getattr(cfg, 'WAYPOINT_TOLERANCE_M', 2.0):
            print(f"[NAV_PROCESS] {current_task} Tamamlandı! (Hedefe {distance_m:.1f}m)")
            next_task = NEXT_TASK_MAP.get(current_task, 'FINISHED')
            shared_state['current_task'] = next_task
            continue

        # 3. Harita / Lidar verisini al (GERÇEK A* PLANLAMASI)
        lidar_map = shared_state.get('lidar_map', [])
        virtual_obs = shared_state.get('camera_virtual_obstacles', [])

        # YEREL COSTMAP OLUŞTURMA
        # 127 = Bilinmeyen, 255 = Boş, 0 = Engel
        nav_map = np.full((COSTMAP_SIZE_PX[1], COSTMAP_SIZE_PX[0]), 255, dtype=np.uint8)
        costmap_center_m = (0.0, 0.0) # Local Map: Robot is at (0,0) locally!

        # Lidar engellerini haritaya ekle (Basit Işınlama Yöntemi)
        if len(lidar_map) == 72:
            for idx, dist in enumerate(lidar_map):
                if dist < 10.0: # Engel varsa
                    angle_deg = idx * 5.0
                    global_angle = current_heading - angle_deg # Yön hesabı
                    # Dünyadaki ofset (metre)
                    obs_x_m = dist * math.cos(math.radians(global_angle))
                    obs_y_m = dist * math.sin(math.radians(global_angle))

                    # Piksele çevir
                    cw, ch = COSTMAP_SIZE_PX[0] // 2, COSTMAP_SIZE_PX[1] // 2
                    px = int(cw + (obs_x_m / COSTMAP_RES_M_PER_PX))
                    py = int(ch - (obs_y_m / COSTMAP_RES_M_PER_PX))

                    if 0 <= px < COSTMAP_SIZE_PX[0] and 0 <= py < COSTMAP_SIZE_PX[1]:
                        cv2.circle(nav_map, (px, py), 6, 0, -1)

        # Sanal Kamere Engellerini haritaya ekle
        for obs_dx, obs_dy in virtual_obs:
            cw, ch = COSTMAP_SIZE_PX[0] // 2, COSTMAP_SIZE_PX[1] // 2
            px = int(cw + (obs_dx / COSTMAP_RES_M_PER_PX))
            py = int(ch - (obs_dy / COSTMAP_RES_M_PER_PX))
            if 0 <= px < COSTMAP_SIZE_PX[0] and 0 <= py < COSTMAP_SIZE_PX[1]:
                cv2.circle(nav_map, (px, py), 6, 0, -1)

        # Şişirme (Inflation - Engel etrafını güvenli hale getirme)
        obstacles_mask = (nav_map < 100).astype(np.uint8) * 255
        inflation_m = getattr(cfg, 'INFLATION_MARGIN_M', 0.25)
        kernel_size = (int((ROBOT_RADIUS_M + inflation_m) / COSTMAP_RES_M_PER_PX) * 2) + 1
        if kernel_size % 2 == 0: kernel_size += 1
        inflated_obstacles = cv2.dilate(obstacles_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

        nav_map[:] = 255
        nav_map[inflated_obstacles > 0] = 0

        # 4. A* PATH PLANNING ÇAĞRISI (FIXED: Local Metric Coordinate Conversion)
        # We need to project the target GPS to local metric offsets relative to the robot.
        target_bearing = nav.calculate_bearing(lat, lon, target_lat, target_lon)
        target_dist = nav.haversine(lat, lon, target_lat, target_lon)

        # Limit the target projection to the map size
        max_map_dist = (COSTMAP_SIZE_PX[0] // 2) * COSTMAP_RES_M_PER_PX
        effective_target_dist = min(target_dist, max_map_dist - 0.5)

        tx_local = effective_target_dist * math.cos(math.radians(target_bearing))
        ty_local = effective_target_dist * math.sin(math.radians(target_bearing))

        # Line of sight kontrolü
        path_is_clear = planner.check_line_of_sight(
            (0.0, 0.0), (tx_local, ty_local), nav_map,
            costmap_center_m, COSTMAP_RES_M_PER_PX, COSTMAP_SIZE_PX
        )

        current_path = []
        if not path_is_clear:
            current_path = planner.get_path_plan(
                (0.0, 0.0), (tx_local, ty_local), nav_map,
                costmap_center_m, COSTMAP_RES_M_PER_PX, COSTMAP_SIZE_PX
            )
        else:
            current_path = [(0.0, 0.0), (tx_local, ty_local)]

        # 5. PURE PURSUIT KONTROLCÜSÜ İLE PWM HESABI
        if current_path:
            cur_spd = controller.get_horizontal_speed()
            if cur_spd is None: cur_spd = 0.0

            base_pwm = getattr(cfg, 'BASE_PWM', 1500)
            cruise_pwm = getattr(cfg, 'CRUISE_PWM', 100)

            # Follow path locally
            pp_sol, pp_sag, raw_target, current_error, pruned_path = planner.pure_pursuit_control(
                0.0, 0.0, math.radians(current_heading), current_path,
                current_speed=cur_spd,
                base_speed=cruise_pwm,
                prev_error=0.0
            )

            left_pwm = base_pwm + pp_sol - 1500
            right_pwm = base_pwm + pp_sag - 1500

            left_pwm = clamp_pwm(left_pwm)
            right_pwm = clamp_pwm(right_pwm)
        else:
            # Rota bulunamadıysa dur
            left_pwm = 1500
            right_pwm = 1500

        # 7. Motorlara Güç Ver
        controller.set_servo(getattr(cfg, 'SOL_MOTOR', 1), left_pwm)
        controller.set_servo(getattr(cfg, 'SAG_MOTOR', 2), right_pwm)

        # 8. Telemetri İçin PWM Değerlerini Paylaşımlı Belleğe Yaz
        shared_state['motor_pwm_left'] = left_pwm
        shared_state['motor_pwm_right'] = right_pwm

        # F. DÖNGÜ ZAMANLAMASI (Loop Frequency)
        # Saniyede ~20 döngü (50ms) için bekleme süresi ayarla
        elapsed = time.time() - start_time
        sleep_time = max(0.01, 0.05 - elapsed)
        time.sleep(sleep_time)

    # GÜVENLİ KAPANIŞ (Shutdown bayrağı True olduysa buraya düşer)
    print("[NAV_PROCESS] Kapanış sinyali alındı. Motorlar durduruluyor...")
    try:
        controller.set_servo(cfg.SOL_MOTOR, cfg.BASE_PWM)
        controller.set_servo(cfg.SAG_MOTOR, cfg.BASE_PWM)
        # controller.disarm_vehicle() # Gerekirse disarm edilebilir
    except Exception:
        pass
    print("[NAV_PROCESS] Kapandı.")