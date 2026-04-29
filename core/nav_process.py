import time
import math
import sys

# Proje ana dizininden modülleri çekebilmek için (Eğer script kendi klasöründen çalıştırılırsa diye)
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from hardware.MainSystem2 import USVController
from utils import navigasyon as nav


def nav_worker(shared_state, command_queue):
    """
    Otonom Navigasyon ve Beyin Prosesi.
    OrangeCube ile haberleşir, GPS okur, PID hesaplar ve motorlara PWM basar.
    """
    print("[NAV_PROCESS] Başlatılıyor...")

    # 1. ORANGECUBE (MAVLINK) BAĞLANTISI
    try:
        controller = USVController(connection_string=cfg.ORANGE_PORT, baud=cfg.ORANGE_BAUD)
        print("[NAV_PROCESS] OrangeCube bağlantısı başarılı!")

        # Otonom sürüş için Pixhawk'ı MANUAL moda alıp motorları arm etmeliyiz
        # (Gerçek testlerde Arm komutunu manuel vermek isteyebilirsin, şimdilik otomatik yapıyoruz)
        controller.set_mode("MANUAL")
        print("[NAV_PROCESS] Mod MANUAL olarak ayarlandı.")
    except Exception as e:
        print(f"[NAV_PROCESS] KRİTİK HATA! OrangeCube'a bağlanılamadı: {e}")
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
        target_lat, target_lon = cfg.TASK_WAYPOINTS[current_task]

        # 1. Mesafe Hesapla
        distance_m = nav.haversine(lat, lon, target_lat, target_lon)

        # 2. Hedefe Vardık Mı? (Transition)
        if distance_m < cfg.WAYPOINT_TOLERANCE_M:
            print(f"[NAV_PROCESS] {current_task} Tamamlandı! (Hedefe {distance_m:.1f}m)")
            next_task = NEXT_TASK_MAP.get(current_task, 'FINISHED')
            shared_state['current_task'] = next_task
            continue  # Döngünün başına dön ve yeni hedef için hesap yap

        # 3. Yön (Bearing) ve Hata Açısı (Error) Hesapla
        target_bearing = nav.calculate_bearing(lat, lon, target_lat, target_lon)
        angle_error = nav.signed_angle_difference(current_heading, target_bearing)

        # 4. P-Kontrolcü ile Dönüş Şiddetini Hesapla
        turn_cmd = angle_error * cfg.Kp_HEADING

        # Dönüş komutunu sınırla (Max dönüş hızını aşmamak için)
        if turn_cmd > cfg.MAX_TURN_PWM:
            turn_cmd = cfg.MAX_TURN_PWM
        elif turn_cmd < -cfg.MAX_TURN_PWM:
            turn_cmd = -cfg.MAX_TURN_PWM

        # 5. Motor PWM Değerlerini Oluştur (Diferansiyel Sürüş)
        # angle_error pozitifse (hedef sağda), sol motor daha hızlı dönmeli
        left_pwm = cfg.BASE_PWM + cfg.CRUISE_PWM + turn_cmd
        right_pwm = cfg.BASE_PWM + cfg.CRUISE_PWM - turn_cmd

        left_pwm = clamp_pwm(left_pwm)
        right_pwm = clamp_pwm(right_pwm)

        # 6. Motorlara Güç Ver
        controller.set_servo(cfg.SOL_MOTOR, left_pwm)
        controller.set_servo(cfg.SAG_MOTOR, right_pwm)

        # 7. Telemetri İçin PWM Değerlerini Paylaşımlı Belleğe Yaz
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