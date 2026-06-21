import time
import math
import sys
import threading
from rplidar import RPLidar

# Import config (assuming it has LIDAR_PORT_NAME, LIDAR_BAUDRATE, LIDAR_MAX_DIST, LIDAR_ACIL_DURMA_M)
import config as cfg

def process_lidar_sectors(scan_data, max_dist=3.0):
    """
    Analyzes Lidar data for Left, Center, and Right sectors.
    Returns: (center_blocked, left_dist, center_dist, right_dist)
    """
    left_min_dist = float('inf')
    center_min_dist = float('inf')
    right_min_dist = float('inf')
    center_blocked = False

    if not scan_data:
        return False, float('inf'), float('inf'), float('inf')

    for quality, angle, distance_mm in scan_data:
        dist_m = distance_mm / 1000.0

        # Noise filter (ignore below 0.4m and above max_dist)
        if dist_m < 0.4 or dist_m > max_dist:
            continue

        # Normalize angles (-180 to +180)
        norm_angle = angle
        if angle > 180:
            norm_angle = angle - 360

        # 1. CENTER (DANGER) ZONE (-15 to +15 degrees)
        if -15 <= norm_angle <= 15:
            if dist_m < center_min_dist:
                center_min_dist = dist_m
            lidar_limit = getattr(cfg, 'LIDAR_ACIL_DURMA_M', 1.5)
            if dist_m < lidar_limit:
                center_blocked = True

        # 2. RIGHT CORRIDOR (+15 to +60 degrees)
        elif 15 < norm_angle <= 60:
            if dist_m < right_min_dist:
                right_min_dist = dist_m

        # 3. LEFT CORRIDOR (-60 to -15 degrees)
        elif -60 <= norm_angle < -15:
            if dist_m < left_min_dist:
                left_min_dist = dist_m

    return center_blocked, left_min_dist, center_min_dist, right_min_dist

def lidar_worker(shared_state):
    """
    Independent process handling high-frequency Lidar point cloud reading.
    Updates the shared_state with emergency obstacle sectors and optionally raw points.
    """
    print("[LIDAR_PROCESS] Starting Lidar Worker...")

    port_name = getattr(cfg, 'LIDAR_PORT_NAME', '/dev/ttyUSB1')
    baud_rate = getattr(cfg, 'LIDAR_BAUDRATE', 1000000)
    lidar_max_d = getattr(cfg, 'LIDAR_MAX_DIST', 10.0)

    lidar = None

    def setup_lidar():
        nonlocal lidar
        try:
            print(f"[LIDAR_PROCESS] Connecting to Lidar on {port_name} @ {baud_rate}...")
            lidar = RPLidar(port_name, baudrate=baud_rate, timeout=3)
            health = lidar.get_health()
            print(f"[LIDAR_PROCESS] Lidar Health: {health}")
            if health[0] != 'Good':
                print(f"[LIDAR_PROCESS][WARNING] Lidar health: {health}")
            return True
        except Exception as e:
            print(f"[LIDAR_PROCESS][ERROR] Failed to connect to Lidar: {e}", file=sys.stderr)
            return False

    while not shared_state['shutdown']:
        if lidar is None:
            if not setup_lidar():
                time.sleep(2)
                continue

        try:
            lidar.start_motor()
            time.sleep(0.5) # Give the motor a moment to spin up

            # Continuously read scans
            for scan in lidar.iter_scans(max_buf_meas=5000, min_len=5):
                if shared_state['shutdown']:
                    break

                valid_points = []
                for quality, angle, distance in scan:
                    if distance > 0:
                        valid_points.append((quality, angle, distance))

                if valid_points:
                    # Process sectors immediately
                    center_blocked, left_d, center_d, right_d = process_lidar_sectors(valid_points, max_dist=lidar_max_d)

                    # Update Shared State (Lightweight numbers only)
                    shared_state['lidar_center_blocked'] = center_blocked
                    shared_state['lidar_left_dist'] = float(left_d)
                    shared_state['lidar_center_dist'] = float(center_d)
                    shared_state['lidar_right_dist'] = float(right_d)

                    # Store sampled points for mapping later (if pitch/roll is stable)
                    if shared_state['lidar_wave_stable']:
                        # Downsample points for IPC (e.g. taking every 3rd point if needed)
                        # To prevent huge IPC overhead, we only pass a subsampled list or ignore entirely if not needed.
                        # Legacy code downsampled 3:1 for distances < 5m.
                        downsampled = []
                        for i, p in enumerate(valid_points):
                            dist_m = p[2] / 1000.0
                            if dist_m > 5.0 or (i % 3 == 0):
                                downsampled.append(p)
                        shared_state['lidar_points'] = downsampled
                    else:
                        shared_state['lidar_points'] = []

        except Exception as e:
            print(f"[LIDAR_PROCESS][ERROR] Connection lost or data corrupt: {e}")
            print("[LIDAR_PROCESS] Hard Resetting Lidar...")
            try:
                if lidar:
                    lidar.stop_motor()
                    lidar.disconnect()
            except:
                pass
            lidar = None
            time.sleep(1.0) # Rest the port

    # Shutdown sequence
    print("[LIDAR_PROCESS] Shutting down...")
    try:
        if lidar:
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
    except Exception as e:
        print(f"[LIDAR_PROCESS] Error during shutdown: {e}")
