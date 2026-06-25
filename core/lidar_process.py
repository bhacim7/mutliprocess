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

import queue
import numpy as np
import cv2

def lidar_worker(shared_state, lidar_queue):
    """
    Independent process handling high-frequency Lidar point cloud reading.
    Updates the shared_state with emergency obstacle sectors and generates a local costmap
    to send to the Nav process via Queue (Update 4-B).
    """
    print("[LIDAR_PROCESS] Starting Lidar Worker...")

    port_name = getattr(cfg, 'LIDAR_PORT_NAME', '/dev/ttyUSB1')
    baud_rate = getattr(cfg, 'LIDAR_BAUDRATE', 1000000)
    lidar_max_d = getattr(cfg, 'LIDAR_MAX_DIST', 10.0)

    # --- 4-B UPDATE: Setup local costmap in Lidar process ---
    COSTMAP_SIZE_PX = (800, 800)
    COSTMAP_RES_M_PER_PX = 0.10
    costmap_center_m = (0, 0)
    temporal_lidar_buffer = []

    def world_to_pixel(world_x, world_y):
        cw, ch = COSTMAP_SIZE_PX[0] // 2, COSTMAP_SIZE_PX[1] // 2
        dx_m = world_x - costmap_center_m[0]
        dy_m = world_y - costmap_center_m[1]
        px = int(cw + (dx_m / COSTMAP_RES_M_PER_PX))
        py = int(ch - (dy_m / COSTMAP_RES_M_PER_PX))
        h, w = COSTMAP_SIZE_PX
        if 0 <= px < w and 0 <= py < h:
            return (px, py)
        return None
    # --------------------------------------------------------

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
                        current_time = time.time()
                        # --- 4-B UPDATE: Process Costmap locally instead of sending raw points ---
                        # Fetch current robot pose from shared state (updated by ZED)
                        rx = shared_state.get('zed_x', 0.0)
                        ry = shared_state.get('zed_y', 0.0)
                        ryaw = math.radians(shared_state.get('magnetic_heading', 0.0)) # Needs to be global magnetic

                        # Add to local temporal buffer
                        temporal_lidar_buffer.append({'time': current_time, 'points': valid_points, 'rx': rx, 'ry': ry, 'ryaw': ryaw})

                        # Prune old points
                        retention_time = 3.0
                        temporal_lidar_buffer = [b for b in temporal_lidar_buffer if (current_time - b['time']) <= retention_time]

                        # Build Accumulation Map
                        accumulation_map = np.zeros(COSTMAP_SIZE_PX, dtype=np.float32)
                        free_gain = float(getattr(cfg, 'LIDAR_FREE_GAIN', 25))
                        occ_gain = float(getattr(cfg, 'LIDAR_OCCUPIED_GAIN', 80))

                        for buf_entry in temporal_lidar_buffer:
                            b_rx, b_ry, b_ryaw = buf_entry['rx'], buf_entry['ry'], buf_entry['ryaw']
                            p_robot = world_to_pixel(b_rx, b_ry)
                            if not p_robot: continue

                            temp_empty = np.zeros(COSTMAP_SIZE_PX, dtype=np.uint8)
                            temp_occ = np.zeros(COSTMAP_SIZE_PX, dtype=np.uint8)

                            for quality, angle_deg, dist_mm in buf_entry['points']:
                                dist_m = dist_mm / 1000.0
                                angle_rad = math.radians(angle_deg)
                                global_angle = b_ryaw - angle_rad
                                obs_x = b_rx + (dist_m * math.cos(global_angle))
                                obs_y = b_ry + (dist_m * math.sin(global_angle))
                                p_obs = world_to_pixel(obs_x, obs_y)

                                if p_obs:
                                    cv2.line(temp_empty, p_robot, p_obs, 1, 1)
                                    cv2.circle(temp_occ, p_obs, 2, 1, -1)

                            accumulation_map += (temp_empty.astype(np.float32) * free_gain)
                            accumulation_map -= (temp_occ.astype(np.float32) * occ_gain)

                        # Finalize costmap
                        final_map = 127.0 + accumulation_map
                        costmap_img = np.clip(final_map, 0, 255).astype(np.uint8)

                        # --- Push compressed Costmap to Queue ---
                        try:
                            # Send the matrix instead of points
                            lidar_queue.put_nowait((current_time, costmap_img))
                        except queue.Full:
                            pass
                        # -------------------------------------------------------------------------

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
