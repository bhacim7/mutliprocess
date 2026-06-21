import time
import math
import numpy as np
import cv2

# Hardware / Utils
import config as cfg
from hardware.MainSystem2 import USVController
import utils.navigasyon as nav
import utils.planner as planner
from utils.navigasyon import calculate_obj_gps

def nav_worker(shared_state, command_queue, hf_data):
    """
    Independent process handling the autonomous state machine,
    A* path planning, local costmap generation, and PID motor control.
    """
    print("[NAV_PROCESS] Starting Navigation Brain...")

    # 1. Hardware Initialization
    try:
        # Eski Hali: controller = USVController("/dev/ttyACM0", baud=57600)
        fc_port = getattr(cfg, 'FC_PORT', '/dev/ttyACM0')
        fc_baud = getattr(cfg, 'FC_BAUD', 57600)
        controller = USVController(fc_port, baud=fc_baud)
        
        controller.set_mode("MANUAL")
        print("[NAV_PROCESS] USV Controller Initialized.")
    except Exception as e:
        print(f"[NAV_PROCESS][ERROR] Failed to init USV Controller: {e}")
        return

    # 2. Local Costmap Variables
    COSTMAP_SIZE_PX = (800, 800)
    COSTMAP_RES_M_PER_PX = 0.10
    costmap_img = np.full(COSTMAP_SIZE_PX, 127, dtype=np.uint8)
    costmap_center_m = (0, 0)
    costmap_ready = True

    # Temporal Point Cloud Buffer for Map Decay
    temporal_lidar_buffer = []  # List of dicts: {'time': t, 'points': lidar_points}
    last_lidar_points_id = id(None)

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

    # 3. State Machine Variables
    task5_dock_timer = 0
    task5_dock_side = "RIGHT"

    # PID Control Variables for Direct Drive
    direct_drive_integral = 0.0
    direct_drive_prev_error = 0.0

    task2_green_verify_count = 0
    task2_circle_center_lat = None
    task2_circle_center_lon = None
    task2_search_phase = 0
    task2_circle_target_phase = 0
    task2_stall_start_time = None
    task2_stall_check_time = None
    task2_last_dist_to_wp = 0.0
    task2_search_accumulated_yaw = 0.0
    task2_search_prev_yaw = None
    task2_search_start_yaw = None

    task3_gate_passed = False

    current_path = []
    plan_timer = 0
    prev_heading_error = 0.0
    prev_pp_target = None
    hybrid_local_target = None
    force_initial_alignment = False
    prev_target_lat = None
    prev_target_lon = None
    returning_home = False
    finished_printed = False

    failsafe_active = False
    failsafe_start_time = 0
    path_lost_time = None

    acil_durum_aktif_mi = False

    # Position state (Stubbed for now, normally fused from GPS/ZED)
    robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0
    last_pos_time = time.time()

    # Store legacy PWM defaults
    extra = 50

    try:
        while not shared_state['shutdown']:
            start_time = time.time()

            # --- A. READ SENSORS FROM HARDWARE ---
            ida_enlem, ida_boylam = controller.get_current_position()
            hf_data['gps_lat'].value = ida_enlem
            hf_data['gps_lon'].value = ida_boylam

            fc_hdg = controller.get_heading()
            if fc_hdg is not None:
                shared_state['fc_heading'] = fc_hdg

            # Compute actual magnetic_heading based on HEADING_SOURCE
            zed_heading = shared_state.get('zed_heading', 0.0)
            heading_source = getattr(cfg, 'HEADING_SOURCE', 'ZED')

            if heading_source == 'FC' and fc_hdg is not None:
                magnetic_heading = fc_hdg
            elif heading_source == 'FUSED' and fc_hdg is not None:
                diff = nav.signed_angle_difference(zed_heading, fc_hdg)
                magnetic_heading = (zed_heading + (diff * 0.5)) % 360
            else:
                magnetic_heading = zed_heading

            hf_data['magnetic_heading'].value = magnetic_heading


            # --- B. PROCESS INCOMING COMMANDS ---
            try:
                while not command_queue.empty():
                    cmd = command_queue.get_nowait()
                    cmd_str = cmd.get("cmd")

                    if cmd_str == "emergency_stop":
                        print("[NAV_PROCESS] Emergency Stop Received!")
                        shared_state['shutdown'] = True
                    elif cmd_str == "report_status":
                        shared_state['send_telemetry'] = True
                    elif cmd_str == "set_gps":
                        idx = cmd.get("index")
                        lat = cmd.get("lat")
                        lon = cmd.get("lon")
                        if idx == 1: cfg.T1_GATE_ENTER_LAT = lat; cfg.T1_GATE_ENTER_LON = lon
                        elif idx == 2: cfg.T1_GATE_MID_LAT = lat; cfg.T1_GATE_MID_LON = lon
                        elif idx == 3: cfg.T1_GATE_EXIT_LAT = lat; cfg.T1_GATE_EXIT_LON = lon
                        elif idx == 4: cfg.T2_ZONE_ENTRY_LAT = lat; cfg.T2_ZONE_ENTRY_LON = lon
                        elif idx == 5: cfg.T2_ZONE_MID_LAT = lat; cfg.T2_ZONE_MID_LON = lon
                        elif idx == 6: cfg.T2_ZONE_MID1_LAT = lat; cfg.T2_ZONE_MID1_LON = lon
                        elif idx == 7: cfg.T2_ZONE_END_LAT = lat; cfg.T2_ZONE_END_LON = lon
                        elif idx == 8: cfg.T3_START_LAT = lat; cfg.T3_START_LON = lon
                        elif idx == 9: cfg.T3_MID_LAT = lat; cfg.T3_MID_LON = lon
                        elif idx == 10: cfg.T3_RIGHT_LAT = lat; cfg.T3_RIGHT_LON = lon
                        elif idx == 11: cfg.T3_END_LAT = lat; cfg.T3_END_LON = lon
                        elif idx == 12: cfg.T3_END1_LAT = lat; cfg.T3_END1_LON = lon
                        elif idx == 13: cfg.T3_LEFT_LAT = lat; cfg.T3_LEFT_LON = lon
                        elif idx == 14: cfg.T5_DOCK_APPROACH_LAT = lat; cfg.T5_DOCK_APPROACH_LON = lon
                        print(f"[NAV_PROCESS] Updated GPS Point {idx}")
                    elif cmd_str == "set_task":
                        new_task = cmd.get("task_name")
                        if new_task:
                            shared_state['current_task'] = new_task
                            print(f"[NAV_PROCESS] Task updated to {new_task}")
            except: pass

            # --- C. SYNC WITH SHARED STATE ---
            magnetic_heading = hf_data['magnetic_heading'].value
            mevcut_gorev = shared_state.get('current_task', 'TASK_1')
            manual_mode = shared_state.get('manual_mode', False)
            mission_started = shared_state.get('mission_started', True)

            # ZED Odometry / Integration
            robot_x = shared_state.get('zed_x', 0.0)
            robot_y = shared_state.get('zed_y', 0.0)
            robot_yaw = math.radians(magnetic_heading)

            center_danger = shared_state.get('lidar_center_blocked', False)
            left_d = shared_state.get('lidar_left_dist', float('inf'))
            center_d = shared_state.get('lidar_center_dist', float('inf'))
            right_d = shared_state.get('lidar_right_dist', float('inf'))
            lidar_points = shared_state.get('lidar_points', [])

            vision_objects = shared_state.get('vision_detected_objects', [])

            # --- D. UPDATE LOCAL COSTMAP (TEMPORAL BUFFER) ---
            if costmap_ready:
                # 1. Update temporal buffer ONLY with fresh points to prevent N^2 bloat
                current_time = time.time()
                if lidar_points and id(lidar_points) != last_lidar_points_id:
                    temporal_lidar_buffer.append({'time': current_time, 'points': lidar_points, 'rx': robot_x, 'ry': robot_y, 'ryaw': robot_yaw})
                    last_lidar_points_id = id(lidar_points)

                # 2. Prune old points from buffer (keep only the last 3 seconds of data to prevent lag)
                retention_time = 3.0
                temporal_lidar_buffer = [b for b in temporal_lidar_buffer if (current_time - b['time']) <= retention_time]

                # 3. Clear the costmap to base value
                costmap_img.fill(127)

                # 4. Accumulate probabilities correctly
                # We use 32-bit floats to accumulate weights without clipping at 255/0 midway
                accumulation_map = np.zeros(COSTMAP_SIZE_PX, dtype=np.float32)

                free_gain = float(getattr(cfg, 'LIDAR_FREE_GAIN', 25))
                occ_gain = float(getattr(cfg, 'LIDAR_OCCUPIED_GAIN', 80))

                for buf_entry in temporal_lidar_buffer:
                    rx, ry, ryaw = buf_entry['rx'], buf_entry['ry'], buf_entry['ryaw']
                    p_robot = world_to_pixel(rx, ry)
                    if not p_robot: continue

                    # Create temporary masks for this specific scan
                    temp_empty = np.zeros(COSTMAP_SIZE_PX, dtype=np.uint8)
                    temp_occ = np.zeros(COSTMAP_SIZE_PX, dtype=np.uint8)

                    for quality, angle_deg, dist_mm in buf_entry['points']:
                        dist_m = dist_mm / 1000.0
                        angle_rad = math.radians(angle_deg)
                        global_angle = ryaw - angle_rad
                        obs_x = rx + (dist_m * math.cos(global_angle))
                        obs_y = ry + (dist_m * math.sin(global_angle))
                        p_obs = world_to_pixel(obs_x, obs_y)

                        if p_obs:
                            cv2.line(temp_empty, p_robot, p_obs, 1, 1)
                            cv2.circle(temp_occ, p_obs, 2, 1, -1)

                    # Accumulate: Free space adds positive probability, occupied subtracts
                    accumulation_map += (temp_empty.astype(np.float32) * free_gain)
                    accumulation_map -= (temp_occ.astype(np.float32) * occ_gain)

                # 5. Apply the accumulated probabilties to the base map and clip
                final_map = 127.0 + accumulation_map
                costmap_img = np.clip(final_map, 0, 255).astype(np.uint8)

            if vision_objects and costmap_ready:
                for obj in vision_objects:
                    dist_m = obj.get('dist', 0)
                    if 0 < dist_m < 15.0:
                        pixel_offset = (obj.get('cx', 1280/2) - (1280 / 2)) / 1280.0
                        angle_offset = -pixel_offset * math.radians(getattr(cfg, 'CAM_HFOV', 110.0))
                        obj_global_angle = robot_yaw + angle_offset

                        obj_world_x = robot_x + (dist_m * math.cos(obj_global_angle))
                        obj_world_y = robot_y + (dist_m * math.sin(obj_global_angle))

                        p_virtual = world_to_pixel(obj_world_x, obj_world_y)
                        if p_virtual:
                            cv2.circle(costmap_img, p_virtual, 6, 0, -1)

            # --- E. FULL STATE MACHINE ---
            target_lat = None
            target_lon = None

            # --- TASK 6 ---
            if mevcut_gorev == "TASK6_SPEED":
                target_lat, target_lon = getattr(cfg, 'T3_START_LAT', 0), getattr(cfg, 'T3_START_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "T3_START"
            elif mevcut_gorev == "TASK6_DOCK":
                target_lat, target_lon = getattr(cfg, 'T5_DOCK_APPROACH_LAT', 0), getattr(cfg, 'T5_DOCK_APPROACH_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "TASK5_APPROACH"

            # --- TASK 1 ---
            elif mevcut_gorev == "TASK1_APPROACH": mevcut_gorev = "TASK1_STATE_ENTER"
            elif mevcut_gorev in ["TASK1_STATE_ENTER", "TASK1_STATE_MID", "TASK1_STATE_EXIT"]:
                if mevcut_gorev == "TASK1_STATE_ENTER": target_lat, target_lon = getattr(cfg, 'T1_GATE_ENTER_LAT', 0), getattr(cfg, 'T1_GATE_ENTER_LON', 0)
                elif mevcut_gorev == "TASK1_STATE_MID": target_lat, target_lon = getattr(cfg, 'T1_GATE_MID_LAT', 0), getattr(cfg, 'T1_GATE_MID_LON', 0)
                else: target_lat, target_lon = getattr(cfg, 'T1_GATE_EXIT_LAT', 0), getattr(cfg, 'T1_GATE_EXIT_LON', 0)

                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0:
                    print(f"[TASK1] Reached {mevcut_gorev}")
                    if mevcut_gorev == "TASK1_STATE_ENTER": mevcut_gorev = "TASK1_STATE_MID"
                    elif mevcut_gorev == "TASK1_STATE_MID": mevcut_gorev = "TASK1_STATE_EXIT"
                    else:
                        if returning_home: mevcut_gorev = "TASK1_RETURN_MID"
                        else: mevcut_gorev = "TASK2_START"

            elif mevcut_gorev in ["TASK1_RETURN_MID", "TASK1_RETURN_ENTER"]:
                if mevcut_gorev == "TASK1_RETURN_MID":
                    target_lat, target_lon = getattr(cfg, 'T1_GATE_MID_LAT', 0), getattr(cfg, 'T1_GATE_MID_LON', 0)
                    if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "TASK1_RETURN_ENTER"
                elif mevcut_gorev == "TASK1_RETURN_ENTER":
                    target_lat, target_lon = getattr(cfg, 'T1_GATE_ENTER_LAT', 0), getattr(cfg, 'T1_GATE_ENTER_LON', 0)
                    if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "FINISHED"

            elif mevcut_gorev == "FINISHED":
                if not finished_printed:
                    print("[TASK1] MISSION COMPLETE")
                    finished_printed = True
                controller.set_servo(cfg.SOL_MOTOR, 1500)
                controller.set_servo(cfg.SAG_MOTOR, 1500)

            # --- TASK 2 ---
            elif mevcut_gorev == "TASK2_START":
                target_lat, target_lon = getattr(cfg, 'T2_ZONE_ENTRY_LAT', 0), getattr(cfg, 'T2_ZONE_ENTRY_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "TASK2_GO_TO_MID"
            elif mevcut_gorev == "TASK2_GO_TO_MID":
                target_lat, target_lon = getattr(cfg, 'T2_ZONE_MID_LAT', 0), getattr(cfg, 'T2_ZONE_MID_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "TASK2_GO_TO_MID1"
            elif mevcut_gorev == "TASK2_GO_TO_MID1":
                target_lat, target_lon = getattr(cfg, 'T2_ZONE_MID1_LAT', 0), getattr(cfg, 'T2_ZONE_MID1_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "TASK2_GO_TO_END"
            elif mevcut_gorev == "TASK2_GO_TO_END":
                target_lat, target_lon = getattr(cfg, 'T2_ZONE_END_LAT', 0), getattr(cfg, 'T2_ZONE_END_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0:
                    mevcut_gorev = "TASK2_SEARCH_PATTERN"
                    task2_search_accumulated_yaw = 0.0
                    task2_search_prev_yaw = magnetic_heading
                    task2_search_start_yaw = magnetic_heading

            elif mevcut_gorev == "TASK2_SEARCH_PATTERN":
                found_green_live = False
                for obj in vision_objects:
                    if obj.get('cid') == 4 and obj.get('dist', 10) < 5.0: # Green Marker
                        found_green_live = True
                        found_green_dist = obj['dist']
                        pixel_offset = (obj['cx'] - (1280 / 2)) / 1280 # Stub width
                        found_green_angle_offset = pixel_offset * getattr(cfg, 'CAM_HFOV', 110.0)
                        break

                if found_green_live:
                    task2_green_verify_count += 1
                    if task2_green_verify_count >= 5:
                        print("[TASK2] GREEN MARKER CONFIRMED! CALCULATING ORBIT")
                        mevcut_gorev = "TASK2_GREEN_MARKER_FOUND"
                        obj_bearing = (magnetic_heading + found_green_angle_offset) % 360
                        task2_circle_center_lat, task2_circle_center_lon = calculate_obj_gps(ida_enlem, ida_boylam, found_green_dist, obj_bearing)
                        bearing_to_robot = nav.calculate_bearing(task2_circle_center_lat, task2_circle_center_lon, ida_enlem, ida_boylam)
                        task2_search_phase = (int(round(bearing_to_robot / 45.0)) % 8) + 1
                        task2_circle_target_phase = task2_search_phase + 8
                        task2_green_verify_count = 0
                else:
                    task2_green_verify_count = 0
                    current_yaw = magnetic_heading
                    if task2_search_prev_yaw is not None and current_yaw is not None:
                        diff = nav.signed_angle_difference(task2_search_prev_yaw, current_yaw)
                        task2_search_accumulated_yaw += abs(diff)
                        task2_search_prev_yaw = current_yaw
                    if task2_search_start_yaw is not None and current_yaw is not None:
                        heading_diff = abs(nav.signed_angle_difference(task2_search_start_yaw, current_yaw))
                        if task2_search_accumulated_yaw > 320.0 and heading_diff < 15.0:
                            print("[TASK2] 360 ROTATION COMPLETE -> RETURN HOME")
                            mevcut_gorev = "TASK2_RETURN_HOME"

            elif mevcut_gorev == "TASK2_GREEN_MARKER_FOUND":
                R = getattr(cfg, 'TASK2_SEARCH_DIAMETER', 2.0) / 2.0
                if task2_search_phase >= task2_circle_target_phase:
                    mevcut_gorev = "TASK2_RETURN_HOME"
                else:
                    target_angle_deg = (task2_search_phase % 8) * 45.0
                    if task2_circle_center_lat is not None:
                        target_lat, target_lon = calculate_obj_gps(task2_circle_center_lat, task2_circle_center_lon, R, target_angle_deg)
                    dist_to_wp = nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon)

                    if task2_stall_check_time is None:
                        task2_stall_check_time = time.time()
                        task2_last_dist_to_wp = dist_to_wp
                    if (time.time() - task2_stall_check_time) > 1.0:
                        if abs(dist_to_wp - task2_last_dist_to_wp) < 0.1:
                            if task2_stall_start_time is None: task2_stall_start_time = task2_stall_check_time
                        else: task2_stall_start_time = None
                        task2_stall_check_time = time.time()
                        task2_last_dist_to_wp = dist_to_wp

                    if task2_stall_start_time and (time.time() - task2_stall_start_time) > 5.0:
                        print("[TASK2] STALL DETECTED -> ABORTING CIRCLING")
                        mevcut_gorev = "TASK2_RETURN_HOME"

                    if dist_to_wp < 1.5:
                        task2_search_phase += 1
                        task2_stall_start_time = None
                        task2_stall_check_time = None

            elif mevcut_gorev == "TASK2_RETURN_HOME": mevcut_gorev = "TASK2_RETURN_END"
            elif mevcut_gorev in ["TASK2_RETURN_END", "TASK2_RETURN_MID", "TASK2_RETURN_MID1", "TASK2_RETURN_ENTRY"]:
                if mevcut_gorev == "TASK2_RETURN_END":
                    target_lat, target_lon = getattr(cfg, 'T2_ZONE_END_LAT', 0), getattr(cfg, 'T2_ZONE_END_LON', 0)
                    if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "TASK2_RETURN_MID1"
                elif mevcut_gorev == "TASK2_RETURN_MID1":
                    target_lat, target_lon = getattr(cfg, 'T2_ZONE_MID1_LAT', 0), getattr(cfg, 'T2_ZONE_MID1_LON', 0)
                    if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "TASK2_RETURN_MID"
                elif mevcut_gorev == "TASK2_RETURN_MID":
                    target_lat, target_lon = getattr(cfg, 'T2_ZONE_MID_LAT', 0), getattr(cfg, 'T2_ZONE_MID_LON', 0)
                    if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "TASK2_RETURN_ENTRY"
                elif mevcut_gorev == "TASK2_RETURN_ENTRY":
                    target_lat, target_lon = getattr(cfg, 'T2_ZONE_ENTRY_LAT', 0), getattr(cfg, 'T2_ZONE_ENTRY_LON', 0)
                    if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "TASK3_APPROACH"

            # --- TASK 3 ---
            elif mevcut_gorev == "TASK3_APPROACH": mevcut_gorev = "T3_START"
            elif mevcut_gorev == "T3_START":
                target_lat, target_lon = getattr(cfg, 'T3_START_LAT', 0), getattr(cfg, 'T3_START_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0:
                    if getattr(cfg, 'ENABLE_TASK3', True): mevcut_gorev = "T3_MID"
                    else: mevcut_gorev = "TASK5_APPROACH"
            elif mevcut_gorev == "T3_MID":
                target_lat, target_lon = getattr(cfg, 'T3_MID_LAT', 0), getattr(cfg, 'T3_MID_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "T3_RIGHT"
            elif mevcut_gorev == "T3_RIGHT":
                target_lat, target_lon = getattr(cfg, 'T3_RIGHT_LAT', 0), getattr(cfg, 'T3_RIGHT_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "T3_END"
            elif mevcut_gorev == "T3_END":
                target_lat, target_lon = getattr(cfg, 'T3_END_LAT', 0), getattr(cfg, 'T3_END_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "T3_END1"
            elif mevcut_gorev == "T3_END1":
                target_lat, target_lon = getattr(cfg, 'T3_END1_LAT', 0), getattr(cfg, 'T3_END1_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "T3_LEFT"
            elif mevcut_gorev == "T3_LEFT":
                target_lat, target_lon = getattr(cfg, 'T3_LEFT_LAT', 0), getattr(cfg, 'T3_LEFT_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "T3_RETURN_MID"
            elif mevcut_gorev == "T3_RETURN_MID":
                target_lat, target_lon = getattr(cfg, 'T3_MID_LAT', 0), getattr(cfg, 'T3_MID_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "T3_RETURN_START"
            elif mevcut_gorev == "T3_RETURN_START":
                target_lat, target_lon = getattr(cfg, 'T3_START_LAT', 0), getattr(cfg, 'T3_START_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0: mevcut_gorev = "TASK5_APPROACH"

            # --- TASK 5 ---
            elif mevcut_gorev == "TASK5_APPROACH":
                target_lat, target_lon = getattr(cfg, 'T5_DOCK_APPROACH_LAT', 0), getattr(cfg, 'T5_DOCK_APPROACH_LON', 0)
                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0:
                    returning_home = True
                    mevcut_gorev = "TASK1_STATE_EXIT"

            # Sync State
            shared_state['current_task'] = mevcut_gorev

            # --- F. NAVIGATION CALCULATIONS & HYBRID LOGIC ---
            aci_farki = 0.0
            adviced_course = 0.0

            if target_lat is not None:
                if (target_lat != prev_target_lat or target_lon != prev_target_lon):
                    force_initial_alignment = True
                    prev_target_lat = target_lat
                    prev_target_lon = target_lon

                adviced_course = nav.calculate_bearing(ida_enlem, ida_boylam, target_lat, target_lon)
                aci_farki = nav.signed_angle_difference(magnetic_heading, adviced_course)
                shared_state['angle_error'] = float(aci_farki)
                shared_state['adviced_course'] = float(adviced_course)
                shared_state['target_dist'] = float(hedefe_mesafe) if 'hedefe_mesafe' in locals() else 0.0
                shared_state['target_lat'] = float(target_lat)
                shared_state['target_lon'] = float(target_lon)

            # Hybrid targeting setup
            tx_world, ty_world = None, None
            if costmap_ready and target_lat is not None:
                if mevcut_gorev in ["TASK5_ENTER"]:
                    need_new_target = True
                    if hybrid_local_target:
                        d_local = math.sqrt((hybrid_local_target[0] - robot_x) ** 2 + (hybrid_local_target[1] - robot_y) ** 2)
                        if d_local > 0.5:
                            need_new_target = False  # Hala gidiyoruz

                    if need_new_target and 'aci_farki' in locals() and aci_farki is not None:
                        step_dist = getattr(cfg, 'HYBRID_STEP_DIST', 2.0)
                        h_tx, h_ty = nav.get_hybrid_point(robot_x, robot_y, robot_yaw, aci_farki, step_dist)
                        hybrid_local_target = (h_tx, h_ty)

                    if hybrid_local_target:
                        tx_world, ty_world = hybrid_local_target
                    else:
                        gps_lookahead = 1.5
                        tx_world = robot_x + (gps_lookahead * math.cos(robot_yaw + math.radians(-aci_farki)))
                        ty_world = robot_y + (gps_lookahead * math.sin(robot_yaw + math.radians(-aci_farki)))
                else:
                    hybrid_local_target = None  # Reset
                    gps_lookahead = 1.5
                    tx_world = robot_x + (gps_lookahead * math.cos(robot_yaw + math.radians(-aci_farki)))
                    ty_world = robot_y + (gps_lookahead * math.sin(robot_yaw + math.radians(-aci_farki)))

            # --- G. CONTROL LOGIC & MOTORS ---
            if manual_mode or not mission_started:
                controller.set_servo(cfg.SOL_MOTOR, 1500)
                controller.set_servo(cfg.SAG_MOTOR, 1500)
            else:
                # 1. Reactive Avoidance
                if center_danger and mevcut_gorev not in ["TASK5_ENTER", "TASK5_DOCK", "TASK5_EXIT"]:
                    if not acil_durum_aktif_mi:
                        shock_brake_pwm = cfg.BASE_PWM - getattr(cfg, 'shock_pwm', 250)
                        controller.set_servo(cfg.SOL_MOTOR, shock_brake_pwm)
                        controller.set_servo(cfg.SAG_MOTOR, shock_brake_pwm)
                        time.sleep(0.1)
                        acil_durum_aktif_mi = True

                    escape_pwm = cfg.BASE_PWM - getattr(cfg, 'ESCAPE_PWM', 300)
                    controller.set_servo(cfg.SOL_MOTOR, escape_pwm)
                    controller.set_servo(cfg.SAG_MOTOR, escape_pwm)
                    time.sleep(0.4)

                    spot_turn_val = getattr(cfg, 'SPOT_TURN_PWM', 200)
                    if left_d > right_d:
                        controller.set_servo(cfg.SOL_MOTOR, cfg.BASE_PWM - spot_turn_val)
                        controller.set_servo(cfg.SAG_MOTOR, cfg.BASE_PWM + spot_turn_val)
                    else:
                        controller.set_servo(cfg.SOL_MOTOR, cfg.BASE_PWM + spot_turn_val)
                        controller.set_servo(cfg.SAG_MOTOR, cfg.BASE_PWM - spot_turn_val)
                    time.sleep(0.3)
                    current_path = None # Force replan
                    continue
                else:
                    acil_durum_aktif_mi = False

                # 2. Task 5 Specific (Blind Lidar Navigation)
                if mevcut_gorev == "TASK5_ENTER":
                    r_val = right_d if not math.isinf(right_d) else 2.0
                    l_val = left_d if not math.isinf(left_d) else 2.0
                    err = r_val - l_val
                    rot = np.clip(err * 50, -100, 100)
                    controller.set_servo(cfg.SOL_MOTOR, int(1580 + rot))
                    controller.set_servo(cfg.SAG_MOTOR, int(1580 - rot))

                elif mevcut_gorev == "TASK5_DOCK":
                    task5_dock_timer += 1
                    turn_pwm_sol, turn_pwm_sag = 1650, 1350
                    if task5_dock_side == "LEFT": turn_pwm_sol, turn_pwm_sag = 1350, 1650

                    if task5_dock_timer < 25:
                        controller.set_servo(cfg.SOL_MOTOR, turn_pwm_sol)
                        controller.set_servo(cfg.SAG_MOTOR, turn_pwm_sag)
                    elif task5_dock_timer < 65:
                        controller.set_servo(cfg.SOL_MOTOR, 1600)
                        controller.set_servo(cfg.SAG_MOTOR, 1600)
                    else:
                        controller.set_servo(cfg.SOL_MOTOR, 1500)
                        controller.set_servo(cfg.SAG_MOTOR, 1500)
                        mevcut_gorev = "TASK5_EXIT"
                        task5_dock_timer = 0

                elif mevcut_gorev == "TASK5_EXIT":
                    task5_dock_timer += 1
                    if task5_dock_timer < 45:
                        controller.set_servo(cfg.SOL_MOTOR, 1400)
                        controller.set_servo(cfg.SAG_MOTOR, 1400)
                    elif task5_dock_timer < 75:
                        turn_pwm_sol, turn_pwm_sag = 1650, 1350
                        if task5_dock_side == "LEFT": turn_pwm_sol, turn_pwm_sag = 1350, 1650
                        controller.set_servo(cfg.SOL_MOTOR, turn_pwm_sol)
                        controller.set_servo(cfg.SAG_MOTOR, turn_pwm_sag)
                    else:
                        r_val = right_d if not math.isinf(right_d) else 2.0
                        l_val = left_d if not math.isinf(left_d) else 2.0
                        rot = np.clip((r_val - l_val) * 50, -100, 100)
                        controller.set_servo(cfg.SOL_MOTOR, int(1580 + rot))
                        controller.set_servo(cfg.SAG_MOTOR, int(1580 - rot))

                # 3. Task 2 Search Rotation overrides
                elif mevcut_gorev == "TASK2_SEARCH_PATTERN":
                    spot_pwm = getattr(cfg, 'SPOT_TURN_PWM', 200)
                    controller.set_servo(cfg.SOL_MOTOR, 1500 + spot_pwm)
                    controller.set_servo(cfg.SAG_MOTOR, 1500 - spot_pwm - extra)

                # 4. Standard A* / Direct Drive
                else:
                    if target_lat is not None:
                        # Initial alignment logic
                        if force_initial_alignment and abs(aci_farki) < 5.0:
                            force_initial_alignment = False

                        should_force_alignment = force_initial_alignment

                        if should_force_alignment:
                            spot_pwm = getattr(cfg, 'SPOT_TURN_PWM', 200)
                            if aci_farki > 0: controller.set_servo(cfg.SOL_MOTOR, 1500 + spot_pwm); controller.set_servo(cfg.SAG_MOTOR, 1500 - spot_pwm - extra)
                            else: controller.set_servo(cfg.SOL_MOTOR, 1500 - spot_pwm - extra); controller.set_servo(cfg.SAG_MOTOR, 1500 + spot_pwm)
                        else:
                            # Run Planner
                            nav_map, _ = planner.get_inflated_nav_map(costmap_img, ignore_green=(mevcut_gorev == "TASK2_GREEN_MARKER_FOUND"))

                            plan_timer += 1
                            if plan_timer > 4:
                                plan_timer = 0
                                if tx_world is not None:
                                    if planner.check_line_of_sight((robot_x, robot_y), (tx_world, ty_world), nav_map, costmap_center_m, COSTMAP_RES_M_PER_PX, COSTMAP_SIZE_PX):
                                        current_path = [(robot_x, robot_y), (tx_world, ty_world)]
                                    else:
                                        new_path = planner.get_path_plan((robot_x, robot_y), (tx_world, ty_world), nav_map, costmap_center_m, COSTMAP_RES_M_PER_PX, COSTMAP_SIZE_PX)
                                        if new_path: current_path = new_path

                            if current_path:
                                base_pwm = getattr(cfg, 'BASE_PWM', 1500)
                                if mevcut_gorev.startswith("T3_"): base_pwm += getattr(cfg, 'T3_SPEED_PWM', 100)

                                pp_sol, pp_sag, raw_target, current_error, pruned_path = planner.pure_pursuit_control(
                                    robot_x, robot_y, robot_yaw, current_path, current_speed=0, base_speed=base_pwm, prev_error=prev_heading_error)

                                current_path = pruned_path
                                prev_heading_error = current_error

                                failsafe_active = False
                                path_lost_time = None

                                controller.set_servo(cfg.SOL_MOTOR, pp_sol)
                                controller.set_servo(cfg.SAG_MOTOR, pp_sag)
                            else:
                                if target_lat is not None and target_lon is not None:
                                    if path_lost_time is None:
                                        path_lost_time = time.time()

                                    if time.time() - path_lost_time < 5.0:
                                        # Direct Drive Grace Period
                                        failsafe_active = True

                                        # --- FIX: Reintroduce spot turn if heading is severely off ---
                                        threshold = getattr(cfg, 'SPOT_TURN_THRESHOLD', 45.0)

                                        if abs(aci_farki) > threshold:
                                            spot_pwm = getattr(cfg, 'SPOT_TURN_PWM', 200)
                                            extra = 50
                                            if aci_farki > 0:  # Target Right
                                                controller.set_servo(cfg.SOL_MOTOR, 1500 + spot_pwm)
                                                controller.set_servo(cfg.SAG_MOTOR, 1500 - spot_pwm - extra)
                                            else:  # Target Left
                                                controller.set_servo(cfg.SOL_MOTOR, 1500 - spot_pwm - extra)
                                                controller.set_servo(cfg.SAG_MOTOR, 1500 + spot_pwm)
                                        else:
                                            base_pwm = getattr(cfg, 'BASE_PWM', 1500) + getattr(cfg, 'CRUISE_PWM', 80)

                                            # Full PID controller for direct steering
                                            kp = 1.5
                                            ki = 0.05
                                            kd = 0.5

                                            # Calculate terms
                                            error = aci_farki
                                            direct_drive_integral += error

                                            # Integral windup limit
                                            windup_limit = 500.0
                                            direct_drive_integral = max(-windup_limit, min(windup_limit, direct_drive_integral))

                                            derivative = error - direct_drive_prev_error
                                            direct_drive_prev_error = error

                                            steering_correction = (error * kp) + (direct_drive_integral * ki) + (derivative * kd)

                                            pp_sol = base_pwm + steering_correction
                                            pp_sag = base_pwm - steering_correction

                                            # Clamp values
                                            pp_sol = max(1100, min(1900, pp_sol))
                                            pp_sag = max(1100, min(1900, pp_sag))

                                            controller.set_servo(cfg.SOL_MOTOR, int(pp_sol))
                                            controller.set_servo(cfg.SAG_MOTOR, int(pp_sag))
                                    else:
                                        # Stop if grace period exceeded and still no path
                                        controller.set_servo(cfg.SOL_MOTOR, 1500)
                                        controller.set_servo(cfg.SAG_MOTOR, 1500)
                                else:
                                    controller.set_servo(cfg.SOL_MOTOR, 1500)
                                    controller.set_servo(cfg.SAG_MOTOR, 1500)

            # Record final PWMs
            # Note: We rely on the USVController object stub tracking state,
            # but we can push directly to shared_state for safety
            shared_state['motor_pwm_left'] = controller.get_servo_pwm(cfg.SOL_MOTOR)
            shared_state['motor_pwm_right'] = controller.get_servo_pwm(cfg.SAG_MOTOR)

            elapsed = time.time() - start_time
            if elapsed < 0.02: time.sleep(0.02 - elapsed)

    except Exception as e:
        print(f"[NAV_PROCESS][ERROR] Brain crashed: {e}")
    finally:
        print("[NAV_PROCESS] Shutting down...")
        try:
            controller.set_servo(cfg.SOL_MOTOR, 1500)
            controller.set_servo(cfg.SAG_MOTOR, 1500)
            controller.disarm_vehicle()
        except: pass
