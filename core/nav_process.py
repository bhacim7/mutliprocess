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

import queue

def apply_motor_mixer(controller, forward_pwm, yaw_correction):
    """
    Control Mixer for 4-Thruster Vectored-Bow architecture.
    Takes abstract forward speed and yaw steering commands and
    distributes them to 5 hardware channels optimally.
    """
    base = getattr(cfg, 'BASE_PWM', 1500)
    steer_max = getattr(cfg, 'STEER_MAX_PWM', 1900)
    steer_min = getattr(cfg, 'STEER_MIN_PWM', 1100)

    # Calculate desired front steering angle
    steer_pwm = int(np.clip(base - yaw_correction, steer_min, steer_max))

    # Check Deadband: If steering correction is small (< 15 degrees equivalent),
    # only use the front steering servo. No differential thrust.
    # We estimate '15 degrees equivalent' by the size of yaw_correction
    if abs(yaw_correction) < 50: # Tune this threshold
        rear_left = forward_pwm
        rear_right = forward_pwm
        front_left = forward_pwm
        front_right = forward_pwm
    else:
        # Blend in differential thrust
        # Subtract some steering correction to use as differential offset
        diff_offset = yaw_correction * 0.5

        rear_left = int(forward_pwm + diff_offset)
        rear_right = int(forward_pwm - diff_offset)

        # Front thrusters can stay relatively equal as they are already vectored
        front_left = int(forward_pwm + (diff_offset * 0.2))
        front_right = int(forward_pwm - (diff_offset * 0.2))

    # Clip all thrusts to safe limits
    rear_left = np.clip(rear_left, 1100, 1900)
    rear_right = np.clip(rear_right, 1100, 1900)
    front_left = np.clip(front_left, 1100, 1900)
    front_right = np.clip(front_right, 1100, 1900)

    controller.set_servo(cfg.STEER_SERVO, steer_pwm)
    controller.set_servo(cfg.SOL_MOTOR, rear_left)
    controller.set_servo(cfg.SAG_MOTOR, rear_right)
    controller.set_servo(cfg.FRONT_SOL_MOTOR, front_left)
    controller.set_servo(cfg.FRONT_SAG_MOTOR, front_right)

def nav_worker(shared_state, command_queue, hf_data, lidar_queue):
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

            # --- 4-A UPDATE: Pull from Queue instead of shared_state ---
            lidar_data = None
            try:
                # Empty the queue, only keeping the absolute newest frame
                while True:
                    lidar_data = lidar_queue.get_nowait()
            except queue.Empty:
                pass

            if lidar_data and isinstance(lidar_data, tuple) and len(lidar_data) == 2:
                lidar_ts, costmap_payload = lidar_data
                # Ensure the payload is a numpy array (it should be)
                if isinstance(costmap_payload, np.ndarray):
                    costmap_img = costmap_payload
            else:
                lidar_ts = 0.0 # Fallback
            # -----------------------------------------------------------

            vision_objects = shared_state.get('vision_detected_objects', [])

            # --- D. UPDATE LOCAL COSTMAP (TEMPORAL BUFFER) ---
            # Most logic has been moved to lidar_process (4-B Update)

            # If lidar is completely disabled or dead, we still need a blank map to draw vision objects on
            if not getattr(cfg, 'ENABLE_LIDAR', True):
                costmap_img.fill(127)

            if vision_objects and costmap_ready:
                # If we are using vision-only or fused, draw vision objects on whatever map we have
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
            # 3-A UPDATE: Refactored modular state machine

            # Helper to execute task routing logic
            def execute_task1(task_state, lat, lon, returning):
                if task_state == "TASK1_APPROACH": return "TASK1_STATE_ENTER", None, None, returning

                targets = {
                    "TASK1_STATE_ENTER": (getattr(cfg, 'T1_GATE_ENTER_LAT', 0), getattr(cfg, 'T1_GATE_ENTER_LON', 0)),
                    "TASK1_STATE_MID": (getattr(cfg, 'T1_GATE_MID_LAT', 0), getattr(cfg, 'T1_GATE_MID_LON', 0)),
                    "TASK1_STATE_EXIT": (getattr(cfg, 'T1_GATE_EXIT_LAT', 0), getattr(cfg, 'T1_GATE_EXIT_LON', 0)),
                    "TASK1_RETURN_MID": (getattr(cfg, 'T1_GATE_MID_LAT', 0), getattr(cfg, 'T1_GATE_MID_LON', 0)),
                    "TASK1_RETURN_ENTER": (getattr(cfg, 'T1_GATE_ENTER_LAT', 0), getattr(cfg, 'T1_GATE_ENTER_LON', 0))
                }

                t_lat, t_lon = targets.get(task_state, (None, None))
                if t_lat and nav.haversine(lat, lon, t_lat, t_lon) < 2.0:
                    if task_state == "TASK1_STATE_ENTER": task_state = "TASK1_STATE_MID"
                    elif task_state == "TASK1_STATE_MID": task_state = "TASK1_STATE_EXIT"
                    elif task_state == "TASK1_STATE_EXIT": task_state = "TASK1_RETURN_MID" if returning else "TASK2_START"
                    elif task_state == "TASK1_RETURN_MID": task_state = "TASK1_RETURN_ENTER"
                    elif task_state == "TASK1_RETURN_ENTER": task_state = "FINISHED"
                return task_state, t_lat, t_lon, returning

            def execute_task2(task_state, lat, lon):
                targets = {
                    "TASK2_START": (getattr(cfg, 'T2_ZONE_ENTRY_LAT', 0), getattr(cfg, 'T2_ZONE_ENTRY_LON', 0), "TASK2_GO_TO_MID"),
                    "TASK2_GO_TO_MID": (getattr(cfg, 'T2_ZONE_MID_LAT', 0), getattr(cfg, 'T2_ZONE_MID_LON', 0), "TASK2_GO_TO_END"),
                    "TASK2_GO_TO_END": (getattr(cfg, 'T2_ZONE_END_LAT', 0), getattr(cfg, 'T2_ZONE_END_LON', 0), "TASK3_APPROACH"),
                }

                if task_state in targets:
                    t_lat, t_lon, next_state = targets[task_state]
                    if nav.haversine(lat, lon, t_lat, t_lon) < 2.0:
                        task_state = next_state
                    return task_state, t_lat, t_lon

                return task_state, None, None

            def execute_task3(task_state, lat, lon):
                if task_state == "TASK3_APPROACH": return "T3_START", None, None

                targets = {
                    "T3_START": (getattr(cfg, 'T3_START_LAT', 0), getattr(cfg, 'T3_START_LON', 0), "T3_MID" if getattr(cfg, 'ENABLE_TASK3', True) else "FINISHED"),
                    "T3_MID": (getattr(cfg, 'T3_MID_LAT', 0), getattr(cfg, 'T3_MID_LON', 0), "TASK3_SEARCH_KAMIKAZE")
                }

                if task_state in targets:
                    t_lat, t_lon, next_state = targets[task_state]
                    if nav.haversine(lat, lon, t_lat, t_lon) < 2.0:
                        task_state = next_state
                    return task_state, t_lat, t_lon

                return task_state, None, None

            # Main State Router
            target_lat = None
            target_lon = None

            if "TASK1" in mevcut_gorev or mevcut_gorev == "FINISHED":
                if mevcut_gorev == "FINISHED":
                    if not finished_printed:
                        print("[TASK1] MISSION COMPLETE")
                        finished_printed = True
                    controller.set_servo(cfg.SOL_MOTOR, 1500)
                    controller.set_servo(cfg.SAG_MOTOR, 1500)
                else:
                    mevcut_gorev, target_lat, target_lon, returning_home = execute_task1(mevcut_gorev, ida_enlem, ida_boylam, returning_home)

            elif "TASK2" in mevcut_gorev:
                mevcut_gorev, target_lat, target_lon = execute_task2(mevcut_gorev, ida_enlem, ida_boylam)

            elif "T3" in mevcut_gorev or mevcut_gorev == "TASK3_APPROACH":
                if mevcut_gorev == "TASK3_SEARCH_KAMIKAZE":
                    found_target = False
                    for obj in vision_objects:
                        # Assuming CIDs: 3=Red, 4=Green, 10=Black (Adjust as per dataset)
                        if obj.get('cid') in [3, 4, 10]:
                            found_target = True
                            dist_m = obj.get('dist', 10.0)

                            # Calculate the global GPS of the buoy to set as target
                            pixel_offset = (obj['cx'] - (1280 / 2)) / 1280.0
                            angle_offset_deg = pixel_offset * getattr(cfg, 'CAM_HFOV', 110.0)
                            obj_bearing = (magnetic_heading + angle_offset_deg) % 360
                            target_lat, target_lon = calculate_obj_gps(ida_enlem, ida_boylam, dist_m, obj_bearing)

                            # Check collision condition
                            if dist_m < 1.0 or obj.get('area', 0) > 300000: # Bounding box fills screen or very close
                                print("[TASK3] KAMIKAZE COLLISION CONFIRMED! RETURNING HOME.")
                                returning_home = True
                                mevcut_gorev = "TASK1_STATE_EXIT" # Triggers return sequence
                            break

                    if not found_target:
                        # Spin slowly to search
                        spot_pwm = getattr(cfg, 'SPOT_TURN_PWM', 150)
                        controller.set_servo(cfg.SOL_MOTOR, 1500 + spot_pwm)
                        controller.set_servo(cfg.SAG_MOTOR, 1500 - spot_pwm - extra)
                        # Ensure we don't drop into A* or PID below while spinning
                        target_lat, target_lon = None, None
                else:
                    mevcut_gorev, target_lat, target_lon = execute_task3(mevcut_gorev, ida_enlem, ida_boylam)

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
                controller.reset_all_servos()
            else:
                # 1. Reactive Avoidance
                if center_danger:
                    if not acil_durum_aktif_mi:
                        # Vector-Assisted Shock Brake
                        # Steer hard away from obstacle, full reverse
                        steer_dir = 1900 if left_d > right_d else 1100
                        brake_pwm = cfg.BASE_PWM - getattr(cfg, 'ESCAPE_PWM', 300)

                        controller.set_servo(cfg.STEER_SERVO, steer_dir)
                        controller.set_servo(cfg.SOL_MOTOR, brake_pwm)
                        controller.set_servo(cfg.SAG_MOTOR, brake_pwm)
                        controller.set_servo(cfg.FRONT_SOL_MOTOR, brake_pwm)
                        controller.set_servo(cfg.FRONT_SAG_MOTOR, brake_pwm)
                        time.sleep(0.4)
                        acil_durum_aktif_mi = True
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
                            # Unified Spot Turn Logic
                            spot_pwm = getattr(cfg, 'SPOT_TURN_PWM', 200)
                            fwd = 1500 + spot_pwm
                            rev = 1500 - spot_pwm

                            if aci_farki > 0: # Turn Right
                                controller.set_servo(cfg.STEER_SERVO, 1900)
                                controller.set_servo(cfg.SOL_MOTOR, fwd)
                                controller.set_servo(cfg.SAG_MOTOR, rev)
                                controller.set_servo(cfg.FRONT_SOL_MOTOR, fwd)
                                controller.set_servo(cfg.FRONT_SAG_MOTOR, rev)
                            else: # Turn Left
                                controller.set_servo(cfg.STEER_SERVO, 1100)
                                controller.set_servo(cfg.SOL_MOTOR, rev)
                                controller.set_servo(cfg.SAG_MOTOR, fwd)
                                controller.set_servo(cfg.FRONT_SOL_MOTOR, rev)
                                controller.set_servo(cfg.FRONT_SAG_MOTOR, fwd)
                        else:
                            current_path = None

                            # Use A* ONLY for Task 2
                            if "TASK2" in mevcut_gorev:
                                # Run Planner
                                # --- 1-C UPDATE: Costmap Cropping for Faster A* ---
                                crop_radius_m = 10.0 # Only look at a 20m x 20m window around the boat
                                crop_radius_px = int(crop_radius_m / COSTMAP_RES_M_PER_PX)

                                cw, ch = COSTMAP_SIZE_PX[0] // 2, COSTMAP_SIZE_PX[1] // 2
                                rx_px = int(cw + ((robot_x - costmap_center_m[0]) / COSTMAP_RES_M_PER_PX))
                                ry_px = int(ch - ((robot_y - costmap_center_m[1]) / COSTMAP_RES_M_PER_PX))

                                x_min = max(0, rx_px - crop_radius_px)
                                x_max = min(COSTMAP_SIZE_PX[0], rx_px + crop_radius_px)
                                y_min = max(0, ry_px - crop_radius_px)
                                y_max = min(COSTMAP_SIZE_PX[1], ry_px + crop_radius_px)

                                cropped_costmap = costmap_img[y_min:y_max, x_min:x_max]
                                cropped_center_m = (
                                    costmap_center_m[0] + ((x_min + x_max)/2 - cw) * COSTMAP_RES_M_PER_PX,
                                    costmap_center_m[1] - ((y_min + y_max)/2 - ch) * COSTMAP_RES_M_PER_PX
                                )
                                cropped_size_px = (x_max - x_min, y_max - y_min)

                                nav_map, _ = planner.get_inflated_nav_map(cropped_costmap, ignore_green=(mevcut_gorev == "TASK2_GREEN_MARKER_FOUND"))

                                plan_timer += 1
                                if plan_timer > 4:
                                    plan_timer = 0
                                    if tx_world is not None:
                                        if planner.check_line_of_sight((robot_x, robot_y), (tx_world, ty_world), nav_map, cropped_center_m, COSTMAP_RES_M_PER_PX, cropped_size_px):
                                            current_path = [(robot_x, robot_y), (tx_world, ty_world)]
                                        else:
                                            new_path = planner.get_path_plan((robot_x, robot_y), (tx_world, ty_world), nav_map, cropped_center_m, COSTMAP_RES_M_PER_PX, cropped_size_px)
                                            if new_path: current_path = new_path
                                # ------------------------------------------------

                            # If we have an A* path (Task 2 only), follow it with Pure Pursuit
                            if current_path:
                                base_pwm = getattr(cfg, 'BASE_PWM', 1500)
                                if mevcut_gorev.startswith("T3_"): base_pwm += getattr(cfg, 'T3_SPEED_PWM', 100)

                                fwd_pwm, yaw_corr, raw_target, current_error, pruned_path = planner.pure_pursuit_control(
                                    robot_x, robot_y, robot_yaw, current_path, current_speed=0, base_speed=base_pwm, prev_error=prev_heading_error)

                                current_path = pruned_path
                                prev_heading_error = current_error
                                failsafe_active = False
                                path_lost_time = None

                                apply_motor_mixer(controller, fwd_pwm, yaw_corr)

                            # If no path, or we are NOT in Task 2 (meaning Task 1 or 3), use PID Direct Drive
                            else:
                                if target_lat is not None and target_lon is not None:
                                    if path_lost_time is None:
                                        path_lost_time = time.time()

                                    # Allow unlimited grace period if we are intentionally skipping A* (Task 1 & 3)
                                    if ("TASK2" not in mevcut_gorev) or (time.time() - path_lost_time < 5.0):
                                        failsafe_active = True

                                        threshold = getattr(cfg, 'SPOT_TURN_THRESHOLD', 45.0)

                                        if abs(aci_farki) > threshold:
                                            # Spot Turn Recovery
                                            spot_pwm = getattr(cfg, 'SPOT_TURN_PWM', 200)
                                            fwd = 1500 + spot_pwm
                                            rev = 1500 - spot_pwm

                                            if aci_farki > 0: # Turn Right
                                                controller.set_servo(cfg.STEER_SERVO, 1900)
                                                controller.set_servo(cfg.SOL_MOTOR, fwd)
                                                controller.set_servo(cfg.SAG_MOTOR, rev)
                                                controller.set_servo(cfg.FRONT_SOL_MOTOR, fwd)
                                                controller.set_servo(cfg.FRONT_SAG_MOTOR, rev)
                                            else: # Turn Left
                                                controller.set_servo(cfg.STEER_SERVO, 1100)
                                                controller.set_servo(cfg.SOL_MOTOR, rev)
                                                controller.set_servo(cfg.SAG_MOTOR, fwd)
                                                controller.set_servo(cfg.FRONT_SOL_MOTOR, rev)
                                                controller.set_servo(cfg.FRONT_SAG_MOTOR, fwd)
                                        else:
                                            base_pwm = getattr(cfg, 'BASE_PWM', 1500)
                                            if "TASK3" in mevcut_gorev or mevcut_gorev.startswith("T3_"):
                                                base_pwm += getattr(cfg, 'T3_SPEED_PWM', 100)
                                            else:
                                                base_pwm += getattr(cfg, 'CRUISE_PWM', 80)

                                            # Full PID controller for direct steering
                                            kp = 1.5
                                            ki = 0.05
                                            kd = 0.5

                                            error = aci_farki

                                            if abs(error) < 15.0:
                                                direct_drive_integral += error
                                            else:
                                                direct_drive_integral *= 0.9

                                            windup_limit = 500.0
                                            direct_drive_integral = max(-windup_limit, min(windup_limit, direct_drive_integral))

                                            derivative = error - direct_drive_prev_error
                                            direct_drive_prev_error = error

                                            yaw_corr = (error * kp) + (direct_drive_integral * ki) + (derivative * kd)

                                            apply_motor_mixer(controller, base_pwm, yaw_corr)
                                    else:
                                        # Stop if grace period exceeded and still no path (Task 2 only)
                                        controller.reset_all_servos()
                                else:
                                    controller.reset_all_servos()

            # Record final PWMs
            shared_state['motor_pwm_left'] = controller.get_servo_pwm(cfg.SOL_MOTOR)
            shared_state['motor_pwm_right'] = controller.get_servo_pwm(cfg.SAG_MOTOR)

            elapsed = time.time() - start_time
            if elapsed < 0.02: time.sleep(0.02 - elapsed)

    except Exception as e:
        print(f"[NAV_PROCESS][ERROR] Brain crashed: {e}")
    finally:
        print("[NAV_PROCESS] Shutting down...")
        try:
            controller.reset_all_servos()
            controller.disarm_vehicle()
        except: pass
