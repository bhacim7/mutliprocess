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


def apply_motor_mixer(controller, forward_pwm, yaw_pwm):
    """
    Control Mixer: Distributes Forward Effort and Yaw Effort across 5 channels.
    Implements Steering Deadband for smooth turns and push-pull differential.
    """
    base = getattr(cfg, 'BASE_PWM', 1500)

    # 1. Yaw Effort mappings
    # Map Yaw to Steering Servo (Direct proportion)
    steer_out = base + yaw_pwm
    steer_max = getattr(cfg, 'STEER_MAX_PWM', 1900)
    steer_min = getattr(cfg, 'STEER_MIN_PWM', 1100)
    steer_out = np.clip(steer_out, steer_min, steer_max)

    # 2. Differential Thrust (Continuous micro-corrections, removed deadband)
    # Applying differential thrust continuously prevents lateral drift by fighting wind/current smoothly
    diff_thrust = yaw_pwm * 1.0  # Slight dampening factor for thruster diff

    # 3. Calculate Individual Thrusters
    # Evasive Braking override: If we are actively braking (reverse thrust),
    # we DO NOT want positive differential thrust pushing us forward on the outside.
    if forward_pwm < 1450:  # Actively reversing
        rear_left = int(forward_pwm)
        rear_right = int(forward_pwm)
        front_left = int(forward_pwm)
        front_right = int(forward_pwm)
    else:
        # Standard turning
        # Right turn (yaw_pwm > 0): Left motors spin faster, Right motors slower
        rear_left = int(forward_pwm + diff_thrust)
        rear_right = int(forward_pwm - diff_thrust)

        # Front thrusters (keep forward effort high, apply smaller diff)
        front_left = int(forward_pwm + (diff_thrust * 0.5))
        front_right = int(forward_pwm - (diff_thrust * 0.5))

    # Clamp all thrusters
    rear_left = int(np.clip(rear_left, 1100, 1900))
    rear_right = int(np.clip(rear_right, 1100, 1900))
    front_left = int(np.clip(front_left, 1100, 1900))
    front_right = int(np.clip(front_right, 1100, 1900))

    # Command the hardware
    controller.set_servo(getattr(cfg, 'SOL_MOTOR', 1), rear_left)
    controller.set_servo(getattr(cfg, 'SAG_MOTOR', 3), rear_right)
    controller.set_servo(getattr(cfg, 'FRONT_SOL_MOTOR', 2), front_left)
    controller.set_servo(getattr(cfg, 'FRONT_SAG_MOTOR', 4), front_right)
    controller.set_servo(getattr(cfg, 'FRONT_STEER_SERVO', 5), int(steer_out))


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
    # PID Control Variables for Direct Drive
    direct_drive_integral = 0.0
    direct_drive_prev_error = 0.0

    current_path = []
    plan_timer = 0
    prev_heading_error = 0.0
    prev_pp_target = None
    hybrid_local_target = None
    force_initial_alignment = False
    prev_target_lat = None
    prev_target_lon = None
    start_lat = None
    start_lon = None
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

    # Task 3 Search State variables
    t3_search_state = "INIT_SEARCH"
    t3_original_heading = None
    t3_search_move_target = None

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
                        if idx == 1:
                            cfg.T1_GATE_ENTER_LAT = lat; cfg.T1_GATE_ENTER_LON = lon
                        elif idx == 2:
                            cfg.T1_GATE_MID_LAT = lat; cfg.T1_GATE_MID_LON = lon
                        elif idx == 3:
                            cfg.T1_GATE_EXIT_LAT = lat; cfg.T1_GATE_EXIT_LON = lon
                        elif idx == 4:
                            cfg.T2_ZONE_ENTRY_LAT = lat; cfg.T2_ZONE_ENTRY_LON = lon
                        elif idx == 5:
                            cfg.T2_ZONE_MID_LAT = lat; cfg.T2_ZONE_MID_LON = lon
                        elif idx == 6:
                            cfg.T2_ZONE_END_LAT = lat; cfg.T2_ZONE_END_LON = lon
                        elif idx == 7:
                            cfg.T3_START_LAT = lat; cfg.T3_START_LON = lon
                        elif idx == 8:
                            cfg.T3_MID_LAT = lat; cfg.T3_MID_LON = lon
                        print(f"[NAV_PROCESS] Updated GPS Point {idx}")
                    elif cmd_str == "set_task":
                        new_task = cmd.get("task_name")
                        if new_task:
                            shared_state['current_task'] = new_task
                            print(f"[NAV_PROCESS] Task updated to {new_task}")
                    elif cmd_str == "set_manual":
                        val = cmd.get("value")
                        if val is True:
                            shared_state['manual_mode'] = True
                            print("[NAV_PROCESS] 🎮 Switched to MANUAL mode.")
                        elif val is False:
                            shared_state['manual_mode'] = False
                            print("[NAV_PROCESS] 🤖 Switched to AUTO mode.")
            except:
                pass

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
                lidar_ts = 0.0  # Fallback
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
                    if "TASK2" in mevcut_gorev and obj.get('cid') not in [1, 3]:
                        continue

                    dist_m = obj.get('dist', 0)
                    if 0 < dist_m < 15.0:
                        pixel_offset = (obj.get('cx', 1280 / 2) - (1280 / 2)) / 1280.0
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
            def execute_task1(task_state, lat, lon):
                if task_state == "TASK1_APPROACH": return "TASK1_STATE_ENTER", None, None

                targets = {
                    "TASK1_STATE_ENTER": (getattr(cfg, 'T1_GATE_ENTER_LAT', 0), getattr(cfg, 'T1_GATE_ENTER_LON', 0)),
                    "TASK1_STATE_MID": (getattr(cfg, 'T1_GATE_MID_LAT', 0), getattr(cfg, 'T1_GATE_MID_LON', 0)),
                    "TASK1_STATE_EXIT": (getattr(cfg, 'T1_GATE_EXIT_LAT', 0), getattr(cfg, 'T1_GATE_EXIT_LON', 0))
                }

                t_lat, t_lon = targets.get(task_state, (None, None))
                if t_lat and nav.haversine(lat, lon, t_lat, t_lon) < 2.0:
                    if task_state == "TASK1_STATE_ENTER":
                        task_state = "TASK1_STATE_MID"
                    elif task_state == "TASK1_STATE_MID":
                        task_state = "TASK1_STATE_EXIT"
                    elif task_state == "TASK1_STATE_EXIT":
                        task_state = "TASK2_START"
                return task_state, t_lat, t_lon

            def execute_task2(task_state, lat, lon):
                targets = {
                    "TASK2_START": (getattr(cfg, 'T2_ZONE_ENTRY_LAT', 0), getattr(cfg, 'T2_ZONE_ENTRY_LON', 0),
                                    "TASK2_GO_TO_MID"),
                    "TASK2_GO_TO_MID": (getattr(cfg, 'T2_ZONE_MID_LAT', 0), getattr(cfg, 'T2_ZONE_MID_LON', 0),
                                        "TASK2_GO_TO_END"),
                    "TASK2_GO_TO_END": (getattr(cfg, 'T2_ZONE_END_LAT', 0), getattr(cfg, 'T2_ZONE_END_LON', 0),
                                        "T3_START"),
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
                    "T3_START": (getattr(cfg, 'T3_START_LAT', 0), getattr(cfg, 'T3_START_LON', 0),
                                 "T3_MID" if getattr(cfg, 'ENABLE_TASK3', True) else "FINISHED"),
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
            skip_default_nav = False

            if "TASK1" in mevcut_gorev or mevcut_gorev == "FINISHED":
                if mevcut_gorev == "FINISHED":
                    if not finished_printed:
                        print("[MISSION] ALL TASKS COMPLETE")
                        finished_printed = True
                    apply_motor_mixer(controller, 1500, 0)

                else:
                    mevcut_gorev, target_lat, target_lon = execute_task1(mevcut_gorev, ida_enlem, ida_boylam)

            elif "TASK2" in mevcut_gorev:
                mevcut_gorev, target_lat, target_lon = execute_task2(mevcut_gorev, ida_enlem, ida_boylam)

            elif "T3" in mevcut_gorev or "TASK3" in mevcut_gorev:
                if mevcut_gorev == "TASK3_SEARCH_KAMIKAZE":
                    found_target = False
                    target_color = getattr(cfg, 'TASK3_KAMIKAZE_COLOR', 'red').lower()
                    target_cid = 0
                    if target_color == "yellow": target_cid = 1
                    elif target_color == "black": target_cid = 2
                    elif target_color == "orange": target_cid = 3
                    elif target_color == "green": target_cid = 4

                    for obj in vision_objects:
                        if obj.get('cid') == target_cid:
                            found_target = True
                            dist_m = obj.get('dist', 10.0)

                            # Visual Servoing logic
                            # Prevent GPS PID from interfering by setting targets to None
                            target_lat, target_lon = None, None

                            pixel_error = obj['cx'] - (1280 / 2) # Assuming 1280 width (ZED HD720)

                            # Simple P controller for pixel error to steering PWM
                            kp_pixel = getattr(cfg, 'Kp_PIXEL', 0.3)

                            # Use inversion toggle to fix the circling bug
                            if getattr(cfg, 'TASK3_INVERT_STEERING', False):
                                steering_correction = -pixel_error * kp_pixel
                            else:
                                steering_correction = pixel_error * kp_pixel

                            base_pwm = getattr(cfg, 'BASE_PWM', 1500) + getattr(cfg, 'T3_SPEED_PWM', 100)
                            apply_motor_mixer(controller, base_pwm, steering_correction)

                            # Check collision condition
                            if dist_m < 1.0 or obj.get('area', 0) > 300000: # Bounding box fills screen or very close
                                print("[TASK3] KAMIKAZE COLLISION CONFIRMED! STOPPING VEHICLE.")
                                apply_motor_mixer(controller, 1500, 0)
                                mevcut_gorev = "FINISHED" # Finish mission
                            break

                    if not found_target:
                        target_lat, target_lon = None, None # Default to not using GPS PID unless moving
                        spot_pwm = getattr(cfg, 'SPOT_TURN_PWM', 150)

                        if t3_search_state == "INIT_SEARCH":
                            t3_original_heading = magnetic_heading
                            t3_search_state = "PAN_LEFT"

                        elif t3_search_state == "PAN_LEFT":
                            target_h = (t3_original_heading - 45) % 360
                            diff = nav.signed_angle_difference(magnetic_heading, target_h)
                            if abs(diff) < 5.0:
                                apply_motor_mixer(controller, 1500, 0)
                                t3_search_state = "PAN_RIGHT"
                            else:
                                apply_motor_mixer(controller, 1500, -spot_pwm if diff < 0 else spot_pwm)

                        elif t3_search_state == "PAN_RIGHT":
                            target_h = (t3_original_heading + 45) % 360
                            diff = nav.signed_angle_difference(magnetic_heading, target_h)
                            if abs(diff) < 5.0:
                                apply_motor_mixer(controller, 1500, 0)
                                t3_search_state = "CALC_MOVE_L60"
                            else:
                                apply_motor_mixer(controller, 1500, spot_pwm if diff > 0 else -spot_pwm)

                        elif t3_search_state == "CALC_MOVE_L60":
                            target_h = (t3_original_heading - 60) % 360
                            t_lat, t_lon = calculate_obj_gps(ida_enlem, ida_boylam, 3.0, target_h)
                            t3_search_move_target = (t_lat, t_lon)
                            t3_search_state = "MOVE_L60"

                        elif t3_search_state == "MOVE_L60":
                            target_lat, target_lon = t3_search_move_target
                            dist = nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon)
                            if dist < 1.5:
                                apply_motor_mixer(controller, 1500, 0)
                                t3_search_state = "PAN_LEFT_2"
                                target_lat, target_lon = None, None

                        elif t3_search_state == "PAN_LEFT_2":
                            target_h = (t3_original_heading - 45) % 360
                            diff = nav.signed_angle_difference(magnetic_heading, target_h)
                            if abs(diff) < 5.0:
                                apply_motor_mixer(controller, 1500, 0)
                                t3_search_state = "PAN_RIGHT_2"
                            else:
                                apply_motor_mixer(controller, 1500, -spot_pwm if diff < 0 else spot_pwm)

                        elif t3_search_state == "PAN_RIGHT_2":
                            target_h = (t3_original_heading + 45) % 360
                            diff = nav.signed_angle_difference(magnetic_heading, target_h)
                            if abs(diff) < 5.0:
                                apply_motor_mixer(controller, 1500, 0)
                                t3_search_state = "CALC_MOVE_R60"
                            else:
                                apply_motor_mixer(controller, 1500, spot_pwm if diff > 0 else -spot_pwm)

                        elif t3_search_state == "CALC_MOVE_R60":
                            target_h = (t3_original_heading + 60) % 360
                            t_lat, t_lon = calculate_obj_gps(ida_enlem, ida_boylam, 3.0, target_h)
                            t3_search_move_target = (t_lat, t_lon)
                            t3_search_state = "MOVE_R60"

                        elif t3_search_state == "MOVE_R60":
                            target_lat, target_lon = t3_search_move_target
                            dist = nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon)
                            if dist < 1.5:
                                apply_motor_mixer(controller, 1500, 0)
                                t3_search_state = "PAN_LEFT"
                                target_lat, target_lon = None, None

                    # Only skip default nav if we didn't set a GPS target to drive to during the search phase
                    if target_lat is None:
                        skip_default_nav = True
                else:
                    mevcut_gorev, target_lat, target_lon = execute_task3(mevcut_gorev, ida_enlem, ida_boylam)

            # Sync State
            shared_state['current_task'] = mevcut_gorev

            # --- F. NAVIGATION CALCULATIONS & HYBRID LOGIC ---
            aci_farki = 0.0
            control_error = 0.0
            adviced_course = 0.0

            if target_lat is not None:
                if (target_lat != prev_target_lat or target_lon != prev_target_lon):
                    force_initial_alignment = True
                    prev_target_lat = target_lat
                    prev_target_lon = target_lon
                    # Reset start_lat so we can capture it
                    start_lat = None
                    start_lon = None

                # Fix 1: Do not lock the start position if the GPS is 0.0 (uninitialized)
                # This prevents the boat from drawing a line from the coast of Africa.
                # It continually attempts to acquire the GPS coordinates if they were initially 0.0
                if start_lat is None and ida_enlem != 0.0 and ida_boylam != 0.0:
                    start_lat = ida_enlem
                    start_lon = ida_boylam

                # Calculate standard bearing and distance to the final target
                adviced_course = nav.calculate_bearing(ida_enlem, ida_boylam, target_lat, target_lon)
                hedefe_mesafe = nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon)

                # Line of Sight (LOS) Guidance Logic (for non-A* paths)
                # If we have a valid start point, we create a virtual target (rabbit) on the line
                # to correct for cross-track error.
                # Fix 2: Disable LOS if we are within 6 meters of the target. This prevents the boat
                # from doing a U-turn or steering wildly backwards if it slightly overshoots the line.
                if start_lat is not None and start_lon is not None and getattr(cfg, 'ENABLE_LOS_GUIDANCE', True) and hedefe_mesafe > 3.5:
                    # How far off the ideal line are we?
                    xte = nav.calculate_cross_track_error(start_lat, start_lon, ida_enlem, ida_boylam, target_lat, target_lon)

                    # LOS lookahead distance (the carrot distance on the line)
                    # Scales with distance, but kept within sane bounds (e.g., look 4-10m ahead)
                    los_lookahead = max(4.0, min(10.0, hedefe_mesafe * 0.5))

                    # Calculate the bearing of the ideal line itself
                    path_bearing = nav.calculate_bearing(start_lat, start_lon, target_lat, target_lon)

                    # Calculate LOS correction angle based on XTE
                    # K_los controls how aggressively we turn back to the line
                    k_los = getattr(cfg, 'LOS_KP', 1.5)
                    # Inverse tangent creates a smooth S-curve back to the line
                    correction_angle = math.degrees(math.atan2(k_los * xte, los_lookahead))

                    # The new desired heading points back to the line, rather than straight at the target
                    los_heading = (path_bearing - correction_angle) % 360

                    # Update control_error to chase the LOS heading
                    control_error = nav.signed_angle_difference(magnetic_heading, los_heading)
                    # Keep aci_farki pure for target bearing
                    aci_farki = nav.signed_angle_difference(magnetic_heading, adviced_course)
                else:
                    # Fallback to direct bearing if LOS is disabled or no start point
                    aci_farki = nav.signed_angle_difference(magnetic_heading, adviced_course)
                    control_error = aci_farki

                shared_state['angle_error'] = float(aci_farki)
                shared_state['control_error'] = float(control_error)
                shared_state['adviced_course'] = float(adviced_course)
                shared_state['target_dist'] = float(hedefe_mesafe)
                shared_state['target_lat'] = float(target_lat)
                shared_state['target_lon'] = float(target_lon)

            # Hybrid targeting setup
            tx_world, ty_world = None, None
            if costmap_ready and target_lat is not None:
                hybrid_local_target = None  # Reset
                gps_lookahead = 1.5
                tx_world = robot_x + (gps_lookahead * math.cos(robot_yaw + math.radians(-aci_farki)))
                ty_world = robot_y + (gps_lookahead * math.sin(robot_yaw + math.radians(-aci_farki)))

            # --- G. CONTROL LOGIC & MOTORS ---
            if manual_mode or not mission_started:
                apply_motor_mixer(controller, 1500, 0)

            else:
                # 1. Reactive Avoidance (Vector-Assisted Braking)
                if center_danger:
                    if not acil_durum_aktif_mi:
                        shock_brake_pwm = cfg.BASE_PWM - getattr(cfg, 'shock_pwm', 250)
                        # Braking: Reverse thrust, hard steer away
                        avoid_turn = 400 if (left_d > right_d) else -400
                        apply_motor_mixer(controller, shock_brake_pwm, avoid_turn)
                        time.sleep(0.1)
                        acil_durum_aktif_mi = True

                    escape_pwm = cfg.BASE_PWM - getattr(cfg, 'ESCAPE_PWM', 300)
                    avoid_turn = 400 if (left_d > right_d) else -400
                    apply_motor_mixer(controller, escape_pwm, avoid_turn)
                    time.sleep(0.4)

                    spot_turn_val = getattr(cfg, 'SPOT_TURN_PWM', 200)
                    if left_d > right_d:
                        apply_motor_mixer(controller, cfg.BASE_PWM, -spot_turn_val)
                    else:
                        apply_motor_mixer(controller, cfg.BASE_PWM, spot_turn_val)
                    time.sleep(0.3)
                    current_path = None  # Force replan
                    continue
                else:
                    acil_durum_aktif_mi = False

                # 2. Standard A* / Direct Drive
                if not skip_default_nav and target_lat is not None:
                    # Initial alignment logic
                    if force_initial_alignment and abs(aci_farki) < 15.0:
                        force_initial_alignment = False

                    should_force_alignment = force_initial_alignment

                    if should_force_alignment:
                        spot_pwm = getattr(cfg, 'SPOT_TURN_PWM', 200)
                        if aci_farki > 0:
                            apply_motor_mixer(controller, 1500, spot_pwm)
                        else:
                            apply_motor_mixer(controller, 1500, -spot_pwm)
                    else:
                        current_path = None

                        # Use A* ONLY for Task 2
                        if "TASK2" in mevcut_gorev:
                            # Run Planner
                            # --- 1-C UPDATE: Costmap Cropping for Faster A* ---
                            crop_radius_m = 10.0  # Only look at a 20m x 20m window around the boat
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
                                costmap_center_m[0] + ((x_min + x_max) / 2 - cw) * COSTMAP_RES_M_PER_PX,
                                costmap_center_m[1] - ((y_min + y_max) / 2 - ch) * COSTMAP_RES_M_PER_PX
                            )
                            cropped_size_px = (x_max - x_min, y_max - y_min)

                            nav_map, _ = planner.get_inflated_nav_map(cropped_costmap, ignore_green=(
                                        mevcut_gorev == "TASK2_GREEN_MARKER_FOUND"))

                            plan_timer += 1
                            if plan_timer > 4:
                                plan_timer = 0
                                if tx_world is not None:
                                    if planner.check_line_of_sight((robot_x, robot_y), (tx_world, ty_world), nav_map,
                                                                   cropped_center_m, COSTMAP_RES_M_PER_PX,
                                                                   cropped_size_px):
                                        current_path = [(robot_x, robot_y), (tx_world, ty_world)]
                                    else:
                                        new_path = planner.get_path_plan((robot_x, robot_y), (tx_world, ty_world),
                                                                         nav_map, cropped_center_m,
                                                                         COSTMAP_RES_M_PER_PX, cropped_size_px)
                                        if new_path: current_path = new_path
                            # ------------------------------------------------

                        # If we have an A* path (Task 2 only), follow it with Pure Pursuit
                        if current_path:
                            base_pwm = getattr(cfg, 'BASE_PWM', 1500)
                            if mevcut_gorev.startswith("T3_"): base_pwm += getattr(cfg, 'T3_SPEED_PWM', 100)

                            # Pure pursuit now returns base_speed and steering_correction instead of left/right pwms
                            p_base, p_steer, raw_target, current_error, pruned_path = planner.pure_pursuit_control(
                                robot_x, robot_y, robot_yaw, current_path, current_speed=base_pwm - 1500, base_speed=base_pwm,
                                prev_error=prev_heading_error)

                            current_path = pruned_path
                            prev_heading_error = current_error

                            failsafe_active = False
                            path_lost_time = None

                            apply_motor_mixer(controller, p_base, p_steer)

                        # If no path, or we are NOT in Task 2 (meaning Task 1 or 3), use PID Direct Drive
                        else:
                            if target_lat is not None and target_lon is not None:
                                # Always use Direct Drive PID (and spot turns) as a fallback when A* has no path
                                # This prevents the boat from freezing if A* inflation blocks the target
                                # or if initial alignment takes longer than 5 seconds.
                                failsafe_active = True

                                # --- FIX: Reintroduce spot turn if heading is severely off ---
                                threshold = getattr(cfg, 'SPOT_TURN_THRESHOLD', 45.0)

                                if abs(aci_farki) > threshold:
                                    spot_pwm = getattr(cfg, 'SPOT_TURN_PWM', 200)
                                    if aci_farki > 0:  # Target Right
                                        apply_motor_mixer(controller, 1500, spot_pwm)
                                    else:  # Target Left
                                        apply_motor_mixer(controller, 1500, -spot_pwm)
                                else:
                                    base_pwm = getattr(cfg, 'BASE_PWM', 1500)
                                    if "TASK3" in mevcut_gorev or mevcut_gorev.startswith("T3_"):
                                        base_pwm += getattr(cfg, 'T3_SPEED_PWM', 100)
                                    else:
                                        base_pwm += getattr(cfg, 'CRUISE_PWM', 80)

                                    # Full PID controller for direct steering
                                    kp = getattr(cfg, 'DIRECT_DRIVE_KP', 1.5)
                                    ki = getattr(cfg, 'DIRECT_DRIVE_KI', 0.05)
                                    kd = getattr(cfg, 'DIRECT_DRIVE_KD', 0.8)

                                    # Calculate terms
                                    error = control_error

                                    # --- 3-B UPDATE: Anti-Windup Logic ---
                                    # Only accumulate integral if the error is relatively small
                                    # This prevents massive windup when pushing against an obstacle or turning sharply
                                    anti_windup_deg = getattr(cfg, 'ANTI_WINDUP_DEG', 15.0)
                                    if abs(error) < anti_windup_deg:
                                        direct_drive_integral += error
                                    else:
                                        # Optional: Reset or decay the integral when outside the linear region
                                        direct_drive_integral *= 0.9

                                    # Add natural decay if error crosses zero (sign change) to prevent residual windup
                                    if (error > 0 and direct_drive_integral < 0) or (error < 0 and direct_drive_integral > 0):
                                        direct_drive_integral *= 0.5

                                    # Integral windup hard limit
                                    windup_limit = getattr(cfg, 'ANTI_WINDUP_LIMIT', 500.0)
                                    direct_drive_integral = max(-windup_limit, min(windup_limit, direct_drive_integral))
                                    # -------------------------------------

                                    derivative = error - direct_drive_prev_error
                                    direct_drive_prev_error = error

                                    steering_correction = (error * kp) + (direct_drive_integral * ki) + (
                                                derivative * kd)

                                    apply_motor_mixer(controller, base_pwm, steering_correction)

                elif not skip_default_nav:
                    apply_motor_mixer(controller, 1500, 0)

            # Record final PWMs
            # Note: We rely on the USVController object stub tracking state,
            # but we can push directly to shared_state for safety
            shared_state['motor_pwm_left'] = controller.get_servo_pwm(getattr(cfg, 'SOL_MOTOR', 1))
            shared_state['motor_pwm_right'] = controller.get_servo_pwm(getattr(cfg, 'SAG_MOTOR', 3))
            shared_state['motor_pwm_front_left'] = controller.get_servo_pwm(getattr(cfg, 'FRONT_SOL_MOTOR', 2))
            shared_state['motor_pwm_front_right'] = controller.get_servo_pwm(getattr(cfg, 'FRONT_SAG_MOTOR', 4))
            shared_state['motor_pwm_steer'] = controller.get_servo_pwm(getattr(cfg, 'FRONT_STEER_SERVO', 5))

            elapsed = time.time() - start_time
            if elapsed < 0.02: time.sleep(0.02 - elapsed)

    except Exception as e:
        print(f"[NAV_PROCESS][ERROR] Brain crashed: {e}")
    finally:
        print("[NAV_PROCESS] Shutting down...")
        try:
            apply_motor_mixer(controller, 1500, 0)

            controller.disarm_vehicle()
        except:
            pass
