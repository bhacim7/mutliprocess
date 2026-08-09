import time
import math
import os
import signal
import numpy as np
import cv2

# Hardware / Utils
import config as cfg
from hardware.MainSystem2 import USVController
import utils.navigasyon as nav
import utils.planner as planner
from utils.navigasyon import calculate_obj_gps

import queue
from utils.costmap_recorder import CostmapRecorder

# Servo channel assignment, read once from config. These used to be inlined as
# getattr(cfg, 'SOL_MOTOR', 1) etc. with fallbacks (1,3,2,4,5) that did not match
# either config.py (6,3,5,2,7) or MainSystem2's channel list.
CH_REAR_LEFT = getattr(cfg, 'SOL_MOTOR', 1)
CH_REAR_RIGHT = getattr(cfg, 'SAG_MOTOR', 3)
CH_FRONT_LEFT = getattr(cfg, 'FRONT_SOL_MOTOR', 2)
CH_FRONT_RIGHT = getattr(cfg, 'FRONT_SAG_MOTOR', 4)
CH_STEER = getattr(cfg, 'FRONT_STEER_SERVO', 5)


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

    # Check if we are doing a pure spot turn (stationary)
    is_spot_turn = (forward_pwm == 1500 and yaw_pwm != 0)

    if is_spot_turn:
        # Disable dynamic multiplier for pure spot turns so we don't multiply max outputs
        diff_thrust = yaw_pwm * 1.0
    else:
        # Dynamic multiplier: Boosts differential thrust at lower speeds where the steering servo is hydrodynamically weak.
        # At 1500 (stopped), multiplier is 2.0. At 1700 (CRUISE_PWM=200), multiplier is ~1.4. At 1850+ (max speed), multiplier is 1.0.
        diff_multiplier = 1.0 + max(0.0, (1850.0 - forward_pwm) / 350.0)
        diff_thrust = yaw_pwm * diff_multiplier

    # 3. Calculate Individual Thrusters
    # Evasive Braking override: If we are actively braking (reverse thrust),
    # we DO NOT want positive differential thrust pushing us forward on the outside.
    if forward_pwm < 1450:  # Actively reversing
        rear_left = int(forward_pwm)
        rear_right = int(forward_pwm)
        front_left = int(forward_pwm)
        front_right = int(forward_pwm)
    elif is_spot_turn:
        # Pure Pivot: Front and rear motors must use identical differential thrust
        # to ensure the boat twists perfectly around its center without moving backward.
        rear_left = int(forward_pwm + diff_thrust)
        rear_right = int(forward_pwm - diff_thrust)
        front_left = int(forward_pwm + diff_thrust)
        front_right = int(forward_pwm - diff_thrust)
    else:
        # Standard turning (forward movement)
        # Right turn (yaw_pwm > 0): Left motors spin faster, Right motors slower
        rear_left = int(forward_pwm + diff_thrust)
        rear_right = int(forward_pwm - diff_thrust)

        # Front thrusters (keep forward effort high, apply smaller diff to prevent crab-walking)
        front_left = int(forward_pwm + (diff_thrust * 0.5))
        front_right = int(forward_pwm - (diff_thrust * 0.5))

    # Clamp all thrusters
    rear_left = int(np.clip(rear_left, 1100, 1900))
    rear_right = int(np.clip(rear_right, 1100, 1900))
    front_left = int(np.clip(front_left, 1100, 1900))
    front_right = int(np.clip(front_right, 1100, 1900))

    # Command the hardware
    controller.set_servo(CH_REAR_LEFT, rear_left)
    controller.set_servo(CH_REAR_RIGHT, rear_right)
    controller.set_servo(CH_FRONT_LEFT, front_left)
    controller.set_servo(CH_FRONT_RIGHT, front_right)
    controller.set_servo(CH_STEER, int(steer_out))


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
    # The planner only ever looks at a A_STAR_CROP_RADIUS_M window around the boat, so
    # the map is kept robot-centred and exactly that size. The previous 4000x4000 (16 MB)
    # global map was cleared with fill(127) on every 50 Hz iteration - ~800 MB/s of memory
    # traffic for a region that was then cropped down to 400x400 anyway.
    COSTMAP_RES_M_PER_PX = getattr(cfg, 'COSTMAP_RES_M_PER_PX', 0.10)
    CROP_RADIUS_M = getattr(cfg, 'A_STAR_CROP_RADIUS_M', 20.0)
    LOCAL_SIZE = int((CROP_RADIUS_M * 2.0) / COSTMAP_RES_M_PER_PX)

    local_costmap = np.full((LOCAL_SIZE, LOCAL_SIZE), 127, dtype=np.uint8)

    # These follow whatever map we are currently holding. When lidar_process pushes a
    # costmap it comes with its own geometry (its own size, centred on world origin),
    # so they must not be hardcoded to the local map's constants.
    costmap_img = local_costmap
    costmap_size_px = (LOCAL_SIZE, LOCAL_SIZE)   # (width, height)
    costmap_center_m = (0.0, 0.0)
    costmap_ready = True

    def world_to_pixel(world_x, world_y):
        w, h = costmap_size_px
        dx_m = world_x - costmap_center_m[0]
        dy_m = world_y - costmap_center_m[1]
        px = int((w // 2) + (dx_m / COSTMAP_RES_M_PER_PX))
        py = int((h // 2) - (dy_m / COSTMAP_RES_M_PER_PX))
        if 0 <= px < w and 0 <= py < h:
            return (px, py)
        return None

    # 3. State Machine Variables
    # PID Control Variables for Direct Drive
    direct_drive_integral = 0.0
    direct_drive_prev_error = 0.0

    current_path = None
    path_progress_idx = 0
    active_controller = None          # 'PP' or 'PID' - used to reset stale derivative state
    plan_timer = 0
    prev_heading_error = 0.0
    hybrid_local_target = None
    force_initial_alignment = False
    alignment_started_at = None
    prev_target_lat = None
    prev_target_lon = None
    start_lat = None
    start_lon = None
    finished_printed = False

    acil_durum_aktif_mi = False

    # Local navigation frame: x = North, y = East, in metres from the first valid GPS fix.
    # This is the SAME frame the obstacles are plotted in (they are placed with
    # cos/sin of a compass bearing), so the map stays self-consistent.
    origin_lat = None
    origin_lon = None
    robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0

    last_loop_time = time.time()
    loop_dt = 0.02

    # Give up on the in-place spot turn after this long and let the normal controllers
    # take over, rather than pivoting forever on a bad heading.
    ALIGNMENT_TIMEOUT_S = 12.0

    # Store legacy PWM defaults
    extra = 50

    # Task 3 Search State variables
    t3_search_state = "INIT_SEARCH"
    t3_original_heading = None
    t3_search_move_target = None

    costmap_recorder = None
    if getattr(cfg, 'RECORD_COSTMAP', False):
        costmap_recorder = CostmapRecorder()
        costmap_recorder.register_exit_handlers()

    # The orchestrator watchdog kills a wedged nav process with p.terminate() (SIGTERM).
    # Python's default SIGTERM handling tears the process down immediately, so the
    # `finally` block below never runs and the thrusters HOLD their last commanded PWM
    # until a fresh nav process finishes booting its MAVLink link - potentially seconds,
    # at full throttle. Neutralise on the way out.
    # Installed after register_exit_handlers() and chained to it so the costmap recorder's
    # own SIGTERM save is not lost.
    _prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _on_terminate(signum, frame):
        print("[NAV_PROCESS] SIGTERM received - neutralising thrusters.")
        try:
            apply_motor_mixer(controller, 1500, 0)
            controller.disarm_vehicle()
        except Exception:
            pass
        if callable(_prev_sigterm):
            _prev_sigterm(signum, frame)
        else:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    try:
        signal.signal(signal.SIGTERM, _on_terminate)
    except (ValueError, AttributeError, OSError) as e:
        print(f"[NAV_PROCESS][WARNING] Could not install SIGTERM handler: {e}")

    try:
        while not shared_state['shutdown']:
            try:
                start_time = time.time()
                # Real elapsed time drives the PID/PP derivative + integral scaling. The loop
                # is nominally 50 Hz but the cycle where A* runs is measurably longer, and an
                # un-normalised D/I term silently changes gain with loop rate.
                loop_dt = min(0.5, max(0.001, start_time - last_loop_time))
                last_loop_time = start_time

                # --- A. READ SENSORS FROM HARDWARE ---
                ida_enlem, ida_boylam = controller.get_current_position()
                hf_data['gps_lat'].value = ida_enlem
                hf_data['gps_lon'].value = ida_boylam

                # Fetch and store horizontal speed
                try:
                    shared_state['horizontal_speed'] = controller.get_horizontal_speed()
                except AttributeError:
                    pass  # Fallback if get_horizontal_speed doesn't exist on dummy controller

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
                                cfg.T1_GATE_ENTER_LAT = lat;
                                cfg.T1_GATE_ENTER_LON = lon
                            elif idx == 2:
                                cfg.T1_GATE_MID_LAT = lat;
                                cfg.T1_GATE_MID_LON = lon
                            elif idx == 3:
                                cfg.T1_GATE_EXIT_LAT = lat;
                                cfg.T1_GATE_EXIT_LON = lon
                            elif idx == 4:
                                cfg.T2_ZONE_ENTRY_LAT = lat;
                                cfg.T2_ZONE_ENTRY_LON = lon
                            elif idx == 5:
                                cfg.T2_ZONE_MID_LAT = lat;
                                cfg.T2_ZONE_MID_LON = lon
                            elif idx == 6:
                                cfg.T2_ZONE_END_LAT = lat;
                                cfg.T2_ZONE_END_LON = lon
                            elif idx == 7:
                                cfg.T3_START_LAT = lat;
                                cfg.T3_START_LON = lon
                            elif idx == 8:
                                cfg.T3_MID_LAT = lat;
                                cfg.T3_MID_LON = lon
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
                        elif cmd_str == "set_target_color":
                            color = cmd.get("color")
                            if color:
                                shared_state['drone_target_color'] = color
                                print(f"[NAV_PROCESS] Drone target color updated to {color}")
                except:
                    pass

                # --- C. SYNC WITH SHARED STATE ---
                magnetic_heading = hf_data['magnetic_heading'].value
                mevcut_gorev = shared_state.get('current_task', 'TASK_1')
                manual_mode = shared_state.get('manual_mode', False)
                mission_started = shared_state.get('mission_started', True)

                # --- Local navigation frame (x = North, y = East, metres from first fix) ---
                # This replaces the ZED odometry. Two things were wrong with reading the pose
                # from shared_state['zed_x'/'zed_y']:
                #   1) init_params.coordinate_system was never set, so the ZED defaults to
                #      COORDINATE_SYSTEM.IMAGE (X=right, Y=DOWN, Z=FORWARD). The code took
                #      t_vec[1] and t_vec[0], i.e. heave and sway - the boat's forward travel
                #      (t_vec[2]) was never read at all, so robot_x/robot_y barely moved.
                #   2) ZED odometry lives in the camera's start-up frame, not a North-referenced
                #      one, while everything else here (target projection, obstacle plotting)
                #      assumes x=cos(bearing), y=sin(bearing) i.e. North/East.
                # With the lidar disabled every obstacle on the map is already GPS-derived, so
                # deriving the pose from GPS puts the boat and the obstacles in one frame with
                # no accumulating drift.
                if origin_lat is None and ida_enlem != 0.0 and ida_boylam != 0.0:
                    origin_lat, origin_lon = ida_enlem, ida_boylam
                    print(f"[NAV_PROCESS] Local frame origin set: {origin_lat:.7f}, {origin_lon:.7f}")

                if origin_lat is not None and ida_enlem != 0.0 and ida_boylam != 0.0:
                    _d = nav.haversine(origin_lat, origin_lon, ida_enlem, ida_boylam)
                    _b = math.radians(nav.calculate_bearing(origin_lat, origin_lon, ida_enlem, ida_boylam))
                    robot_x = _d * math.cos(_b)   # North
                    robot_y = _d * math.sin(_b)   # East

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

                got_lidar_map = False
                if lidar_data and isinstance(lidar_data, tuple) and len(lidar_data) == 2:
                    lidar_ts, costmap_payload = lidar_data
                    # Ensure the payload is a numpy array (it should be)
                    if isinstance(costmap_payload, np.ndarray) and costmap_payload.ndim == 2:
                        # Adopt the payload's OWN geometry. lidar_process builds its map at its
                        # own resolution/size centred on the world origin; assuming the local
                        # map's dimensions here made every crop index out of range (empty slice)
                        # and silently disabled A* altogether.
                        costmap_img = costmap_payload
                        costmap_size_px = (costmap_payload.shape[1], costmap_payload.shape[0])
                        costmap_center_m = (0.0, 0.0)
                        got_lidar_map = True
                else:
                    lidar_ts = 0.0  # Fallback
                # -----------------------------------------------------------

                vision_objects = shared_state.get('vision_detected_objects', [])

                # --- UPDATE SEPARATE COSTMAP RECORDER ---
                if costmap_recorder is not None:
                    # Use current raw gps location to feed global costmap
                    costmap_recorder.update(ida_enlem, ida_boylam, vision_objects)

                # --- D. UPDATE LOCAL COSTMAP (TEMPORAL BUFFER) ---
                # Most logic has been moved to lidar_process (4-B Update)

                # No fresh lidar map: fall back to the robot-centred local map and rebuild it
                # from this cycle's vision detections (which already persist for 5s inside
                # camera_process's ObjectMemoryManager).
                if not got_lidar_map:
                    costmap_img = local_costmap
                    costmap_size_px = (LOCAL_SIZE, LOCAL_SIZE)
                    costmap_center_m = (robot_x, robot_y)
                    costmap_img.fill(127)

                if vision_objects and costmap_ready:
                    # If we are using vision-only or fused, draw vision objects on whatever map we have
                    for obj in vision_objects:
                        if "TASK2" in mevcut_gorev and obj.get('cid') not in [1, 3]:
                            continue

                        obj_lat = obj.get('lat')
                        obj_lon = obj.get('lon')

                        if obj_lat and obj_lon and ida_enlem != 0:
                            real_dist = nav.haversine(ida_enlem, ida_boylam, obj_lat, obj_lon)
                            real_bearing = nav.calculate_bearing(ida_enlem, ida_boylam, obj_lat, obj_lon)

                            if 0 < real_dist < 15.0:
                                angle_in_world = math.radians(real_bearing)
                                obj_world_x = robot_x + (real_dist * math.cos(angle_in_world))
                                obj_world_y = robot_y + (real_dist * math.sin(angle_in_world))

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
                    if t_lat and nav.haversine(lat, lon, t_lat, t_lon) < 3.0:
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
                        if nav.haversine(lat, lon, t_lat, t_lon) < 3.0:
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

                # The state machine both advances mission state AND (for Task 3) issues motor
                # commands directly. Running it unguarded while disarmed meant that manually
                # driving past a waypoint silently advanced the mission and that Task 3 could
                # mark itself FINISHED from the dock.
                # The target waypoint IS still computed below when disarmed, so the GCS keeps
                # showing distance/bearing to the next point during pre-flight - only the
                # state transition is rolled back and the motors are left alone.
                mission_active = mission_started and not manual_mode
                gorev_before_router = mevcut_gorev

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
                    if mevcut_gorev == "TASK3_SEARCH_KAMIKAZE" and not mission_active:
                        # This whole branch drives the thrusters directly (pan/spot turns and
                        # the kamikaze run) and can declare the mission FINISHED. Never run it
                        # while disarmed or in manual.
                        skip_default_nav = True

                    elif mevcut_gorev == "TASK3_SEARCH_KAMIKAZE":
                        found_target = False
                        target_color = getattr(cfg, 'TASK3_KAMIKAZE_COLOR', 'red').lower()

                        if getattr(cfg, 'DRONE_ACTIVE', False):
                            target_color = shared_state.get('drone_target_color', target_color)
                            if target_color is not None:
                                target_color = target_color.lower()

                        target_cid = 0
                        if target_color == "yellow":
                            target_cid = 1
                        elif target_color == "black":
                            target_cid = 2
                        elif target_color == "orange":
                            target_cid = 3
                        elif target_color == "green":
                            target_cid = 4

                        # camera_process keeps a detection in memory (and therefore in this
                        # list) for 5 seconds after it was last actually seen. Steering on a
                        # 'cx' that old makes the boat chase where the target USED to be -
                        # which is what produced the circling behaviour that TASK3_INVERT_STEERING
                        # was added to mask. Only servo on genuinely fresh pixels.
                        T3_MAX_DETECTION_AGE_S = 0.3
                        now_ts = time.time()

                        for obj in vision_objects:
                            if obj.get('cid') == target_cid:
                                cx_val = obj.get('cx')
                                if cx_val is None:
                                    # No fresh pixel data for this tracked object this frame
                                    # (defensive: avoids KeyError if 'cx' was never populated)
                                    continue

                                last_seen = obj.get('last_seen')
                                if last_seen is None or (now_ts - last_seen) > T3_MAX_DETECTION_AGE_S:
                                    continue  # stale memory entry - keep searching instead

                                found_target = True
                                dist_m = obj.get('dist', 10.0)

                                # Visual Servoing logic
                                # Prevent GPS PID from interfering by setting targets to None
                                target_lat, target_lon = None, None

                                pixel_error = cx_val - (1280 / 2)  # Assuming 1280 width (ZED HD720)

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
                                if dist_m < 1.0 or obj.get('area', 0) > 300000:  # Bounding box fills screen or very close
                                    print("[TASK3] KAMIKAZE COLLISION CONFIRMED! STOPPING VEHICLE.")
                                    apply_motor_mixer(controller, 1500, 0)
                                    mevcut_gorev = "FINISHED"  # Finish mission
                                break

                        if not found_target:
                            target_lat, target_lon = None, None  # Default to not using GPS PID unless moving
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

                if not mission_active:
                    # Roll back any state advance the routers above made. Driving the boat
                    # manually past a waypoint (or simply sitting within its 3 m acceptance
                    # radius on the dock) used to walk the mission forward on its own.
                    # target_lat/target_lon are deliberately left as computed so section F
                    # still publishes distance/bearing to the next point for the GCS.
                    mevcut_gorev = gorev_before_router

                # Sync State
                shared_state['current_task'] = mevcut_gorev

                # --- F. NAVIGATION CALCULATIONS & HYBRID LOGIC ---
                aci_farki = 0.0
                control_error = 0.0
                adviced_course = 0.0

                if target_lat is not None:
                    if (target_lat != prev_target_lat or target_lon != prev_target_lon):
                        force_initial_alignment = True
                        alignment_started_at = time.time()
                        prev_target_lat = target_lat
                        prev_target_lon = target_lon
                        # Reset start_lat so we can capture it
                        start_lat = None
                        start_lon = None
                        current_path = None  # Discard old A* path when target changes
                        path_progress_idx = 0

                    # Fix 1: Do not lock the start position if the GPS is 0.0 (uninitialized)
                    # This prevents the boat from drawing a line from the coast of Africa.
                    # It continually attempts to acquire the GPS coordinates if they were initially 0.0
                    # Fix 3: Also do not lock the start position while we are still performing the initial spot-turn alignment.
                    # Capturing it while drifting/turning causes massive LOS path cross-track errors.
                    if start_lat is None and ida_enlem != 0.0 and ida_boylam != 0.0 and not force_initial_alignment:
                        start_lat = ida_enlem
                        start_lon = ida_boylam

                    # Calculate standard bearing and distance to the final target
                    adviced_course = nav.calculate_bearing(ida_enlem, ida_boylam, target_lat, target_lon)
                    hedefe_mesafe = nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon)

                    # Line of Sight (LOS) Guidance Logic (for non-A* paths)
                    # If we have a valid start point, we create a virtual target (rabbit) on the line
                    # to correct for cross-track error.
                    # Fix 2: Disable LOS dynamically based on speed. This prevents the boat
                    # from doing a U-turn or steering wildly backwards if it slightly overshoots the line.

                    # Dynamic speed factor calculation
                    base_pwm_current = getattr(cfg, 'BASE_PWM', 1500)
                    if "TASK3" in mevcut_gorev or mevcut_gorev.startswith("T3_"):
                        base_pwm_current += getattr(cfg, 'T3_SPEED_PWM', 100)
                    else:
                        base_pwm_current += getattr(cfg, 'CRUISE_PWM', 80)

                    speed_factor = max(0.0, (base_pwm_current - 1500) / 100.0)

                    # Dynamic LOS cutoff distance (e.g., 3.5m at base speed, ~8m at high speed)
                    dynamic_los_cutoff = 3.5 + (speed_factor * 1.5)

                    if start_lat is not None and start_lon is not None and getattr(cfg, 'ENABLE_LOS_GUIDANCE',
                                                                                   True) and hedefe_mesafe > dynamic_los_cutoff:
                        # How far off the ideal line are we?
                        xte = nav.calculate_cross_track_error(start_lat, start_lon, ida_enlem, ida_boylam, target_lat,
                                                              target_lon)

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

                        # HARD CLAMP: Prevent massive skidding. Never ask the boat to turn more than 30 degrees sideways
                        correction_angle = max(-30.0, min(30.0, correction_angle))

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
                    # Project the actual GPS target into the local world map
                    # Cap distance so the projected point always fits inside the local A* crop window
                    # (cfg.A_STAR_CROP_RADIUS_M is the SAME constant used below to build that window -
                    # keeping these tied together is what prevents the out-of-bounds-goal freeze from
                    # silently coming back if the crop radius is ever retuned).
                    _a_star_crop_radius_m = getattr(cfg, 'A_STAR_CROP_RADIUS_M', 20.0)
                    projection_dist = min(hedefe_mesafe, _a_star_crop_radius_m * 0.75)

                    # The local costmap is built using raw compass bearings mapped directly into math.cos/sin.
                    # We must plot the target exactly the same way we plot the vision obstacles.
                    angle_in_world = math.radians(adviced_course)

                    tx_world = robot_x + (projection_dist * math.cos(angle_in_world))
                    ty_world = robot_y + (projection_dist * math.sin(angle_in_world))

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
                        path_progress_idx = 0

                        # This branch blocks for ~0.8s and then `continue`s, skipping the
                        # heartbeat at the bottom of the loop. An obstacle that persists for
                        # more than HEARTBEAT_TIMEOUT_S (5s) - i.e. the boat is actually stuck,
                        # exactly when we least want it - made the watchdog treat nav as wedged
                        # and SIGTERM it, which skips the `finally` that neutralises the motors.
                        shared_state['nav_heartbeat'] = time.time()
                        continue
                    else:
                        acil_durum_aktif_mi = False

                    # 2. Standard A* / Direct Drive
                    if not skip_default_nav and target_lat is not None:
                        # Initial alignment logic
                        if force_initial_alignment:
                            if abs(aci_farki) < 15.0:
                                force_initial_alignment = False
                                alignment_started_at = None
                            elif alignment_started_at is not None and \
                                    (time.time() - alignment_started_at) > ALIGNMENT_TIMEOUT_S:
                                # Nothing else clears this flag, so a bad heading source (or a
                                # spot turn the thrusters cannot win against wind/current) used
                                # to leave the boat pivoting in place forever. Give up and let
                                # the normal controllers drive; they steer toward the target too.
                                print(f"[NAV_PROCESS][WARN] Initial alignment timed out after "
                                      f"{ALIGNMENT_TIMEOUT_S:.0f}s (err={aci_farki:.0f} deg). Proceeding.")
                                force_initial_alignment = False
                                alignment_started_at = None

                        should_force_alignment = force_initial_alignment

                        if should_force_alignment:
                            # YENİ EKLENEN KISIM: Hedefe yaklaştıkça yavaşlayan Oransal (P) Dönüş
                            base_spot_pwm = getattr(cfg, 'SPOT_TURN_PWM', 200)

                            # Hedefe 45 derece kala yavaşlamaya başla.
                            # abs(aci_farki) azaldıkça hiz_carpani küçülecek (0.0 ile 1.0 arası)
                            hiz_carpani = min(1.0, abs(aci_farki) / 45.0)

                            # Minimum PWM (örn: 60) tanımlıyoruz ki tekne son derecelerde su direncine takılıp durmasın
                            dinamik_spot_pwm = int(60 + (base_spot_pwm - 60) * hiz_carpani)

                            if aci_farki > 0:
                                apply_motor_mixer(controller, 1500, dinamik_spot_pwm)
                            else:
                                apply_motor_mixer(controller, 1500, -dinamik_spot_pwm)
                        else:
                            # Failsafe: Use A* for Task 2 always, or for all tasks if LOS guidance is explicitly disabled.
                            # Do NOT use A* during the specific TASK3_SEARCH_KAMIKAZE phase as it relies on specific local waypoints.
                            use_astar_planner = ("TASK2" in mevcut_gorev) or (not getattr(cfg, 'ENABLE_LOS_GUIDANCE',
                                                                                          True) and mevcut_gorev != "TASK3_SEARCH_KAMIKAZE")

                            if not use_astar_planner:
                                current_path = None
                                path_progress_idx = 0

                            if use_astar_planner:
                                # Run Planner
                                # --- 1-C UPDATE: Costmap Cropping for Faster A* ---
                                crop_radius_m = getattr(cfg, 'A_STAR_CROP_RADIUS_M',
                                                        20.0)  # Only look at a window around the boat; kept in sync with the projection clamp above
                                crop_radius_px = int(crop_radius_m / COSTMAP_RES_M_PER_PX)

                                map_w, map_h = costmap_size_px
                                cw, ch = map_w // 2, map_h // 2
                                rx_px = int(cw + ((robot_x - costmap_center_m[0]) / COSTMAP_RES_M_PER_PX))
                                ry_px = int(ch - ((robot_y - costmap_center_m[1]) / COSTMAP_RES_M_PER_PX))

                                x_min = max(0, rx_px - crop_radius_px)
                                x_max = min(map_w, rx_px + crop_radius_px)
                                y_min = max(0, ry_px - crop_radius_px)
                                y_max = min(map_h, ry_px + crop_radius_px)

                                cropped_costmap = costmap_img[y_min:y_max, x_min:x_max]
                                cropped_center_m = (
                                    costmap_center_m[0] + ((x_min + x_max) / 2 - cw) * COSTMAP_RES_M_PER_PX,
                                    costmap_center_m[1] - ((y_min + y_max) / 2 - ch) * COSTMAP_RES_M_PER_PX
                                )
                                cropped_size_px = (x_max - x_min, y_max - y_min)

                                nav_map, _ = planner.get_inflated_nav_map(cropped_costmap)

                                # Replan at ~10 Hz, or immediately if we have no path at all.
                                plan_timer += 1
                                if (plan_timer > 4 or not current_path) and tx_world is not None:
                                    plan_timer = 0
                                    new_path = None
                                    if planner.check_line_of_sight((robot_x, robot_y), (tx_world, ty_world), nav_map,
                                                                   cropped_center_m, COSTMAP_RES_M_PER_PX,
                                                                   cropped_size_px):
                                        new_path = [(robot_x, robot_y), (tx_world, ty_world)]
                                    else:
                                        new_path = planner.get_path_plan(
                                            (robot_x, robot_y), (tx_world, ty_world),
                                            nav_map, cropped_center_m,
                                            COSTMAP_RES_M_PER_PX, cropped_size_px,
                                            heuristic_weight=getattr(cfg, 'A_STAR_HEURISTIC_WEIGHT', 2.5))
                                    if new_path and len(new_path) >= 2:
                                        current_path = new_path
                                        path_progress_idx = 0
                                # ------------------------------------------------

                            # Follow the planned path with Pure Pursuit.
                            # The path is NOT consumed by the controller any more: progress is
                            # carried in path_progress_idx. Previously pure_pursuit_control()
                            # returned a pruned path that collapsed a 2-point LOS path to a single
                            # point on its very first call, which failed this len() >= 2 test on
                            # the next iteration and quietly handed steering to the Direct-Drive
                            # PID for 4 out of every 5 cycles - two controllers with different
                            # gains fighting each other, each with a stale prev_error.
                            if current_path and len(current_path) >= 2:
                                base_pwm = getattr(cfg, 'BASE_PWM', 1500)
                                if mevcut_gorev.startswith("T3_") or "TASK3" in mevcut_gorev:
                                    base_pwm += getattr(cfg, 'T3_SPEED_PWM', 100)
                                else:
                                    base_pwm += getattr(cfg, 'CRUISE_PWM', 80)

                                if active_controller != 'PP':
                                    prev_heading_error = 0.0   # avoid a derivative kick on handover
                                    active_controller = 'PP'

                                # Ground speed in m/s - NOT a PWM offset. Feeding PWM here pinned
                                # the lookahead at PURE_PURSUIT_MAX_LOOKAHEAD permanently.
                                speed_mps = shared_state.get('horizontal_speed', 0.0)
                                try:
                                    speed_mps = float(speed_mps)
                                except (TypeError, ValueError):
                                    speed_mps = 0.0

                                p_base, p_steer, raw_target, current_error, path_progress_idx = \
                                    planner.pure_pursuit_control(
                                        robot_x, robot_y, robot_yaw, current_path,
                                        current_speed=speed_mps, base_speed=base_pwm,
                                        prev_error=prev_heading_error, dt=loop_dt,
                                        start_idx=path_progress_idx)

                                prev_heading_error = current_error

                                apply_motor_mixer(controller, p_base, p_steer)

                                # Path consumed: force a fresh plan on the next cycle.
                                end_pt = current_path[-1]
                                if math.hypot(end_pt[0] - robot_x, end_pt[1] - robot_y) < 0.5:
                                    current_path = None
                                    path_progress_idx = 0

                            # If no path, or we are NOT in Task 2 (meaning Task 1 or 3), use PID Direct Drive
                            else:
                                current_path = None  # Force a fresh A* attempt on the next plan_timer cycle
                                path_progress_idx = 0
                                if target_lat is not None and target_lon is not None:
                                    # Always use Direct Drive PID (and spot turns) as a fallback when A* has no path
                                    # This prevents the boat from freezing if A* inflation blocks the target
                                    # or if initial alignment takes longer than 5 seconds.
                                    if active_controller != 'PID':
                                        direct_drive_prev_error = control_error  # no derivative kick on handover
                                        active_controller = 'PID'

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

                                        # Dynamic KD scaling for high speed momentum braking
                                        base_kd = getattr(cfg, 'DIRECT_DRIVE_KD', 0.8)
                                        speed_factor_pid = max(0.0, (base_pwm - 1500) / 100.0)
                                        kd = base_kd + (speed_factor_pid * 0.3)

                                        # Calculate terms
                                        error = control_error

                                        # dt normalisation: scale is 1.0 at the nominal 50 Hz, so
                                        # the existing config gains keep their current meaning, but
                                        # a slow cycle (e.g. the one where A* runs) no longer
                                        # inflates KD or starves KI.
                                        dts = planner.dt_scale(loop_dt)

                                        # --- 3-B UPDATE: Anti-Windup Logic ---
                                        # Only accumulate integral if the error is relatively small
                                        # This prevents massive windup when pushing against an obstacle or turning sharply
                                        anti_windup_deg = getattr(cfg, 'ANTI_WINDUP_DEG', 15.0)
                                        if abs(error) < anti_windup_deg:
                                            direct_drive_integral += error * dts
                                        else:
                                            # Optional: Reset or decay the integral when outside the linear region
                                            direct_drive_integral *= 0.9

                                        # Add natural decay if error crosses zero (sign change) to prevent residual windup
                                        if (error > 0 and direct_drive_integral < 0) or (
                                                error < 0 and direct_drive_integral > 0):
                                            direct_drive_integral *= 0.5

                                        # Integral windup hard limit
                                        windup_limit = getattr(cfg, 'ANTI_WINDUP_LIMIT', 500.0)
                                        direct_drive_integral = max(-windup_limit, min(windup_limit, direct_drive_integral))
                                        # -------------------------------------

                                        derivative = (error - direct_drive_prev_error) / dts
                                        direct_drive_prev_error = error

                                        steering_correction = (error * kp) + (direct_drive_integral * ki) + (
                                                derivative * kd)

                                        apply_motor_mixer(controller, base_pwm, steering_correction)

                    elif not skip_default_nav:
                        apply_motor_mixer(controller, 1500, 0)

                # Record final PWMs
                # Note: We rely on the USVController object stub tracking state,
                # but we can push directly to shared_state for safety
                shared_state['motor_pwm_left'] = controller.get_servo_pwm(CH_REAR_LEFT)
                shared_state['motor_pwm_right'] = controller.get_servo_pwm(CH_REAR_RIGHT)
                shared_state['motor_pwm_front_left'] = controller.get_servo_pwm(CH_FRONT_LEFT)
                shared_state['motor_pwm_front_right'] = controller.get_servo_pwm(CH_FRONT_RIGHT)
                shared_state['motor_pwm_steer'] = controller.get_servo_pwm(CH_STEER)

                # Heartbeat: p.is_alive() in the orchestrator watchdog only detects a dead process,
                # not one blocked on a hung call (e.g. a serial write). This timestamp lets the
                # watchdog notice "alive but not making progress" and force a restart.
                shared_state['nav_heartbeat'] = time.time()
            except Exception as loop_err:
                # A transient failure in a single iteration (numpy shape, None arithmetic,
                # a MAVLink parse hiccup) used to break out of the whole while loop, fall
                # into the `finally` and DISARM the vehicle. Since arm_vehicle() is never
                # called anywhere in this codebase, the boat then stayed dead in the water
                # until it was re-armed by hand from the RC. Neutralise, log, keep looping.
                print(f"[NAV_PROCESS][ERROR] Iteration failed: {loop_err}")
                try:
                    apply_motor_mixer(controller, 1500, 0)
                except Exception:
                    pass
                shared_state['nav_heartbeat'] = time.time()


            elapsed = time.time() - start_time
            if elapsed < 0.02: time.sleep(0.02 - elapsed)

    except Exception as e:
        print(f"[NAV_PROCESS][ERROR] Brain crashed: {e}")
    finally:
        print("[NAV_PROCESS] Shutting down...")
        if costmap_recorder is not None:
            costmap_recorder.save()
        try:
            apply_motor_mixer(controller, 1500, 0)

            controller.disarm_vehicle()
        except:
            pass
