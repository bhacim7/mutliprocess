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


def _slew(target, prev, max_delta):
    """Limit one channel's per-cycle change to +/- max_delta."""
    if max_delta is None or max_delta <= 0 or prev is None:
        return int(target)
    if target > prev + max_delta:
        return int(prev + max_delta)
    if target < prev - max_delta:
        return int(prev - max_delta)
    return int(target)


def apply_motor_mixer(controller, forward_pwm, yaw_pwm, slew=True):
    """
    Control Mixer: Distributes Forward Effort and Yaw Effort across 5 channels.
    Implements Steering Deadband for smooth turns and push-pull differential.

    `slew=False` bypasses the rate limiter - use it for emergency braking and for the
    SIGTERM neutralise, where an instantaneous command is the whole point.
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
    steer_out = int(steer_out)

    # --- Slew-rate limiting (cfg.MAX_PWM_CHANGE) ---
    # This limit has been sitting in config.py unused since the beginning. Flight data
    # shows why it matters: at 23:48:08 the mixer went 1100/1900 -> 1831/1528 -> 1900/1109
    # in 0.25 s, an ~800 PWM reversal on every thruster inside a quarter second. That is
    # not control, it is chatter - and with the hull's ~1 s yaw time constant the boat
    # simply keeps rotating through it. The previous PWM is read back from the controller's
    # own cache, so no extra state has to be threaded through here.
    if slew:
        max_delta = getattr(cfg, 'MAX_PWM_CHANGE', 60)
        rear_left = _slew(rear_left, controller.get_servo_pwm(CH_REAR_LEFT), max_delta)
        rear_right = _slew(rear_right, controller.get_servo_pwm(CH_REAR_RIGHT), max_delta)
        front_left = _slew(front_left, controller.get_servo_pwm(CH_FRONT_LEFT), max_delta)
        front_right = _slew(front_right, controller.get_servo_pwm(CH_FRONT_RIGHT), max_delta)
        steer_out = _slew(steer_out, controller.get_servo_pwm(CH_STEER), max_delta)

    # Command the hardware
    force = not slew  # emergency / neutralise paths must not be rate limited downstream
    controller.set_servo(CH_REAR_LEFT, rear_left, force=force)
    controller.set_servo(CH_REAR_RIGHT, rear_right, force=force)
    controller.set_servo(CH_FRONT_LEFT, front_left, force=force)
    controller.set_servo(CH_FRONT_RIGHT, front_right, force=force)
    controller.set_servo(CH_STEER, steer_out, force=force)


def _latlon_to_local(origin_lat, origin_lon, lat, lon):
    """(North, East) metres from the local frame origin - the same frame robot_x/robot_y use."""
    d = nav.haversine(origin_lat, origin_lon, lat, lon)
    b = math.radians(nav.calculate_bearing(origin_lat, origin_lon, lat, lon))
    return d * math.cos(b), d * math.sin(b)


def _task2_axis_local(origin_lat, origin_lon):
    """
    Task 2 reference axis in local coordinates: first waypoint -> last waypoint.

    This is the only geometry the rules guarantee - a straight line between those two points
    stays inside the course - and unlike a boat->goal axis it does not rotate with the
    approach, so "left" and "right" stay meaningful when entering diagonally.
    Returns (p0, p1) as numpy arrays, or None if the points are unset or degenerate.
    """
    e_lat = getattr(cfg, 'T2_ZONE_ENTRY_LAT', 0.0)
    e_lon = getattr(cfg, 'T2_ZONE_ENTRY_LON', 0.0)
    x_lat = getattr(cfg, 'T2_ZONE_END_LAT', 0.0)
    x_lon = getattr(cfg, 'T2_ZONE_END_LON', 0.0)
    if not (e_lat and e_lon and x_lat and x_lon):
        return None
    if nav.haversine(e_lat, e_lon, x_lat, x_lon) < 5.0:
        return None
    p0 = np.array(_latlon_to_local(origin_lat, origin_lon, e_lat, e_lon), dtype=float)
    p1 = np.array(_latlon_to_local(origin_lat, origin_lon, x_lat, x_lon), dtype=float)
    return p0, p1


def _confirmed_boundary_local(vision_objects, boat_lat, boat_lon, robot_x, robot_y):
    """
    Local (x, y) of the boundary buoys that are trustworthy enough to constrain on.

    Confirmation matters here in a way it does not for plain obstacle avoidance: a spurious
    orange narrows the corridor, and a corridor that is too narrow is as bad as none at all.
    A buoy counts once it has been seen CORRIDOR_CONFIRM_SIGHTINGS times and was seen in the
    last CORRIDOR_CONFIRM_MAX_AGE_S seconds.
    """
    if not vision_objects or not boat_lat:
        return []

    min_seen = int(getattr(cfg, 'CORRIDOR_CONFIRM_SIGHTINGS', 3))
    max_age = float(getattr(cfg, 'CORRIDOR_CONFIRM_MAX_AGE_S', 1.5))
    now = time.time()

    out = []
    for obj in vision_objects:
        if obj.get('cid') != 3:          # orange = boundary
            continue
        # Position not yet confirmed by a close sighting - see MAP_MAX_RANGE_M. Building the
        # corridor from distant projections is what put orange boundary marks 2-3 m from where
        # the buoys actually were.
        if not obj.get('pos_ok', True):
            continue
        if int(obj.get('seen', 0)) < min_seen:
            continue
        last_seen = obj.get('last_seen')
        if last_seen is None or (now - last_seen) > max_age:
            continue
        o_lat, o_lon = obj.get('lat'), obj.get('lon')
        if not o_lat or not o_lon:
            continue
        dist = nav.haversine(boat_lat, boat_lon, o_lat, o_lon)
        if not (0.0 < dist < 20.0):
            continue
        brg = math.radians(nav.calculate_bearing(boat_lat, boat_lon, o_lat, o_lon))
        out.append((robot_x + dist * math.cos(brg), robot_y + dist * math.sin(brg)))
    return out


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

    # Physical buoy footprint painted into the costmap. Safety margin is NOT included here -
    # it belongs to the inflation step, which applies it with the correct geometry.
    buoy_radius_px = max(1, int(round(getattr(cfg, 'BUOY_RADIUS_M', 0.25) / COSTMAP_RES_M_PER_PX)))

    # These follow whatever map we are currently holding. When lidar_process pushes a
    # costmap it comes with its own geometry (its own size, centred on world origin),
    # so they must not be hardcoded to the local map's constants.
    costmap_img = local_costmap
    costmap_size_px = (LOCAL_SIZE, LOCAL_SIZE)   # (width, height)
    costmap_center_m = (0.0, 0.0)
    costmap_ready = True

    def publish_mission_points():
        """
        Mirror the current waypoint set into shared_state.

        `set_gps` below mutates this process's `cfg` module. With mp.set_start_method('spawn')
        every worker imports its OWN copy of config.py, so telem_process never saw those
        updates - it kept echoing the config-file defaults back to the GCS while nav steered
        to the operator's points. Publishing them here is what makes the GCS display (and the
        change-detection in telem_process) reflect reality.
        """
        try:
            shared_state['mission_points'] = {
                "GPS1": (float(cfg.T1_GATE_ENTER_LAT), float(cfg.T1_GATE_ENTER_LON)),
                "GPS2": (float(cfg.T1_GATE_MID_LAT), float(cfg.T1_GATE_MID_LON)),
                "GPS3": (float(cfg.T1_GATE_EXIT_LAT), float(cfg.T1_GATE_EXIT_LON)),
                "GPS4": (float(cfg.T2_ZONE_ENTRY_LAT), float(cfg.T2_ZONE_ENTRY_LON)),
                "GPS5": (float(cfg.T2_ZONE_MID_LAT), float(cfg.T2_ZONE_MID_LON)),
                "GPS6": (float(cfg.T2_ZONE_END_LAT), float(cfg.T2_ZONE_END_LON)),
                "GPS7": (float(cfg.T3_START_LAT), float(cfg.T3_START_LON)),
                "GPS8": (float(cfg.T3_MID_LAT), float(cfg.T3_MID_LON)),
            }
        except Exception as e:
            print(f"[NAV_PROCESS][WARNING] Could not publish mission points: {e}")

    publish_mission_points()

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
    current_path_ts = 0.0             # when current_path was produced (A_STAR_MAX_PATH_AGE_S)
    path_progress_idx = 0
    active_controller = None          # 'PP' or 'PID' - used to reset stale derivative state
    pp_spot_turn_active = False       # hysteresis latch for the PP spot-turn guard
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
    NAV_PERIOD_S = 1.0 / max(1.0, getattr(cfg, 'NAV_LOOP_HZ', 25.0))
    loop_dt = NAV_PERIOD_S

    # --- IPC throttling (see cfg.VISION_READ_HZ / cfg.NAV_PUBLISH_HZ) ---
    # shared_state is a multiprocessing.Manager dict: EVERY access is a pickle round trip
    # over a socket. The nav loop was making ~25 of them per iteration at 50 Hz (~1250/s),
    # and the single most expensive entry - 'vision_detected_objects', a list of 10-30
    # dicts - was being unpickled on every single cycle. That cost is what pushed loop_dt
    # around, and because dt_scale() rescales the D/I terms by loop_dt, the effective
    # damping was changing from cycle to cycle. That is the direct cause of "sometimes it
    # runs beautifully, sometimes it does not" on identical code.
    VISION_READ_PERIOD_S = 1.0 / max(1.0, getattr(cfg, 'VISION_READ_HZ', 10.0))
    PUBLISH_PERIOD_S = 1.0 / max(1.0, getattr(cfg, 'NAV_PUBLISH_HZ', 10.0))
    last_vision_read = 0.0
    last_publish = 0.0
    vision_objects = []
    # Task 2 entrance alignment latch. A dict rather than a bare flag because the state
    # machine helpers are closures rebuilt every cycle, so a local would not survive. Cleared
    # whenever the boat is in a Task 1 state, i.e. before every Task 2 run.
    t2_entry = {'aligned': False}
    # Throttle for the A*-failure log below - planning runs at 5 Hz and a genuinely blocked
    # goal would otherwise print five times a second.
    last_astar_fail_log = 0.0

    # Same list with distant-only positions removed; see MAP_MAX_RANGE_M. Initialised here
    # because the vision read below is throttled to VISION_READ_HZ and the costmap block runs
    # every cycle.
    mappable_objects = []
    lidar_enabled = getattr(cfg, 'ENABLE_LIDAR', True)
    speed_mps = 0.0

    # Give up on the in-place spot turn after this long and let the normal controllers
    # take over, rather than pivoting forever on a bad heading.
    ALIGNMENT_TIMEOUT_S = 12.0

    # Store legacy PWM defaults
    extra = 50

    # --- Task 3 search v2 state ---
    # The old pattern was an unbounded zigzag: pan +/-45, crawl 3 m, repeat forever along
    # whatever heading the boat happened to arrive with. Against the real field - three
    # buoys on a line spanning up to 30 m, T3_MID dropped by eye ~15 m short of it, and a
    # hard 15 m detection ceiling from the ZED depth gate - that pattern relied on luck:
    # no stationary pan can cover a 30 m line from 15 m away, so the search MUST move,
    # and it must stop moving when it leaves the plausible field.
    t3_state = 'INIT'
    t3_heading0 = None            # search frame axis, frozen at first search entry
    t3_move_target = None         # (lat, lon) of the current pattern waypoint
    t3_patrol_side = 1            # first V-arm/pan side; flipped when a full fan comes up empty
    t3_v_round = 0                # expanding V fan: which round (index into T3_V_ROUND_DEG)
    t3_v_arm = 0                  # 0 = first arm of the round, 1 = mirrored arm
    t3_v_wps = None               # [tip, probe] waypoints of the arm being flown
    t3_v_idx = 0
    t3_lock_id = None             # visual-servo target lock (see selection block)
    t3_lock_seen_ts = 0.0         # when the locked track was last actually fresh
    t3_lock_cx = None             # its last pixel column, steered on during the lock grace
    t3_last_servo_dist = None     # for the loss-of-lock hold
    t3_last_servo_ts = None
    T3_MAX_DETECTION_AGE_S = 0.3  # steer only on genuinely fresh pixels

    # --- Relay (motor power) ---
    # Retried rather than fired once: command_long_send waits for no ACK, so a dropped
    # safety command would go unnoticed. Retries stop as soon as RELAY_STATUS confirms the
    # vehicle agrees, and they are spread across nav cycles rather than slept through, so
    # the control loop keeps running.
    MY_ID = getattr(cfg, 'VEHICLE_ID', 1)
    RELAY_MAX_RETRIES = getattr(cfg, 'RELAY_COMMAND_RETRIES', 5)
    RELAY_RETRY_INTERVAL_S = getattr(cfg, 'RELAY_RETRY_INTERVAL_S', 0.25)
    relay_target = None          # state we are trying to reach, None = nothing pending
    relay_retries = 0
    relay_last_tx = 0.0
    relay_reported = None        # last state read back from RELAY_STATUS
    last_cmd_cq = None           # sequence of the last processed GCS command (R1 ack)

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
            apply_motor_mixer(controller, 1500, 0, slew=False)
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

                # Fetch horizontal speed. Kept in a LOCAL variable - Pure Pursuit reads it
                # every cycle and round-tripping it through the Manager dict just to read it
                # back was pure overhead. It is published to shared_state at NAV_PUBLISH_HZ
                # further down for the HUD/telemetry.
                try:
                    _spd = controller.get_horizontal_speed()
                    if _spd is not None:
                        speed_mps = float(_spd)
                except (AttributeError, TypeError, ValueError):
                    pass  # Fallback if get_horizontal_speed doesn't exist on dummy controller

                fc_hdg = controller.get_heading()

                # --- Relay: read back, then retry any pending command ---
                # The transmitter can change this relay too (RC7_OPTION=28), so the truth
                # comes from the vehicle rather than from what we last sent.
                try:
                    relay_reported = controller.get_relay_state()
                except AttributeError:
                    relay_reported = None

                if relay_target is not None:
                    if relay_reported is not None and int(relay_reported) == relay_target:
                        relay_target = None          # vehicle agrees, done
                    elif relay_retries <= 0:
                        print("[NAV_PROCESS][WARNING] Relay command not confirmed after "
                              f"{RELAY_MAX_RETRIES} attempts.")
                        relay_target = None
                    elif (start_time - relay_last_tx) >= RELAY_RETRY_INTERVAL_S:
                        relay_last_tx = start_time
                        relay_retries -= 1
                        controller.set_relay(relay_target)

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

                        # The RFD link is shared with the drone, and CommandReceiver puts
                        # every line it hears on this queue regardless of who it was for.
                        # Ignoring a command addressed elsewhere matters most for the relay,
                        # which cuts motor power.
                        _tgt = cmd.get("target_id")
                        if _tgt is not None and int(_tgt) != MY_ID:
                            continue

                        # R1 reliable commands: the GCS resends until our telemetry echoes
                        # the command's sequence number back. A resend of something already
                        # processed is skipped here - the ack is already on the air - so
                        # retries cannot double-execute or spam the log. Commands without
                        # cq (old GCS) flow through unchanged.
                        _cq = cmd.get("cq")
                        if _cq is not None and _cq == last_cmd_cq:
                            continue

                        if cmd_str == "set_relay":
                            want = 1 if cmd.get("value") else 0
                            relay_target = want
                            relay_retries = RELAY_MAX_RETRIES
                            relay_last_tx = 0.0     # send on this very cycle
                            print(f"[NAV_PROCESS] Relay command received: "
                                  f"{'ON' if want else 'OFF'}")

                        elif cmd_str == "emergency_stop":
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
                            publish_mission_points()
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

                        # Ack AFTER the dispatch above ran to completion: if it raised, the
                        # outer except swallows the iteration and no ack goes out, so the
                        # GCS retries and the command gets a second chance instead of being
                        # falsely confirmed.
                        if _cq is not None:
                            last_cmd_cq = int(_cq)
                            shared_state['last_cmd_ack'] = last_cmd_cq
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

                # With ENABLE_LIDAR off these four Manager-dict reads happen every cycle and
                # can never be anything but their defaults - the lidar process is not even
                # started (see main_orchestrator). Skip them entirely.
                if lidar_enabled:
                    center_danger = shared_state.get('lidar_center_blocked', False)
                    left_d = shared_state.get('lidar_left_dist', float('inf'))
                    center_d = shared_state.get('lidar_center_dist', float('inf'))
                    right_d = shared_state.get('lidar_right_dist', float('inf'))
                else:
                    center_danger = False
                    left_d = center_d = right_d = float('inf')

                # --- 4-A UPDATE: Pull from Queue instead of shared_state ---
                lidar_data = None
                if lidar_enabled:
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

                # Throttled read - this is the heaviest Manager-dict entry by far.
                # camera_process only refreshes it at the ZED frame rate anyway, so reading
                # it 50x/s never produced new information.
                if (start_time - last_vision_read) >= VISION_READ_PERIOD_S:
                    last_vision_read = start_time
                    vision_objects = shared_state.get('vision_detected_objects', [])
                    # Objects whose position rests only on distant sightings are dropped here,
                    # once, so the costmap, the corridor and the recorder all see the same set.
                    # Keeping the recorder in step matters: it is the instrument used to judge
                    # whether the map is good, so it has to show what A* actually plans on.
                    # Task 3 reads shared_state directly and is unaffected - it steers on the
                    # pixel column, not on lat/lon.
                    mappable_objects = [o for o in vision_objects if o.get('pos_ok', True)]

                # --- UPDATE SEPARATE COSTMAP RECORDER ---
                if costmap_recorder is not None:
                    # Use current raw gps location to feed global costmap
                    costmap_recorder.update(ida_enlem, ida_boylam, mappable_objects)

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

                if mappable_objects and costmap_ready:
                    # If we are using vision-only or fused, draw vision objects on whatever map we have
                    for obj in mappable_objects:
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
                                    # The radius used to be a hardcoded 6 px (0.60 m), which
                                    # silently doubled as safety margin. It now represents the
                                    # buoy's actual size; ALL clearance comes from the
                                    # inflation step in planner.get_inflated_nav_map(), where
                                    # the robot radius is applied with the correct geometry.
                                    cv2.circle(costmap_img, p_virtual, buoy_radius_px, 0, -1)

                # --- E. FULL STATE MACHINE ---
                # 3-A UPDATE: Refactored modular state machine

                # Helper to execute task routing logic
                def execute_task1(task_state, lat, lon):
                    # Back in Task 1 means a fresh run at the Task 2 mouth is still to come,
                    # so drop the entrance latch here rather than trying to spot the exact
                    # moment Task 2 ends.
                    t2_entry['aligned'] = False
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
                        dist_to_target = nav.haversine(lat, lon, t_lat, t_lon)
                        if dist_to_target < 3.0:
                            return next_state, t_lat, t_lon

                        # Entrance approach. Arriving at the mouth very obliquely, the
                        # straight line to the entry waypoint can pass between the first and
                        # second buoy of a boundary chain instead of through the mouth - the
                        # chain spacing leaves a gap wide enough for A* to slip through.
                        # Aiming at a point set back along the corridor axis first forces the
                        # final leg to be aligned with the corridor, from either diagonal.
                        if task_state == "TASK2_START":
                            end_lat = getattr(cfg, 'T2_ZONE_END_LAT', 0)
                            end_lon = getattr(cfg, 'T2_ZONE_END_LON', 0)
                            if end_lat and end_lon and \
                                    nav.haversine(t_lat, t_lon, end_lat, end_lon) > 5.0:
                                offset = getattr(cfg, 'TASK2_APPROACH_OFFSET_M', 12.0)
                                reached = getattr(cfg, 'TASK2_APPROACH_REACHED_M', 3.0)
                                axis_brg = nav.calculate_bearing(t_lat, t_lon, end_lat, end_lon)
                                a_lat, a_lon = calculate_obj_gps(
                                    t_lat, t_lon, offset, (axis_brg + 180.0) % 360.0)

                                # Position relative to the corridor axis, measured from the
                                # entry: s along the axis (negative = still outside, in front
                                # of the mouth) and d across it.
                                brg_to_boat = nav.calculate_bearing(t_lat, t_lon, lat, lon)
                                r_boat = nav.haversine(t_lat, t_lon, lat, lon)
                                rel = math.radians(brg_to_boat - axis_brg)
                                s_along = r_boat * math.cos(rel)
                                d_across = abs(r_boat * math.sin(rel))

                                # The approach point exists for one reason: make the last leg
                                # into the mouth run ALONG the corridor, so an oblique line
                                # cannot slip between the first two buoys of a boundary
                                # chain. It is therefore needed only while we are off to the
                                # side - once we are lined up in front of the mouth, heading
                                # straight for the entry is already aligned.
                                #
                                # Keying it on lateral offset also makes the switch one-way.
                                # Two earlier versions oscillated instead:
                                #   * comparing distance to the ENTRY against an 8 m
                                #     threshold while the approach point sat 12 m further
                                #     back meant that on reaching it the boat was still 12 m
                                #     from the entry, so it kept targeting the point it was
                                #     already standing on and never left TASK2_START. Flight
                                #     log 03:59:02-03:59:31: HEDEFE_MESAFE cycling 0.6-3.5 m
                                #     with HEDEF_HDG swinging 143, 273, 296, 177, 100, 256.
                                #   * a simple "within `reached` of the approach point" test
                                #     flipped back the moment the boat moved past it toward
                                #     the entry and left that radius again.
                                #
                                # The lateral test alone was not enough. On the 2026-08-17 run
                                # the boat sat 16.3 m in front of the mouth but only 3.68 m off
                                # the axis, so 3.68 > 8.0 was false, the approach point was
                                # skipped, and it ran the whole 16.7 m diagonally - which is how
                                # it came to enter on the far side of a boundary buoy when the
                                # entry waypoint was placed near the edge of the course.
                                #
                                # So there is a second way in: if the boat is still BEHIND the
                                # approach point, aim at it. Comparing s_along against -offset
                                # is exactly the test for "have I passed it yet", because the
                                # point sits at s = -offset by construction. Behind it, going
                                # there is forward motion and costs almost nothing - 0.9 m of
                                # extra travel for the run above. Past it, going there would
                                # mean REVERSING away from the mouth, which is what made the
                                # earlier 3 m lateral setting look like it was aligning from
                                # much too far out, and why it was raised to 8 to suppress that.
                                # This branch cannot reverse.
                                #
                                # The lateral branch stays at its tuned 8 m as a last resort:
                                # close to the mouth but far off to the side is a bad enough
                                # entry angle to be worth backing up for.
                                lateral_tol = getattr(cfg, 'TASK2_APPROACH_LATERAL_M', 3.0)
                                dist_to_approach = nav.haversine(lat, lon, a_lat, a_lon)

                                # Latch. Without it the boat ping-pongs: on leaving the approach
                                # point towards the entry, its distance to that point grows past
                                # `reached` again, the condition re-arms, and it turns back.
                                # Flight log 03:59:02-03:59:31 recorded exactly that, HEDEF_HDG
                                # swinging 143, 273, 296, 177, 100, 256. Once aligned, this run
                                # never targets the point again.
                                if dist_to_approach <= reached:
                                    t2_entry['aligned'] = True

                                if (not t2_entry['aligned']
                                        and s_along < 0.0
                                        and dist_to_approach > reached
                                        and (s_along < -offset or d_across > lateral_tol)):
                                    return task_state, a_lat, a_lon

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

                def t3_target_cid():
                    """The colour Task 3 hunts, as a YOLO class id.

                    Config first; the drone's reported colour overrides it when DRONE_ACTIVE
                    and one has actually arrived. If the drone never delivers, the config
                    colour stands - the boat hunts something rather than nothing, and the
                    GCS shows RENK YOK so the operator knows which case is live."""
                    color = getattr(cfg, 'TASK3_KAMIKAZE_COLOR', 'red')
                    if getattr(cfg, 'DRONE_ACTIVE', False):
                        color = shared_state.get('drone_target_color', color) or color
                    return {'red': 0, 'yellow': 1, 'black': 2,
                            'orange': 3, 'green': 4}.get(str(color).lower(), 0)

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
                        target_cid = t3_target_cid()
                        now_ts = time.time()
                        # T3-specific turn authority. The search patterns are corner-heavy
                        # and the measured 35-57 deg/s pirouettes both looked frantic and
                        # pushed the ZED-filtered heading further behind the hull. 140 gives
                        # ~22-38 deg/s; Task 1/2 keep the full 200.
                        spot_pwm = getattr(cfg, 'T3_SPOT_TURN_PWM', 140)
                        t3_base_pwm = getattr(cfg, 'BASE_PWM', 1500) + getattr(cfg, 'T3_SPEED_PWM', 100)

                        # ---- Target selection: nearest fresh candidate, held by an ID lock ----
                        # The old loop took the FIRST fresh match in memory-insertion order,
                        # which is neither the nearest nor stable: with two objects of the
                        # target colour in view, a single missed YOLO frame on the first one
                        # hopped the servo to the other and back, and a stale false-positive
                        # track that happened to be inserted first outranked the real buoy
                        # dead ahead. Selection is now by distance, and the winner's track id
                        # is locked so the choice cannot flicker frame to frame. The lock is
                        # identity, not position: the locked object's cx is re-read every
                        # frame, so steering keeps correcting. It drops when the track goes
                        # stale - or when the drone changes the colour mid-run, because a
                        # different cid simply never enters `fresh`.
                        fresh = []
                        for obj in vision_objects:
                            if obj.get('cid') != target_cid or obj.get('cx') is None:
                                continue
                            ls = obj.get('last_seen')
                            if ls is None or (now_ts - ls) > T3_MAX_DETECTION_AGE_S:
                                continue
                            fresh.append(obj)

                        target_obj = None
                        grace_steer = False
                        if t3_lock_id is not None:
                            for o in fresh:
                                if o.get('id') == t3_lock_id:
                                    target_obj = o
                                    break
                            if target_obj is None:
                                # Lock grace. The lock used to drop the instant its track
                                # missed one freshness window, and selection immediately
                                # re-locked the nearest OTHER candidate - the 83<->85
                                # ping-pong of the 2026-08-18 run, where the rudder swung
                                # to an object 8.6 m away while the real target sat at
                                # 3.4 m, weaving the final approach. For a short grace the
                                # lock now stays faithful: steer on the track's last pixel
                                # column and wait for it to come back before considering
                                # anyone else.
                                if (t3_lock_cx is not None
                                        and (now_ts - t3_lock_seen_ts)
                                        <= getattr(cfg, 'T3_LOCK_GRACE_S', 0.6)):
                                    grace_steer = True
                                else:
                                    t3_lock_id = None
                        if target_obj is None and not grace_steer and fresh:
                            target_obj = min(fresh, key=lambda o: float(o.get('dist', 1e9)))
                            t3_lock_id = target_obj.get('id')
                            print(f"[TASK3] Locked target id={t3_lock_id} at "
                                  f"{float(target_obj.get('dist', -1)):.1f} m")

                        if target_obj is not None:
                            # --- Visual servoing (control law unchanged) ---
                            target_lat, target_lon = None, None
                            dist_m = float(target_obj.get('dist', 10.0))
                            t3_last_servo_dist = dist_m
                            t3_last_servo_ts = now_ts
                            t3_lock_seen_ts = now_ts
                            t3_lock_cx = target_obj.get('cx')

                            pixel_error = target_obj.get('cx') - (1280 / 2)
                            kp_pixel = getattr(cfg, 'Kp_PIXEL', 0.3)
                            if getattr(cfg, 'TASK3_INVERT_STEERING', False):
                                steering_correction = -pixel_error * kp_pixel
                            else:
                                steering_correction = pixel_error * kp_pixel
                            apply_motor_mixer(controller, t3_base_pwm, steering_correction)

                            if dist_m < 1.0 or target_obj.get('area', 0) > 300000:
                                print("[TASK3] KAMIKAZE COLLISION CONFIRMED! STOPPING VEHICLE.")
                                apply_motor_mixer(controller, 1500, 0)
                                mevcut_gorev = "FINISHED"

                        elif grace_steer:
                            # Steer on where the locked target last was. One or two missed
                            # YOLO frames no longer bend the approach; if the track returns
                            # inside the grace it is picked up above as if nothing happened.
                            target_lat, target_lon = None, None
                            pixel_error = t3_lock_cx - (1280 / 2)
                            kp_pixel = getattr(cfg, 'Kp_PIXEL', 0.3)
                            if getattr(cfg, 'TASK3_INVERT_STEERING', False):
                                steering_correction = -pixel_error * kp_pixel
                            else:
                                steering_correction = pixel_error * kp_pixel
                            apply_motor_mixer(controller, t3_base_pwm, steering_correction)

                        elif (t3_last_servo_ts is not None
                              and t3_last_servo_dist is not None
                              and t3_last_servo_dist <= getattr(cfg, 'T3_HOLD_MAX_DIST_M', 5.0)
                              and (now_ts - t3_last_servo_ts) <= getattr(cfg, 'T3_HOLD_S', 2.0)):
                            # --- Loss-of-lock hold ---
                            # Spray and glare routinely break detection for a few frames in
                            # the last metres. Dropping straight back into the search here
                            # made the boat yank sideways right next to its target; holding
                            # course briefly lets the detector re-acquire, and if the target
                            # was genuinely hit the collision check ends the run on
                            # re-acquisition anyway.
                            target_lat, target_lon = None, None
                            apply_motor_mixer(controller, t3_base_pwm, 0)

                        else:
                            # ==================== SEARCH v2 ====================
                            # Wide pan first; any Task 3 buoy (not just the target colour)
                            # found during a scan is used as an ANCHOR: the three buoys sit
                            # on one line, so finding any of them is finding the line, and
                            # the target is on it within ~30 m. Patrol runs parallel to that
                            # line from a standoff point. Everything is bounded to a box
                            # around T3_MID, and hitting a bound returns there and restarts
                            # mirrored - no more unbounded zigzag toward the horizon.
                            target_lat, target_lon = None, None

                            mid_lat = getattr(cfg, 'T3_MID_LAT', 0.0)
                            mid_lon = getattr(cfg, 'T3_MID_LON', 0.0)
                            have_mid = bool(mid_lat and mid_lon)

                            if t3_state == 'INIT':
                                if t3_heading0 is None:
                                    # Frozen once: the search frame must not rotate with the
                                    # boat, or "lateral" would mean something new every leg.
                                    t3_heading0 = magnetic_heading
                                t3_v_round = 0
                                t3_v_arm = 0
                                t3_v_wps = None
                                t3_v_idx = 0
                                t3_state = 'PAN_A'
                                print(f"[TASK3] Search start: axis {t3_heading0:.0f} deg, "
                                      f"first side {'L' if t3_patrol_side < 0 else 'R'}")

                            # --- Bounds: stay inside the plausible field ---
                            # Frame is anchored at the T3_MID waypoint (the operator's own
                            # reference), axis = t3_heading0. The old pattern had no bounds
                            # at all and crawled ~3 m per cycle toward wherever the boat
                            # happened to be pointing when the search began.
                            if have_mid and t3_heading0 is not None and t3_state != 'RETURN':
                                d_mid = nav.haversine(mid_lat, mid_lon, ida_enlem, ida_boylam)
                                brg_mid = nav.calculate_bearing(mid_lat, mid_lon, ida_enlem, ida_boylam)
                                rel_mid = math.radians(brg_mid - t3_heading0)
                                s_fwd = d_mid * math.cos(rel_mid)
                                d_side = abs(d_mid * math.sin(rel_mid))
                                if (d_side > getattr(cfg, 'T3_BOUND_LATERAL_M', 30.0)
                                        or s_fwd > getattr(cfg, 'T3_BOUND_FORWARD_M', 20.0)
                                        or s_fwd < -10.0):
                                    print("[TASK3] Search bound reached - returning to T3_MID.")
                                    t3_state = 'RETURN'

                            # --- Anchor scan (only while scanning, not mid-patrol) ---
                            # pos_ok is deliberately NOT required: a buoy first seen at
                            # 13-15 m carries a position error of a metre or so, and the
                            # standoff point sits 8 m short of it - plenty of slack. Waiting
                            # for a <=12 m confirmation would mean never anchoring on
                            # exactly the sightings this search exists to exploit.
                            # T3_ANCHOR_ENABLED False turns the shortcut off entirely, so
                            # the expanding V fan runs pure - the operator's experiment to
                            # see what the pattern does on its own. Default True: any Task 3
                            # buoy seen mid-search still snaps to the standoff/patrol path.
                            if (getattr(cfg, 'T3_ANCHOR_ENABLED', True)
                                    and t3_state in ('PAN_A', 'PAN_B', 'PAN_C', 'V_LEG',
                                                     'V_TIPPAN_A', 'V_TIPPAN_B', 'V_HOME')):
                                anchor = None
                                a_best = 1e9
                                for obj in vision_objects:
                                    if obj.get('cid') not in (0, 2, 4):
                                        continue
                                    ls = obj.get('last_seen')
                                    if ls is None or (now_ts - ls) > T3_MAX_DETECTION_AGE_S:
                                        continue
                                    if int(obj.get('seen', 0)) < 3:
                                        continue
                                    o_lat, o_lon = obj.get('lat'), obj.get('lon')
                                    if not o_lat or not o_lon:
                                        continue
                                    a_d = float(obj.get('dist', 1e9))
                                    if a_d < a_best:
                                        a_best = a_d
                                        anchor = (o_lat, o_lon)
                                if anchor is not None:
                                    brg_away = nav.calculate_bearing(
                                        anchor[0], anchor[1], ida_enlem, ida_boylam)
                                    t3_move_target = calculate_obj_gps(
                                        anchor[0], anchor[1],
                                        getattr(cfg, 'T3_STANDOFF_M', 8.0), brg_away)
                                    t3_state = 'GOTO_STANDOFF'
                                    print(f"[TASK3] Anchor buoy at {a_best:.1f} m - "
                                          "moving to standoff on the line.")

                            if t3_state in ('PAN_A', 'PAN_B', 'PAN_C'):
                                pan_deg = getattr(cfg, 'T3_PAN_WIDE_DEG', 90.0)
                                if t3_state == 'PAN_A':
                                    target_h = (t3_heading0 - t3_patrol_side * pan_deg) % 360
                                elif t3_state == 'PAN_B':
                                    target_h = (t3_heading0 + t3_patrol_side * pan_deg) % 360
                                else:
                                    target_h = t3_heading0
                                diff = nav.signed_angle_difference(magnetic_heading, target_h)
                                # Proportional slow-down, same law as force_initial_alignment.
                                # These pans used to run at a flat SPOT_TURN_PWM with a 5 deg
                                # stop window - at the measured 35-57 deg/s the ZED heading
                                # lags a few degrees behind the hull, the window was missed,
                                # and the boat kept spinning: full extra circles on the
                                # 2026-08-18 run (HDG 118 -> 20 -> 248 while chasing a 97 deg
                                # target), 15-25 s lost per pan cycle and loops drawn all
                                # over the costmap. Slowing into the target with a wider stop
                                # window is exactly what the waypoint-alignment branch already
                                # does, and that one does not overshoot.
                                pan_tol = getattr(cfg, 'T3_PAN_TOL_DEG', 10.0)
                                if abs(diff) < pan_tol:
                                    apply_motor_mixer(controller, 1500, 0)
                                    t3_state = {'PAN_A': 'PAN_B', 'PAN_B': 'PAN_C',
                                                'PAN_C': 'V_CALC'}[t3_state]
                                else:
                                    ramp = min(1.0, abs(diff) / 45.0)
                                    pan_pwm = int(60 + (spot_pwm - 60) * ramp)
                                    apply_motor_mixer(controller, 1500,
                                                      pan_pwm if diff > 0 else -pan_pwm)

                            elif t3_state == 'V_CALC':
                                # Expanding V fan - the operator's design, replacing the
                                # rectangular transect sweep after the 2026-08-18 run.
                                #
                                # The prior it encodes: T3_MID is dropped BY HAND, facing
                                # the buoy line from ~15 m. So the targets are almost
                                # certainly in the forward cone - yet the rectangle's first
                                # move was 26 m to a SIDE corner (bearing 272 on that run),
                                # spending its opening leg on the least likely part of the
                                # box. The fan spends it on the most likely part instead:
                                # diagonal arms at +/-45 deg first, then wider and longer
                                # each round (70 deg/20 m, 100 deg/30 m), so angle AND
                                # radius grow together and round 3 covers the flanks and
                                # slightly behind the beam.
                                #
                                # Each arm carries a PROBE: from the arm tip, 8 m further
                                # along the arrival axis, and a short tip pan there. That
                                # probe is what plugs the dead-ahead 20-30 m zone that a
                                # pure V leaves outside both arms' 15 m detection ceiling -
                                # panning from 15 m forward sees out to ~30 m.
                                degs = getattr(cfg, 'T3_V_ROUND_DEG', (45.0, 70.0, 100.0))
                                legs = getattr(cfg, 'T3_V_ROUND_LEG_M', (10.0, 20.0, 30.0))
                                if not have_mid or t3_heading0 is None \
                                        or t3_v_round >= len(degs):
                                    t3_state = 'RETURN'
                                else:
                                    v_deg = float(degs[t3_v_round])
                                    v_leg = float(legs[t3_v_round])
                                    v_sgn = t3_patrol_side if t3_v_arm == 0 else -t3_patrol_side
                                    tip = calculate_obj_gps(
                                        mid_lat, mid_lon, v_leg,
                                        (t3_heading0 + v_sgn * v_deg) % 360)
                                    probe = calculate_obj_gps(
                                        tip[0], tip[1],
                                        getattr(cfg, 'T3_V_PROBE_M', 8.0), t3_heading0)
                                    t3_v_wps = [tip, probe]
                                    t3_v_idx = 0
                                    t3_state = 'V_LEG'
                                    print(f"[TASK3] V round {t3_v_round + 1}: "
                                          f"{'+' if v_sgn > 0 else '-'}{v_deg:.0f} deg / "
                                          f"{v_leg:.0f} m arm")

                            elif t3_state == 'V_LEG':
                                if not t3_v_wps or t3_v_idx >= len(t3_v_wps):
                                    t3_state = 'V_TIPPAN_A'
                                else:
                                    target_lat, target_lon = t3_v_wps[t3_v_idx]
                                    if nav.haversine(ida_enlem, ida_boylam,
                                                     target_lat, target_lon) < 2.0:
                                        t3_v_idx += 1
                                        target_lat, target_lon = None, None

                            elif t3_state in ('V_TIPPAN_A', 'V_TIPPAN_B'):
                                # Short sweep at the probe point, facing the arrival axis:
                                # +/-T3_V_TIP_PAN_DEG rather than the full +/-90 opening pan,
                                # per the agreed design - the forward band is what matters
                                # here and the narrow sweep saves ~30 s per round.
                                tp = getattr(cfg, 'T3_V_TIP_PAN_DEG', 45.0)
                                target_h = (t3_heading0 - tp if t3_state == 'V_TIPPAN_A'
                                            else t3_heading0 + tp) % 360
                                diff = nav.signed_angle_difference(magnetic_heading, target_h)
                                pan_tol = getattr(cfg, 'T3_PAN_TOL_DEG', 10.0)
                                if abs(diff) < pan_tol:
                                    apply_motor_mixer(controller, 1500, 0)
                                    t3_state = ('V_TIPPAN_B' if t3_state == 'V_TIPPAN_A'
                                                else 'V_HOME')
                                else:
                                    ramp = min(1.0, abs(diff) / 45.0)
                                    pan_pwm = int(60 + (spot_pwm - 60) * ramp)
                                    apply_motor_mixer(controller, 1500,
                                                      pan_pwm if diff > 0 else -pan_pwm)

                            elif t3_state == 'V_HOME':
                                if not have_mid:
                                    t3_state = 'RETURN'
                                else:
                                    target_lat, target_lon = mid_lat, mid_lon
                                    if nav.haversine(ida_enlem, ida_boylam,
                                                     mid_lat, mid_lon) < 2.5:
                                        target_lat, target_lon = None, None
                                        if t3_v_arm == 0:
                                            t3_v_arm = 1
                                            t3_state = 'V_CALC'
                                        else:
                                            t3_v_arm = 0
                                            t3_v_round += 1
                                            degs = getattr(cfg, 'T3_V_ROUND_DEG',
                                                           (45.0, 70.0, 100.0))
                                            if t3_v_round >= len(degs):
                                                # Whole fan flown empty: mirror and start
                                                # over - the operator's chosen loop, kept
                                                # safe by the bounds box.
                                                t3_v_round = 0
                                                t3_patrol_side = -t3_patrol_side
                                                t3_state = 'INIT'
                                                print("[TASK3] V fan exhausted - "
                                                      "mirrored restart.")
                                            else:
                                                t3_state = 'V_CALC'

                            elif t3_state == 'GOTO_STANDOFF':
                                target_lat, target_lon = t3_move_target
                                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0:
                                    # Patrol parallel to the buoy line (perpendicular to the
                                    # approach axis). First leg one way, second leg twice as
                                    # far the other way; driving each leg the camera faces
                                    # along the line, so the 15 m ceiling extends coverage
                                    # well past the leg ends. 15 + 30 covers a 30 m span
                                    # from any anchor position on it, whichever side the
                                    # target is on - the order is random by the rules.
                                    lat_brg = (t3_heading0 + t3_patrol_side * 90.0) % 360
                                    t3_move_target = calculate_obj_gps(
                                        ida_enlem, ida_boylam,
                                        getattr(cfg, 'T3_PATROL_FIRST_LEG_M', 15.0), lat_brg)
                                    target_lat, target_lon = None, None
                                    t3_state = 'PATROL_1'

                            elif t3_state == 'PATROL_1':
                                target_lat, target_lon = t3_move_target
                                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0:
                                    lat_brg = (t3_heading0 - t3_patrol_side * 90.0) % 360
                                    t3_move_target = calculate_obj_gps(
                                        ida_enlem, ida_boylam,
                                        getattr(cfg, 'T3_PATROL_SECOND_LEG_M', 30.0), lat_brg)
                                    target_lat, target_lon = None, None
                                    t3_state = 'PATROL_2'

                            elif t3_state == 'PATROL_2':
                                target_lat, target_lon = t3_move_target
                                if nav.haversine(ida_enlem, ida_boylam, target_lat, target_lon) < 2.0:
                                    target_lat, target_lon = None, None
                                    t3_state = 'RETURN'

                            elif t3_state == 'RETURN':
                                # User's choice over "stop and wait": come home to T3_MID
                                # and run the whole pattern again mirrored. The bounding box
                                # is what makes this loop safe to repeat indefinitely.
                                if have_mid:
                                    target_lat, target_lon = mid_lat, mid_lon
                                    if nav.haversine(ida_enlem, ida_boylam, mid_lat, mid_lon) < 2.5:
                                        target_lat, target_lon = None, None
                                        t3_patrol_side = -t3_patrol_side
                                        t3_state = 'INIT'
                                        print("[TASK3] Back at T3_MID - restarting mirrored.")
                                else:
                                    t3_patrol_side = -t3_patrol_side
                                    t3_state = 'INIT'

                        if target_lat is None:
                            skip_default_nav = True
                    else:
                        mevcut_gorev, target_lat, target_lon = execute_task3(mevcut_gorev, ida_enlem, ida_boylam)
                        # --- Early engage, T3_MID leg only ---
                        # Until now the boat drove all the way to the MID waypoint with the
                        # target plainly in view, because the servo scan only ran inside the
                        # search state. Three gates keep this from firing on garbage: the
                        # track must have been seen >= 3 times (one YOLO misfire cannot
                        # launch a full-throttle dive), it must be fresh, and the colour must
                        # be COMMITTED - config-decided, or actually delivered by the drone.
                        # While the GCS shows RENK YOK, no early dive. T3_START is excluded
                        # deliberately: red and green are also Task 1 gate colours, and that
                        # leg can still be near the gates.
                        if mevcut_gorev == "T3_MID" and mission_active:
                            _committed = ((not getattr(cfg, 'DRONE_ACTIVE', False))
                                          or bool(shared_state.get('drone_target_color')))
                            if _committed:
                                _tc = t3_target_cid()
                                _now_e = time.time()
                                for obj in vision_objects:
                                    if (obj.get('cid') == _tc
                                            and obj.get('cx') is not None
                                            and int(obj.get('seen', 0)) >= 3
                                            and obj.get('last_seen') is not None
                                            and (_now_e - obj['last_seen']) <= T3_MAX_DETECTION_AGE_S):
                                        print("[TASK3] Target sighted on the T3_MID leg - engaging early.")
                                        mevcut_gorev = "TASK3_SEARCH_KAMIKAZE"
                                        target_lat, target_lon = None, None
                                        break

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
                        current_path_ts = 0.0
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
                    # Kept shorter than the 15 m at which obstacles stop being drawn, so the
                    # goal always sits inside well-observed space rather than on the ragged
                    # outer edge of the costmap.
                    projection_dist = min(hedefe_mesafe,
                                          getattr(cfg, 'A_STAR_GOAL_PROJECTION_M', 10.0))

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
                        # Emergency braking bypasses the slew limiter: an instantaneous
                        # reversal is exactly what is wanted here.
                        if not acil_durum_aktif_mi:
                            shock_brake_pwm = cfg.BASE_PWM - getattr(cfg, 'shock_pwm', 250)
                            # Braking: Reverse thrust, hard steer away
                            avoid_turn = 400 if (left_d > right_d) else -400
                            apply_motor_mixer(controller, shock_brake_pwm, avoid_turn, slew=False)
                            time.sleep(0.1)
                            acil_durum_aktif_mi = True

                        escape_pwm = cfg.BASE_PWM - getattr(cfg, 'ESCAPE_PWM', 300)
                        avoid_turn = 400 if (left_d > right_d) else -400
                        apply_motor_mixer(controller, escape_pwm, avoid_turn, slew=False)
                        time.sleep(0.4)

                        spot_turn_val = getattr(cfg, 'SPOT_TURN_PWM', 200)
                        if left_d > right_d:
                            apply_motor_mixer(controller, cfg.BASE_PWM, -spot_turn_val, slew=False)
                        else:
                            apply_motor_mixer(controller, cfg.BASE_PWM, spot_turn_val, slew=False)
                        time.sleep(0.3)
                        current_path = None  # Force replan
                        current_path_ts = 0.0
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
                            # Task 3 turns on a lower authority: its search patterns are
                            # corner-heavy and the full 200 spun the boat at 35-57 deg/s,
                            # which both looked frantic and left the ZED-filtered heading
                            # lagging the hull. Task 1/2 keep the original value.
                            if "T3" in mevcut_gorev or "TASK3" in mevcut_gorev:
                                base_spot_pwm = getattr(cfg, 'T3_SPOT_TURN_PWM', 140)
                            else:
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
                                current_path_ts = 0.0
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

                                # --- Task 2 corridor cap (soft) ---
                                # Orange boundary buoys are otherwise indistinguishable from
                                # yellow obstacles, so routing around the OUTSIDE of the
                                # course is both legal and shorter whenever a waypoint sits
                                # near the boundary. That is what took the boat out of the
                                # course on the 2026-08-12 run.
                                corridor_cost = None
                                if (getattr(cfg, 'ENABLE_TASK2_CORRIDOR', True)
                                        and "TASK2" in mevcut_gorev and origin_lat is not None):
                                    axis = _task2_axis_local(origin_lat, origin_lon)
                                    if axis is not None:
                                        boundary = _confirmed_boundary_local(
                                            vision_objects, ida_enlem, ida_boylam,
                                            robot_x, robot_y)
                                        corridor_cost = planner.build_corridor_cost(
                                            nav_map.shape, cropped_center_m,
                                            COSTMAP_RES_M_PER_PX, axis[0], axis[1],
                                            boundary, (robot_x, robot_y))

                                # --- Planning is decoupled from control ---
                                # A* runs every A_STAR_PLAN_DIVISOR cycles (5 Hz at 25 Hz
                                # loop) with A_STAR_TIME_BUDGET_S; Pure Pursuit below runs
                                # every cycle on the last good path.
                                plan_divisor = max(1, int(getattr(cfg, 'A_STAR_PLAN_DIVISOR', 5)))
                                plan_timer += 1
                                if (plan_timer >= plan_divisor or not current_path) and tx_world is not None:
                                    plan_timer = 0
                                    new_path = None
                                    _los_cost = getattr(cfg, 'CORRIDOR_LOS_BLOCK_COST', 3.0)
                                    if planner.check_line_of_sight((robot_x, robot_y), (tx_world, ty_world), nav_map,
                                                                   cropped_center_m, COSTMAP_RES_M_PER_PX,
                                                                   cropped_size_px,
                                                                   corridor_cost, _los_cost):
                                        new_path = [(robot_x, robot_y), (tx_world, ty_world)]
                                    else:
                                        new_path = planner.get_path_plan(
                                            (robot_x, robot_y), (tx_world, ty_world),
                                            nav_map, cropped_center_m,
                                            COSTMAP_RES_M_PER_PX, cropped_size_px,
                                            heuristic_weight=getattr(cfg, 'A_STAR_HEURISTIC_WEIGHT', 2.5),
                                            max_expansions=40000, time_budget_s=0.15,
                                            cost_map=corridor_cost, los_max_cost=_los_cost)

                                        # Fallback ladder. Staying inside the corridor is a
                                        # preference, not a hard requirement: if the cap
                                        # leaves no route at all, an obstacle-avoiding path
                                        # that steps outside beats the alternative, which is
                                        # dropping to the obstacle-BLIND Direct-Drive PID.
                                        if new_path is None and corridor_cost is not None:
                                            new_path = planner.get_path_plan(
                                                (robot_x, robot_y), (tx_world, ty_world),
                                                nav_map, cropped_center_m,
                                                COSTMAP_RES_M_PER_PX, cropped_size_px,
                                                heuristic_weight=getattr(cfg, 'A_STAR_HEURISTIC_WEIGHT', 2.5))
                                            if new_path:
                                                print("[NAV_PROCESS] Corridor cap left no route "
                                                      "- replanning without it.")
                                    if new_path and len(new_path) >= 2:
                                        current_path = new_path
                                        current_path_ts = time.time()
                                        path_progress_idx = 0
                                    else:
                                        # A* gave up (no route, expansion cap, or time budget).
                                        # The old code left `current_path` untouched here, so
                                        # the boat kept following a path planned from a pose it
                                        # had long since left - and find_lookahead_point would
                                        # then latch onto a node behind it. Dropping the path
                                        # hands control to the Direct-Drive PID below: obstacle
                                        # blind, but smooth, and far safer than a stale route.
                                        current_path = None
                                        current_path_ts = 0.0
                                        path_progress_idx = 0
                                        # Say so. This fallback used to be completely silent,
                                        # which made a contact impossible to diagnose after the
                                        # fact: "position error ate the margin" and "the planner
                                        # quit and the blind PID drove straight at it" look
                                        # identical on the water. Now the log tells them apart -
                                        # if a hit happens with no line here, it was the margin.
                                        if (start_time - last_astar_fail_log) >= 1.0:
                                            last_astar_fail_log = start_time
                                            print("[NAV_PROCESS] A* found no path - falling "
                                                  "back to obstacle-blind PID toward target.")

                                # Never follow a path older than A_STAR_MAX_PATH_AGE_S.
                                max_age = getattr(cfg, 'A_STAR_MAX_PATH_AGE_S', 0.6)
                                if current_path and (time.time() - current_path_ts) > max_age:
                                    current_path = None
                                    current_path_ts = 0.0
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
                                cruise_offset = getattr(cfg, 'CRUISE_PWM', 80)
                                if mevcut_gorev.startswith("T3_") or "TASK3" in mevcut_gorev:
                                    cruise_offset = getattr(cfg, 'T3_SPEED_PWM', 100)
                                base_pwm = getattr(cfg, 'BASE_PWM', 1500) + cruise_offset

                                if active_controller != 'PP':
                                    prev_heading_error = 0.0   # avoid a derivative kick on handover
                                    active_controller = 'PP'
                                    pp_spot_turn_active = False

                                # Ground speed in m/s - NOT a PWM offset. Feeding PWM here pinned
                                # the lookahead at PURE_PURSUIT_MAX_LOOKAHEAD permanently.
                                # `speed_mps` is now a local updated once per cycle from the FC
                                # (see section A) instead of a Manager-dict round trip.
                                p_base, p_steer, raw_target, current_error, path_progress_idx = \
                                    planner.pure_pursuit_control(
                                        robot_x, robot_y, robot_yaw, current_path,
                                        current_speed=speed_mps, base_speed=base_pwm,
                                        prev_error=prev_heading_error, dt=loop_dt,
                                        start_idx=path_progress_idx)

                                prev_heading_error = current_error

                                # --- Throttle taper + spot-turn guard ---
                                # The Direct-Drive PID branch below has always had
                                # SPOT_TURN_THRESHOLD; the Pure Pursuit branch had nothing. So
                                # the boat held full cruise thrust while commanding maximum
                                # differential at 150 deg of heading error - which traces a
                                # circle by construction. That is precisely what the 23:47:59 to
                                # 23:48:08 flight segment shows.
                                abs_err = abs(current_error)
                                enter_deg = getattr(cfg, 'PP_SPOT_TURN_ENTER_DEG', 60.0)
                                exit_deg = getattr(cfg, 'PP_SPOT_TURN_EXIT_DEG', 35.0)

                                if pp_spot_turn_active:
                                    if abs_err < exit_deg:
                                        pp_spot_turn_active = False
                                elif abs_err > enter_deg:
                                    pp_spot_turn_active = True

                                if pp_spot_turn_active:
                                    # Pivot in place until we are pointing roughly the right way.
                                    p_base = getattr(cfg, 'BASE_PWM', 1500)
                                else:
                                    # Smoothly bleed off forward thrust as the error grows, so the
                                    # boat stops skidding sideways through its own turn.
                                    taper_min = getattr(cfg, 'PP_THROTTLE_TAPER_MIN', 0.25)
                                    taper = math.cos(math.radians(min(abs_err, 90.0)))
                                    taper = max(taper_min, taper)
                                    p_base = getattr(cfg, 'BASE_PWM', 1500) + (cruise_offset * taper)

                                apply_motor_mixer(controller, p_base, p_steer)

                                # Path consumed: force a fresh plan on the next cycle.
                                end_pt = current_path[-1]
                                if math.hypot(end_pt[0] - robot_x, end_pt[1] - robot_y) < 0.5:
                                    current_path = None
                                    current_path_ts = 0.0
                                    path_progress_idx = 0

                            # If no path, or we are NOT in Task 2 (meaning Task 1 or 3), use PID Direct Drive
                            else:
                                current_path = None  # Force a fresh A* attempt on the next plan_timer cycle
                                current_path_ts = 0.0
                                path_progress_idx = 0
                                pp_spot_turn_active = False
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
                                        if "T3" in mevcut_gorev or "TASK3" in mevcut_gorev:
                                            spot_pwm = getattr(cfg, 'T3_SPOT_TURN_PWM', 140)
                                        else:
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

                # Record final PWMs + speed for the HUD and the GCS.
                # These are display-only: nothing in the control loop reads them back, so
                # publishing them at 50 Hz was 6 pickled Manager-dict writes per cycle for
                # values that are consumed by a 30 fps overlay and a 2 Hz telemetry link.
                if (start_time - last_publish) >= PUBLISH_PERIOD_S:
                    last_publish = start_time
                    shared_state['motor_pwm_left'] = controller.get_servo_pwm(CH_REAR_LEFT)
                    shared_state['motor_pwm_right'] = controller.get_servo_pwm(CH_REAR_RIGHT)
                    shared_state['motor_pwm_front_left'] = controller.get_servo_pwm(CH_FRONT_LEFT)
                    shared_state['motor_pwm_front_right'] = controller.get_servo_pwm(CH_FRONT_RIGHT)
                    shared_state['motor_pwm_steer'] = controller.get_servo_pwm(CH_STEER)
                    shared_state['horizontal_speed'] = speed_mps
                    if fc_hdg is not None:
                        shared_state['fc_heading'] = fc_hdg
                    # -1 means the vehicle never reported RELAY_STATUS, so the GCS can say
                    # "unknown" instead of showing a state it cannot actually vouch for.
                    shared_state['relay_state'] = (-1 if relay_reported is None
                                                   else int(relay_reported))

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
                    apply_motor_mixer(controller, 1500, 0, slew=False)
                except Exception:
                    pass
                shared_state['nav_heartbeat'] = time.time()


            # Loop rate is cfg.NAV_LOOP_HZ (25 Hz), down from a hardcoded 50 Hz. A USV's yaw
            # time constant is 1-2 s, so 25 Hz is still 12-25x oversampled - but it halves the
            # MAVLink command rate to the flight controller and the Manager-dict IPC traffic,
            # both of which were measurably starving the rest of the system (camera FPS fell
            # 30 -> 17 and GPS updates stalled for ~1.5 s during the failure window).
            elapsed = time.time() - start_time
            if elapsed < NAV_PERIOD_S:
                time.sleep(NAV_PERIOD_S - elapsed)

    except Exception as e:
        print(f"[NAV_PROCESS][ERROR] Brain crashed: {e}")
    finally:
        print("[NAV_PROCESS] Shutting down...")
        if costmap_recorder is not None:
            costmap_recorder.save()
        try:
            apply_motor_mixer(controller, 1500, 0, slew=False)

            controller.disarm_vehicle()
        except:
            pass