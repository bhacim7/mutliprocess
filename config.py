# ==========================================
# CONFIGURATION FILE
# ==========================================

# --- SYSTEM SETTINGS ---
NAV_MODE = "GPS"
STREAM = True
GCS_IP = "192.168.1.105"
RECORD_VIDEO = False
RECORD_COSTMAP = True
SHOW_LOCAL_WINDOW = False

# --- HARDWARE SPECS ---
ROBOT_RADIUS_M = 0.4
# Physical radius of a buoy as painted into the costmap. This used to be a hardcoded
# 6 px (0.60 m) inside nav_process; it is now a real, tunable quantity so that ALL the
# safety margin lives in INFLATION_MARGIN_M where the geometry is applied correctly.
BUOY_RADIUS_M = 0.25
# Total clearance A* keeps from a buoy centre = BUOY_RADIUS_M + ROBOT_RADIUS_M + INFLATION_MARGIN_M
#   0.25 + 0.40 + 0.55 = 1.20 m   (was 0.60 + 0.50 = 1.10 m, but mis-decomposed)
#
# TUNING RULE - measure the real gate width before raising this:
#   free_gap = corridor_width - 2 * (BUOY_RADIUS_M + ROBOT_RADIUS_M + INFLATION_MARGIN_M)
#   free_gap must stay >= 1.0 m or A* will find NO path and fall back to the
#   obstacle-blind PID, which is worse than the current behaviour.
#     corridor 3.0 m -> INFLATION_MARGIN_M <= 0.35
#     corridor 4.0 m -> INFLATION_MARGIN_M <= 0.85
#     corridor 5.0 m -> INFLATION_MARGIN_M <= 1.35
# 0.55 -> 0.25 -> 0.45.
#
# 0.25 was set from the per-track scatter (0.11 m), which was the WRONG statistic: that
# figure is precision within one track, not how well the buoy's position is actually known.
# Clustering the tracks gave the real number - between-track spread, mean 0.50 m - and with
# it the boat hit two yellow buoys it had detected correctly at 2.0 m.
#
# The geometry is exact:
#
#     hull-to-buoy gap = INFLATION_MARGIN_M - position_error
#
# because A* keeps the boat CENTRE at BUOY + ROBOT + INFLATION from the MAPPED position,
# and the true buoy may be `position_error` closer than that. At 0.25 with 0.50 m of error
# the gap is -0.25 m, i.e. contact, which is exactly what happened.
#
#     INFL   gap at 0.50 m error   narrowest passable surface gap
#     0.25          -0.25 m                 1.30 m
#     0.45          -0.05 m                 1.70 m
#     0.55          +0.05 m                 1.90 m
#
# There is no value that both avoids contact and threads the 1.5 m gaps on this course -
# that needs the position error itself to come down. 0.45 is the compromise: grazing rather
# than hitting, and passages from 1.7 m up stay open.
INFLATION_MARGIN_M = 0.45
MAX_TILT_ANGLE = 5.0

# --- PINS / CHANNELS ---
SOL_MOTOR = 6
SAG_MOTOR = 3
FRONT_SOL_MOTOR = 5
FRONT_SAG_MOTOR = 2
FRONT_STEER_SERVO = 7

BASE_PWM = 1500
CRUISE_PWM = 180
T3_SPEED_PWM = 200

STEER_MAX_PWM = 1900
STEER_MIN_PWM = 1100
STEER_DEADBAND_DEG = 5.0

# --- SENSORS ---
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 57600
# Hard ceiling on one telemetry line. At 57600 baud (5760 B/s) a 0.5 s write_timeout can
# only push 2880 bytes, and a truncated line has no trailing '\n' - the GCS then glues it
# onto the next packet and json.loads() throws, so BOTH packets are lost silently.
# The payload is kept well under this; the guard in utils/telem.py is the safety net.
TELEM_MAX_PAYLOAD_B = 1500
# Mission waypoints are ~513 B of the packet. Send them only when they change, plus a
# periodic refresh so a GCS that connects late still gets them.
TELEM_WAYPOINT_REFRESH_S = 10.0
HEADING_SOURCE = 'ZED'

FC_PORT = "/dev/ttyACM0"  # OrangeCube Uçuş Kontrolcüsü Portu
FC_BAUD = 57600

LIDAR_PORT_NAME = "/dev/ttyUSB1"
LIDAR_BAUDRATE = 1000000
LIDAR_MAX_DIST = 10.0
LIDAR_ACIL_DURMA_M = 1.5
LIDAR_FREE_GAIN = 25
ENABLE_LIDAR = False
LIDAR_OCCUPIED_GAIN = 80
LIDAR_KORIDOR_KP = 30.0
MAP_DECAY_AMOUNT = 1

# --- POST-MISSION COSTMAP RECORDER (diagnostics only, no effect on control) ---
# The old recorder stamped every 1 Hz sighting into a fixed 500 m canvas at 0.5 m/px and
# never erased anything, so a single buoy smeared into a ~14 m blob and ~90% of the saved
# PNG was empty black. It now records in metres and renders once, auto-fitting the canvas.
COSTMAP_REC_RES_M_PER_PX = 0.2
COSTMAP_REC_TRACK_HZ = 5.0    # boat track sampling - 1 Hz turned a 10 s circle into a decagon
COSTMAP_REC_OBJECT_HZ = 1.0   # raw sighting sampling (feeds the noise-cloud layer)

CAM_RES = 2
CAM_FPS = 30
CAM_HFOV = 110.0
YOLO_CONFIDENCE = 0.40
MODEL_PATH = "/home/arge/PycharmProjects/PythonProject/best (2).engine"
Kp_PIXEL = 0.3
Kd_PIXEL = 0.1

# --- NAVIGATION CONTROL ---
SPOT_TURN_THRESHOLD = 30.0
SPOT_TURN_PWM = 200
ESCAPE_PWM = 300
# Per-cycle slew limit on every thruster/servo channel. This existed in config for a long
# time but was NEVER referenced by any code; measured behaviour was ~800 PWM reversals in
# 0.25 s. Now enforced in nav_process.apply_motor_mixer().
# At NAV_LOOP_HZ = 25 this is 1500 PWM/s, i.e. the full 1100..1900 range in ~0.53 s.
# Emergency braking and the SIGTERM neutralise deliberately bypass it.
MAX_PWM_CHANGE = 60
HYBRID_STEP_DIST = 2.0
HYBRID_HEADING_THRESHOLD = 30.0
ENABLE_LOS_GUIDANCE = True
LOS_KP = 1.2
DIRECT_DRIVE_KP = 1.5
DIRECT_DRIVE_KI = 0.05
DIRECT_DRIVE_KD = 1.2
ANTI_WINDUP_DEG = 15.0
ANTI_WINDUP_LIMIT = 500.0

# --- PURE PURSUIT (A* PATH FOLLOWING) ---
PURE_PURSUIT_MIN_LOOKAHEAD = 0.8
# Raised from 1.5: with get_horizontal_speed() fixed the lookahead is finally allowed to
# grow with speed, and 1.5 m at ~1.8 m/s is still under a second of horizon.
PURE_PURSUIT_MAX_LOOKAHEAD = 2.5
PURE_PURSUIT_K_SPEED = 1.0
# KP was 5.5 / KD 2.5. With BASE 1500 + CRUISE 180 = 1680 and the mixer's 1.486 dynamic
# multiplier, the rear thrusters saturate at |correction| >= 148, i.e. a heading error of
# only 27 deg at KP=5.5. Measured in flight: 1100/1900 rails at ACI_FARKI = 4.
# KP = 2.0 moves the saturation point out to ~74 deg, matching the Direct-Drive PID
# (KP 1.5 -> 99 deg) instead of being 3.7x more aggressive than it.
# KD 0.8 keeps damping but stops a single replan (error stepping 80 deg in one cycle)
# from contributing 200 counts of correction on its own.
PURE_PURSUIT_KP = 2.0
PURE_PURSUIT_KD = 0.8
A_STAR_HEURISTIC_WEIGHT = 2.5
COSTMAP_RES_M_PER_PX = 0.10
A_STAR_CROP_RADIUS_M = 20.0  # Local A* window radius.

# How far ahead the GPS target is projected into the local map for A* to aim at.
# This used to be A_STAR_CROP_RADIUS_M * 0.75 = 15 m, which is exactly the range at which
# obstacles stop being drawn into the costmap (see nav_process). The planner was therefore
# aiming at the outer edge of what it could see, where detections are newest, smallest in
# the image and least reliable. Keeping the goal inside the well-observed region is the
# point of making this its own number.
A_STAR_GOAL_PROJECTION_M = 10.0

# --- TASK 2 CORRIDOR (lateral cap) ---
# The boat left the course when a waypoint sat near the boundary: orange boundary buoys are
# just point obstacles to A*, so routing around the OUTSIDE of them is both legal and
# shorter. The fix is a soft rule - "do not pass on the far side of an orange buoy" -
# expressed as extra cost rather than a wall, so a genuinely blocked corridor can still be
# escaped instead of deadlocking.
#
# Lateral offsets are measured against the line from the first Task 2 point to the last one.
# That line is the ONE thing the rules guarantee stays inside the course, and unlike a
# boat->goal axis it does not rotate with the approach, so left/right stays correct even
# when entering the corridor diagonally.
ENABLE_TASK2_CORRIDOR = True
# Only buoys whose distance along the axis falls in [boat - back, boat + ahead] set the cap.
# Anything further ahead says nothing about how far sideways we may go right now.
CORRIDOR_WINDOW_BACK_M = 5.0
CORRIDOR_WINDOW_AHEAD_M = 15.0
# The cap is only applied when BOTH sides are visible. Seeing one chain (which is what
# happens on the diagonal approach to the entrance, and whenever a chain is momentarily
# lost) tells us nothing about where the corridor ends, so no constraint is imposed.
CORRIDOR_REQUIRE_BOTH_SIDES = True
# A buoy counts once it has been seen this many times and was seen recently.
CORRIDOR_CONFIRM_SIGHTINGS = 3
CORRIDOR_CONFIRM_MAX_AGE_S = 1.5
# Extra clearance kept inside the boundary buoys, and the cost charged per cell beyond it.
# The penalty is per grid cell, against a base step cost of 1.0, so 6.0 makes a 1 m
# excursion (10 cells at 0.10 m/px) cost about 30 - far more than any sane detour.
# Measured: 3.0 and 6.0 give the same route, but a smaller penalty keeps the search cheap.
CORRIDOR_MARGIN_M = 0.5
CORRIDOR_PENALTY = 3.0
# Straight-line shortcuts may not cross cells costing more than this, otherwise the
# line-of-sight fast path would tunnel straight through the cap.
CORRIDOR_LOS_BLOCK_COST = 3.0
# Longitudinal bin width used to turn the buoys into a cap profile along the axis.
CORRIDOR_BIN_M = 2.0

# Approach point for the Task 2 entrance, set back along the axis. Coming in very obliquely,
# the shortest line to the entry waypoint can slip between the first and second buoys of a
# boundary chain rather than through the mouth. Aiming at a point behind the entrance first
# forces the last leg to be aligned with the corridor, whichever diagonal we arrive from.
TASK2_APPROACH_OFFSET_M = 12.0
TASK2_APPROACH_REACHED_M = 3.0  # approach point counts as reached inside this radius
# Only route via the approach point when we are this far off the corridor axis.
# Lined up in front of the mouth, a straight run at the entry is already aligned,
# and keying on lateral offset keeps the switch one-way instead of oscillating.
TASK2_APPROACH_LATERAL_M = 3.0

# Planning is decoupled from control: Pure Pursuit runs every nav cycle on the last good
# path, A* only every A_STAR_PLAN_DIVISOR cycles (25 Hz / 5 = 5 Hz) but with double the
# old time budget. 0.06 s only bought ~1-2k node expansions in pure Python, so A* returned
# None constantly and the boat followed a stale path.
A_STAR_PLAN_DIVISOR = 5
A_STAR_TIME_BUDGET_S = 0.12
# A path older than this is discarded rather than followed.
A_STAR_MAX_PATH_AGE_S = 0.6

# Throttle taper + spot-turn guard for the Pure Pursuit branch. The Direct-Drive PID had
# SPOT_TURN_THRESHOLD; the PP branch had nothing, so the boat held full cruise thrust while
# commanding maximum differential at 150 deg of error - which draws a circle by construction.
PP_THROTTLE_TAPER_MIN = 0.25   # never drop below 25% of CRUISE_PWM while still driving
PP_SPOT_TURN_ENTER_DEG = 60.0  # above this: cut forward thrust, pivot in place
PP_SPOT_TURN_EXIT_DEG = 35.0   # below this: resume normal cruise (hysteresis)

# --- LOOP RATES / IPC ---
# 50 Hz meant 5 set_servo calls x 50 = 250 COMMAND_LONG/s to the flight controller (each
# answered with a COMMAND_ACK) and ~1000 Manager-dict IPC round trips/s. Measured effect:
# GPS position froze for ~1.5 s and camera FPS collapsed 30 -> 17. A USV's yaw time
# constant is 1-2 s; 25 Hz is still 12-25x oversampled.
NAV_LOOP_HZ = 25.0
# Skip a servo write entirely if the commanded PWM moved less than this. Combined with the
# slew limiter this cuts the MAVLink command rate by roughly 60%.
SERVO_MIN_PWM_DELTA = 3
# ...but always refresh each channel at least this often so a dropped command cannot leave
# the FC latched on a stale value.
SERVO_REFRESH_S = 0.5
# The vision object list is the single most expensive shared_state entry (a pickled list of
# 10-30 dicts). Read it at this rate instead of every nav cycle.
VISION_READ_HZ = 10.0
# Display/telemetry-only values are published at this rate; nothing in the control loop
# reads them back.
NAV_PUBLISH_HZ = 10.0

# --- TASK WAYPOINTS (Legacy values from IDA1) ---
T1_GATE_ENTER_LAT =40.8090735
T1_GATE_ENTER_LON = 29.2622975
T1_GATE_MID_LAT = 40.8089036
T1_GATE_MID_LON = 29.2622449
T1_GATE_EXIT_LAT = 40.8088117
T1_GATE_EXIT_LON = 29.2624693

T2_ZONE_ENTRY_LAT = 40.8090289
T2_ZONE_ENTRY_LON = 29.262524
T2_ZONE_MID_LAT = 40.8095851
T2_ZONE_MID_LON = 29.2622612
T2_ZONE_END_LAT = 40.8095851
T2_ZONE_END_LON =  29.2622612

ENABLE_TASK3 = True
T3_START_LAT = 40.809666
T3_START_LON = 29.2622792
T3_MID_LAT = 40.8096887
T3_MID_LON = 29.2622795

TASK3_KAMIKAZE_COLOR = "black"  # Options: "red", "green", "black"
TASK3_INVERT_STEERING = False  # Toggle if boat turns away from target

DRONE_ACTIVE = False

MEVCUT_GOREV = "TASK1_STATE_ENTER"