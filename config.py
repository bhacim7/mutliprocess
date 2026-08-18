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
# Total clearance A* keeps from a buoy CENTRE = BUOY_RADIUS_M + ROBOT_RADIUS_M + this.
#
# History, because the value was got wrong twice in both directions:
#   0.55 -> too wide. A 1.5 m gap between buoy surfaces came out negative, so tight
#           passages were invisible and going around the outside of the course was the
#           only route A* could see.
#   0.25 -> too narrow. Set from the per-track scatter (0.11 m), which is the WRONG
#           statistic - that is precision within one track, not how well a buoy's
#           position is known. The boat then hit two yellow buoys it had detected
#           correctly at 2.0 m.
#   0.45 -> grazed in the water.
#   0.50 -> current. Completed Task 2 without contact and without leaving the course.
#
# The geometry is exact:
#
#     hull-to-buoy gap = INFLATION_MARGIN_M - position_error
#
# because A* keeps the boat CENTRE at BUOY + ROBOT + INFLATION from the MAPPED position
# while the true buoy may be `position_error` closer. Measured position error (between-track
# spread, i.e. how far the ~10 tracks of one physical buoy disagree) is 0.50 m mean,
# 0.85 m max.
#
#     INFL   gap at 0.50 m error   narrowest passable surface gap
#     0.25          -0.25 m                 1.30 m
#     0.45          -0.05 m                 1.70 m
#     0.50           0.00 m                 1.80 m
#     0.55          +0.05 m                 1.90 m
#
# No value satisfies both constraints on this course: avoiding contact wants INFL > 0.50,
# threading its 1.5 m gaps wants INFL <= 0.35. Closing that gap means reducing the position
# error itself - 0.50 m is about 2.9 deg of heading error at 10 m. Until then the corridor
# cap is what stops a blocked 1.5 m gap being answered by leaving the course.
INFLATION_MARGIN_M = 0.5
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

# Telemetry leaves on the boat's OWN clock at this interval - the GCS no longer polls.
# The poll round trip (GCS GUI timer -> air -> CommandReceiver sleep -> nav_process flag ->
# telem_process flag poll -> air) had jitter comparable to the interval itself, which is
# what an uneven, stuttering panel actually was. Same rate as the old 300 ms polling.
TELEM_BROADCAST_S = 0.3
# Mission waypoints are ~513 B of the packet. Send them only when they change, plus a
# periodic refresh so a GCS that connects late still gets them.
TELEM_WAYPOINT_REFRESH_S = 10.0
HEADING_SOURCE = 'ZED'

FC_PORT = "/dev/ttyACM0"  # OrangeCube Uçuş Kontrolcüsü Portu
FC_BAUD = 57600

# --- MOTOR POWER RELAY ---
# Cube side (already set, nothing new needed for the GCS path):
#   SERVO9_FUNCTION = -1   AUX1 becomes GPIO instead of a servo output
#   RELAY1_PIN      = 50   relay 1 is bound to AUX1
#   RELAY1_FUNCTION = 1    relay function enabled
#   RELAY1_DEFAULT  = 0    boots de-energised, i.e. motors unpowered  <- keep at 0
#   RC7_OPTION      = 28   transmitter switch drives the same relay
#
# The first three define the relay itself; RC7_OPTION is merely one input to it and
# MAVLink DO_SET_RELAY is another. Both write the same state inside ArduPilot, so the last
# one wins - the transmitter keeps working exactly as before.
#
# MAV_CMD_DO_SET_RELAY's instance argument is ZERO based: RELAY1 is instance 0.
RELAY_INSTANCE = 0
# Retries, because command_long_send waits for no ACK. They stop early once RELAY_STATUS
# confirms the vehicle agrees, and are spread over nav cycles rather than slept through.
RELAY_COMMAND_RETRIES = 5
RELAY_RETRY_INTERVAL_S = 0.25

# Used to ignore commands addressed to another vehicle. The RFD link is shared with the
# drone and CommandReceiver queues every line it hears, whoever it was meant for.
VEHICLE_ID = 1

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

# --- Run replay video ---
# The raw recording (.npz) is ALWAYS written next to the PNG; it costs milliseconds and a
# few hundred KB, and it is what utils/costmap_video.py replays. This flag only controls
# whether the mp4 is also rendered during shutdown, which adds 15-30 s to every exit.
# Leave it False for competition runs and render the video afterwards from the .npz:
#     python utils/costmap_video.py final_costmap.npz
COSTMAP_REC_VIDEO = False
COSTMAP_REC_VIDEO_SAMPLE_HZ = 1.0    # map snapshots per second of RUN time
# Playback rate. Sampling and playback are different things: 1 Hz sampled and played at
# 1 fps would make a ten minute run a ten minute video. 1 Hz at 15 fps plays it 15x, so a
# ten minute run becomes 40 s - and each frame carries its elapsed time.
COSTMAP_REC_VIDEO_FPS = 15.0
COSTMAP_REC_VIDEO_SCALE = 2.0        # a 55 m course at 0.2 m/px is only ~275 px across
COSTMAP_REC_VIDEO_MAX_FRAMES = 900   # long runs sample slower instead of growing forever

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
# How long a confirmed boundary buoy keeps counting after it was last actually SEEN.
#
# 1.5 s was far too strict, and it interacted badly with the camera geometry: the FOV is
# +/-55 deg, so a buoy being passed abeam leaves the frame and 1.5 s later stopped being a
# boundary - even though its position sits safely in the 20 s object memory. The cap was
# therefore built only from buoys inside the forward cone AT THAT MOMENT, and on a course
# with irregular spacing one side regularly had none, which voids the whole cap
# (CORRIDOR_REQUIRE_BOTH_SIDES). Those gaps are when the boat slipped outside the oranges
# and came back in - the intermittent on/off of the cap is exactly that signature, seen
# about 1 run in 10.
#
# 6 s keeps a passed buoy constraining for ~3 m of travel at survey speed. The safety
# filters that justify trusting it are unchanged: pos_ok (confirmed within 12 m), seen >= 3,
# and the buoys are anchored - their position does not go stale the way a moving target's
# would. This also makes CORRIDOR_WINDOW_BACK_M actually do something: with a 1.5 s age
# limit nothing behind the boat could ever qualify.
CORRIDOR_CONFIRM_MAX_AGE_S = 6.0
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
# How far BACK along the corridor axis the alignment point sits. This is the "from how far
# out does it line itself up" number - not TASK2_APPROACH_LATERAL_M, which is a sideways
# threshold and was once raised on the assumption that it controlled this.
#
# 12 m made the aligned run longer than it needed to be. 5 m still gives the boat about ten
# seconds on the axis at survey speed, and the approach to the point stays cheap because the
# point is on the way in: for the 2026-08-17 geometry (16.3 m out, 3.68 m off axis) the whole
# detour works out at 0.14 m of extra travel.
TASK2_APPROACH_OFFSET_M = 5.0
# Radius counting as "arrived" at the alignment point. Scaled down with the offset above:
# at 3 m it would have swallowed more than half of a 5 m run, so the latch could close with
# only 2 m of aligned approach left. If the boat somehow never enters this radius nothing
# jams - once it passes the point, s_along > -offset makes the branch fall through anyway.
TASK2_APPROACH_REACHED_M = 2.0
# Only route via the approach point when we are this far off the corridor axis.
# Lined up in front of the mouth, a straight run at the entry is already aligned,
# and keying on lateral offset keeps the switch one-way instead of oscillating.
TASK2_APPROACH_LATERAL_M = 8.0

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

# --- ObjectMemoryManager: how long a buoy is remembered, and how its velocity is used ---
#
# The 2026-08-12 run created 292 tracks for 25-30 buoys, because a track was dropped after
# 5 s and a buoy is routinely out of frame for longer than that. Each recycle re-converged
# to a slightly different position, and A* - which re-reads the map every planning cycle -
# saw the obstacle jump ~0.5 m and could change which side it passed on. That is the
# between-track spread the costmap recorder prints; per-track precision was already 0.22 m.
#
# 20 s is a starting point, not a measured optimum. Raise it and stale detections linger
# proportionally longer; lower it and tracks start recycling again. The recorder's
# `tracks:` count is the number to watch - it should fall towards `buoys (clustered):`.
OBJECT_MEMORY_S = 20.0

# Velocity guards. These are what make the longer memory safe, and they are a fix in their
# own right: buoys are anchored, so any velocity attributed to them is noise. It was
# estimated over a single ~0.1 s frame gap, which turns 0.2 m of position noise into an
# apparent 2 m/s. Require a real baseline before believing a velocity, and never project it
# further ahead than a second - unclamped, a 0.4 m/s phantom velocity over a 20 s memory
# predicts the buoy 8 m away, the match fails, and the track duplicates instead of merging.
OBJECT_VEL_MIN_DT_S = 0.5       # ignore velocity updates computed over a shorter gap
OBJECT_VEL_MAX_PREDICT_S = 1.0  # cap forward projection during matching

# Maximum range at which a sighting is trusted to PLACE a buoy on the map.
#
# A detection becomes a position as (boat GPS) + (distance) along (heading + pixel offset), so
# heading error turns into a tangential position error proportional to range. Measured on the
# 2026-08-17 run: 6 of 9 orange sighting clouds were elongated ACROSS the line of sight rather
# than along it (median 73.5 deg, 0.68 m at 1 sigma) - so heading, not ZED depth, dominates.
# At the implied ~2.6 deg that is 0.55 m at 12 m but 1.36 m at 30 m, and two encounters differ
# by about twice that. Past the 2.5 m merge radius one real buoy becomes several tracks: the
# run logged 33 orange "buoys" at 2.91 m median spacing, 6 of them sitting within hull-contact
# distance of the boat's own track - positions a buoy cannot physically occupy.
#
# Yellow, seen at 4-11 m, produced 7 clean tracks from identical code. 12 m puts orange in the
# same regime. It also improves accuracy, not just the count: the stored position stops being
# an average contaminated by 30 m sightings.
#
# Tuning: if `tracks:` in the costmap output still exceeds the real buoy count, lower it; if
# obstacles appear too late, raise it. At 0.5 m/s, 12 m is 24 s of warning and the boat turns
# in about 2 s, so there is a lot of headroom. Note A_STAR_GOAL_PROJECTION_M is 10 m, so the
# planner's horizon stays inside this gate.
#
# Objects beyond the gate are NOT discarded - they stay in memory with pos_ok False, so Task 3
# still sees them (it servos on the pixel column, never on lat/lon). Only map consumers skip
# them, and the first close sighting replaces the bad position outright rather than averaging
# into it.
MAP_MAX_RANGE_M = 12.0

# --- TASK 3 search v2 (nav_process TASK3_SEARCH_KAMIKAZE) ---
# The field this is sized for: three buoys (red/black/green, RANDOM order) on a line,
# spacing up to 15 m so the span is <= 30 m; T3_MID is dropped by eye roughly 15 m short
# of the line, not necessarily opposite the middle buoy. The ZED depth gate caps detection
# at 15 m, so from T3_MID the far buoy can be ~30 m away - twice the ceiling. No stationary
# pan can cover that; the search must move, and it must stop moving when it leaves the
# plausible field.
T3_STANDOFF_M = 8.0          # stop this far in front of an anchor buoy - an observation
                             # distance, not a ramming one: from 8 m back the 15 m ceiling
                             # keeps both directions of the line in view
T3_PATROL_FIRST_LEG_M = 15.0   # first sweep along the line...
T3_PATROL_SECOND_LEG_M = 30.0  # ...then twice as far the other way (order is random)
# --- Expanding V fan (operator's design, replaces the rectangular transect sweep) ---
# T3_MID is dropped by hand FACING the buoy line from ~15 m, so the targets are almost
# certainly in the forward cone - the fan searches there first instead of opening with a
# 26 m drive to a side corner like the rectangle did. Per round, both arms are flown:
# diagonal leg from T3_MID at +/-deg, then an 8 m PROBE further along the arrival axis,
# a short tip pan there (the probe is what covers the dead-ahead 20-30 m zone), back to
# T3_MID, mirror arm. Angle AND radius grow per round; round 3 reaches the flanks and
# slightly behind the beam. All three rounds empty -> mirrored restart from round 1.
# NOTE: round 3's lateral reach (30*sin100 = 29.5 m) runs right at T3_BOUND_LATERAL_M;
# drift can trip the bound there, which safely restarts the fan mirrored.
T3_V_ROUND_DEG = (45.0, 70.0, 100.0)
T3_V_ROUND_LEG_M = (10.0, 20.0, 30.0)
T3_V_PROBE_M = 8.0
T3_V_TIP_PAN_DEG = 45.0        # tip pans face the axis and sweep short - the forward band
                               # is what matters there; the +/-90 opening pan stays as is
T3_PAN_TOL_DEG = 10.0          # pan stop window; 5 deg was missed at speed (ZED heading
                               # lags the hull) and the boat spun full extra circles

# Anchor shortcut on/off. True: any Task 3 buoy seen mid-search snaps to the
# standoff/patrol path (proved itself on the red-buoy run). False: the V fan runs pure -
# for field experiments comparing the pattern's own performance.
T3_ANCHOR_ENABLED = True

# Turn authority in Task 3 only. The search is corner-heavy and 200 spun the hull at a
# measured 35-57 deg/s - frantic to watch, and fast enough that the ZED-filtered heading
# lagged the hull and pan stop windows were missed. 140 gives ~22-38 deg/s. Task 1/2
# alignment and in-drive spot turns keep SPOT_TURN_PWM.
T3_SPOT_TURN_PWM = 140

# Servo lock fidelity window. Without it the lock dropped on ONE missed freshness check
# and instantly re-locked the nearest other candidate - on 2026-08-18 the rudder swung to
# an object 8.6 m away while the true target sat 3.4 m ahead, weaving the final approach.
# Within this grace the boat steers on the locked track's last pixel column instead.
T3_LOCK_GRACE_S = 0.6
T3_BOUND_LATERAL_M = 30.0    # search box around the T3_MID waypoint; leaving it triggers
T3_BOUND_FORWARD_M = 20.0    # a return to T3_MID and a mirrored restart
T3_PAN_WIDE_DEG = 90.0       # opening sweep; with the +/-55 deg FOV this sees ~all around
T3_HOLD_S = 2.0              # lost the visual lock inside T3_HOLD_MAX_DIST_M: hold course
T3_HOLD_MAX_DIST_M = 5.0     # this long before falling back to the search

TASK3_KAMIKAZE_COLOR = "black"  # Options: "red", "green", "black"
TASK3_INVERT_STEERING = False  # Toggle if boat turns away from target

# When True, TASK3 hunts the colour the drone reported (relayed by the GCS as
# `set_target_color`) instead of TASK3_KAMIKAZE_COLOR above. This was False while the
# link was being built, which meant the boat received the colour, stored it and even
# logged "Drone target color updated to ..." - and then ignored it.
#
# TASK3_KAMIKAZE_COLOR stays the fallback: if the drone never delivers a colour, the
# boat hunts the configured one rather than sitting with no target at all. The GCS shows
# which of the two is in force (see the TASK3 RENK field), so a silent fallback is
# visible on the ground instead of being discovered on the water.
DRONE_ACTIVE = True

MEVCUT_GOREV = "TASK1_STATE_ENTER"