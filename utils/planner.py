import math
import time
import numpy as np
import cv2
import heapq

import config as cfg
import utils.navigasyon as nav

# Nominal nav loop period. The PID/PP derivative and integral terms are scaled by
# (dt / NOMINAL_DT) so that a loop that runs slower than 50 Hz (e.g. the cycle where
# A* runs) does not silently change the effective KD/KI. At dt == NOMINAL_DT the
# behaviour is identical to the old un-normalised code, so existing config gains
# stay valid.
NOMINAL_DT = 0.02


def dt_scale(dt):
    if dt is None or dt <= 0.0:
        return 1.0
    return min(5.0, max(0.2, dt / NOMINAL_DT))


# --- UTILITY & HYBRID ---
def get_hybrid_point(robot_x, robot_y, robot_yaw, aci_farki, step_dist=2.0):
    target_angle = robot_yaw - math.radians(aci_farki)
    tx = robot_x + (step_dist * math.cos(target_angle))
    ty = robot_y + (step_dist * math.sin(target_angle))
    return tx, ty

def get_inflated_nav_map(raw_costmap):
    """
    Prepares map for A* by inflating obstacles to account for robot radius.
    """
    if raw_costmap is None or raw_costmap.size == 0:
        return None, None

    nav_map = raw_costmap.copy()

    # Obstacles are < 100 in the grayscale costmap
    obstacles_mask = (nav_map < 100).astype(np.uint8) * 255

    inflation_m = getattr(cfg, 'INFLATION_MARGIN_M', 0.25)
    robot_radius = getattr(cfg, 'ROBOT_RADIUS_M', 0.25)
    res = getattr(cfg, 'COSTMAP_RES_M_PER_PX', 0.10)

    kernel_size = (int((robot_radius + inflation_m) / res) * 2) + 1
    inflated_obstacles = cv2.dilate(obstacles_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

    nav_map[:] = 255 # Assume clear everywhere
    nav_map[inflated_obstacles > 0] = 0 # Mark inflated obstacles as black

    return nav_map, inflated_obstacles


def _segment_is_clear(nav_map, p1, p2):
    """
    O(L) obstacle check along a pixel segment. Samples one pixel per unit of the
    dominant axis instead of rasterising the line into a full-size mask and scanning
    the whole image (which cost ~160k element ops per call and made string_pulling
    the single most expensive thing in the planning cycle).
    """
    h, w = nav_map.shape[:2]

    x0, y0 = float(p1[0]), float(p1[1])
    x1, y1 = float(p2[0]), float(p2[1])

    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if n < 2:
        n = 2

    xi = np.rint(np.linspace(x0, x1, n)).astype(np.int32)
    yi = np.rint(np.linspace(y0, y1, n)).astype(np.int32)

    # Leaving the planning window is treated as blocked (we cannot vouch for it).
    if xi.min() < 0 or yi.min() < 0 or xi.max() >= w or yi.max() >= h:
        return False

    return not np.any(nav_map[yi, xi] == 0)


def check_line_of_sight(start, end, nav_map, center_m, res, size_px):
    """
    Raycast on the grid to check if a straight line between start and end is obstacle-free.
    """
    if nav_map is None: return True

    # Helper for world -> pixel inside planner
    def w2p(x, y):
        cw, ch = size_px[0] // 2, size_px[1] // 2
        dx = x - center_m[0]
        dy = y - center_m[1]
        px = int(cw + (dx / res))
        py = int(ch - (dy / res))
        if 0 <= px < size_px[0] and 0 <= py < size_px[1]:
            return px, py
        return None

    p1 = w2p(start[0], start[1])
    p2 = w2p(end[0], end[1])

    if p1 is None or p2 is None: return False

    return _segment_is_clear(nav_map, p1, p2)


# --- A* PATH PLANNING ---
def heuristic(a, b, weight=1.0):
    # Using Octile distance (Chebyshev variant) instead of Euclidean for much faster grid computation
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    return weight * (dx + dy + (1.414 - 2) * min(dx, dy))

def string_pulling(path, nav_map, center_m, res, size_px):
    """
    Smoothens the jagged A* grid path by drawing straight lines between furthest
    visible nodes, reducing zigzagging significantly.
    """
    if not path or len(path) <= 2:
        return path

    smoothed_path = [path[0]]
    current_idx = 0

    while current_idx < len(path) - 1:
        furthest_visible = current_idx + 1
        # Try to find the furthest node we can draw a straight line to without hitting an obstacle
        for test_idx in range(current_idx + 2, len(path)):
            if check_line_of_sight(path[current_idx], path[test_idx], nav_map, center_m, res, size_px):
                furthest_visible = test_idx
            else:
                # Visibility broken, keep the last visible
                break

        smoothed_path.append(path[furthest_visible])
        current_idx = furthest_visible

    return smoothed_path


def _find_free_cell(nav_map, cell, max_radius_px):
    """
    Nearest free cell to `cell`, searched outward in Chebyshev rings.

    With ROBOT_RADIUS_M + INFLATION_MARGIN_M = 2.1 m of inflation, ANY obstacle within
    2.1 m swallows the boat's own cell, and A* used to return None right when the
    obstacle was closest - handing control to the obstacle-blind PID at the worst
    possible moment. Relocating the search start (rather than punching a free disc
    around the boat, which can leave it on an isolated island of free cells with no
    way out) keeps the planner useful in exactly that situation.
    """
    h, w = nav_map.shape[:2]
    cx, cy = int(cell[0]), int(cell[1])

    if 0 <= cx < w and 0 <= cy < h and nav_map[cy, cx] != 0:
        return (cx, cy)

    for r in range(1, int(max_radius_px) + 1):
        best = None
        best_d2 = None
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue  # only the new ring
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h and nav_map[y, x] != 0:
                    d2 = dx * dx + dy * dy
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = (x, y)
        if best is not None:
            return best
    return None


def _nudge_goal_to_free(goal, start, nav_map):
    """
    If the goal cell sits inside an inflated obstacle, walk back along the goal->start
    line and take the first free cell. Without this, a target that happens to be behind
    (or on) a buoy makes A* give up entirely and hand control to the obstacle-blind PID.
    Returns None if the whole segment is blocked.
    """
    h, w = nav_map.shape[:2]
    if nav_map[goal[1], goal[0]] != 0:
        return goal

    x0, y0 = float(goal[0]), float(goal[1])
    x1, y1 = float(start[0]), float(start[1])
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if n < 2:
        return None

    xi = np.rint(np.linspace(x0, x1, n)).astype(np.int32)
    yi = np.rint(np.linspace(y0, y1, n)).astype(np.int32)
    np.clip(xi, 0, w - 1, out=xi)
    np.clip(yi, 0, h - 1, out=yi)

    free = np.nonzero(nav_map[yi, xi] != 0)[0]
    if free.size == 0:
        return None
    k = int(free[0])
    return (int(xi[k]), int(yi[k]))


def get_path_plan(start_world, end_world, nav_map, center_m, res, size_px,
                  heuristic_weight=2.5, max_expansions=20000, time_budget_s=0.06,
                  goal_tolerance_px=3):
    """
    A* algorithm implementation over the local costmap grid.
    Returns list of world coordinates (x,y).

    max_expansions / time_budget_s bound the search. A 20m crop at 0.10 m/px is a
    400x400 = 160k cell grid; on an unreachable goal an unbounded pure-Python A*
    sweeps the whole grid and blocks the single-threaded nav loop for seconds - long
    enough for the orchestrator watchdog (5s) to SIGTERM the process, which skips the
    `finally` that neutralises the motors. Giving up early and falling back to the PID
    is always the safer outcome.
    """
    if nav_map is None: return None

    # Helper for world <-> pixel
    def w2p(x, y):
        cw, ch = size_px[0] // 2, size_px[1] // 2
        px = int(cw + ((x - center_m[0]) / res))
        py = int(ch - ((y - center_m[1]) / res))
        return (px, py)

    def p2w(px, py):
        cw, ch = size_px[0] // 2, size_px[1] // 2
        x = center_m[0] + ((px - cw) * res)
        y = center_m[1] - ((py - ch) * res)
        return (x, y)

    start = w2p(*start_world)
    raw_goal = w2p(*end_world)

    # Safety Clamp: Ensure goal is strictly within the map boundaries.
    # If the target is pushed outside the cropped window (e.g. 50m target on a 20m window),
    # clamp it to the edge so A* doesn't get stuck in a full-grid search for an impossible point.
    goal = (max(0, min(size_px[0] - 1, raw_goal[0])),
            max(0, min(size_px[1] - 1, raw_goal[1])))

    # Start must be inside the window for the search to make sense.
    if not (0 <= start[0] < size_px[0] and 0 <= start[1] < size_px[1]):
        return None

    # Boat inside an inflated obstacle: plan from the nearest free cell instead of
    # giving up (see _find_free_cell). Search out to a bit past the inflation radius.
    inflation_px = int((getattr(cfg, 'ROBOT_RADIUS_M', 0.25) +
                        getattr(cfg, 'INFLATION_MARGIN_M', 0.25)) / res) + 4
    start = _find_free_cell(nav_map, start, inflation_px)
    if start is None:
        return None

    goal = _nudge_goal_to_free(goal, start, nav_map)
    if goal is None:
        return None  # Goal (and everything between it and us) is unreachable

    tol_sq = goal_tolerance_px * goal_tolerance_px
    deadline = time.time() + time_budget_s

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, heuristic_weight)}

    # 8-way movement (dx, dy)
    neighbors = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    # Direction tracking for kinematic penalty (heading change).
    # Entry format: (f_score, (px, py), (dx_from_parent, dy_from_parent))
    open_set = []
    heapq.heappush(open_set, (f_score[start], start, (0, 0)))

    expansions = 0

    while open_set:
        current_f, current, prev_dir = heapq.heappop(open_set)

        # Skip stale heap entries: a cell can be pushed several times with different
        # arrival directions, and only the cheapest one is worth expanding.
        if current_f > f_score.get(current, float('inf')) + 1e-9:
            continue

        dgx = current[0] - goal[0]
        dgy = current[1] - goal[1]
        if (dgx * dgx + dgy * dgy) <= tol_sq:
            path = []
            node = current
            while node in came_from:
                path.append(p2w(node[0], node[1]))
                node = came_from[node]
            path.append(p2w(start[0], start[1]))
            path.reverse()
            # Apply path smoothing post-processing
            return string_pulling(path, nav_map, center_m, res, size_px)

        expansions += 1
        if expansions >= max_expansions:
            return None
        if (expansions & 0x3FF) == 0 and time.time() > deadline:
            return None

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < size_px[0] and 0 <= neighbor[1] < size_px[1]:
                if nav_map[neighbor[1], neighbor[0]] == 0:
                    continue # Hit obstacle

                # Cost is 1 for straight, 1.414 for diagonal
                step_cost = 1 if dx == 0 or dy == 0 else 1.414

                # --- Kinematic Constraint Penalty ---
                # NOTE: this makes the true state space (cell, arrival direction) while
                # g_score/came_from are keyed by cell only, so the penalty biases the
                # search without being strictly enforced. It is kept because it visibly
                # reduces zigzag; string_pulling below does the rest.
                if prev_dir != (0, 0):
                    dot_product = prev_dir[0]*dx + prev_dir[1]*dy
                    mag_prev = math.sqrt(prev_dir[0]**2 + prev_dir[1]**2)
                    mag_curr = math.sqrt(dx**2 + dy**2)
                    if mag_prev > 0 and mag_curr > 0:
                        cos_angle = min(1.0, max(-1.0, dot_product / (mag_prev * mag_curr)))
                        angle_diff = math.degrees(math.acos(cos_angle))

                        # Penalize sharp turns (e.g., > 45 degrees)
                        if angle_diff > 45.0:
                            step_cost += (angle_diff * 0.1)  # Tunable weight
                        if angle_diff >= 90.0:
                            step_cost += 5.0 # Heavy penalty for reversals/right angles
                # ------------------------------------------------

                tentative_g_score = g_score[current] + step_cost

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_n = tentative_g_score + heuristic(neighbor, goal, heuristic_weight)
                    f_score[neighbor] = f_n
                    heapq.heappush(open_set, (f_n, neighbor, (dx, dy)))

    return None # No path found

# --- PURE PURSUIT CONTROLLER ---
def _circle_segment_intersection(cx, cy, r, p1, p2):
    """Furthest intersection of the lookahead circle with segment p1->p2, or None."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    a = dx * dx + dy * dy
    if a < 1e-9:
        return None

    fx = p1[0] - cx
    fy = p1[1] - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None

    disc = math.sqrt(disc)
    for t in (((-b + disc) / (2.0 * a)), ((-b - disc) / (2.0 * a))):
        if 0.0 <= t <= 1.0:
            return (p1[0] + t * dx, p1[1] + t * dy)
    return None


def find_lookahead_point(x, y, path, lookahead_dist, start_idx=0):
    """
    Returns (target_point, segment_index).

    The target is the actual intersection of the lookahead circle with the path, so
    the effective lookahead really is `lookahead_dist`. The previous version returned
    "the furthest node inside the circle, plus one", which - after string_pulling
    leaves nodes metres apart - made the effective lookahead essentially arbitrary.

    start_idx makes progress monotonic: the search never looks behind the segment we
    already committed to, so the boat cannot latch onto a piece of path it has passed.
    """
    n = len(path)
    if n == 0:
        return None, start_idx
    if n == 1:
        return path[0], 0

    start_idx = max(0, min(start_idx, n - 1))

    # 1. Closest node at or after our committed progress point
    closest_idx = start_idx
    min_d2 = float('inf')
    for i in range(start_idx, n):
        d2 = (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2
        if d2 < min_d2:
            min_d2 = d2
            closest_idx = i

    # 2. Blown off the path entirely - steer back to the nearest point first
    if min_d2 > lookahead_dist * lookahead_dist:
        return path[closest_idx], closest_idx

    # 3. First segment (walking forward) that crosses the lookahead circle
    for i in range(closest_idx, n - 1):
        hit = _circle_segment_intersection(x, y, lookahead_dist, path[i], path[i + 1])
        if hit is not None:
            return hit, i

    # 4. Whole remaining path is inside the circle - aim at the end of it
    return path[n - 1], n - 1


def pure_pursuit_control(rx, ry, ryaw, path, current_speed=0.0, base_speed=1500,
                         prev_error=0.0, dt=NOMINAL_DT, start_idx=0):
    """
    Executes the Pure Pursuit algorithm to generate Base Speed and Steering Correction.

    current_speed is the ground speed in m/s (NOT a PWM offset - feeding PWM here pinned
    the lookahead at its max and killed the speed-adaptive behaviour entirely).

    Returns (base_speed, correction, target_pt, heading_err, next_start_idx).
    The path itself is never modified: progress is carried in next_start_idx. Destructively
    pruning the path used to collapse it to a single point on the first call, which failed
    the caller's len(path) >= 2 test and silently handed control back to the Direct-Drive
    PID for 4 out of every 5 cycles.
    """
    if not path:
        return base_speed, 0, None, 0.0, start_idx

    # 1. Dynamic Lookahead Distance
    min_ld = getattr(cfg, 'PURE_PURSUIT_MIN_LOOKAHEAD', 1.0)
    max_ld = getattr(cfg, 'PURE_PURSUIT_MAX_LOOKAHEAD', 3.0)
    k_ld = getattr(cfg, 'PURE_PURSUIT_K_SPEED', 0.5) # Lookahead multiplier

    lookahead_dist = float(np.clip(float(current_speed) * k_ld, min_ld, max_ld))

    # 2. Find target point
    target_pt, t_idx = find_lookahead_point(rx, ry, path, lookahead_dist, start_idx)
    if target_pt is None:
        return base_speed, 0, None, 0.0, start_idx

    # 3. Calculate steering error
    tx, ty = target_pt

    # Degenerate target (we are sitting exactly on it): atan2(0, 0) would report a bearing
    # of due North and command a bogus turn. Hold the current heading instead.
    if math.hypot(tx - rx, ty - ry) < 1e-3:
        return base_speed, 0, target_pt, 0.0, t_idx

    # The world frame here is x = North, y = East (built from cos/sin of raw compass
    # bearings), so atan2(dy, dx) yields the compass bearing to the target directly.
    target_bearing_deg = math.degrees(math.atan2(ty - ry, tx - rx)) % 360
    current_heading_deg = math.degrees(ryaw) % 360

    # Same convention as navigasyon.signed_angle_difference(current, target):
    # positive means the target is to the right.
    heading_err = (target_bearing_deg - current_heading_deg + 180) % 360 - 180

    # 4. PD calculation (dt-normalised - see NOMINAL_DT)
    kp = getattr(cfg, 'PURE_PURSUIT_KP', 2.0)
    kd = getattr(cfg, 'PURE_PURSUIT_KD', 0.5)

    scale = dt_scale(dt)
    P = heading_err * kp
    D = ((heading_err - prev_error) / scale) * kd
    correction = P + D

    return base_speed, correction, target_pt, heading_err, t_idx
