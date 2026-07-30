import math
import numpy as np
import cv2
import heapq

import config as cfg
import utils.navigasyon as nav

# --- UTILITY & HYBRID ---
def get_hybrid_point(robot_x, robot_y, robot_yaw, aci_farki, step_dist=2.0):
    target_angle = robot_yaw - math.radians(aci_farki)
    tx = robot_x + (step_dist * math.cos(target_angle))
    ty = robot_y + (step_dist * math.sin(target_angle))
    return tx, ty

def get_inflated_nav_map(raw_costmap, ignore_green=False, ignore_yellow=False):
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

    if not p1 or not p2: return False

    # Create empty mask to draw the line
    line_mask = np.zeros_like(nav_map)
    cv2.line(line_mask, p1, p2, 255, 1)

    # Check collision: where line_mask is 255 AND nav_map is 0 (obstacle)
    collision = np.logical_and(line_mask == 255, nav_map == 0)

    return not np.any(collision)


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

def get_path_plan(start_world, end_world, nav_map, center_m, res, size_px, bias_to_goal_line=0.0, heuristic_weight=2.5, cone_deg=180.0):
    """
    A* algorithm implementation over the local costmap grid.
    Returns list of world coordinates (x,y).
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

    if not start or not raw_goal: return None

    # Safety Clamp: Ensure goal is strictly within the map boundaries.
    # If the target is pushed outside the cropped window (e.g. 50m target on a 20m window),
    # clamp it to the edge so A* doesn't get stuck in a full-grid search for an impossible point.
    clamped_goal_x = max(0, min(size_px[0] - 1, raw_goal[0]))
    clamped_goal_y = max(0, min(size_px[1] - 1, raw_goal[1]))
    goal = (clamped_goal_x, clamped_goal_y)

    # Check if start is inside obstacle (safety fallback)
    if 0 <= start[0] < size_px[0] and 0 <= start[1] < size_px[1]:
        if nav_map[start[1], start[0]] == 0:
            # If boat is inside an obstacle, A* cannot escape. Return None to trigger fallback behavior (e.g. PID)
            return None

    # Check if goal is inside obstacle. If so, fallback logic could be applied here
    if nav_map[goal[1], goal[0]] == 0:
        return None # Goal is unreachable

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, heuristic_weight)}

    # Vector calculation for bias
    dx_line = goal[0] - start[0]
    dy_line = goal[1] - start[1]
    line_length = math.sqrt(dx_line**2 + dy_line**2)

    if line_length > 0:
        norm_line_x = dx_line / line_length
        norm_line_y = dy_line / line_length
    else:
        norm_line_x, norm_line_y = 0, 0

    # 8-way movement (dx, dy)
    neighbors = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    # Direction tracking for kinematic penalty (heading change)
    # Store (f_score, (px, py), (dx_from_parent, dy_from_parent))
    open_set = []
    heapq.heappush(open_set, (0, start, (0, 0)))

    while open_set:
        _, current, prev_dir = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(p2w(current[0], current[1]))
                current = came_from[current]
            path.append(p2w(start[0], start[1]))
            path.reverse()
            # Apply path smoothing post-processing
            return string_pulling(path, nav_map, center_m, res, size_px)

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < size_px[0] and 0 <= neighbor[1] < size_px[1]:
                if nav_map[neighbor[1], neighbor[0]] == 0:
                    continue # Hit obstacle

                # Cost is 1 for straight, 1.414 for diagonal
                step_cost = 1 if dx == 0 or dy == 0 else 1.414

                # Apply bias penalty if it strays from straight line
                if bias_to_goal_line > 0 and line_length > 0:
                    vec_nx = neighbor[0] - start[0]
                    vec_ny = neighbor[1] - start[1]
                    cross_product = abs(vec_nx * norm_line_y - vec_ny * norm_line_x)
                    step_cost += (cross_product * bias_to_goal_line)

                # --- 1-A UPDATE: Kinematic Constraint Penalty ---
                kinematic_penalty = 0.0
                if prev_dir != (0, 0):
                    # Calculate angle difference between previous direction and current direction
                    dot_product = prev_dir[0]*dx + prev_dir[1]*dy
                    mag_prev = math.sqrt(prev_dir[0]**2 + prev_dir[1]**2)
                    mag_curr = math.sqrt(dx**2 + dy**2)
                    if mag_prev > 0 and mag_curr > 0:
                        cos_angle = np.clip(dot_product / (mag_prev * mag_curr), -1.0, 1.0)
                        angle_diff = math.degrees(math.acos(cos_angle))

                        # Penalize sharp turns (e.g., > 45 degrees)
                        if angle_diff > 45.0:
                            kinematic_penalty += (angle_diff * 0.1)  # Tunable weight
                        if angle_diff >= 90.0:
                            kinematic_penalty += 5.0 # Heavy penalty for reversals/right angles

                step_cost += kinematic_penalty
                # ------------------------------------------------

                tentative_g_score = g_score[current] + step_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal, heuristic_weight)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor, (dx, dy)))

    return None # No path found

# --- PURE PURSUIT CONTROLLER ---
def find_lookahead_point(x, y, path, lookahead_dist):
    """
    Finds the furthest point on the path within the lookahead circle.
    """
    if not path: return None

    target_idx = -1
    for i in range(len(path) - 1, -1, -1):
        pt = path[i]
        dist = math.sqrt((pt[0] - x)**2 + (pt[1] - y)**2)
        if dist <= lookahead_dist:
            target_idx = i
            break

    # If no point is within lookahead, we target the first point if it's further away,
    # or the closest point to the lookahead circle
    if target_idx == -1:
        # Fallback to closest point
        min_d = float('inf')
        for i, pt in enumerate(path):
            dist = math.sqrt((pt[0] - x)**2 + (pt[1] - y)**2)
            if dist < min_d:
                min_d = dist
                target_idx = i

    # Try to interpolate between target_idx and target_idx+1 to exactly hit lookahead circle
    # (Simplified approach: just take the next point if available)
    if target_idx < len(path) - 1:
        target_idx += 1

    return path[target_idx], target_idx

def pure_pursuit_control(rx, ry, ryaw, path, current_speed=0, base_speed=1500, prev_error=0):
    """
    Executes the Pure Pursuit algorithm to generate Base Speed and Steering Correction (Yaw Effort).
    Dynamic lookahead based on speed.
    """
    if not path or len(path) < 2:
        return base_speed, 0, None, 0.0, path

    # 1. Dynamic Lookahead Distance
    min_ld = getattr(cfg, 'PURE_PURSUIT_MIN_LOOKAHEAD', 1.0)
    max_ld = getattr(cfg, 'PURE_PURSUIT_MAX_LOOKAHEAD', 3.0)
    k_ld = getattr(cfg, 'PURE_PURSUIT_K_SPEED', 0.5) # Lookahead multiplier

    lookahead_dist = np.clip(current_speed * k_ld, min_ld, max_ld)

    # 2. Find target point
    target_pt, t_idx = find_lookahead_point(rx, ry, path, lookahead_dist)
    if target_pt is None:
        return base_speed, 0, None, 0.0, path

    # 3. Calculate steering error (Alpha)
    tx, ty = target_pt

    # In this codebase, the A* grid (rx, ry) is built using math.cos/sin on raw compass bearings.
    # Therefore, we can reverse this by using math.atan2 to get the angle in radians,
    # but since it was built with compass logic (0 = North, clockwise),
    # the atan2 result directly corresponds to a math.radians(compass_bearing).
    # Since ryaw is passed as math.radians(magnetic_heading), we can just subtract them.
    # WAIT - standard atan2(y, x) gives 0 for East, positive Counter-Clockwise.
    # If the map was built using math.cos(compass_bearing) and math.sin(compass_bearing):
    # - compass_bearing = 0 (North): cos(0)=1, sin(0)=0 -> (x=1, y=0) -> mapped to East on the grid.
    # - compass_bearing = 90 (East): cos(90)=0, sin(90)=1 -> (x=0, y=1) -> mapped to North on the grid.
    # This means math.atan2(ty - ry, tx - rx) gives the original math.radians(compass_bearing)!

    target_bearing_rad = math.atan2(ty - ry, tx - rx)
    alpha = target_bearing_rad - ryaw

    # Normalize alpha to [-pi, pi]
    alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

    # Convert to Degrees for PID
    # (Since alpha is target - current, positive alpha means target is to the right if compass is clockwise.
    # Let's verify: compass 90, boat 0 -> alpha +90. We want to turn right.
    # Legacy PID logic: positive heading_err turns right.
    # Wait, the old code had `heading_err = -math.degrees(alpha)`. Let's remove the negative sign if alpha is already correct.)

    # Old code had: heading_err = -math.degrees(alpha).
    # Let's check signed_angle_difference in navigasyon.py: diff = (angle2 - angle1 + 180) % 360 - 180
    # where angle2 is target, angle1 is current. Positive diff means target is to the right.
    # We will use exactly that logic to be absolutely safe and consistent with Direct Drive PID.
    target_bearing_deg = math.degrees(target_bearing_rad) % 360
    current_heading_deg = math.degrees(ryaw) % 360

    heading_err = (target_bearing_deg - current_heading_deg + 180) % 360 - 180

    # 4. PID calculation
    kp = getattr(cfg, 'PURE_PURSUIT_KP', 2.0)
    kd = getattr(cfg, 'PURE_PURSUIT_KD', 0.5)

    P = heading_err * kp
    D = (heading_err - prev_error) * kd
    correction = P + D

    # Prune path: remove points we have already passed
    pruned_path = path[t_idx:] if t_idx < len(path) else path[-1:]

    return base_speed, correction, target_pt, heading_err, pruned_path
