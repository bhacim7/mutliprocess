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
    if raw_costmap is None: return None, None

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
    goal = w2p(*end_world)

    if not start or not goal: return None

    # Check if goal is inside obstacle. If so, fallback logic could be applied here
    if 0 <= goal[0] < size_px[0] and 0 <= goal[1] < size_px[1]:
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

def calculate_path_curvature(path, lookahead_idx, max_points=3):
    """
    Estimates the curvature of the upcoming path segment.
    Returns a ratio from 0.0 (straight) to 1.0 (sharp turn).
    """
    if not path or len(path) - lookahead_idx < 3:
        return 0.0

    pts = path[lookahead_idx : lookahead_idx + max_points]
    if len(pts) < 3:
        return 0.0

    # Simple angle check between first, middle, and last point in the segment
    p1, p2, p3 = pts[0], pts[len(pts)//2], pts[-1]

    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    angle_diff = math.degrees(math.acos(cos_angle))

    # Normalize curvature penalty (e.g., 0-90 degrees maps to 0.0-1.0)
    curvature_ratio = np.clip(angle_diff / 90.0, 0.0, 1.0)
    return curvature_ratio

def pure_pursuit_control(rx, ry, ryaw, path, current_speed=0, base_speed=1500, prev_error=0):
    """
    Executes the Pure Pursuit algorithm.
    Returns: (base_speed, steering_correction, target_pt, heading_err, pruned_path)
    to be consumed by the central motor mixer.
    """
    if not path or len(path) < 2:
        return base_speed, 0.0, None, 0.0, path

    # 1. Dynamic Lookahead Distance (Speed Base)
    min_ld = getattr(cfg, 'PURE_PURSUIT_MIN_LOOKAHEAD', 1.0)
    max_ld = getattr(cfg, 'PURE_PURSUIT_MAX_LOOKAHEAD', 3.0)
    k_ld = getattr(cfg, 'PURE_PURSUIT_K_SPEED', 0.5) # Lookahead multiplier

    base_lookahead = np.clip(current_speed * k_ld, min_ld, max_ld)

    # 2. Find initial target point to check curvature
    temp_pt, temp_idx = find_lookahead_point(rx, ry, path, base_lookahead)

    # --- Adaptive Lookahead based on curvature ---
    curvature = calculate_path_curvature(path, temp_idx)

    # Reduce lookahead distance sharply if curvature is high (tight corners)
    adaptive_lookahead = base_lookahead - (curvature * (base_lookahead - min_ld))
    lookahead_dist = np.clip(adaptive_lookahead, min_ld, max_ld)

    # Find final target point with adapted lookahead
    target_pt, t_idx = find_lookahead_point(rx, ry, path, lookahead_dist)
    if target_pt is None:
        return base_speed, 0.0, None, 0.0, path

    # 3. Calculate steering error (Alpha)
    tx, ty = target_pt
    alpha = math.atan2(ty - ry, tx - rx) - ryaw

    # Normalize alpha to [-pi, pi]
    alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

    # Convert to Degrees for PID (invert: Pos Err -> Turn Right)
    heading_err = -math.degrees(alpha)

    # 4. PID calculation
    kp = getattr(cfg, 'PURE_PURSUIT_KP', 2.0)
    kd = getattr(cfg, 'PURE_PURSUIT_KD', 0.5)

    P = heading_err * kp
    D = (heading_err - prev_error) * kd
    steering_correction = P + D

    # Prune path: remove points we have already passed
    pruned_path = path[t_idx:] if t_idx < len(path) else path[-1:]

    return base_speed, steering_correction, target_pt, heading_err, pruned_path
