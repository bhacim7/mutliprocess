# Legacy vs Multiprocessing Architecture Analysis

Based on a detailed comparison between the legacy `IDA1.py` monolith and the new multiprocessing architecture (`main_orchestrator.py`, `core/camera_process.py`, `core/lidar_process.py`, `core/nav_process.py`, `core/telem_process.py`), the following details were identified as skipped, changed, or incorrectly ported.

## 1. Camera & Vision (`core/camera_process.py`)

*   **Buoy Depth Processing Calculation:**
    *   **Legacy (`IDA1.py`):** Calculates buoy depth by focusing on the bottom 15% of the bounding box where it meets the water. `target_cy = int(y2 - (box_h * 0.15))`
    *   **New:** Uses the exact center of the bounding box `cy = int((y2 + y1) / 2)`. This is a critical mathematical omission that will cause incorrect depth measurements.
*   **Virtual Obstacles in Costmap:**
    *   **Legacy:** Injects detected objects directly into `costmap_img` as virtual obstacles (radius 6px).
    *   **New:** Handled correctly. `camera_worker` sends `vision_detected_objects` to `shared_state`, and `nav_worker` draws them into its local `costmap_img`.
*   **FPS Terminal Print Limiting:**
    *   **Legacy:** Printed FPS, but the new version has it commented out. The new version's commented logic correctly limits prints to once per second if uncommented.

## 2. Navigation (`core/nav_process.py`)

*   **Target Selection & Alignment Fallback:**
    *   **Legacy:** For normal A* navigation (`if current_path:` -> `else:`), if the path is lost and a fallback is triggered, it uses a P-controller on `aci_farki` (bearing difference) to directly steer toward the target GPS. Crucially, if the angle is too large during direct drive (`force_initial_alignment` or explicit conditions), it uses a `spot_turn` (tank turn) to correct heading before driving.
    *   **New:** The `else:` block logic (`if target_lat is not None...`) has a 5-second direct drive grace period, but misses the logic to properly handle spot turns if the heading is severely off during this direct drive phase. It simply applies `base_pwm + steering_correction`.
*   **ZED Coordinate Axis Swap:**
    *   **Legacy:** X and Y are swapped: `robot_x = t_vec[1]`, `robot_y = -t_vec[0]`.
    *   **New:** Correctly ported in `camera_worker` when writing to `shared_state`.

## 3. Telemetry (`core/telem_process.py`)

*   **Report Status Payload Structure:**
    *   **Legacy:** When commanded via `report_status`, the payload included many fields: `FPS` (from zed), `hlth` (magnetometer state), `dist` (distance to target), `trg_hdg` (adviced course), `spd` (horizontal speed), and `GÖREV_NOKTALARI` (the 14 global GPS waypoints from config).
    *   **New:** The payload in `telem_worker` is severely stripped down and is missing `FPS`, `hlth`, `dist`, `trg_hdg`, `spd`, and `GÖREV_NOKTALARI`.
*   **Command Handling (`set_gps`, `set_task`):**
    *   **Legacy:** Handled incoming commands for `set_gps` (dynamically changing config waypoints) and `set_task`.
    *   **New:** `telem_worker` receives commands and pushes them to the `command_queue`. However, `nav_worker` only processes `emergency_stop` and `report_status`. It completely ignores `set_gps` and `set_task` commands, meaning they are silently dropped.

## 4. Orchestrator (`main_orchestrator.py`)

*   **Missing Shared State Keys:** The orchestrator's `shared_state` dictionary needs to be initialized with several missing keys to prevent KeyErrors or logical failures when workers try to read/write them. Examples:
    *   `fc_heading`
    *   `zed_x`, `zed_y`
    *   `send_telemetry`

## 5. Intentional Omissions (Confirmed via Memory)

*   **Acoustic / Signal Interrupt Task:** Deprecated and permanently removed. Note: `nav_process.py` still contains dead logic for this (`interrupt = shared_state.get('interrupt_request')`) which should ideally be removed for cleanliness.
*   **Lidar Buoy Density Verification:** Intentionally omitted. ZED camera buoy detections are correctly passed via `vision_detected_objects` without Lidar validation.
*   **Polling Telemetry Architecture:** Telemetry uses a polling architecture correctly triggered by `send_telemetry` flag (when `report_status` is received).
