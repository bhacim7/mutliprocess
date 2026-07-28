import multiprocessing as mp
import time
import sys
import signal

import config as cfg

# Import worker processes
from core.camera_process import camera_worker
from core.lidar_process import lidar_worker
from core.nav_process import nav_worker
from core.telem_process import telem_worker


def signal_handler(sig, frame):
    """Catches OS kill signals."""
    print("\n[ORCHESTRATOR] Kill signal received. Securing system...")
    sys.exit(0)


def pre_flight_check(shared_state, hf_data):
    """Blocks execution until critical sensors come online in the worker processes."""
    print("\n" + "=" * 60)
    print("[PRE-FLIGHT CHECK] System Diagnostics Started...")
    print("Waiting for sensors to populate data...")
    print("=" * 60)

    # We need to verify that Nav, Camera, and Lidar are writing real values to shared_state
    # instead of their default zeros.
    sensors_ready = False

    while not sensors_ready and not shared_state['shutdown']:
        lat = hf_data['gps_lat'].value
        heading = hf_data['magnetic_heading'].value

        gps_ok = (lat != 0.0)
        hdg_ok = (heading != 0.0)

        # In a real scenario, you can expand these to check Lidar points or ZED positional state
        if gps_ok and hdg_ok:
            sensors_ready = True
            print("   ✅ GPS: ONLINE & Locked")
            print("   ✅ Compass/Heading: ONLINE")
            print("   ✅ Sensor Data Validated!")
        else:
            time.sleep(1.0)

    if sensors_ready:
        print("=" * 60)
        print("✅ SYSTEM READY FOR MISSION! ARMING...")
        print("=" * 60)
        shared_state['mission_started'] = True


def main():
    print("=" * 50)
    print("🚀 RoboBoat 2026 - IDA System Orchestrator Starting...")
    print("=" * 50)

    signal.signal(signal.SIGINT, signal_handler)

    # 1. HIGH-FREQUENCY MEMORY (Avoids dictionary lock contention)
    # Using 'd' for double precision floats
    hf_data = {
        'gps_lat': mp.Value('d', 0.0),
        'gps_lon': mp.Value('d', 0.0),
        'magnetic_heading': mp.Value('d', 0.0)
    }

    # 2. SHARED MEMORY DICTIONARY (For metadata and low-frequency states)
    manager = mp.Manager()
    shared_state = manager.dict({
        # Global Control Flags
        'shutdown': False,
        'mission_started': False,  # <--- Starts False, blocks motors until pre-flight passes
        'manual_mode': True,
        'current_task': 'TASK_1',

        'target_dist': 0.0,
        'adviced_course': 0.0,
        'angle_error': 0.0,
        'target_lat': 0.0,
        'target_lon': 0.0,

        # Sensors & Tracking
        'fc_heading': 0.0,
        'zed_x': 0.0,
        'zed_y': 0.0,
        'send_telemetry': False,

        # Lidar Process Output (Emergency Distances)
        'lidar_center_blocked': False,
        'lidar_left_dist': float('inf'),
        'lidar_center_dist': float('inf'),
        'lidar_right_dist': float('inf'),
        'lidar_wave_stable': True,  # Pitch/Roll stability flag
        'lidar_points': [],  # Downsampled lightweight points for mapping

        # Vision Process Output (Metadata Only)
        'vision_detected_objects': [],  # List of dicts: {'type': x, 'color': y, 'lat': z, 'lon': w, 'dist': d}
        'vision_frame_ready': False,

        # Motor State (For telemetry & debugging)
        'motor_pwm_left': 1500,
        'motor_pwm_right': 1500,
        'motor_pwm_front_left': 1500,
        'motor_pwm_front_right': 1500,
        'motor_pwm_steer': 1500
    })

    # --- 1.5. INITIAL TASK ASSIGNMENT ---
    default_task = getattr(cfg, 'MEVCUT_GOREV', 'TASK1_STATE_ENTER')
    shared_state['current_task'] = default_task

    if shared_state['manual_mode']:
        print(f"[ORCHESTRATOR] Manual Mode Active. Loaded default task from config: {default_task}")
        print("[ORCHESTRATOR] Waiting for GCS commands to start auto mode or change task...\n")
    else:
        print(f"[ORCHESTRATOR] Auto Mode Active. Loaded default task from config: {default_task}\n")

    # 3. QUEUES FOR IPC
    command_queue = mp.Queue()
    # --- 4-A UPDATE: Queue for heavy Lidar points ---
    lidar_queue = mp.Queue(maxsize=10)  # Small maxsize to prevent memory bloat if nav lags

    # 4. DEFINE PROCESSES
    processes = []

    # Process initialization
    p_nav = mp.Process(target=nav_worker, args=(shared_state, command_queue, hf_data, lidar_queue), name="NavProcess")
    p_telem = mp.Process(target=telem_worker, args=(shared_state, command_queue, hf_data), name="TelemProcess")
    p_cam = mp.Process(target=camera_worker, args=(shared_state, hf_data), name="CameraProcess")

    processes.extend([p_nav, p_telem, p_cam])

    # Conditional Lidar Process
    if getattr(cfg, 'ENABLE_LIDAR', True):
        p_lidar = mp.Process(target=lidar_worker, args=(shared_state, lidar_queue), name="LidarProcess")
        processes.append(p_lidar)
    else:
        print("[ORCHESTRATOR] ENABLE_LIDAR is False. Lidar process will not be started.")

    # 5. START PROCESSES
    print("[ORCHESTRATOR] Launching processes...")
    for p in processes:
        p.start()
        print(f"[ORCHESTRATOR] Started: {p.name} (PID: {p.pid})")

    # --- 5.5 RUN PRE FLIGHT CHECKS ---
    # Blocks here while the worker processes boot up their hardware drivers
    pre_flight_check(shared_state, hf_data)

    # 6. WATCHDOG LOOP
    try:
        while True:
            if shared_state['shutdown']:
                print("[ORCHESTRATOR] Shutdown command detected!")
                break

            # Watchdog: Check if all processes are alive, restart if necessary
            for i, p in enumerate(processes):
                if not p.is_alive():
                    print(f"[ORCHESTRATOR][WARNING] Process {p.name} (PID: {p.pid}) has died unexpectedly!")
                    # Identify the correct target and args based on the name
                    if p.name == "NavProcess":
                        target = nav_worker
                        args = (shared_state, command_queue, hf_data)
                    elif p.name == "TelemProcess":
                        target = telem_worker
                        args = (shared_state, command_queue, hf_data)
                    elif p.name == "CameraProcess":
                        target = camera_worker
                        args = (shared_state, hf_data)
                    elif p.name == "LidarProcess":
                        target = lidar_worker
                        args = (shared_state,)
                    else:
                        continue

                    print(f"[ORCHESTRATOR] Restarting {p.name}...")
                    new_p = mp.Process(target=target, args=args, name=p.name)
                    new_p.start()
                    processes[i] = new_p
                    print(f"[ORCHESTRATOR] {p.name} restarted successfully with new PID: {new_p.pid}")

            time.sleep(1.0)
    except SystemExit:
        shared_state['shutdown'] = True

    # 6. GRACEFUL SHUTDOWN
    print("\n[ORCHESTRATOR] Sending stop signal to all processes...")
    for p in processes:
        if p.is_alive():
            p.join(timeout=3.0)
            if p.is_alive():
                print(f"[ORCHESTRATOR] {p.name} refused to close, terminating (SIGTERM)!")
                p.terminate()

    print("[ORCHESTRATOR] System shutdown complete. Have a good day Captain.")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
