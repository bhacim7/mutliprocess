import multiprocessing as mp
import time
import sys
import signal

# Import worker processes
from core.camera_process import camera_worker
from core.lidar_process import lidar_worker
from core.nav_process import nav_worker
from core.telem_process import telem_worker

def signal_handler(sig, frame):
    """Catches OS kill signals."""
    print("\n[ORCHESTRATOR] Kill signal received. Securing system...")
    sys.exit(0)

def main():
    print("=" * 50)
    print("🚀 RoboBoat 2026 - IDA System Orchestrator Starting...")
    print("=" * 50)

    signal.signal(signal.SIGINT, signal_handler)

    # 1. SHARED MEMORY DICTIONARY
    manager = mp.Manager()
    shared_state = manager.dict({
        # Global Control Flags
        'shutdown': False,
        'mission_started': True,
        'manual_mode': False,
        'current_task': 'TASK_1',

        # Sensors & Tracking
        'gps_lat': 0.0,
        'gps_lon': 0.0,
        'magnetic_heading': 0.0,

        # Lidar Process Output (Emergency Distances)
        'lidar_center_blocked': False,
        'lidar_left_dist': float('inf'),
        'lidar_center_dist': float('inf'),
        'lidar_right_dist': float('inf'),
        'lidar_wave_stable': True, # Pitch/Roll stability flag
        'lidar_points': [], # Downsampled lightweight points for mapping

        # Vision Process Output (Metadata Only)
        'vision_detected_objects': [], # List of dicts: {'type': x, 'color': y, 'lat': z, 'lon': w, 'dist': d}
        'vision_frame_ready': False,

        # Motor State (For telemetry & debugging)
        'motor_pwm_left': 1500,
        'motor_pwm_right': 1500,

        # Acoustic / Interrupt State
        'interrupt_request': None,
        'detected_freq': 0
    })

    # 2. QUEUES FOR IPC
    command_queue = mp.Queue()

    # 3. DEFINE PROCESSES
    processes = []

    # Process initialization
    p_nav = mp.Process(target=nav_worker, args=(shared_state, command_queue), name="NavProcess")
    p_telem = mp.Process(target=telem_worker, args=(shared_state, command_queue), name="TelemProcess")
    p_cam = mp.Process(target=camera_worker, args=(shared_state,), name="CameraProcess")
    p_lidar = mp.Process(target=lidar_worker, args=(shared_state,), name="LidarProcess")

    processes.extend([p_nav, p_telem, p_cam, p_lidar])

    # 4. START PROCESSES
    print("[ORCHESTRATOR] Launching processes...")
    for p in processes:
        p.start()
        print(f"[ORCHESTRATOR] Started: {p.name} (PID: {p.pid})")

    # 5. WATCHDOG LOOP
    try:
        while True:
            if shared_state['shutdown']:
                print("[ORCHESTRATOR] Shutdown command detected!")
                break
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
