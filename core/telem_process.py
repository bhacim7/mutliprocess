import time
import datetime
import config as cfg
import utils.telem as telem

def telem_worker(shared_state, command_queue):
    """
    Independent process handling GCS communication.
    Broadcasts real-time telemetry from shared_state and listens for incoming commands.
    """
    print("[TELEM_PROCESS] Starting Telemetry/Comm Worker...")

    # 1. Initialization
    port = getattr(cfg, 'SERIAL_PORT', '/dev/ttyUSB0')
    baud = getattr(cfg, 'SERIAL_BAUD', 57600)

    telemetry_sender = telem.TelemetrySender(port, baud)
    tx = telem.TelemetryTx(telemetry_sender, max_hz=10)

    # For a real system, the receiver might push to a local queue,
    # which we then forward to the multiprocess command_queue.
    # Here we simulate the setup:
    cmd_rx = telem.CommandReceiver(telemetry_sender, command_queue)
    cmd_rx.start()

    my_id = 1

    # 2. Main Loop
    try:
        while not shared_state['shutdown']:
            start_time = time.time()

            # Extract data from shared_state without blocking others
            current_lat = shared_state.get('gps_lat', 0.0)
            current_lon = shared_state.get('gps_lon', 0.0)
            heading = shared_state.get('magnetic_heading', 0.0)
            mevcut_gorev = shared_state.get('current_task', 'TASK_UNKNOWN')
            pwm_l = shared_state.get('motor_pwm_left', 1500)
            pwm_r = shared_state.get('motor_pwm_right', 1500)
            objects = shared_state.get('vision_detected_objects', [])
            manual_mode = shared_state.get('manual_mode', False)

            # In a real scenario, incoming commands from GCS (set_gps, emergency_stop, set_task)
            # would be read by the serial thread and pushed into command_queue for NavProcess to handle.

            # --- TELEMETRY BROADCAST ---
            # Periodically report status (or respond to 'report_status' commands from queue)
            # For this structure, we'll auto-broadcast at ~2 Hz as an example.

            payload = {
                "id": my_id,
                "t_ms": datetime.datetime.now().strftime('%H:%M:%S'),
                "pwm_L": pwm_l,
                "pwm_R": pwm_r,
                "hdg": f"{heading:.0f}" if heading is not None else "0",
                "task": mevcut_gorev,
                "objects": objects, # Lightweight dicts, safe to serialize
                "MEVCUT_KONUM": {"lat": current_lat, "lon": current_lon},
                "mod": bool(manual_mode),
            }

            tx.send(payload)

            # Sleep to maintain frequency (~2Hz loop)
            elapsed = time.time() - start_time
            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)

    except Exception as e:
        print(f"[TELEM_PROCESS][ERROR] Loop crashed: {e}")
    finally:
        print("[TELEM_PROCESS] Shutting down...")
        try:
            cmd_rx.stop()
            telemetry_sender.close()
        except:
            pass
