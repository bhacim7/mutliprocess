import time
import datetime
import config as cfg
import utils.telem as telem

def telem_worker(shared_state, command_queue, hf_data):
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
            current_lat = hf_data['gps_lat'].value
            current_lon = hf_data['gps_lon'].value
            heading = hf_data['magnetic_heading'].value
            mevcut_gorev = shared_state.get('current_task', 'TASK_UNKNOWN')
            pwm_l = shared_state.get('motor_pwm_left', 1500)
            pwm_r = shared_state.get('motor_pwm_right', 1500)
            pwm_fl = shared_state.get('motor_pwm_front_left', 1500)
            pwm_fr = shared_state.get('motor_pwm_front_right', 1500)
            pwm_steer = shared_state.get('motor_pwm_steer', 1500)
            objects = shared_state.get('vision_detected_objects', [])
            manual_mode = shared_state.get('manual_mode', False)


            # In a real scenario, incoming commands from GCS (set_gps, emergency_stop, set_task)
            # would be read by the serial thread and pushed into command_queue for NavProcess to handle.

            # --- TELEMETRY BROADCAST ---

            if shared_state.get('send_telemetry', False):
                gps_points = {
                    "id": my_id,
                    "GPS1": {"lat": float(getattr(cfg, "T1_GATE_ENTER_LAT", 0.0)), "lon": float(getattr(cfg, "T1_GATE_ENTER_LON", 0.0))},
                    "GPS2": {"lat": float(getattr(cfg, "T1_GATE_MID_LAT", 0.0)), "lon": float(getattr(cfg, "T1_GATE_MID_LON", 0.0))},
                    "GPS3": {"lat": float(getattr(cfg, "T1_GATE_EXIT_LAT", 0.0)), "lon": float(getattr(cfg, "T1_GATE_EXIT_LON", 0.0))},
                    "GPS4": {"lat": float(getattr(cfg, "T2_ZONE_ENTRY_LAT", 0.0)), "lon": float(getattr(cfg, "T2_ZONE_ENTRY_LON", 0.0))},
                    "GPS5": {"lat": float(getattr(cfg, "T2_ZONE_MID_LAT", 0.0)), "lon": float(getattr(cfg, "T2_ZONE_MID_LON", 0.0))},
                    "GPS6": {"lat": float(getattr(cfg, "T2_ZONE_END_LAT", 0.0)), "lon": float(getattr(cfg, "T2_ZONE_END_LON", 0.0))},
                    "GPS7": {"lat": float(getattr(cfg, "T3_START_LAT", 0.0)), "lon": float(getattr(cfg, "T3_START_LON", 0.0))},
                    "GPS8": {"lat": float(getattr(cfg, "T3_MID_LAT", 0.0)), "lon": float(getattr(cfg, "T3_MID_LON", 0.0))},
                }

                payload = {
                    "id": my_id,
                    "t_ms": datetime.datetime.now().strftime('%H:%M:%S'),
                    "pwm_L": pwm_l,
                    "pwm_R": pwm_r,
                    "pwm_FL": pwm_fl,
                    "pwm_FR": pwm_fr,
                    "pwm_STEER": pwm_steer,
                    "spd": shared_state.get('horizontal_speed', 0.0),
                    "hdg": f"{heading:.0f}" if heading is not None else "0",
                    "trg_hdg": shared_state.get('adviced_course', 0.0),
                    "err_ang": shared_state.get('angle_error', 0.0),
                    "ctrl_err": shared_state.get('control_error', 0.0),
                    "hlth": "GOOD", # Simplified for now, or read from shared state
                    "task": mevcut_gorev,
                    "objects": objects, # Lightweight dicts, safe to serialize
                    "MEVCUT_KONUM": {"lat": current_lat, "lon": current_lon},
                    "HEDEF_KONUM": {"lat": shared_state.get('target_lat', 0.0), "lon": shared_state.get('target_lon', 0.0)},
                    "dist": shared_state.get('target_dist', 0.0),
                    "mod": bool(manual_mode),
                    "FPS": shared_state.get('camera_fps', 0),
                    "GÖREV_NOKTALARI": gps_points
                }

                tx.send(payload)
                shared_state['send_telemetry'] = False

            # Sleep to maintain frequency (~20Hz loop for polling responsiveness)
            elapsed = time.time() - start_time
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)

    except Exception as e:
        print(f"[TELEM_PROCESS][ERROR] Loop crashed: {e}")
    finally:
        print("[TELEM_PROCESS] Shutting down...")
        try:
            cmd_rx.stop()
            telemetry_sender.close()
        except:
            pass
