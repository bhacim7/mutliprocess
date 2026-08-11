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

    # --- Bandwidth budget ---------------------------------------------------
    # The GCS polls report_status and the reply used to carry the whole object memory
    # plus all 8 mission waypoints on EVERY packet: 2-4 KB each. A 57600 baud radio
    # link tops out around 5.7 KB/s, so at the GCS's poll rate this asked for several
    # times the available bandwidth. The write buffer backs up, TelemetrySender's
    # write_timeout starts raising, and the COMMAND channel (which shares this link,
    # emergency stop included) degrades along with it.
    MAX_OBJECTS_IN_PAYLOAD = 8
    last_gps_points = None
    last_gps_points_sent = 0.0
    GPS_POINTS_REFRESH_S = 10.0   # also resend periodically so a GCS that reconnects catches up

    def compact_objects(objs):
        """Nearest few detections, rounded, with only the fields the GCS renders."""
        try:
            usable = [o for o in objs if o.get('lat') and o.get('lon')]
            usable.sort(key=lambda o: o.get('dist') if o.get('dist') is not None else 1e9)
        except Exception:
            usable = list(objs)[:MAX_OBJECTS_IN_PAYLOAD]

        out = []
        for o in usable[:MAX_OBJECTS_IN_PAYLOAD]:
            try:
                out.append({
                    "id": o.get('id'),
                    "cid": o.get('cid'),
                    "type": o.get('type'),
                    "color": o.get('color'),
                    "lat": round(float(o.get('lat', 0.0)), 7),
                    "lon": round(float(o.get('lon', 0.0)), 7),
                    "dist": round(float(o.get('dist') or 0.0), 1),
                })
            except (TypeError, ValueError):
                continue
        return out

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
                    "pwm_L": int(pwm_l),
                    "pwm_R": int(pwm_r),
                    "pwm_FL": int(pwm_fl),
                    "pwm_FR": int(pwm_fr),
                    "pwm_STEER": int(pwm_steer),
                    "spd": round(float(shared_state.get('horizontal_speed', 0.0)), 2),
                    "hdg": f"{heading:.0f}" if heading is not None else "0",
                    "trg_hdg": round(float(shared_state.get('adviced_course', 0.0)), 1),
                    "err_ang": round(float(shared_state.get('angle_error', 0.0)), 1),
                    "ctrl_err": round(float(shared_state.get('control_error', 0.0)), 1),
                    "hlth": "GOOD", # Simplified for now, or read from shared state
                    "task": mevcut_gorev,
                    "objects": compact_objects(objects),
                    "MEVCUT_KONUM": {"lat": round(float(current_lat), 7), "lon": round(float(current_lon), 7)},
                    "HEDEF_KONUM": {"lat": round(float(shared_state.get('target_lat', 0.0)), 7),
                                    "lon": round(float(shared_state.get('target_lon', 0.0)), 7)},
                    "dist": round(float(shared_state.get('target_dist', 0.0)), 1),
                    "mod": bool(manual_mode),
                    "FPS": shared_state.get('camera_fps', 0),
                }

                # The mission waypoints only change when the GCS uploads new ones, so
                # re-sending ~450 bytes of them 10x/s was pure link tax.
                if gps_points != last_gps_points or \
                        (start_time - last_gps_points_sent) > GPS_POINTS_REFRESH_S:
                    payload["GÖREV_NOKTALARI"] = gps_points
                    last_gps_points = gps_points
                    last_gps_points_sent = start_time

                tx.send(payload)
                shared_state['send_telemetry'] = False

            shared_state['telem_heartbeat'] = time.time()

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
