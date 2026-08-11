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

    # --- Waypoint block state (see the GÖREV_NOKTALARI handling below) ---
    waypoint_refresh_s = float(getattr(cfg, 'TELEM_WAYPOINT_REFRESH_S', 10.0))
    last_waypoints_sent = None
    last_waypoints_ts = 0.0

    # Config-file defaults, used only until nav_process publishes the live set.
    _CFG_WAYPOINT_ATTRS = (
        ("GPS1", "T1_GATE_ENTER_LAT", "T1_GATE_ENTER_LON"),
        ("GPS2", "T1_GATE_MID_LAT", "T1_GATE_MID_LON"),
        ("GPS3", "T1_GATE_EXIT_LAT", "T1_GATE_EXIT_LON"),
        ("GPS4", "T2_ZONE_ENTRY_LAT", "T2_ZONE_ENTRY_LON"),
        ("GPS5", "T2_ZONE_MID_LAT", "T2_ZONE_MID_LON"),
        ("GPS6", "T2_ZONE_END_LAT", "T2_ZONE_END_LON"),
        ("GPS7", "T3_START_LAT", "T3_START_LON"),
        ("GPS8", "T3_MID_LAT", "T3_MID_LON"),
    )

    def current_waypoints():
        """
        Live waypoint set as {"GPS1": (lat, lon), ...}.

        Reads shared_state first: `set_gps` is handled inside nav_process and mutates THAT
        process's cfg module. Because the orchestrator uses mp.set_start_method('spawn'),
        this process has its own independent copy of config.py that never sees those
        updates - so reading cfg here echoed the config-file defaults back to the GCS while
        the boat was actually steering to the operator's points.
        """
        pts = shared_state.get('mission_points')
        if isinstance(pts, dict) and pts:
            return {k: (float(v[0]), float(v[1])) for k, v in pts.items()}
        return {name: (float(getattr(cfg, la, 0.0)), float(getattr(cfg, lo, 0.0)))
                for name, la, lo in _CFG_WAYPOINT_ATTRS}

    def r(value, digits, default=0.0):
        """Round for the wire. json.dumps writes floats with repr(), i.e. 17 significant
        digits - '40.809465000000004' costs 18 bytes to say 1 cm. 7 decimals of latitude is
        ~1.1 cm, which is far below GPS noise."""
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return default

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
            # 'vision_detected_objects' is no longer read here. It is the heaviest entry in
            # the Manager dict (a pickled list of 10-30 dicts) and this loop was unpickling
            # it 20x/s only to serialise it into a packet the GCS never looked at.
            manual_mode = shared_state.get('manual_mode', False)


            # In a real scenario, incoming commands from GCS (set_gps, emergency_stop, set_task)
            # would be read by the serial thread and pushed into command_queue for NavProcess to handle.

            # --- TELEMETRY BROADCAST ---

            if shared_state.get('send_telemetry', False):
                # --- Payload construction ---
                # Field NAMES are deliberately unchanged: GCSv1000.on_packet() looks up
                # 'pwm_L', 'spd', 'trg_hdg', 'MEVCUT_KONUM' and friends by name, so renaming
                # them to save bytes would silently blank the whole GCS panel. All the size
                # reduction comes from dropping dead weight and rounding instead.
                payload = {
                    "id": my_id,
                    "t_ms": datetime.datetime.now().strftime('%H:%M:%S'),
                    "pwm_L": int(pwm_l),
                    "pwm_R": int(pwm_r),
                    "pwm_FL": int(pwm_fl),
                    "pwm_FR": int(pwm_fr),
                    "pwm_STEER": int(pwm_steer),
                    "spd": r(shared_state.get('horizontal_speed', 0.0), 2),
                    "hdg": r(heading, 0) if heading is not None else 0,
                    "trg_hdg": r(shared_state.get('adviced_course', 0.0), 1),
                    "err_ang": r(shared_state.get('angle_error', 0.0), 1),
                    "ctrl_err": r(shared_state.get('control_error', 0.0), 1),
                    "hlth": "GOOD",  # Simplified for now, or read from shared state
                    "task": mevcut_gorev,
                    "MEVCUT_KONUM": {"lat": r(current_lat, 7), "lon": r(current_lon, 7)},
                    "HEDEF_KONUM": {"lat": r(shared_state.get('target_lat', 0.0), 7),
                                    "lon": r(shared_state.get('target_lon', 0.0), 7)},
                    "dist": r(shared_state.get('target_dist', 0.0), 1),
                    "mod": bool(manual_mode),
                    "FPS": int(shared_state.get('camera_fps', 0) or 0),
                }

                # 'objects' is deliberately NOT sent any more.
                #
                # It was the single biggest field in the packet - ~200 B per tracked buoy,
                # and ObjectMemoryManager keeps every detection for 5 s, so the Task 2 buoy
                # field routinely produced 10-30 entries. Measured: 2960 B at 10 objects,
                # 4960 B at 20, against a hard 2880 B ceiling imposed by write_timeout at
                # 57600 baud. Every one of those writes was truncated mid-JSON.
                #
                # And it was never read: GCSv1000 has no consumer for the key at all. Pure
                # airtime. If the GCS ever needs detections, send a separate, trimmed,
                # low-rate packet rather than reviving this field.

                # Waypoints are ~513 B - over half the remaining packet. Send them only when
                # they actually change, plus a slow refresh so a GCS that connects late (or
                # missed the packet) still populates its map. on_packet() already guards its
                # read with `if "GÖREV_NOKTALARI" in d`, so omitting it is safe.
                wp = current_waypoints()
                now_ts = time.time()
                if wp != last_waypoints_sent or (now_ts - last_waypoints_ts) >= waypoint_refresh_s:
                    payload["GÖREV_NOKTALARI"] = {
                        "id": my_id,
                        **{name: {"lat": r(lat, 7), "lon": r(lon, 7)}
                           for name, (lat, lon) in wp.items()},
                    }
                    last_waypoints_sent = wp
                    last_waypoints_ts = now_ts

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