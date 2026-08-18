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
    wp_cycle_idx = 0        # which point of the set goes in this packet

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

    # --- TASK3 target colour, as the boat actually holds it ---
    #
    # DRONE_ACTIVE is a static config value - nothing mutates it at runtime, unlike the GPS
    # points - so reading this process's own cfg copy is safe here even under spawn.
    drone_active = bool(getattr(cfg, 'DRONE_ACTIVE', False))
    cfg_color = str(getattr(cfg, 'TASK3_KAMIKAZE_COLOR', 'red')).lower()

    def task3_color():
        """
        Returns (colour, source) for the GCS.

        The GCS uses this two ways: to show the operator which colour the boat will
        actually hunt, and to notice when what the boat holds differs from what the drone
        reported - i.e. that a set_target_color command was lost - and say it again. Before
        this field existed the GCS sent the colour once and had no way of ever learning
        whether it arrived.

        source "cfg"   -> DRONE_ACTIVE is off, config.py decides, the drone is irrelevant
        source "drone" -> the drone decides; an empty colour means none has arrived yet
        """
        if not drone_active:
            return cfg_color, "cfg"
        col = shared_state.get('drone_target_color')
        return (str(col).lower() if col else ""), "drone"

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
                # Keys are SHORT on the wire and expanded back to their long names by the
                # GCS the moment the line is parsed, so every consumer downstream still sees
                # 'pwm_L', 'MEVCUT_KONUM' and friends untouched.
                #
                # Why the size matters so much: the radio has a per-frame limit and splits
                # anything larger, and it runs with Mavlink framing OFF, so it has no idea
                # where a JSON line begins or ends - it cuts wherever its buffer fills. With
                # three nodes on one channel, the drone's turn can fall BETWEEN two fragments
                # of a boat packet, and its bytes land in the middle of our JSON. The result
                # is a line the GCS cannot parse. Measured: the old packet was 378 B (750 B
                # with waypoints) while every command is 38-76 B and the drone's is 36 B -
                # the boat's telemetry was the only thing on the link big enough to be split,
                # which is exactly why commands kept arriving while telemetry did not.
                #
                # The GCS also reassembles interrupted lines now, so this is the second line
                # of defence rather than the only one - but a packet that never gets split
                # cannot be corrupted this way at all.
                payload = {
                    "i": my_id,
                    "t": datetime.datetime.now().strftime('%H:%M:%S'),
                    "a": int(pwm_l),
                    "b": int(pwm_r),
                    "c": int(pwm_fl),
                    "d": int(pwm_fr),
                    "s": int(pwm_steer),
                    "v": r(shared_state.get('horizontal_speed', 0.0), 2),
                    "h": r(heading, 0) if heading is not None else 0,
                    "th": r(shared_state.get('adviced_course', 0.0), 1),
                    "ea": r(shared_state.get('angle_error', 0.0), 1),
                    "ce": r(shared_state.get('control_error', 0.0), 1),
                    "k": mevcut_gorev,
                    # Position flattened rather than nested: {"lat":..,"lon":..} costs 14 B of
                    # punctuation and key names per point for no information.
                    "la": r(current_lat, 7), "lo": r(current_lon, 7),
                    "xa": r(shared_state.get('target_lat', 0.0), 7),
                    "xo": r(shared_state.get('target_lon', 0.0), 7),
                    "ds": r(shared_state.get('target_dist', 0.0), 1),
                    "m": 1 if manual_mode else 0,
                    "f": int(shared_state.get('camera_fps', 0) or 0),
                    # Motor-power relay as the VEHICLE reports it (RELAY_STATUS), not as we
                    # last commanded it - the RC transmitter can change it independently.
                    # -1 = the vehicle never reported, so the GCS shows "unknown".
                    "r": int(shared_state.get('relay_state', -1)),
                }
                # 'hlth' is not transmitted: it has always been the constant "GOOD" (see the
                # note this replaced), so the GCS fills it in locally instead of paying for
                # it 3 times a second. If real health ever exists it gets its own field.

                # TASK3 target colour + source. Source travels as 0/1 rather than
                # "cfg"/"drone"; the GCS expands it.
                _tc, _tsrc = task3_color()
                payload["tc"] = _tc
                payload["ts"] = 1 if _tsrc == "drone" else 0

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

                # Waypoints: ONE point per packet, cycled, instead of all eight in a block.
                #
                # The block was 353 B on its own and pushed the packet to 750 B - three radio
                # frames, two seams for another node to cut across, and losing any one of them
                # lost all eight points. A single point is ~38 B, the packet stays in one
                # frame, and a lost point is re-sent one cycle later instead of taking the
                # whole set with it. The GCS accumulates them and rebuilds the same
                # GÖREV_NOKTALARI dict its map code has always consumed.
                wp = current_waypoints()
                now_ts = time.time()
                if wp != last_waypoints_sent or (now_ts - last_waypoints_ts) >= waypoint_refresh_s:
                    names = sorted(wp.keys(), key=lambda n: int(n.replace("GPS", "")))
                    if names:
                        name = names[wp_cycle_idx % len(names)]
                        lat, lon = wp[name]
                        payload["w"] = int(name.replace("GPS", ""))
                        payload["wa"] = r(lat, 7)
                        payload["wo"] = r(lon, 7)
                        wp_cycle_idx += 1
                        # The set counts as sent only once every point has had a turn, so a
                        # change part-way through simply keeps the cycle running rather than
                        # restarting it and starving the points at the end of the list.
                        if wp_cycle_idx % len(names) == 0:
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