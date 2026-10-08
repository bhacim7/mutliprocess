#!/usr/bin/env python3
"""
Turn a recorded run into an mp4 showing the map filling in.

    python utils/costmap_video.py final_costmap.npz
    python utils/costmap_video.py final_costmap.npz --fps 10 --scale 3 --out jury.mp4

The input is the .npz the boat writes next to final_costmap.png at shutdown - NOT the PNG.
A video cannot be made from the PNG: it is a single frame, the state at the end of the
run, and the chronology is not in it. Every buoy is already there.

Running this separately is the point. Rendering hundreds of frames inside the shutdown
path would add half a minute to every Ctrl+C, and an interrupted encode leaves an mp4
with no index, which will not open at all. Here the boat's shutdown stays as fast as it
has always been, this can run on a laptop rather than the Jetson, and the video can be
re-made with different speed or scale without going back on the water.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.costmap_recorder import CostmapRecorder   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Render a costmap run as mp4.")
    ap.add_argument("npz", help="recording written next to final_costmap.png")
    ap.add_argument("--out", default=None, help="output mp4 (default: alongside the npz)")
    ap.add_argument("--sample-hz", type=float, default=1.0,
                    help="map snapshots per second of RUN time (default 1)")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="playback rate; 1 Hz sampled at 15 fps plays 15x (default 15)")
    ap.add_argument("--scale", type=float, default=2.0,
                    help="pixels per recorded pixel (default 2)")
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        print(f"Not found: {args.npz}")
        return 1

    rec = CostmapRecorder.from_npz(args.npz)
    print(f"[COSTMAP] {len(rec.track)} track points, {len(rec.observations)} sightings, "
          f"{len(rec.objects)} tracks")

    out = rec.render_video(path=args.out or (args.npz.rsplit('.', 1)[0] + ".mp4"),
                           sample_hz=args.sample_hz, fps=args.fps, scale=args.scale)
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
