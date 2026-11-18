"""Render the MP3D frames of the three Place tasks with habitat-sim, from the released camera poses.

    python scripts/render_mp3d.py --src <mp3d root> [--out data/frames] [--gpu 0] [--scenes ...]

`--src` is the habitat release of Matterport3D: <src>/<house>/<house>.glb.
Needs habitat-sim 0.3.1 (headless).  Frames are 512x384, vertical field of view 53.7 degrees, JPEG quality 90.
data/mp3d_camera_poses.npz holds, per frame, the habitat agent state: position (x, y, z) and rotation (float32
quaternion x, y, z, w).  `md5_exact` marks the frames whose render was confirmed byte-identical to the benchmark frame.
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W, H, VFOV = 512, 384, 53.7


def simulator(glb, gpu):
    import habitat_sim

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(glb)
    cfg.enable_physics = False
    cfg.gpu_device_id = gpu
    s = habitat_sim.CameraSensorSpec()
    s.uuid = "rgb"
    s.sensor_type = habitat_sim.SensorType.COLOR
    s.resolution = [H, W]
    s.position = [0.0, 0.0, 0.0]
    s.hfov = math.degrees(2 * math.atan(math.tan(math.radians(VFOV) / 2) * W / H))
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = [s]
    ac.height = 1.5
    ac.radius = 0.15
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
    return sim, sim.initialize_agent(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="habitat MP3D root: <src>/<house>/<house>.glb")
    ap.add_argument("--out", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--poses", default=str(ROOT / "data" / "mp3d_camera_poses.npz"))
    ap.add_argument("--scenes", nargs="*", help="only these scene ids (default: all)")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs

    z = np.load(a.poses)
    todo = defaultdict(list)  # house -> [(scene_id, frame_id, position, rotation)]
    for sc, f, p, q in zip(z["scene_id"], z["frame_id"], z["position"], z["rotation"]):
        if a.scenes and sc not in a.scenes:
            continue
        if not (Path(a.out) / "mp3d" / sc / f"frame_{f:06d}.jpg").exists():
            todo[sc.split("_")[0]].append((sc, int(f), p, q))
    missing = [h for h in todo if not (Path(a.src) / h / f"{h}.glb").exists()]
    if missing:
        raise SystemExit(f"missing {Path(a.src) / missing[0] / (missing[0] + '.glb')} ({len(missing)} houses)")
    n = 0
    for house in sorted(todo):
        sim, agent = simulator(Path(a.src) / house / f"{house}.glb", a.gpu)
        for sc, f, p, q in todo[house]:
            st = habitat_sim.AgentState()
            st.position = p
            st.rotation = quat_from_coeffs(q.astype(np.float64))
            agent.set_state(st)
            rgb = np.ascontiguousarray(sim.get_sensor_observations()["rgb"][..., :3])
            dst = Path(a.out) / "mp3d" / sc
            dst.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(dst / f"frame_{f:06d}.jpg", quality=90)
            n += 1
        sim.close()
        print(f"{house}: {len(todo[house])} frames", flush=True)
    print(f"wrote {n} frames to {Path(a.out) / 'mp3d'}")


if __name__ == "__main__":
    main()
