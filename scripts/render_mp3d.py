"""Render the Matterport3D frames of the three Place tasks with habitat-sim, from the released camera poses.

    python scripts/render_mp3d.py --src <mp3d root> [--set bench | train] [--gpu 0] [--houses ...] [--scenes ...]

`--src` is the habitat release of Matterport3D: <src>/<house>/<house>.glb.
Needs habitat-sim 0.3.1 (headless).  Frames are 512x384, vertical field of view 53.7 degrees, JPEG quality 90, written to
data/frames/mp3d/<scene_id>/frame_<frame_id:06d>.jpg; frames that already exist are skipped, so the script can be run
again after an interruption.

The poses are data/prospace_<set>/mp3d_camera_poses.npz, fetched from Hugging Face when missing: per frame scene_id,
frame_id, the habitat agent state (position x, y, z and rotation as a quaternion x, y, z, w, both float32) and md5_exact,
the frames whose render was confirmed byte-identical to ours.  --set bench: 100,925 frames (18 houses), which include
every frame the benchmark streams read; --set train: the 1,326,929 Matterport3D frames of ProSpace-Train (both sets and
the validation files, 45 houses).
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prospace_bench.data import FRAMES, check_layout, fetch  # noqa: E402

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
    ap.add_argument("--set", default="bench", choices=["bench", "train"])
    ap.add_argument("--out", default=str(FRAMES), help="frame root (default data/frames); frames go to <out>/mp3d/")
    ap.add_argument("--poses", default=None, help="default data/prospace_<set>/mp3d_camera_poses.npz")
    ap.add_argument("--houses", nargs="*", help="only the scenes of these houses")
    ap.add_argument("--scenes", nargs="*", help="only these scene ids")
    ap.add_argument("--gpu", type=int, default=0, help="index among the visible GPUs (CUDA_VISIBLE_DEVICES), not the physical id")
    ap.add_argument("--list-houses", action="store_true", help="print the houses of --set and exit")
    a = ap.parse_args()
    check_layout()
    z = np.load(a.poses or fetch(a.set, "mp3d_camera_poses.npz"))
    if a.list_houses:
        print(" ".join(sorted({str(s).split("_")[0] for s in z["scene_id"]})))
        return
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs

    houses, scenes = set(a.houses or []), set(a.scenes or [])
    out = Path(a.out) / "mp3d"
    todo = defaultdict(list)  # house -> [(scene_id, frame_id, position, rotation)]
    for sc, f, p, q in zip(z["scene_id"], z["frame_id"], z["position"], z["rotation"]):
        h = sc.split("_")[0]
        if (houses and h not in houses) or (scenes and sc not in scenes):
            continue
        if not (out / sc / f"frame_{f:06d}.jpg").exists():
            todo[h].append((sc, int(f), p, q))
    missing = [h for h in todo if not (Path(a.src) / h / f"{h}.glb").exists()]
    if missing:
        raise SystemExit(f"missing {Path(a.src) / missing[0] / (missing[0] + '.glb')} ({len(missing)} houses)")
    print(f"{sum(map(len, todo.values()))} frames to render in {len(todo)} houses", flush=True)
    n, t0 = 0, time.time()
    for house in sorted(todo):
        sim, agent = simulator(Path(a.src) / house / f"{house}.glb", a.gpu)
        made = set()
        for sc, f, p, q in todo[house]:
            st = habitat_sim.AgentState()
            st.position = p
            st.rotation = quat_from_coeffs(q.astype(np.float64))
            agent.set_state(st)
            rgb = np.ascontiguousarray(sim.get_sensor_observations()["rgb"][..., :3])
            if sc not in made:
                (out / sc).mkdir(parents=True, exist_ok=True)
                made.add(sc)
            dst = out / sc / f"frame_{f:06d}.jpg"
            Image.fromarray(rgb).save(dst.with_suffix(".part"), format="JPEG", quality=90)
            dst.with_suffix(".part").replace(dst)
            n += 1
        sim.close()
        print(f"{house}: {len(todo[house])} frames ({n / (time.time() - t0):.0f} frames/s overall)", flush=True)
    print(f"wrote {n} frames to {out}")


if __name__ == "__main__":
    main()
