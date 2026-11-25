"""Where the data lives, and the small Hugging Face files fetched on first use.

    data/prospace_bench/   the files of the Hugging Face dataset shamanneo/ProSpace-Bench
    data/prospace_train/   the files of the Hugging Face dataset shamanneo/ProSpace-Train
    data/frames/<dataset>/<scene_id>/frame_XXXXXX.jpg   benchmark and training frames (disjoint scenes)

`fetch()` downloads a missing file anonymously (huggingface_hub when installed, plain HTTPS otherwise, so it also works in
the habitat environment); no token is read or stored.  PROSPACE_HF_MIRROR=<dir> copies the file from
<dir>/ProSpace-Bench/ or <dir>/ProSpace-Train/ instead (a local copy of the two datasets).

    python -m prospace_bench.data bench          # fetch prospace_bench.jsonl and frames_manifest.tsv
    python -m prospace_bench.data train 100k     # fetch the 100K set and the validation files

Python 3.9+, standard library only.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BENCH = DATA / "prospace_bench"
TRAIN = DATA / "prospace_train"
FRAMES = DATA / "frames"
REPOS = {"bench": "shamanneo/ProSpace-Bench", "train": "shamanneo/ProSpace-Train"}
DATASETS = ["scannetpp", "scannet", "ca1m", "mp3d"]

OLD = {
    "data/prospace_bench.jsonl": "data/prospace_bench/prospace_bench.jsonl",
    "data/frames_manifest.tsv": "data/prospace_bench/frames_manifest.tsv",
    "data/mp3d_camera_poses.npz": "data/prospace_bench/mp3d_camera_poses.npz",
    "data/train": "data/prospace_train",
    "data/train_frames": "data/frames (training frames now share the benchmark's folder)",
}


def check_layout() -> None:
    """Stop with a clear message when files of the earlier layout are still there."""
    old = [(o, n) for o, n in OLD.items() if (ROOT / o).exists()]
    if old:
        lines = "\n".join(f"  {o:30s} -> {n}" for o, n in old)
        raise SystemExit(
            "This checkout still has files of the earlier data layout:\n"
            f"{lines}\n"
            "Move (or delete) them; the small files are fetched again on first use.  "
            "Training frames built earlier under data/train_frames have to be rebuilt in data/frames (docs/data.md)."
        )


def fetch(which: str, name: str) -> Path:
    """data/prospace_<which>/<name>, downloaded when it is missing."""
    folder = BENCH if which == "bench" else TRAIN
    path = folder / name
    if path.exists():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    repo = REPOS[which]
    mirror = os.environ.get("PROSPACE_HF_MIRROR")
    if mirror:
        src = Path(mirror) / repo.split("/")[1] / name
        if not src.exists():
            raise SystemExit(f"PROSPACE_HF_MIRROR is set, but {src} does not exist")
        print(f"copying {src} -> {path}", file=sys.stderr, flush=True)
        shutil.copyfile(src, str(path) + ".part")
        os.replace(str(path) + ".part", path)
        return path
    print(f"downloading {name} from huggingface.co/datasets/{repo} -> {folder}", file=sys.stderr, flush=True)
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo, name, repo_type="dataset", local_dir=str(folder), token=False)
    except ImportError:
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{name}"
        try:
            with urllib.request.urlopen(url) as r, open(str(path) + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        except OSError as e:
            raise SystemExit(
                f"could not download {url} ({e}); download it by hand:\n"
                f"  hf download {repo} {name} --repo-type dataset --local-dir {folder}"
            )
        os.replace(str(path) + ".part", path)
    return path


def train_frame(rel: str) -> str:
    """A ProSpace-Train frame path -> its place in data/frames.

    The published files name Matterport3D training frames mp3d/<scene>/frames/frame_XXXXXX.jpg and the generated
    history before a mid-stream request mp3d/<scene>/history_b/frame_JJJJ.jpg (or history/); in data/frames they are
    mp3d/<scene>/frame_XXXXXX.jpg and mp3d/<scene>_history/frame_<J:06d>.jpg.  Other paths are unchanged.
    """
    p = rel.split("/")
    if p[0] != "mp3d" or len(p) != 4:
        return rel
    if p[2] == "frames":
        return f"mp3d/{p[1]}/{p[3]}"
    return f"mp3d/{p[1]}_history/frame_{int(p[3][6:-4]):06d}.jpg"


def main() -> None:
    check_layout()
    which = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if which == "bench":
        names = ["prospace_bench.jsonl", "frames_manifest.tsv"]
    elif which == "train":
        sets = sys.argv[2:] or ["100k"]
        names = [f"prospace_train_{s}.jsonl.gz" for s in sets] + ["val_poll.jsonl", "val_es.jsonl", "frames_manifest.tsv.gz"]
    else:
        raise SystemExit("usage: python -m prospace_bench.data bench | train [100k] [400k]")
    for n in names:
        print(fetch(which, n))


if __name__ == "__main__":
    main()
