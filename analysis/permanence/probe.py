"""Object permanence measures from the dumped states (CPU).

    python analysis/permanence/probe.py            # after truth.py and dump.py; writes results/permanence/

1. Bearing probe.  A ridge probe from the last-layer decision-token state to (sin, cos) of the asked object's bearing
   (camera heading to object, left positive), fitted on the `fit` streams (states standardised, a bias column, lambda
   by a 4-fold split of the fitting streams) and scored on the out-of-view seconds of the 78 `test` streams, all three
   questions: median bearing error, overall and by seconds since the object was last seen.
2. Read-out accuracy.  Each `readout` stream is asked three times (left / right / behind); at every out-of-view second
   whose true bearing is in one of the three sectors, the question with the highest p(respond) (4 decimals, as in the
   evaluation records) is the answer.
3. Rotation correlation.  The same probe recipe, cross-fitted over 5 folds of scenes (`fit` and `test` streams
   together), so that every decoded bearing comes from a probe fitted on other scenes; the Fisher-Lee circular
   correlation between the decoded and the true bearing over the out-of-view seconds of each stream's own question.
4. Error curve.  The probe of 1, unchanged, applied to the `test` and `long` streams: median bearing error at each
   second since the object left the view (1-60 s, seconds with at least 20 rows, 3-s rolling median) with a
   95 % stream-bootstrap band.
5. Top-view examples (topview.py): example A reads the cross-fitted bearings of 3; example B, a `long` stream, a probe
   fitted on all `fit` and `test` streams.

Outputs: summary.json, curve.json, curve.png / .pdf, examples.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DIRS, HALF_HFOV, RESULTS, load_runs, load_truth, per_second, wrap  # noqa: E402

BINS = [(0, 3), (3, 6), (6, 10), (10, 20), (20, 999)]
LAMBDAS = (1e2, 1e3, 1e4, 1e5)
EXAMPLES = {
    "A": "ca1m:47331084:object_direction:90:45c80d54-127f-4d2b-a6c0-033248b25222:behind_v1",
    "B": "ca1m:45261495:object_direction:58:b8530020-1898-4d16-906a-731991992657:right_v0",
}


# ----------------------------------------------------------------------------- rows
def rows_of(runs, truth, sets):
    """One row per second of every run in `sets`: its state and labels."""
    X, M = [], {k: [] for k in ("run", "stream", "scene", "t", "part", "theta", "vis", "since_first", "since_last", "own_q")}
    for r in runs:
        if r["set"] not in sets:
            continue
        g = per_second(r, truth)
        f = CACHE / "states" / f"{r['run']:04d}.npz"
        if g is None or not f.exists():
            continue
        h = np.load(f)["h"]
        T = min(len(h), len(g["theta"]))
        X.append(h[:T])
        M["run"] += [r["run"]] * T
        M["stream"] += [r["stream"]] * T
        M["scene"] += [r["scene_id"]] * T
        M["t"] += list(range(T))
        M["part"] += ["fit" if r["set"] == "fit" else "test"] * T
        M["theta"] += g["theta"][:T].tolist()
        M["vis"] += g["vis"][:T].tolist()
        M["since_first"] += g["since_first"][:T].tolist()
        M["since_last"] += g["since_last"][:T].tolist()
        M["own_q"] += [r["asked"] == r["own"]] * T
    M = {k: np.array(v) for k, v in M.items()}
    return np.concatenate(X, 0).astype(np.float32), M


# ----------------------------------------------------------------------------- probe
def ridge(X, Y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def ang_err(pred, th):
    bel = np.degrees(np.arctan2(pred[:, 0], pred[:, 1]))
    return np.abs(wrap(bel - th)), bel


def fit_probe(X, M, fit_mask):
    """Standardisation, lambda (4 folds of the fitting streams, sorted by name) and weights fitted on `fit_mask` rows
    (rows after the object was first seen).  Returns a function: states -> decoded bearing (deg), and the lambda."""
    th = M["theta"]
    Y = np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], 1)
    fit = fit_mask & (M["since_first"] > 0)
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
    Z = lambda A: np.concatenate([(A - mu) / sd, np.ones((len(A), 1), A.dtype)], 1).astype(np.float64)
    Zf = Z(X[fit])
    names = sorted(set(M["stream"][fit]))
    fold = {s: i % 4 for i, s in enumerate(names)}
    fo = np.array([fold[s] for s in M["stream"][fit]])
    vis = M["vis"][fit]
    best, best_lam = 1e9, LAMBDAS[0]
    for lam in LAMBDAS:
        e = []
        for k in range(4):
            tr, va = fo != k, (fo == k) & ~vis
            e.append(np.median(ang_err(Zf[va] @ ridge(Zf[tr], Y[fit][tr], lam), th[fit][va])[0]))
        if np.mean(e) < best:
            best, best_lam = float(np.mean(e)), lam
    W = ridge(Zf, Y[fit], best_lam)
    return (lambda A: ang_err(Z(A) @ W, np.zeros(len(A)))[1]), best_lam


def scores(err, M, mask):
    gap = M["since_last"]
    return {
        "median_error_deg": float(np.median(err[mask])),
        "within_45_deg_pct": float(100 * np.mean(err[mask] < 45)),
        "seconds": int(mask.sum()),
        "streams": int(len(set(M["stream"][mask]))),
        "by_seconds_since_last_seen": {
            f"{lo}-{hi}" if hi < 999 else f">{lo}": float(np.median(err[mask & (gap >= lo) & (gap < hi)])) for lo, hi in BINS
        },
    }


def fisher_lee(a, b):
    """Fisher & Lee (1983) circular correlation of two angle samples (deg)."""
    a, b = np.radians(a), np.radians(b)
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    num = (sa * sb).sum() * (ca * cb).sum() - (sa * cb).sum() * (ca * sb).sum()
    da = (sa * sa).sum() * (ca * ca).sum() - (sa * ca).sum() ** 2
    db = (sb * sb).sum() * (cb * cb).sum() - (sb * cb).sum() ** 2
    return float(num / np.sqrt(da * db))


def sector(th):
    a = np.abs(th)
    return np.where(a > 135, 2, np.where((th >= 45) & (th <= 135), 0, np.where((th <= -45) & (th >= -135), 1, -1)))


# ----------------------------------------------------------------------------- read-out
def readout(runs, truth):
    by = {}
    for r in runs:
        f = CACHE / "states" / f"{r['run']:04d}.npz"
        if r["set"] == "readout" and f.exists():
            # p(respond) rounded to 4 decimals, as the evaluation records it (prospace_vlm/eval.py)
            by.setdefault(r["episode_id"], {})[r["asked"]] = (r, np.round(np.load(f)["p"].astype(np.float64), 4))
    ok, n_streams = [], 0
    for eid, d in by.items():
        if len(d) != 3:
            continue
        g = per_second(d["left"][0], truth)
        if g is None:
            continue
        n_streams += 1
        P = np.stack([d[k][1] for k in DIRS])
        th = g["theta"]
        T = min(len(th), P.shape[1])
        vis, sec, last = np.abs(th[:T]) < HALF_HFOV, sector(th[:T]), None
        for t in range(T):
            if vis[t]:
                last = t
                continue
            if last is not None and sec[t] >= 0:
                ok.append(int(np.argmax(P[:, t]) == sec[t]))
    return {"accuracy_pct": float(100 * np.mean(ok)), "seconds": len(ok), "streams": n_streams}


# ----------------------------------------------------------------------------- curve
def roll3(a):
    return np.array([np.nanmedian(a[max(0, i - 1) : i + 2]) if np.isfinite(a[max(0, i - 1) : i + 2]).any() else np.nan for i in range(len(a))])


def wmedian(v, w):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return v[np.searchsorted(c, c[-1] / 2)] if c[-1] > 0 else np.nan


def curve(stream, gap, err, smax=60, boot=500, seed=0):
    S = np.arange(1, smax + 1)
    med = roll3(np.array([np.median(err[gap == s]) if (gap == s).sum() >= 20 else np.nan for s in S]))
    names = sorted(set(stream))
    si = np.array([names.index(s) for s in stream])
    rng, bs = np.random.default_rng(seed), []
    for _ in range(boot):
        w = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))[si]
        bs.append(roll3(np.array([wmedian(err[gap == s], w[gap == s]) if (gap == s).sum() >= 20 else np.nan for s in S])))
    bs = np.array(bs)
    return {
        "seconds": S.tolist(),
        "median_error_deg": med.tolist(),
        "band_95": [np.nanpercentile(bs, 2.5, 0).tolist(), np.nanpercentile(bs, 97.5, 0).tolist()],
        "rows": int(len(err)),
        "streams": len(names),
    }


def plot_curve(c, dst):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    S = np.array(c["seconds"])
    ax.fill_between(S, *c["band_95"], color="#5C5EA3", alpha=0.18, lw=0)
    ax.plot(S, c["median_error_deg"], color="#5C5EA3", lw=1.8, label="ProSpace-VLM")
    ax.set_xlim(1, S[-1])
    ax.set_ylim(10, 50)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Seconds since the object left the view")
    ax.set_ylabel("Bearing error (°)")
    ax.legend(frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(dst / f"curve.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    runs, truth = load_runs(), load_truth()
    RESULTS.mkdir(parents=True, exist_ok=True)
    X, M = rows_of(runs, truth, ("fit", "test"))
    print(f"probe streams: {len(set(M['stream']))}, {len(X)} seconds", flush=True)
    oov = (M["since_first"] > 0) & ~M["vis"]
    test = (M["part"] == "test") & oov

    # 1. bearing probe
    read, lam = fit_probe(X, M, M["part"] == "fit")
    err = np.abs(wrap(read(X) - M["theta"]))
    probe = {"lambda": lam, "fit_seconds": int(((M["part"] == "fit") & (M["since_first"] > 0)).sum()), **scores(err, M, test)}
    print(f"1. bearing probe: median error {probe['median_error_deg']:.1f} deg on {probe['seconds']} out-of-view seconds of "
          f"{probe['streams']} streams x 3 questions | by seconds since last seen: "
          + " ".join(f"{k}: {v:.0f}" for k, v in probe["by_seconds_since_last_seen"].items()), flush=True)

    # 2. read-out
    ro = readout(runs, truth)
    print(f"2. read-out accuracy: {ro['accuracy_pct']:.1f} % on {ro['seconds']} out-of-view seconds of {ro['streams']} streams", flush=True)

    # 3. rotation correlation (scene cross-fit)
    scenes = sorted(set(M["scene"]))
    fold = {s: i % 5 for i, s in enumerate(np.random.default_rng(0).permutation(scenes))}
    fo = np.array([fold[s] for s in M["scene"]])
    cross = np.full(len(X), np.nan)
    for k in range(5):
        rd, _ = fit_probe(X, M, fo != k)
        cross[fo == k] = rd(X[fo == k])
        print(f"   cross-fit fold {k + 1}/5", flush=True)
    prim = oov & M["own_q"]
    rot = {
        "circular_correlation": fisher_lee(M["theta"][prim], cross[prim]),
        "median_error_deg": float(np.median(np.abs(wrap(cross[prim] - M["theta"][prim])))),
        "seconds": int(prim.sum()),
        "streams": int(len(set(M["stream"][prim]))),
        "scenes": int(len(set(M["scene"][prim]))),
    }
    print(f"3. rotation correlation: {rot['circular_correlation']:.2f} on {rot['seconds']} out-of-view seconds of "
          f"{rot['streams']} streams ({rot['scenes']} scenes)", flush=True)

    # 4. curve: the probe of 1 on the test and long streams
    XL, ML = rows_of(runs, truth, ("long",))
    keep = test
    st = np.concatenate([M["stream"][keep], ML["stream"][(ML["since_first"] > 0) & ~ML["vis"]]])
    gap = np.concatenate([M["since_last"][keep], ML["since_last"][(ML["since_first"] > 0) & ~ML["vis"]]])
    errL = np.abs(wrap(read(XL) - ML["theta"]))
    e = np.concatenate([err[keep], errL[(ML["since_first"] > 0) & ~ML["vis"]]])
    cv = curve(st, gap, e)
    (RESULTS / "curve.json").write_text(json.dumps(cv, indent=1))
    plot_curve(cv, RESULTS)
    print(f"4. error curve: {cv['rows']} out-of-view rows of {cv['streams']} streams -> {RESULTS / 'curve.json'}", flush=True)

    # 5. examples
    ex = {}
    sel = lambda MM, eid: (MM["stream"] == "test:" + eid) & MM["own_q"] & (MM["since_first"] > 0) & ~MM["vis"]
    m = sel(M, EXAMPLES["A"])
    ex["A"] = (EXAMPLES["A"], M["t"][m], cross[m], M["theta"][m], "scene cross-fit (3)")
    scB = EXAMPLES["B"].split(":")[1]
    rdB, _ = fit_probe(X, M, M["scene"] != scB)
    m = sel(ML, EXAMPLES["B"])
    ex["B"] = (EXAMPLES["B"], ML["t"][m], rdB(XL[m]), ML["theta"][m], "probe fitted on all fit and test streams")
    out = {}
    for k, (eid, t, bel, th, how) in ex.items():
        e_ = np.abs(wrap(bel - th))
        out[k] = {"episode_id": eid, "probe": how, "median_error_deg": float(np.median(e_)),
                  "seconds": [[int(a), float(b), float(c)] for a, b, c in zip(t, bel, th)]}
        print(f"5. example {k} ({eid.split(':')[0]} {eid.split(':')[1]}): median error {np.median(e_):.1f} deg over {len(t)} out-of-view seconds", flush=True)
    (RESULTS / "examples.json").write_text(json.dumps(out, indent=1))
    summary = {"bearing_probe": probe, "readout": ro, "rotation": rot,
               "curve": {k: cv[k] for k in ("rows", "streams")}, "examples": {k: v["median_error_deg"] for k, v in out.items()}}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"written: {RESULTS}")


if __name__ == "__main__":
    main()
