"""
Step 3: H2 consistency defenses (built on fogsearch.py + step2_attacks.py, unchanged).

Two checks run every round on the heard bearing z; a flagged round skips the audio update
(trust = 0 for that round only, i.e. falls back to visual exploration).

  temporal - take the last K_WIN (position, bearing) pairs; find the single free cell that
             best explains all of them (min mean angular residual). If even the best cell
             leaves a mean residual > TAU, the bearings do not triangulate -> flag.
  visual   - the bearing must point at some cell that could still hold the target
             (free and not yet seen empty). If the nearest such cell is more than W away in
             angle from z, the implied source is on an obstacle / seen-empty cell -> flag.
  both     - flag if either check flags.
  oracle   - flags exactly the attacked rounds (perfect detector). Separates detector error
             from the cost of falling back to vision.

Thresholds TAU and W are set on a separate set of calibration layouts (seeds 5000+) under
honest audio, so each check alone has a TARGET_FAR honest false-alarm rate. Test layouts are
the same 200 seeded layouts as Steps 1-2, so every number is paired with Step 2.

Outputs: step3_results.csv, step3_defense.png
"""
import csv
import sys
from multiprocessing import Pool

import numpy as np

import fogsearch as fs
import step2_attacks as s2

K_WIN = 4             # rounds in the temporal window
TARGET_FAR = 0.05     # honest false-alarm rate per check, fixed on calibration layouts
N_TEST = 200
N_CALIB = 100
MAX_ROUNDS = 80
NOISY_SIGMA_DEG = 90  # "honest but noisy" control: worse SNR than the agent's model assumes

RR, CC = np.indices((fs.N, fs.N))
wrap = s2.wrap


# ---------- attacks: same draws as step2_attacks, but also report whether the round was attacked
def make_naive(s_deg, p, obst, start, target, rng):
    sign = 1.0 if s_deg >= 180 else rng.choice([-1.0, 1.0])
    off = sign * np.deg2rad(s_deg)

    def f(pos, tgt, t, r):
        att = r.random() < p
        b = fs.true_bearing(pos, tgt) + (off if att else 0.0)
        return b + r.vonmises(0.0, fs.KAPPA), att
    return f


def make_decoy(s_deg, p, obst, start, target, rng):
    decoy, _ = s2.choose_decoy(s_deg, obst, start, target, rng)

    def f(pos, tgt, t, r):
        att = r.random() < p
        return fs.true_bearing(pos, decoy if att else tgt) + r.vonmises(0.0, fs.KAPPA), att
    return f


ATTACKS = {"naive": make_naive, "decoy": make_decoy}


# ---------- the two check statistics (larger = more suspicious)
def temporal_stat(hist, obst):
    """Min over free cells of the mean angular residual of the window's bearings."""
    if len(hist) < K_WIN:
        return 0.0
    res = np.zeros((fs.N, fs.N))
    bad = obst.copy()
    for (pr, pc), z in hist:
        res += np.abs(wrap(z - np.arctan2(RR - pr, CC - pc)))
        bad[pr, pc] = True                     # a source on the robot's own cell is undefined
    return float(np.min(np.where(bad, np.inf, res / len(hist))))


def visual_stat(pos, z, obst, observed):
    """Angle from z to the nearest cell that could still hold the target."""
    cand = ~obst & ~observed
    cand[pos] = False
    if not cand.any():
        return np.pi
    d = np.abs(wrap(z - np.arctan2(RR - pos[0], CC - pos[1])))
    return float(d[cand].min())


# ---------- one episode (same planner as fogsearch.run_episode, plus the defense hook)
def episode(obst, start, target, rng, trust=1.0, attack=None, noise_kappa=fs.KAPPA,
            defense=None, tau=np.inf, w=np.inf, collect=False):
    """Return (rounds, log). With defense=None and noise_kappa=KAPPA this reproduces
    fogsearch.run_episode draw for draw (checked in main)."""
    pos = start
    observed = np.zeros((fs.N, fs.N), bool)
    logb = np.zeros((fs.N, fs.N))
    hist = []
    log = dict(heard=0, attacked=0, flagged=0, flagged_attacked=0, t_stats=[], v_stats=[])

    for t in range(MAX_ROUNDS):
        win = fs.seen_from(pos)
        if target in win:
            return t, log
        for cell in win:
            observed[cell] = True

        if trust > 0:
            if attack is None:
                z, att = fs.true_bearing(pos, target) + rng.vonmises(0.0, noise_kappa), False
            else:
                z, att = attack(pos, target, t, rng)
            hist = (hist + [(pos, z)])[-K_WIN:]

            flag = False
            if defense is not None or collect:
                ts = temporal_stat(hist, obst)
                vs = visual_stat(pos, z, obst, observed)
                if collect:
                    if len(hist) == K_WIN:
                        log["t_stats"].append(ts)
                    log["v_stats"].append(vs)
                if defense == "temporal":
                    flag = ts > tau
                elif defense == "visual":
                    flag = vs > w
                elif defense == "both":
                    flag = ts > tau or vs > w
                elif defense == "oracle":
                    flag = att
            log["heard"] += 1
            log["attacked"] += att
            log["flagged"] += flag
            log["flagged_attacked"] += flag and att

            if not flag:
                ang = np.arctan2(RR - pos[0], CC - pos[1])
                logb += trust * fs.KAPPA * np.cos(z - ang)

        lb = np.where(observed | obst, -np.inf, logb)
        b = np.exp(lb - lb.max())
        b /= b.sum()
        dpos = fs.bfs(obst, pos)
        score = np.where(np.isfinite(dpos) & ~observed & ~obst, b / (dpos + 1.0), -1.0)
        goal = np.unravel_index(np.argmax(score), score.shape)

        dg = fs.bfs(obst, goal)
        best, best_d = pos, dg[pos]
        for dr, dc in fs.MOVES:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < fs.N and 0 <= nc < fs.N and not obst[nr, nc] and dg[nr, nc] < best_d:
                best, best_d = (nr, nc), dg[nr, nc]
        pos = best
    return MAX_ROUNDS, log


# ---------- run one condition over a layout set
_LAYOUTS = {}


def layouts(kind):
    if kind not in _LAYOUTS:
        seeds = range(N_TEST) if kind == "test" else range(5000, 5000 + N_CALIB)
        _LAYOUTS[kind] = [(s, fs.make_layout(np.random.default_rng(s))) for s in seeds]
    return _LAYOUTS[kind]


def run(job):
    """job = dict(kind, attack, s, p, trust, defense, tau, w, noise_sigma, collect)."""
    T, agg = [], dict(heard=0, attacked=0, flagged=0, flagged_attacked=0, t_stats=[], v_stats=[])
    for i, (seed, (obst, start, target)) in enumerate(layouts(job["kind"])):
        rng_att = np.random.default_rng(20_000 + seed)
        rng_ep = np.random.default_rng(10_000 + seed)
        attack = None
        if job.get("attack"):
            attack = ATTACKS[job["attack"]](job["s"], job["p"], obst, start, target, rng_att)
        kappa = 1.0 / np.deg2rad(job.get("noise_sigma", fs.SIGMA_DEG)) ** 2
        t, log = episode(obst, start, target, rng_ep, trust=job.get("trust", 1.0), attack=attack,
                         noise_kappa=kappa, defense=job.get("defense"), tau=job.get("tau", np.inf),
                         w=job.get("w", np.inf), collect=job.get("collect", False))
        T.append(t)
        for k in agg:
            agg[k] += log[k]
    return job, np.array(T, float), agg


def boot_ci(x, n=4000, seed=0):
    """Paired bootstrap 95% CI of the mean, resampling layouts."""
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def rates(a):
    fp = a["flagged"] - a["flagged_attacked"]
    honest = a["heard"] - a["attacked"]
    return dict(
        recall=a["flagged_attacked"] / a["attacked"] if a["attacked"] else np.nan,
        precision=a["flagged_attacked"] / a["flagged"] if a["flagged"] else np.nan,
        fpr=fp / honest if honest else np.nan,
    )


if __name__ == "__main__":
    pool = Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2)

    # ---- 1. calibrate thresholds on held-out layouts, honest audio, no defense
    _, _, cal = run(dict(kind="calib", trust=1.0, collect=True))
    TAU = float(np.quantile(cal["t_stats"], 1 - TARGET_FAR))
    W = float(np.quantile(cal["v_stats"], 1 - TARGET_FAR))
    print(f"calibration ({N_CALIB} layouts, honest): TAU = {np.rad2deg(TAU):.1f} deg, "
          f"W = {np.rad2deg(W):.1f} deg  (each check {TARGET_FAR:.0%} honest false alarms)\n")

    DEFS = ["temporal", "visual", "both", "oracle"]
    jobs = [dict(kind="test", trust=0.0, tag="blind"),
            dict(kind="test", trust=1.0, tag="honest")]
    jobs += [dict(kind="test", trust=1.0, defense=d, tau=TAU, w=W, tag="honest")
             for d in ["temporal", "visual", "both"]]
    jobs += [dict(kind="test", trust=1.0, noise_sigma=NOISY_SIGMA_DEG, tag="noisy")]
    jobs += [dict(kind="test", trust=1.0, noise_sigma=NOISY_SIGMA_DEG, defense=d, tau=TAU, w=W,
                  tag="noisy") for d in ["temporal", "visual", "both"]]
    for a in ATTACKS:
        for s in s2.S_GRID:
            for p in s2.P_GRID:
                base = dict(kind="test", attack=a, s=s, p=p, tag="attack")
                for tr in [0.25, 0.5, 1.0]:
                    jobs.append(dict(base, trust=tr))
                for d in DEFS:
                    jobs.append(dict(base, trust=1.0, defense=d, tau=TAU, w=W))
    print(f"running {len(jobs)} jobs x {N_TEST} layouts ...", flush=True)
    out = pool.map(run, jobs, chunksize=1)

    def get(**q):
        for job, T, agg in out:
            if all(job.get(k) == v for k, v in q.items()) and all(
                    k in q or job.get(k) is None for k in ["defense", "attack", "noise_sigma"]):
                return T, agg
        raise KeyError(q)

    blind, _ = get(tag="blind", trust=0.0)

    # ---- 2. check: undefended runs reproduce Step 2 exactly
    hers = {(r["attack"], int(r["s"]), float(r["p"])): float(r["T_fused"])
            for r in csv.DictReader(open("step2_results_hers.csv"))}
    mism = [k for k, v in hers.items() if abs(get(tag="attack", attack=k[0], s=k[1], p=k[2],
                                                    trust=1.0)[0].mean() - v) > 1e-9]
    print("reproduces step2 T_fused for all 24 conditions:", not mism, mism or "")

    # ---- 3. honest / honest-noisy controls
    print(f"\n{'audio':8s} {'defense':9s} {'T':>6s} {'dT vs none':>11s} {'false alarms':>13s}")
    for tag, label in [("honest", "honest"), ("noisy", f"sig={NOISY_SIGMA_DEG}")]:
        T0, _ = get(tag=tag, trust=1.0, **({"noise_sigma": NOISY_SIGMA_DEG} if tag == "noisy" else {}))
        print(f"{label:8s} {'none':9s} {T0.mean():6.2f} {'':>11s} {'':>13s}")
        for d in ["temporal", "visual", "both"]:
            q = dict(tag=tag, trust=1.0, defense=d)
            if tag == "noisy":
                q["noise_sigma"] = NOISY_SIGMA_DEG
            T, a = get(**q)
            print(f"{label:8s} {d:9s} {T.mean():6.2f} {T.mean() - T0.mean():+11.2f} "
                  f"{a['flagged'] / a['heard']:13.1%}")

    # ---- 4. attack grid
    rows = []
    print(f"\n{'atk':5s} {'s':>4s} {'p':>5s} {'T_und':>6s} {'T_cal':>6s} | "
          + " ".join(f"{d[:4]:>5s}:T  rho  rec  fpr" for d in DEFS))
    for a in ATTACKS:
        for s in s2.S_GRID:
            for p in s2.P_GRID:
                base = dict(tag="attack", attack=a, s=s, p=p)
                Tu, _ = get(**base, trust=1.0)
                cand = {0.0: blind.mean(), **{tr: get(**base, trust=tr)[0].mean() for tr in [0.25, 0.5, 1.0]}}
                best_tr = min(cand, key=cand.get)
                Tc = cand[best_tr]
                line = f"{a:5s} {s:4d} {p:5.2f} {Tu.mean():6.2f} {Tc:6.2f} |"
                for d in DEFS:
                    Td, ag = get(**base, trust=1.0, defense=d)
                    gap = Tu.mean() - Tc
                    rho = (Tu.mean() - Td.mean()) / gap if gap >= 2.0 else np.nan
                    gain = Tu - Td
                    lo, hi = boot_ci(gain)
                    dlo, dhi = boot_ci(blind - Td)
                    rt = rates(ag)
                    rows.append(dict(attack=a, s=s, p=p, defense=d, T_undef=Tu.mean(), T_calib=Tc,
                                     best_trust=best_tr, T_def=Td.mean(), gain=gain.mean(),
                                     gain_ci_lo=lo, gain_ci_hi=hi, rho=rho,
                                     delta_def=(blind - Td).mean(), delta_def_ci_lo=dlo,
                                     delta_def_ci_hi=dhi, success_def=(Td < MAX_ROUNDS).mean(), **rt))
                    line += f" {Td.mean():6.2f} {rho:5.2f} {rt['recall']:4.2f} {rt['fpr']:4.2f}"
                print(line)

    with open("step3_results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---- 5. figure: Delta after each defense, same colour scale as step 2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    und = {(r["attack"], r["s"], r["p"]): r["T_undef"] for r in rows}
    cols = [("none", "No defense")] + [(d, t) for d, t in
                                        [("temporal", "Temporal only"), ("visual", "Visual only"),
                                         ("both", "Temporal + visual"), ("oracle", "Perfect detector")]]
    vals = [blind.mean() - v for v in und.values()] + [r["delta_def"] for r in rows]
    vmax = np.abs(vals).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(2, len(cols), figsize=(3.1 * len(cols), 6.4), facecolor="#fcfcfb")
    for i, a in enumerate(ATTACKS):
        for j, (d, title) in enumerate(cols):
            ax = axes[i, j]
            M = np.zeros((len(s2.S_GRID), len(s2.P_GRID)))
            for si, s in enumerate(s2.S_GRID):
                for pj, p in enumerate(s2.P_GRID):
                    if d == "none":
                        M[si, pj] = blind.mean() - und[(a, s, p)]
                    else:
                        M[si, pj] = next(r["delta_def"] for r in rows
                                         if r["attack"] == a and r["s"] == s and r["p"] == p and r["defense"] == d)
            ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
            for si in range(len(s2.S_GRID)):
                for pj in range(len(s2.P_GRID)):
                    ax.text(pj, si, f"{M[si, pj]:+.1f}", ha="center", va="center", fontsize=9, color="#0b0b0b")
            ax.set_xticks(range(len(s2.P_GRID)), [f"{p:g}" for p in s2.P_GRID])
            ax.set_yticks(range(len(s2.S_GRID)), [f"{s}°" for s in s2.S_GRID])
            if i == 1:
                ax.set_xlabel("attack frequency p", color="#52514e")
            if j == 0:
                ax.set_ylabel(("Naive attack" if a == "naive" else "Coherent decoy") + "\nattack strength s",
                              color="#0b0b0b")
            if i == 0:
                ax.set_title(title, color="#0b0b0b")
            ax.tick_params(colors="#898781", length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle("Δ = rounds(audio-blind) − rounds(agent) after each defense:  blue = audio helps,  red = audio hurts",
                 color="#0b0b0b", fontsize=11)
    fig.tight_layout()
    fig.savefig("step3_defense.png", dpi=180, facecolor="#fcfcfb")
    print("\nSaved step3_results.csv and step3_defense.png")
