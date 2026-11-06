"""
Step 2: directional audio spoofing + RQ1 crossover grid.

Two attacks, each round attacked independently with probability p:
  naive   - bearing = true bearing + s (sign fixed per episode), recomputed from the
            agent's current cell, so successive bearings do NOT triangulate to one source.
  decoy   - a fixed false source cell is chosen per episode at angular offset ~s from the
            true target (as seen from the start); attacked rounds sound as if they come
            from that cell, so bearings DO triangulate consistently (to the wrong place).
Non-attacked rounds are honest. Noise sigma is the same in both cases (fogsearch.SIGMA_DEG).

Outputs (in this folder):
  step2_results.csv          per-condition summary
  step2_crossover.png        heatmap of Delta = T(blind) - T(fused) for both attacks
"""
import csv
import numpy as np
import fogsearch as fs

S_GRID = [45, 90, 180]                  # attack strength, degrees
P_GRID = [0.25, 0.5, 0.75, 1.0]         # attack frequency
TRUST_GRID = [0.0, 0.25, 0.5, 1.0]      # used for the trust-calibrated reference
N_SEEDS = 200
MAX_ROUNDS = 80


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------- attack factories: called once per episode, return f(pos, target, t, rng) ----------
def make_naive(s_deg, p, obst, start, target, rng):
    sign = 1.0 if s_deg >= 180 else rng.choice([-1.0, 1.0])
    off = sign * np.deg2rad(s_deg)

    def f(pos, tgt, t, r):
        b = fs.true_bearing(pos, tgt)
        if r.random() < p:
            b = b + off
        return b + r.vonmises(0.0, fs.KAPPA)
    return f, None


def choose_decoy(s_deg, obst, start, target, rng):
    """Free cell whose bearing from start is closest to (true bearing +/- s), similar range,
    and not visible from the target's neighbourhood."""
    sign = 1.0 if s_deg >= 180 else rng.choice([-1.0, 1.0])
    want = fs.true_bearing(start, target) + sign * np.deg2rad(s_deg)
    rng_t = np.hypot(target[0] - start[0], target[1] - start[1])
    best, best_cost = None, np.inf
    for r in range(fs.N):
        for c in range(fs.N):
            if obst[r, c] or (r, c) == start:
                continue
            if max(abs(r - target[0]), abs(c - target[1])) <= 2:
                continue
            ang_err = abs(wrap(fs.true_bearing(start, (r, c)) - want))
            rng_err = abs(np.hypot(r - start[0], c - start[1]) - rng_t) / max(rng_t, 1)
            cost = ang_err + 0.3 * rng_err
            if cost < best_cost:
                best, best_cost = (r, c), cost
    achieved = abs(np.rad2deg(wrap(fs.true_bearing(start, best) - fs.true_bearing(start, target))))
    return best, achieved


def make_decoy(s_deg, p, obst, start, target, rng):
    decoy, achieved = choose_decoy(s_deg, obst, start, target, rng)

    def f(pos, tgt, t, r):
        src = decoy if r.random() < p else tgt
        return fs.true_bearing(pos, src) + r.vonmises(0.0, fs.KAPPA)
    return f, achieved


# ---------- run one condition on the shared seeded layouts ----------
LAYOUTS = None


def layouts():
    global LAYOUTS
    if LAYOUTS is None:
        LAYOUTS = [fs.make_layout(np.random.default_rng(s)) for s in range(N_SEEDS)]
    return LAYOUTS


def run(factory=None, s=None, p=None, trust=1.0, oracle=False):
    T, achieved = [], []
    for i, (obst, start, target) in enumerate(layouts()):
        rng_att = np.random.default_rng(20_000 + i)   # attack setup (sign, decoy) - same across trust
        rng_ep = np.random.default_rng(10_000 + i)    # per-round noise / attack coin flips
        attack = None
        if factory is not None:
            attack, a = factory(s, p, obst, start, target, rng_att)
            if a is not None:
                achieved.append(a)
        T.append(fs.run_episode(obst, start, target, rng_ep, trust=trust,
                                attack=attack, max_rounds=MAX_ROUNDS, oracle=oracle))
    return np.array(T, float), (np.mean(achieved) if achieved else np.nan)


def ci(x):
    se = x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - 1.96 * se, x.mean() + 1.96 * se


if __name__ == "__main__":
    oracle, _ = run(oracle=True)
    blind, _ = run(trust=0.0)
    honest, _ = run(trust=1.0)
    print(f"sigma = {fs.SIGMA_DEG:.0f} deg, N = {N_SEEDS}")
    print(f"Oracle {oracle.mean():.2f} | audio-blind {blind.mean():.2f} | honest fused {honest.mean():.2f}\n")

    rows = []
    hdr = f"{'attack':6s} {'s':>4s} {'p':>5s} {'T_fused':>8s} {'Delta':>7s} {'95% CI':>17s} {'T_calib':>8s} {'best_trust':>10s} {'decoy_off':>9s}"
    print(hdr)
    for name, fac in [("naive", make_naive), ("decoy", make_decoy)]:
        for s in S_GRID:
            for p in P_GRID:
                fused, ach = run(fac, s, p, trust=1.0)
                d = blind - fused
                lo, hi = ci(d)
                # trust-calibrated reference: best fixed trust for this condition (0 = audio-blind)
                best_t, best_T = 1.0, fused.mean()
                for tr in TRUST_GRID:
                    if tr == 1.0:
                        continue
                    Tt = blind if tr == 0.0 else run(fac, s, p, trust=tr)[0]
                    if Tt.mean() < best_T:
                        best_t, best_T = tr, Tt.mean()
                rows.append(dict(attack=name, s=s, p=p, T_fused=fused.mean(), delta=d.mean(),
                                 ci_lo=lo, ci_hi=hi, T_calibrated=best_T, best_trust=best_t,
                                 success=(fused < MAX_ROUNDS).mean(), decoy_offset_deg=ach))
                print(f"{name:6s} {s:4d} {p:5.2f} {fused.mean():8.2f} {d.mean():7.2f} "
                      f"[{lo:6.2f},{hi:6.2f}] {best_T:8.2f} {best_t:10.2f} {ach:9.1f}")

    with open("step2_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- heatmap ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    deltas = np.array([r["delta"] for r in rows])
    vmax = np.abs(deltas).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")
    for ax, name in zip(axes, ["naive", "decoy"]):
        M = np.array([[next(r["delta"] for r in rows if r["attack"] == name and r["s"] == s and r["p"] == p)
                       for p in P_GRID] for s in S_GRID])
        sig = np.array([[next(r["ci_hi"] < 0 or r["ci_lo"] > 0 for r in rows
                              if r["attack"] == name and r["s"] == s and r["p"] == p)
                         for p in P_GRID] for s in S_GRID])
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(S_GRID)):
            for j in range(len(P_GRID)):
                ax.text(j, i, f"{M[i, j]:+.1f}" + ("" if sig[i, j] else "\nn.s."),
                        ha="center", va="center", fontsize=10, color="#0b0b0b")
        ax.set_xticks(range(len(P_GRID)), [f"{p:g}" for p in P_GRID])
        ax.set_yticks(range(len(S_GRID)), [f"{s}°" for s in S_GRID])
        ax.set_xlabel("attack frequency p", color="#52514e")
        ax.set_ylabel("attack strength s", color="#52514e")
        ax.set_title("Naive attack" if name == "naive" else "Coherent decoy", color="#0b0b0b")
        ax.tick_params(colors="#898781", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_facecolor("#fcfcfb")
    fig.suptitle("Δ = rounds(audio-blind) − rounds(fused):  blue = audio helps,  red = audio is a liability",
                 color="#0b0b0b", fontsize=11)
    fig.tight_layout()
    fig.savefig("step2_crossover.png", dpi=200, facecolor="#fcfcfb")
    print("\nSaved step2_results.csv and step2_crossover.png")
