"""
Step 1: minimal closed loop for the CIS 7000 project.

Single-agent 12x12 fog-of-war search. The agent knows the obstacle map, sees only a
radius-1 neighbourhood, and (optionally) hears a noisy azimuth to the target each round.
One belief-map planner covers every condition; the only knob is `trust`:
    trust = 0  -> audio-blind agent  (pure frontier exploration)
    trust = 1  -> agent that treats the bearing as an honest noisy measurement
Later steps plug an attack into `attack(...)` and a detector into the trust decision.
"""
import numpy as np
from collections import deque

N = 12
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIGMA_DEG = 60.0                       # honest bearing noise (chosen so audio helps but is not near-oracle)
KAPPA = 1.0 / np.deg2rad(SIGMA_DEG) ** 2   # von Mises concentration ~ Gaussian sigma


# ---------- map + paths ----------
def bfs(obst, src):
    """Shortest-path distance from src to every cell (inf = unreachable / obstacle)."""
    d = np.full(obst.shape, np.inf)
    d[src] = 0
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and not obst[nr, nc] and d[nr, nc] == np.inf:
                d[nr, nc] = d[r, c] + 1
                q.append((nr, nc))
    return d


def make_layout(rng, density=0.15, min_dist=8):
    """Random connected map plus start/target at least `min_dist` apart."""
    while True:
        obst = rng.random((N, N)) < density
        free = np.argwhere(~obst)
        if len(free) < 20:
            continue
        d0 = bfs(obst, tuple(free[0]))
        if np.isinf(d0[~obst]).any():
            continue
        start = tuple(free[rng.integers(len(free))])
        ds = bfs(obst, start)
        far = [tuple(c) for c in free if ds[tuple(c)] >= min_dist]
        if not far:
            continue
        target = far[rng.integers(len(far))]
        return obst, start, target


def seen_from(pos):
    """Cells inside the radius-1 (Chebyshev) visibility window."""
    r, c = pos
    return [(i, j) for i in range(max(0, r - 1), min(N, r + 2))
            for j in range(max(0, c - 1), min(N, c + 2))]


# ---------- audio ----------
def true_bearing(frm, to):
    """Azimuth from `frm` to `to` (radians); x = column, y = row."""
    return np.arctan2(to[0] - frm[0], to[1] - frm[1])


def honest_audio(pos, target, rng, kappa=KAPPA):
    return true_bearing(pos, target) + rng.vonmises(0.0, kappa)


# ---------- one episode ----------
def run_episode(obst, start, target, rng, trust=1.0, kappa=KAPPA,
                attack=None, max_rounds=80, oracle=False):
    """Return rounds-to-target (max_rounds if not found).

    attack: optional callable (pos, target, t, rng) -> observed azimuth. None = honest.
    """
    pos = start
    observed = np.zeros((N, N), bool)          # visual memory: cells seen to be empty
    logb = np.zeros((N, N))                    # log belief over target cell
    rr, cc = np.indices((N, N))

    for t in range(max_rounds):
        win = seen_from(pos)
        if target in win:
            return t
        for cell in win:
            observed[cell] = True

        # ---- choose a goal ----
        if oracle:
            goal = target
        else:
            if trust > 0:
                if attack is None:
                    z = honest_audio(pos, target, rng, kappa)
                else:
                    z = attack(pos, target, t, rng)
                ang = np.arctan2(rr - pos[0], cc - pos[1])
                logb += trust * kappa * np.cos(z - ang)   # von Mises log-likelihood
            lb = np.where(observed | obst, -np.inf, logb)
            b = np.exp(lb - lb.max())
            b /= b.sum()
            dpos = bfs(obst, pos)
            score = np.where(np.isfinite(dpos) & ~observed & ~obst, b / (dpos + 1.0), -1.0)
            goal = np.unravel_index(np.argmax(score), score.shape)

        # ---- step toward the goal along a shortest path ----
        dg = bfs(obst, goal)
        best, best_d = pos, dg[pos]
        for dr, dc in MOVES:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < N and 0 <= nc < N and not obst[nr, nc] and dg[nr, nc] < best_d:
                best, best_d = (nr, nc), dg[nr, nc]
        pos = best
    return max_rounds


# ---------- experiment ----------
def run_condition(seeds, **kw):
    out = []
    for s in seeds:
        rng_map = np.random.default_rng(s)
        obst, start, target = make_layout(rng_map)
        rng_ep = np.random.default_rng(10_000 + s)   # independent stream for audio noise
        out.append(run_episode(obst, start, target, rng_ep, **kw))
    return np.array(out, float)


if __name__ == "__main__":
    seeds = range(200)                      # same layouts for every condition -> paired
    conds = {
        "Oracle":                 dict(oracle=True),
        "Perfect bearing":        dict(trust=1.0, kappa=500.0),
        "Honest audio (fused)":   dict(trust=1.0),
        "Audio-blind":            dict(trust=0.0),
    }
    res = {k: run_condition(seeds, **v) for k, v in conds.items()}

    print(f"Honest bearing noise sigma = {SIGMA_DEG:.0f} deg, N = {len(seeds)} paired layouts\n")
    print(f"{'condition':24s} {'mean rounds':>12s} {'median':>8s} {'success@80':>11s}")
    for k, v in res.items():
        print(f"{k:24s} {v.mean():12.2f} {np.median(v):8.1f} {(v < 80).mean():11.2f}")

    delta = res["Audio-blind"] - res["Honest audio (fused)"]      # > 0 means audio helps
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    print(f"\nDelta = T(blind) - T(fused): mean {delta.mean():.2f}  "
          f"95% CI [{delta.mean() - 1.96 * se:.2f}, {delta.mean() + 1.96 * se:.2f}]  "
          f"(fused faster in {(delta > 0).mean():.0%} of episodes)")
