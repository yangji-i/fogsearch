"""
Spatial-audio front end (docx Sec. 3, "Inputs and outputs").

Replaces "true bearing + von Mises noise" with a bearing ESTIMATED from simulated
6-microphone audio:
  1. room: the 12x12 grid is a 6 m x 6 m 2-D room (0.5 m cells). Following the docx scope,
     obstacles do NOT affect sound; only the outer walls reflect (pyroomacoustics ShoeBox,
     image-source method).
  2. each (agent cell, source cell) pair: compute the room impulse responses (RIR) to a
     6-mic circular array centred on the agent cell, convolve a 0.5 s speech-band signal,
     add sensor noise at SNR_DB, and estimate the azimuth with SRP-PHAT.
  3. repeat with N_DRAWS different signals/noise -> lookup table est[agent, source, draw].
At run time the agent hears est[agent, source, random draw] (docx: "precomputed with
several noise draws and looked up per round").

Build the table once (on your own laptop, needs `pip install pyroomacoustics`):
    python audio_frontend.py                       -> doa_table.npz      (SNR 10 dB)
    python audio_frontend.py --snr 0 --out doa_table_noisy.npz          (honest-but-noisy control)
Preview without pyroomacoustics (own image-source stand-in / direct path only):
    python audio_frontend.py --rir ism --out doa_table_ism.npz
    python audio_frontend.py --rir freefield --out doa_table_freefield.npz

Angles use the same convention as fogsearch.true_bearing: atan2(d_row, d_col), radians.
"""
import argparse
import time
from multiprocessing import Pool

import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt

N = 12                 # grid size (cells)
CELL = 0.5             # metres per cell
ROOM = [N * CELL, N * CELL]
FS = 16000
DUR = 0.5              # seconds of audio per round
C = 343.0              # speed of sound, m/s
M = 6                  # microphones
RADIUS = 0.05          # array radius, m (mic spacing 5 cm -> no spatial aliasing below ~3.4 kHz)
NFFT, HOP = 512, 256
FMIN, FMAX = 300, 3400  # SRP-PHAT frequency band, Hz
AZ_GRID = np.deg2rad(np.arange(0.0, 360.0, 1.0))   # 1 degree search grid

CFG = dict(snr_db=10.0, absorption=0.3, max_order=8, n_draws=8, rir="pra", seed=0)


# ---------- geometry
def cell_xy(r, c):
    """Cell centre in metres: x = column, y = row (matches fogsearch.true_bearing)."""
    return np.array([(c + 0.5) * CELL, (r + 0.5) * CELL])


MIC_OFF = RADIUS * np.stack([np.cos(2 * np.pi * np.arange(M) / M),
                             np.sin(2 * np.pi * np.arange(M) / M)])        # 2 x M


# ---------- room impulse responses
def rir_pyroomacoustics(src, mics, absorption, max_order):
    import pyroomacoustics as pra
    room = pra.ShoeBox(ROOM, fs=FS, materials=pra.Material(absorption), max_order=max_order)
    room.add_source(src)
    room.add_microphone_array(pra.MicrophoneArray(mics, FS))
    room.compute_rir()
    return [np.asarray(room.rir[m][0]) for m in range(M)]


def _render(images, gains, mics):
    """Sum of fractional-delay impulses, one per image source, for each mic."""
    half = 64
    out = []
    for m in range(M):
        d = np.linalg.norm(images - mics[:, m:m + 1], axis=0)
        delay = half + d / C * FS
        L = int(delay.max()) + half + 2
        h = np.zeros(L)
        k = np.arange(-half, half + 1)
        for dl, g, dd in zip(delay, gains, d):
            i0 = int(np.floor(dl))
            idx = i0 + k
            h[idx] += g / max(dd, 0.1) * np.sinc(idx - dl) * np.hanning(2 * half + 1)
        out.append(h)
    return out


def rir_freefield(src, mics, *_):
    """Direct path only (what pyroomacoustics gives with max_order=0). For testing."""
    return _render(np.asarray(src, float)[:, None], np.ones(1), mics)


def rir_ism(src, mics, absorption, max_order):
    """Stand-in for pyroomacoustics when it is not installed: 2-D shoebox image-source
    method (Allen & Berkley 1979), same room/absorption/max_order. For previews only;
    report results from the real pyroomacoustics table."""
    beta = np.sqrt(1 - absorption)          # pressure reflection coefficient per bounce
    axes = []
    for dim in range(2):
        Ld, x0 = ROOM[dim], src[dim]
        cand = []
        for k in range(-max_order, max_order + 1):
            cand.append((2 * k * Ld + x0, abs(2 * k)))
            cand.append((2 * k * Ld - x0, abs(2 * k - 1)))
        axes.append(cand)
    pos, gains = [], []
    for x, nx in axes[0]:
        for y, ny in axes[1]:
            if nx + ny <= max_order:
                pos.append((x, y))
                gains.append(beta ** (nx + ny))
    return _render(np.array(pos).T, np.array(gains), mics)


# ---------- one simulated recording + SRP-PHAT
_BAND = butter(4, [100, 4000], btype="band", fs=FS, output="sos")


def speech_like(rng):
    """Speech-band noise with a ~4 Hz syllable-rate envelope (no speech dataset needed)."""
    n = int(DUR * FS)
    x = sosfilt(_BAND, rng.standard_normal(n + 2000))[2000:]
    t = np.arange(n) / FS
    env = 0.55 + 0.45 * np.sin(2 * np.pi * rng.uniform(3, 5) * t + rng.uniform(0, 2 * np.pi))
    return x * env


def record(rirs, rng, snr_db):
    s = speech_like(rng)
    n = len(s)
    L = max(len(h) for h in rirs)
    Y = np.stack([fftconvolve(s, np.pad(h, (0, L - len(h)))) for h in rirs])[:, :n]
    p = np.mean(Y ** 2)
    return Y + rng.normal(0.0, np.sqrt(p / 10 ** (snr_db / 10)), Y.shape)


_PAIRS = [(i, j) for i in range(M) for j in range(i + 1, M)]
_FREQ = np.fft.rfftfreq(NFFT, 1 / FS)
_BINS = np.where((_FREQ >= FMIN) & (_FREQ <= FMAX))[0]
# expected phase for each (pair, bin, angle): plane wave from direction u(theta) reaches mic m
# earlier by p_m . u / c  ->  X_i X_j^* has phase  +omega * (p_i - p_j) . u / c
_U = np.stack([np.cos(AZ_GRID), np.sin(AZ_GRID)])                          # 2 x A
_STEER = np.exp(-1j * 2 * np.pi * _FREQ[_BINS][None, :, None]
                * np.stack([(MIC_OFF[:, i] - MIC_OFF[:, j]) @ _U for i, j in _PAIRS])[:, None, :] / C)


def srp_phat(Y):
    """Azimuth (radians, [0, 2pi)) maximising the SRP-PHAT power. Array-relative geometry only."""
    win = np.hanning(NFFT)
    starts = range(0, Y.shape[1] - NFFT, HOP)
    X = np.stack([np.fft.rfft(Y[:, s:s + NFFT] * win, axis=1) for s in starts], axis=2)  # M x F x T
    X = X[:, _BINS, :]
    G = np.stack([np.mean((lambda c: c / (np.abs(c) + 1e-12))(X[i] * np.conj(X[j])), axis=1)
                  for i, j in _PAIRS])                                        # P x B
    power = np.real(np.einsum("pb,pba->a", G, _STEER))
    return AZ_GRID[np.argmax(power)]


# ---------- table
def _row(a):
    """All source cells for agent cell index a."""
    cfg = CFG
    rir_fn = {"pra": rir_pyroomacoustics, "freefield": rir_freefield, "ism": rir_ism}[cfg["rir"]]
    ar, ac = divmod(a, N)
    mics = cell_xy(ar, ac)[:, None] + MIC_OFF
    rng = np.random.default_rng([cfg["seed"], a])
    row = np.full((N * N, cfg["n_draws"]), np.nan)
    for s in range(N * N):
        if s == a:
            continue
        rirs = rir_fn(cell_xy(*divmod(s, N)), mics, cfg["absorption"], cfg["max_order"])
        for d in range(cfg["n_draws"]):
            row[s, d] = srp_phat(record(rirs, rng, cfg["snr_db"]))
    return a, row


def _init(cfg):
    CFG.update(cfg)


def true_table():
    rr, cc = np.divmod(np.arange(N * N), N)
    return np.arctan2(rr[None, :] - rr[:, None], cc[None, :] - cc[:, None])   # [agent, source]


def build(cfg, procs, out):
    t0 = time.time()
    est = np.full((N * N, N * N, cfg["n_draws"]), np.nan)
    with Pool(procs, initializer=_init, initargs=(cfg,)) as pool:
        for k, (a, row) in enumerate(pool.imap_unordered(_row, range(N * N))):
            est[a] = row
            if (k + 1) % 12 == 0:
                el = time.time() - t0
                print(f"  {k + 1}/{N * N} agent cells, {el:.0f}s elapsed, ~{el / (k + 1) * (N * N - k - 1):.0f}s left",
                      flush=True)
    np.savez_compressed(out, est=est, **{k: np.array(v) for k, v in cfg.items()})
    print(f"saved {out} in {time.time() - t0:.0f}s")
    return est


# ---------- runtime sensor used by step3_defense.py
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TableSensor:
    def __init__(self, path):
        z = np.load(path)
        self.est = z["est"]
        self.path = path
        self.info = {k: z[k].item() for k in z.files if k != "est"}

    def bearing(self, pos, src, rng):
        return self.est[pos[0] * N + pos[1], src[0] * N + src[1], rng.integers(self.est.shape[2])]

    def errors(self, min_cheb=2):
        """Estimation errors for every pair the agent can actually hear (target not yet visible)."""
        e = wrap(self.est - true_table()[:, :, None])
        rr, cc = np.divmod(np.arange(N * N), N)
        cheb = np.maximum(np.abs(rr[:, None] - rr[None, :]), np.abs(cc[:, None] - cc[None, :]))
        return e[cheb >= min_cheb].ravel()

    def fit_kappa(self):
        """von Mises maximum-likelihood concentration of the errors (Best & Fisher approx.)."""
        e = self.errors()
        R = np.abs(np.mean(np.exp(1j * e)))
        if R < 0.53:
            k = 2 * R + R ** 3 + 5 * R ** 5 / 6
        elif R < 0.85:
            k = -0.4 + 1.39 * R + 0.43 / (1 - R)
        else:
            k = 1 / (R ** 3 - 4 * R ** 2 + 3 * R)
        return float(k)


def report(path):
    s = TableSensor(path)
    e = np.rad2deg(np.abs(s.errors()))
    k = s.fit_kappa()
    print(f"{path}: {s.info}")
    print(f"  |error| median {np.median(e):.1f} deg, 90th pct {np.percentile(e, 90):.1f}, "
          f"99th pct {np.percentile(e, 99):.1f}, >30 deg {np.mean(e > 30):.1%}")
    print(f"  fitted von Mises kappa = {k:.1f}  (~sigma {np.rad2deg(1 / np.sqrt(k)):.1f} deg; "
          f"analytic sensor used kappa 0.91 ~ sigma 60 deg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rir", choices=["pra", "ism", "freefield"], default="pra")
    ap.add_argument("--snr", type=float, default=CFG["snr_db"])
    ap.add_argument("--absorption", type=float, default=CFG["absorption"])
    ap.add_argument("--max-order", type=int, default=CFG["max_order"])
    ap.add_argument("--draws", type=int, default=CFG["n_draws"])
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--out", default="doa_table.npz")
    a = ap.parse_args()
    cfg = dict(CFG, rir=a.rir, snr_db=a.snr, absorption=a.absorption, max_order=a.max_order,
               n_draws=a.draws)
    build(cfg, a.procs, a.out)
    report(a.out)
