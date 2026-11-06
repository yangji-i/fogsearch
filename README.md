# CIS 7000 – Step 1: baseline search loop

Single-agent 12x12 fog-of-war search. Checks that honest directional audio
(bearing noise sigma = 60 deg) makes the fused agent clearly faster than an
audio-blind agent, before any attack is added.

## Run (Windows, PowerShell)

    cd $HOME\Documents\cis7000-fogsearch
    python -m pip install -r requirements.txt
    python fogsearch.py

## Expected output (200 paired layouts, ~5 s)

    condition                 mean rounds   median  success@80
    Oracle                           9.83      9.0        1.00
    Perfect bearing                  9.88      9.0        1.00
    Honest audio (fused)            14.62     13.0        1.00
    Audio-blind                     32.55     32.0        0.99

    Delta = T(blind) - T(fused): mean 17.92  95% CI [15.50, 20.34]

Numbers should match exactly (seeded); small differences can appear with a
very different numpy version.

## Next step
Plug an attack into run_episode(..., attack=f) where
f(pos, target, t, rng) -> observed azimuth (radians).

---

# Step 2: attacks + RQ1 crossover grid

    py -m pip install -r requirements.txt   # adds matplotlib
    py step2_attacks.py                      # ~3-4 min

Writes `step2_results.csv` and `step2_crossover.png` in this folder.
Grid: s in {45, 90, 180} deg x p in {0.25, 0.5, 0.75, 1.0} x {naive, decoy}, N = 200 paired layouts.
`T_calib` is the best fixed trust in {0, 0.25, 0.5, 1} for that condition (trust-calibrated reference).
`decoy_off` is the decoy's actual mean angular offset (the 12x12 grid can't always fit the requested s).
