# Simulated-audio front end (replaces "true bearing + noise")

Put these files in the same folder as fogsearch.py and step2_attacks.py:
audio_frontend.py, step3_defense.py, step2_results_hers.csv (a copy of step2_results.csv).

## 1. Install
    py -m pip install numpy scipy matplotlib pyroomacoustics

## 2. Build the lookup tables (once, about 10-30 min each, depending on the laptop)
    py audio_frontend.py --procs 4                                   # -> doa_table.npz (SNR 10 dB)
    py audio_frontend.py --procs 4 --snr 0 --out doa_table_noisy.npz # honest-but-noisy control

Each prints the error summary (median / 90th pct / fitted kappa). Expect median of a few degrees.
If pyroomacoustics will not install, `--rir ism` uses the built-in image-source stand-in
(same method, no extra package). Use it only for previews.

## 3. Run RQ1 + RQ2 with the simulated audio (about 10 min)
    py step3_defense.py --sensor table --table doa_table.npz --noisy-table doa_table_noisy.npz --procs 4

Outputs: step3_results_table_doa_table.csv, step3_defense_table_doa_table.png.
The "No defense" column of the figure is the RQ1 heatmap.

## Old sensor (sigma 60 deg), for comparison
    py step3_defense.py --procs 4      # reproduces step 2 exactly; prints a check line

## Settings (top of audio_frontend.py)
2-D 6 m x 6 m room, 0.5 m cells; obstacles do not affect sound (docx scope).
6-mic circular array, radius 5 cm; 16 kHz; 0.5 s speech-band noise per round;
wall energy absorption 0.3; image-source order 8; SRP-PHAT 300-3400 Hz, 1 degree grid;
8 recordings per (agent cell, source cell) pair. The agent's von Mises kappa is fitted to the table.
