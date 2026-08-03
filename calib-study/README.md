# Calibration stability study artifacts

`REPORT.md` is the human-readable verdict. `calibration_stability.py` is the
self-contained CPU-only analysis. The JSON files contain full summaries and
provenance; `curves/` contains CSV curve data and dependency-free SVG plots.

Run from the repository root:

```bash
ionice -c 2 -n 7 nice -n 10 \
  env CUDA_VISIBLE_DEVICES='' \
  python3 calib-study/calibration_stability.py \
  --output calib-study --repeats 40 --seed 20260803
```

No Docker or CUDA path is used. Reference inputs under `/home/rob/dq-runs` are
opened read-only; all writes go to `calib-study/`.
