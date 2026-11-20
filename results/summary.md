# Serving benchmark (H100 PCIe, GPU 0, 512x512 image-to-image, 50% avatar / 50% sticker, 3 styles)

Server numbers on an H100, not phone numbers.

## Summary

| config | model | steps | peak VRAM GB | CCU1 img/s (p50 s) | CCU8 img/s (p95 s) | closed-loop peak img/s | GPU-s/img | max CCU p95≤1.5s | RPS p95≤1s | RPS p95≤1.5s | RPS p95≤2s | foreign GPU activity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdxs_timingonly_b8 | sdxs | 2 | 5.86 | 4.831 (0.2058) | 13.378 (0.8337) | 13.378 | 0.0747 | 8 | 10.336 | 10.336 | 10.336 | 86.5% |

## Closed loop: img/s / p95 latency at each CCU

| config | CCU1 | CCU2 | CCU3 | CCU4 | CCU5 | CCU6 | CCU7 | CCU8 |
|---|---|---|---|---|---|---|---|---|
| sdxs_timingonly_b8 | 4.83 / 0.22s | 9.41 / 0.22s | 10.47 / 0.30s | 10.43 / 0.40s | 12.04 / 0.59s | 12.22 / 0.64s | 12.64 / 0.76s | 13.38 / 0.83s |

## Open loop (Poisson arrivals)

| config | offered RPS | served img/s | p50 s | p95 s | p99 s | CCU avg/peak | errors |
|---|---|---|---|---|---|---|---|
| sdxs_timingonly_b8 | 4.01 | 3.934 | 0.2094 | 0.2225 | 0.2272 | 0.83/2 | 0 |
| sdxs_timingonly_b8 | 6.69 | 5.361 | 0.2103 | 0.2718 | 0.2762 | 1.19/3 | 0 |
| sdxs_timingonly_b8 | 9.36 | 7.241 | 0.2395 | 0.371 | 0.3807 | 1.92/4 | 0 |
| sdxs_timingonly_b8 | 11.37 | 8.755 | 0.2671 | 0.4932 | 0.5004 | 2.71/6 | 0 |
| sdxs_timingonly_b8 | 13.38 | 10.336 | 0.3369 | 0.6174 | 0.6677 | 3.73/7 | 0 |
