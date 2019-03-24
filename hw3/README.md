# Homework 3

- Yucheng Zhang

## 1. Approximating Special Functions Using Taylor Series & Vectorization

## 2. Parallel Scan in OpenMP

|   `Number of threads`   |    `1`     |    `2`     |    `4`     |    `8`     |    `16`    |
| :---------------------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| `Time (s)` (sequential) | `0.284320` | `0.279731` | `0.292832` | `0.283727` | `0.282735` |
|  `Time (s)` (parallel)  | `0.286587` | `0.230436` | `0.181172` | `0.150826` | `0.136962` |

- CPU: `Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz`, with `24` cores and `1` thread per core.
