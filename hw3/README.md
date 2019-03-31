# Homework 3

- Yucheng Zhang

## 1. Approximating Special Functions Using Taylor Series & Vectorization

- I improved the accuracy to 12-digits by adding more terms to both `SSE` and `AVX` parts of the function `sin4_intrin()`, although my processor seems to support `SSE`.
- I also did the same thing for the function `sin4_vec()`.
- To evaluate the function outside of the interval `[-pi/4, pi/4]`, we need to compute `cos(x)` for `x` in the interval `[-pi/4, pi/4]` by doing the same method we have done for `sin(x)`. Then `sin(x) = sin(a)`, with `a = x mod 2pi`.
  - `sin(a) = sin(a - 2pi)` for `a in [7/4pi, 2pi]`
  - `sin(a) = -cos(a - 3/2pi)` for `a in [5/4pi, 7/4pi]`
  - `sin(a) = -sin(a - pi)` for `a in [3/4pi, 5/4pi]`
  - `sin(a) = cos(a - pi/2)` for `a in [1/4pi, 3/4pi]`
  - `sin(a) = sin(a)` for `a in [0, pi/4]`

## 2. Parallel Scan in OpenMP

|   `Number of threads`   |    `1`     |    `2`     |    `4`     |    `8`     |    `16`    |
| :---------------------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| `Time (s)` (sequential) | `0.284320` | `0.279731` | `0.292832` | `0.283727` | `0.282735` |
|  `Time (s)` (parallel)  | `0.286587` | `0.230436` | `0.181172` | `0.150826` | `0.136962` |

- CPU: `Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz`, with `24` cores and `1` thread per core.
