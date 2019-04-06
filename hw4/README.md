# Homework 4

- Yucheng Zhang

## 1. Matrix-vector operations on a GPU.

- `vvmulti.cu` includes code for vector-vector inner product.
- The table below shows the memory band for vector size `N = 2^{25}` on different GPUs provided at CIMS. CUDA version: `cuda-10.0`.

|         `GPU`          | `cuda1`  | `cuda2`  | `cuda3`  | `cuda4`  | `cuda5`  |
| :--------------------: | :------: | :------: | :------: | :------: | :------: |
| `CPU Bandwidth (GB/s)` | `30.79`  | `27.91`  | `11.12`  | `14.65`  | `21.92`  |
| `GPU Bandwidth (GB/s)` | `132.16` | `287.95` | `492.96` | `201.29` | `107.67` |

- GPU & CPU models
  - `cuda1: GeForce GTX TITAN Black & Intel Xeon E5-2680 (2.50 GHz)`
  - `cuda2: GeForce RTX 2080 Ti & Intel Xeon E5-2660 (2.60 GHz)`
  - `cuda3: TITAN V & Intel Xeon Gold 5118 (2.30 GHz)`
  - `cuda4: GeForce GTX TITAN X & Intel Xeon Gold 5118 (2.30 GHz)`
  - `cuda5: GeForce GTX TITAN Z & Intel Xeon E5-2650 (2.60 GHz)`

- `mvmulti.cu` includes code for matrix-vector multiplication.
  - Because of the limitation of memory on host and especially device, matrix with `N = 2^{25}` is too large and results in `Segmentation fault`.
  - We can either copy the whole matrix from host to device at one time or row by row. The second method should work for larger matrix but might be slower. The code implements the first one.

## 2. 2D Jacobi method on a GPU.

## 3. Pitch your final project.
