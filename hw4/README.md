# Homework 4

- Yucheng Zhang

## 1. Matrix-vector operations on a GPU.

- GPU & CPU models
  - `cuda1: GeForce GTX TITAN Black & Intel Xeon E5-2680 (2.50 GHz)`
  - `cuda2: GeForce RTX 2080 Ti & Intel Xeon E5-2660 (2.60 GHz)`
  - `cuda3: TITAN V & Intel Xeon Gold 5118 (2.30 GHz)`
  - `cuda4: GeForce GTX TITAN X & Intel Xeon Gold 5118 (2.30 GHz)`
  - `cuda5: GeForce GTX TITAN Z & Intel Xeon E5-2650 (2.60 GHz)`

- `vvmulti.cu` includes code for vector-vector inner product.
  - The table below shows the memory band for vector size `N = 2^{25}` on different GPUs provided at CIMS. CUDA version: `cuda-10.0`.

|         `GPU`          | `cuda1`  | `cuda2`  | `cuda3`  | `cuda4`  | `cuda5`  |
| :--------------------: | :------: | :------: | :------: | :------: | :------: |
| `CPU Bandwidth (GB/s)` | `30.79`  | `27.91`  | `11.12`  | `14.65`  | `21.92`  |
| `GPU Bandwidth (GB/s)` | `132.16` | `287.95` | `492.96` | `201.29` | `107.67` |



- `mvmulti.cu` includes code for matrix-vector multiplication.
  - Because of the limitation of memory on host and especially device, matrix with `N = 2^{25}` is too large and results in `Segmentation fault`.
  - We can either copy the whole matrix from host to device at one time or row by row. The second method should work for larger matrix but might be slower. The code implements the first one.
  - The table below shows the memory band for matrix (N*N ) size `N = 2^{13}`.

|         `GPU`          | `cuda1`  | `cuda2`  |  `cuda3`   | `cuda4`  | `cuda5`  |
| :--------------------: | :------: | :------: | :--------: | :------: | :------: |
| `CPU Bandwidth (GB/s)` | `0.0200` | `0.0607` |  `0.0078`  | `0.0400` | `0.0205` |
| `GPU Bandwidth (GB/s)` | `0.1127` | `4.5419` | `4.642851` | `0.0888` | `0.0892` |

- One thing to notice is that on CIMS CPUs and GPUs are shared by everyone at the same time, so the bandwidth may not be accurate and stable.

## 2. 2D Jacobi method on a GPU.

- For this problem, I run it on the same five GPUs as used in Problem 1.
- 


## 3. Pitch your final project.

For the final project, `Kaizhe Wang` and I will do the computation of correlation function of galaxies, which is the HPC application I mentioned in the first homework. The most computation intensive part is to calculate the distance between two sets of points and fit them into proper bins. The things we plan to do include,

- Figure out the mesh algorithm. Naive pair count requires `O(N^2)` time complexity, where `N` is the size of the data. With the mesh algorithm, we hope to reduce the time complexity to `O(NlogN)`.
- Include jackknife resampling in the code. The input data points are divided into `njk` parts, where `njk` stands for the number of jackknife regions. For more information about jackknife resampling, see https://en.wikipedia.org/wiki/Jackknife_resampling.
- Parallelize the code with `OpenMP` and `MPI` in `C`.
- Wrap the parallelized `C` with `Python`, which works better in loading data and plotting results.
