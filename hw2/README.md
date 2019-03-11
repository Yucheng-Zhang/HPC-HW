# Homework 2

- Yucheng Zhang

## 1. Finding Memory bugs.

- See comments in the code.

## 2. Optimizing matrix-matrix multiplication.

- Try different loop arrangements.
  - The outputs are shown in [files/2-order-jpi.txt](./files/2-order-jpi.txt), [files/2-order-jip.txt](./files/2-order-jip.txt) and [files/2-order-ipj.txt](./files/2-order-ipj.txt).
  - We can see that the performance of order jpi is the best. This is because the matrices are stored in column major order. For order jpi, `double A_ip = a[i + p * m]; double B_pj = b[p + j * k]; double C_ij = c[i + j * m];` are all read in continuous memory location, which can save a lot of time.
  - Similarly, we get the worst performance for order ipj, which would be the best if the matrices were stored in row major order.

- Implement a one level blocking scheme by using BLOCK_SIZE macro as the block size.

- Experiment with different values for `BLOCK_SIZE`.
  - From the table below, we can see that we get better performance around `BLOCK_SIZE = 64`.

| `BLOCK_SIZE` | `Gflop/s` (Average) | `GB/s` (Average) |
| :----------: | :-----------------: | :--------------: |
|     `4`      |     `6.089491`      |   `97.431855`    |
|     `8`      |     `3.242940`      |   `51.887047`    |
|     `16`     |     `2.426508`      |   `38.824121`    |
|     `32`     |     `15.953180`     |   `255.250880`   |
|     `64`     |     `19.541914`     |   `312.670619`   |
|    `128`     |     `16.749192`     |   `267.987077`   |
|    `256`     |     `14.572623`     |   `233.161976`   |

- Parallelize your matrix-matrix multiplication code using OpenMP.
  - I parallelize the code on the for loop over blocks in C.
  - One thing to notice is how the cache is shared by all the threads. The optimal `BLOCK_SIZE` may be different when we use different number of threads. For example, when I use OpenMP with more than one thread, `BLOCK_SIZE = 32` works better than `BLOCK_SIZE = 64` in the serial case.

- What percentage of the peak FLOP-rate do you achieve with your code?
  - I can achieve up to `22.6 %` of the peak FLOP-rate.

## 3. Finding OpenMP bugs.

- See comments in the code.

## 4. OpenMP version of 2D Jacobi/Gauss-Seidel smoothing.
