# Homework 5

- Yucheng Zhang

## 1. MPI ring communication.

- To make sure that the communication must go through the network, we run the program with `mpirun -np 4 --map-by node --hostfile nodes ./int_ring`. In file `node`, we include `4` nodes on CIMS: `crunchy1, crunchy3, crunchy5, crunchy6`.

- Send an integer for `N` loops.

|      `N`      |   `1000`   |  `10000`   |  `100000`  |
| :-----------: | :--------: | :--------: | :--------: |
| `Latency (s)` | `0.000068` | `0.000045` | `0.000052` |

- Send large array for `N` loops.

|        `N`         |   `10`    |   `100`   |  `1000`   |
| :----------------: | :-------: | :-------: | :-------: |
| `Bandwidth (MB/s)` | `109.218` | `102.136` | `105.439` |

## 2. Provide details regarding your final project.

- Project: Parallel Computing of Galaxy Correlation Function

|    `Week`     |                                       `Work`                                        |      `Who`      |
| :-----------: | :---------------------------------------------------------------------------------: | :-------------: |
| 04/14 - 04/20 |                    Learn about the mesh lattice algorithm used.                     | Yucheng, Kaizhe |
| 04/21 - 04/27 |                               Write the serial code.                                | Yucheng, Kaizhe |
| 04/28 - 05/04 |                          Parallelize the code with OpenMP.                          | Yucheng, Kaizhe |
| 05/05 - 05/11 | Test the performance of OpenMP, try MPI & CUDA. Discuss with prof for any problems. | Kaizhe, Yucheng |
| 05/12 - 05/18 |                     Write report and prepare for presentation.                      | Kaizhe, Yucheng |
