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
