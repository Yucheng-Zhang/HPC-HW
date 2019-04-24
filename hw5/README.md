# Homework 5

- Yucheng Zhang

## 1. MPI ring communication.

- To make sure that the communication must go through the network, we run the program with `mpirun -np 4 --map-by node --hostfile nodes ./int_ring`. In file `node`, we include `4` nodes on CIMS: `crunchy1, crunchy3, crunchy5, crunchy6`.

- Send an integer for `N` loops.

|      `N`      | `1000` |
| :-----------: | :----: |
| `Latency (s)` |        |

- Send large array for `N` loops.

|        `N`         | `10`  |
| :----------------: | :---: |
| `Bandwidth (MB/s)` |       |

## 2. Provide details regarding your final project.
