/******************************************************************************
 * FILE: omp_bug2.c
 * DESCRIPTION:
 *   Another OpenMP program with a bug.
 * AUTHOR: Blaise Barney
 * LAST REVISED: 04/06/05
 ******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int nthreads, tid;
  long i;             // int range may not be long enough
  double total = 0.0; // initialize before parallel

/*** Spawn parallel region ***/
// declare shared and private
#pragma omp parallel shared(total) private(tid)
  {
    /* Obtain thread number */
    tid = omp_get_thread_num();
    /* Only master thread does this */
    if (tid == 0) {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
    printf("Thread %d is starting...\n", tid);

#pragma omp barrier

    /* do some work */
    // race condition, use reduction
#pragma omp for reduction(+ : total)
    for (i = 0; i < 1000000; i++)
      total += i * 1.0;

    printf("Thread %d is done! Total= %e\n", tid, total);

  } /*** End of parallel region ***/
}
