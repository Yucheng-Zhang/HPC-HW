CC=g++

CFLAGS = -O3 -Wall -march=native

default:
	$(CC) $(CFLAGS) val_test01_solved.cpp -o val_test01_solved
	$(CC) $(CFLAGS) val_test02_solved.cpp -o val_test02_solved
	$(CC) $(CFLAGS) -fopenmp MMult1.cpp -o MMult1
	$(CC) $(CFLAGS) -fopenmp omp_solved2.c -o omp_solved2
	$(CC) $(CFLAGS) -fopenmp omp_solved3.c -o omp_solved3
	$(CC) $(CFLAGS) -lm jacobi2D-omp.cpp -o jacobi2D-omp
	$(CC) $(CFLAGS) -lm gs2D-omp.cpp -o gs2D-omp

EXES = val_test01_solved val_test02_solved MMult1 \
		omp_solved2 omp_solved3 \
		jacobi2D-omp gs2D-omp

clean:
	rm -rf $(EXES)
