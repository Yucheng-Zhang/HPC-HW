CC=g++

CFLAGS = -O3 -Wall

all:
#	$(CC) $(CFLAGS) val_test01_solved.cpp -o val_test01_solved
#	$(CC) $(CFLAGS) val_test02_solved.cpp -o val_test02_solved
	$(CC) $(CFLAGS) -fopenmp -march=native MMult1.cpp -o MMult1
#	$(CC) $(CFLAGS) -fopenmp omp_solved2.c -o omp_solved2
#	$(CC) $(CFLAGS) -fopenmp omp_solved3.c -o omp_solved3
#	$(CC) $(CFLAGS) -fopenmp omp_solved4.c -o omp_solved4
#	$(CC) $(CFLAGS) -fopenmp omp_solved5.c -o omp_solved5
#	$(CC) $(CFLAGS) -fopenmp omp_solved6.c -o omp_solved6
#	$(CC) $(CFLAGS) -fopenmp -lm jacobi2D-omp.cpp -o jacobi2D-omp
#	$(CC) $(CFLAGS) -fopenmp -lm gs2D-omp.cpp -o gs2D-omp

EXES = val_test01_solved val_test02_solved MMult1 \
		omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 \
		jacobi2D-omp gs2D-omp 

clean:
	rm -rf $(EXES)
