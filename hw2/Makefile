CCP=g++
CC=gcc

CFLAGS = -O3 -Wall -march=native

default:
	$(CCP) $(CFLAGS) val_test01_solved.cpp -o val_test01_solved
	$(CCP) $(CFLAGS) val_test02_solved.cpp -o val_test02_solved
	$(CCP) $(CFLAGS) -fopenmp MMult1.cpp -o MMult1
	$(CC) $(CFLAGS) -fopenmp omp_solved2.c -o omp_solved2
	$(CC) $(CFLAGS) -fopenmp omp_solved3.c -o omp_solved3

EXES = val_test01_solved val_test02_solved MMult1 \
		omp_solved2 omp_solved3

clean:
	rm -rf $(EXES)
