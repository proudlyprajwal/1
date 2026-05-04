Print ‘Hello, World’ in Sequential and Parallel in OpenMP


#include <stdio.h>
#include <omp.h>
int main() {
int n_threads;
printf("Enter the number of threads: "); scanf("%d", &n_threads);
omp_set_num_threads(n_threads);
printf("\n--- Sequential Execution ---\n"); for(int i = 0; i < n_threads; i++) {
printf("Hello World (Sequential) - iteration %d\n", i);
}
printf("\n--- Parallel Execution (Basic) ---\n"); #pragma omp parallel
{
printf("Hello World from thread %d\n", omp_get_thread_num());
}
printf("\n--- Parallel Execution (with omp for) ---\n"); #pragma omp parallel
{
#pragma omp for
for(int i = 0; i < n_threads; i++) {
printf("Hello World (omp for) - iteration %d from thread %d\n", i, omp_get_thread_num());
}
}
return 0;
}


------------------- end -----------------------

1) Implement following Programs using OpenMP with C:


1.	Vector Scalar Addition
Code:
#include <stdio.h> 
#include <stdlib.h> #include <omp.h>
#define SIZE 10 int main() {
int vector[SIZE];
int scalar = 5; int i;
for (i = 0; i < SIZE; i++) { vector[i] = i * 10;
}
printf("Original Vector: ");
for (i = 0; i < SIZE; i++) printf("%d ", vector[i]); printf("\nScalar to add: %d\n", scalar);
#pragma omp parallel for shared(vector, scalar) private(i) for (i = 0; i < SIZE; i++) {
vector[i] = vector[i] + scalar;
}
printf("\nResulting Vector: "); for (i = 0; i < SIZE; i++) {
printf("%d ", vector[i]);
}
printf("\n");
return 0;
}

2) calculate pi
  
#include <stdio.h> #include <omp.h>
static long num_steps = 100000000; double step;
int main() { int i;
double x, pi, sum = 0.0; double start_time, end_time;
step = 1.0 / (double)num_steps;
start_time = omp_get_wtime();
#pragma omp parallel for private(x) reduction(+:sum) for (i = 0; i < num_steps; i++) {
x = (i + 0.5) * step;
sum = sum + 4.0 / (1.0 + x * x);
}
pi = step * sum;
end_time = omp_get_wtime();
printf("Calculated value of Pi: %.15f\n", pi);
printf("Time taken: %f seconds\n", end_time - start_time);
return 0;
}


