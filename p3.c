Fibonacci Computation:


CODE:
#include <stdio.h>
#include <omp.h>
long fib(int n) { long i, j;
if (n < 2) return n;
#pragma omp task shared(i) firstprivate(n) i = fib(n - 1);
#pragma omp task shared(j) firstprivate(n)
j = fib(n - 2);
#pragma omp taskwait return i + j;
}
int main() { int n = 30; long result;
double start, end;
start = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
{
result = fib(n);
}
}
end = omp_get_wtime();
printf("Fibonacci(%d) = %ld\n", n, result); printf("Time taken: %f seconds\n", end - start);
return 0;
}

------------------------------- end -------------------------------------

Producer Consumer Problem


CODE:
#include <stdio.h> 
#include <stdlib.h>
#include <omp.h>
#define BUFFER_SIZE 5
#define TOTAL_ITEMS 10
 
int buffer[BUFFER_SIZE]; int count = 0;
int in = 0; int out = 0;
void producer() {
for (int i = 0; i < TOTAL_ITEMS; i++) { int produced = 0;
while (!produced) { #pragma omp critical
{
if (count < BUFFER_SIZE) { buffer[in] = i + 100;
printf("Producer: Inserted %d at %d\n", buffer[in], in); in = (in + 1) % BUFFER_SIZE;
count++; produced = 1;
}
}
if (!produced) { omp_set_num_threads(omp_get_num_threads()); }
}
}
}
void consumer() {
for (int i = 0; i < TOTAL_ITEMS; i++) { int consumed = 0;
while (!consumed) { #pragma omp critical
{
if (count > 0) {
int val = buffer[out];
printf("Consumer: Removed %d from %d\n", val, out); out = (out + 1) % BUFFER_SIZE;
count--; consumed = 1;
}
}
if (!consumed) { omp_set_num_threads(omp_get_num_threads()); }
}
}
}
int main() {
#pragma omp parallel sections
{
#pragma omp section
{
producer();
}
#pragma omp section
{
consumer();
}
}
return 0;
}
 
--------------------------------- end ----------------------------
 
