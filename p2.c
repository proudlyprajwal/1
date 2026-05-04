C Program to find the minimum scalar product of two vectors (dot product)

#include <stdio.h> #include <stdlib.h> #include <omp.h>
#define N 1000000
int main() {
int *vecA = (int*)malloc(sizeof(int) * N); int *vecB = (int*)malloc(sizeof(int) * N); long long dot_product = 0;
double start_time, end_time;
for (int i = 0; i < N; i++) {
vecA[i] = 1;
vecB[i] = 2;
}
printf("Calculating Dot Product for %d elements...\n", N);
start_time = omp_get_wtime();
#pragma omp parallel for reduction(+:dot_product) for (int i = 0; i < N; i++) {
dot_product += (long long)vecA[i] * vecB[i];
}
end_time = omp_get_wtime();
printf("Result: %lld\n", dot_product);
printf("Time taken: %f seconds\n", end_time - start_time);
free(vecA); free(vecB); 
return 0;
}

----------------- end --------------------
  
Write OpenMP code for two 2D Matrix addition

#include <stdio.h> 
#include <stdlib.h>
#include <omp.h>

void matrix_addition(int size, int threads) { int i, j;
double **A = (double **)malloc(size * sizeof(double *)); double **B = (double **)malloc(size * sizeof(double *)); double **C = (double **)malloc(size * sizeof(double *)); for (i = 0; i < size; i++) {
A[i] = (double *)malloc(size * sizeof(double)); B[i] = (double *)malloc(size * sizeof(double)); C[i] = (double *)malloc(size * sizeof(double));
}
for (i = 0; i < size; i++) { for (j = 0; j < size; j++) {
A[i][j] = 1.0;
B[i][j] = 2.0;
}
}
double start_time = omp_get_wtime();
#pragma omp parallel for num_threads(threads) private(i, j) collapse(2) for (i = 0; i < size; i++) {
for (j = 0; j < size; j++) { C[i][j] = A[i][j] + B[i][j];
}
}
double end_time = omp_get_wtime();
printf("Size: %d | Threads: %d | Time: %f seconds\n", size, threads, end_time - start_time);
for (i = 0; i < size; i++) { free(A[i]); free(B[i]); free(C[i]);
}
free(A); free(B); free(C);
}
int main() {
int sizes[] = {250, 500, 750, 1000, 2000};
int thread_counts[] = {1, 2, 4, 8};
for (int s = 0; s < 5; s++) { for (int t = 0; t < 4; t++) {
matrix_addition(sizes[s], thread_counts[t]);
}
printf("
}
 \n");
 return 0;
}

--------------------------- end -----------------------------------
  
For 1D Vector (size=200) and scalar addition, write a OpenMP code with the following

#include <stdio.h> 
#include <stdlib.h> 
#include <omp.h>
#define SIZE 200
void vector_scalar_add(int chunk_size, int use_dynamic) { float vec[SIZE];
float scalar = 5.5f;
for(int i = 0; i < SIZE; i++) vec[i] = (float)i;
double start = omp_get_wtime();
#pragma omp parallel
{
if (use_dynamic) {
#pragma omp for schedule(dynamic, chunk_size) nowait for(int i = 0; i < SIZE; i++) vec[i] += scalar;
} else {
#pragma omp for schedule(static, chunk_size) nowait for(int i = 0; i < SIZE; i++) vec[i] += scalar;
}
#pragma omp master
{
}
}
double end = omp_get_wtime();
printf("%s (Chunk %d) | Time: %f s\n", use_dynamic ? "Dynamic" : "Static ", chunk_size, end - start);
}
int main() {
int chunks[] = {1, 10, 50, 100};
printf("--- Static Schedule ---\n");
for(int i = 0; i < 4; i++) vector_scalar_add(chunks[i], 0);
printf("\n--- Dynamic Schedule ---\n");
for(int i = 0; i < 4; i++) vector_scalar_add(chunks[i], 1);
return 0;
}


