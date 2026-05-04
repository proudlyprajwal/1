Implementation of Matrix-scalar Multiplication

#include <stdio.h> #include <stdlib.h> #include <omp.h> #include <time.h>

#define ROWS 4
#define COLS 4

void printMatrix(int matrix[ROWS][COLS], const char *label) { printf("\n%s:\n", label);
printf("┌");
for (int j = 0; j < COLS; j++) printf("────────"); printf("┐\n");

for (int i = 0; i < ROWS; i++) {
printf("│");
for (int j = 0; j < COLS; j++) printf(" %4d ", matrix[i][j]);
printf("│\n");
}

printf("└");
for (int j = 0; j < COLS; j++) printf("────────"); printf("┘\n");
}

void matrixScalarSeq(int A[ROWS][COLS], int scalar, int result[ROWS][COLS]) { for (int i = 0; i < ROWS; i++)
for (int j = 0; j < COLS; j++) result[i][j] = A[i][j] * scalar;
}

void matrixScalarParallel(int A[ROWS][COLS], int scalar, int result[ROWS][COLS]) { int i, j;
#pragma omp parallel for collapse(2) shared(A, result, scalar) private(i, j) for (i = 0; i < ROWS; i++)
for (j = 0; j < COLS; j++) result[i][j] = A[i][j] * scalar;
}

int main() {
int A[ROWS][COLS] = {
{ 1, 2, 3, 4},
{ 5, 6, 7, 8},
{ 9, 10, 11, 12},
{13, 14, 15, 16}
};

int scalar = 3;

int resultSeq[ROWS][COLS]; int resultPar[ROWS][COLS];


printf("╔══════════════════════════════════════════════╗\n"
);
printf("║ Matrix-Scalar Multiplication using OpenMP ║\n");

printf("╚══════════════════════════════════════════════╝\n"
);
printMatrix(A, "Input Matrix A"); printf("\n Scalar value : %d\n", scalar);

double t1 = omp_get_wtime(); matrixScalarSeq(A, scalar, resultSeq); double t2 = omp_get_wtime();
printMatrix(resultSeq, "Result (Sequential)"); printf(" Sequential time : %.6f seconds\n", t2 - t1);


int max_threads; #pragma omp parallel
{
#pragma omp single
max_threads = omp_get_num_threads();
}
printf("\n OpenMP threads available : %d\n", max_threads);

double t3 = omp_get_wtime(); matrixScalarParallel(A, scalar, resultPar); double t4 = omp_get_wtime();
printMatrix(resultPar, "Result (Parallel with OpenMP)"); printf(" Parallel time : %.6f seconds\n", t4 - t3);

int match = 1;
for (int i = 0; i < ROWS && match; i++) for (int j = 0; j < COLS && match; j++)
if (resultSeq[i][j] != resultPar[i][j]) match = 0;

printf("\n Verification : %s\n\n",
match ? "✔ Sequential == Parallel (CORRECT)" :
"✘ Results DIFFER (check your environment)");

printf("─────────────────────────────────────────────────\n");
printf("OpenMP Directives Used\n"); printf("─────────────────────────────────────────────────\n");
printf(" #pragma omp parallel for collapse(2)\n");
printf("	parallel for → splits loop across threads\n"); 
printf("	collapse(2) → merges 2 nested loops into one\n");
printf("		parallel iteration space\n");
printf("	shared(...) → matrix arrays shared by all threads\n"); 
printf("	private(i,j) → each thread has its own i,j copy\n");

printf("─────────────────────────────────────────────────\n\n");

return 0;
}

---------------------------- end ------------------------------

Implementation of Prefix sum.

#include <stdio.h> #include <stdlib.h> #include <string.h> #include <omp.h>

#define N 16
#define MAX_THREADS

void printArray(int *arr, int size, const char *label) { 
printf("\n %-28s: [", label);
for (int i = 0; i < size; i++)
printf("%4d%s", arr[i], i < size - 1 ? "," : ""); printf(" ]\n");
}

void prefixSumSequential(int *input, int *output, int size) { output[0] = input[0];
for (int i = 1; i < size; i++)
output[i] = output[i - 1] + input[i];
}

void prefixSumParallel(int *input, int *output, int size) { int nthreads;
int chunk_sums[MAX_THREADS]; memcpy(output, input, size * sizeof(int));
#pragma omp parallel num_threads(MAX_THREADS) shared(output, chunk_sums)
{
int tid	= omp_get_thread_num(); int nthds	= omp_get_num_threads();

/* store thread count once (inside parallel region) */ #pragma omp single
nthreads = nthds;

int chunk		= size / nthds; int start	= tid * chunk;
int end	= (tid == nthds - 1) ? size : start + chunk;


for (int i = start + 1; i < end; i++) output[i] += output[i - 1];
chunk_sums[tid] = output[end - 1];

/* wait until every thread has finished its local scan */ #pragma omp barrier


#pragma omp single
{
 
for (int t = 1; t < nthreads; t++)
chunk_sums[t] += chunk_sums[t - 1];
}
if (tid > 0) {
for (int i = start; i < end; i++) output[i] += chunk_sums[tid - 1];
}
} /* end parallel */
}
void prefixSumReduction(int *input, int *output, int size) {
/* Step 1: parallel copy */
#pragma omp parallel for shared(input, output) for (int i = 0; i < size; i++)
output[i] = input[i];

/* Step 2: up-sweep (reduce) — serial for clarity */ for (int stride = 1; stride < size; stride *= 2) {
#pragma omp parallel for shared(output, stride) for (int i = stride; i < size; i += stride * 2)
output[i + stride - 1] += output[i - 1];
/* NOTE: this naive parallelisation has write conflicts for arbitrary strides — it is included as a demonstration of
the reduction idiom only. Use the two-phase version above for production correctness. */
}
}

int main() {
int input[N] = {1, 2, 3, 4, 5, 6, 7, 8,
1, 2, 3, 4, 5, 6, 7, 8};
int seqOut[N], parOut[N];


printf("╔═════════════════════════════════════════════════
═╗\n");
printf("║	Prefix Sum (Scan) using OpenMP	║\n");

printf("╚═════════════════════════════════════════════════
═╝\n");

printArray(input, N, "Input array");

 
double t1 = omp_get_wtime();
prefixSumSequential(input, seqOut, N); double t2 = omp_get_wtime();
printArray(seqOut, N, "Sequential prefix sum"); printf(" Sequential time : %.6f sec\n", t2 - t1);


double t3 = omp_get_wtime(); prefixSumParallel(input, parOut, N); double t4 = omp_get_wtime();
printArray(parOut, N, "Parallel prefix sum"); printf(" Parallel time	: %.6f sec\n", t4 - t3);

int ok = 1;
for (int i = 0; i < N && ok; i++)
if (seqOut[i] != parOut[i]) ok = 0; printf("\n Verification	: %s\n",
ok ? "✔ Sequential == Parallel (CORRECT)"
: "✘ Mismatch detected");

printf("\n────────────────────────────────────────────────────
─\n");
printf(" Two-phase parallel algorithm trace (4 threads)\n");

printf("─────────────────────────────────────────────────────\ n");
printf(" Chunk size = %d / 4 = %d elements per thread\n\n", N, N / 4);

const char *phase1_heads[] = { "Thread 0 [0..3]", "Thread 1 [4..7]",
"Thread 2 [8..11]", "Thread 3 [12..15]"
};
int chunk = N / 4;
for (int t = 0; t < 4; t++) { int start = t * chunk; int local[4];
local[0] = input[start];
for (int i = 1; i < chunk; i++)
local[i] = local[i-1] + input[start + i];
printf(" Phase 1 %-18s → [%3d,%3d,%3d,%3d] chunk_sum=%d\n", phase1_heads[t],
local[0], local[1], local[2], local[3],
local[chunk - 1]);
}

int cs[4];
cs[0] = input[0]; for (int i=1;i<chunk;i++) cs[0]+=input[i]; cs[1] = input[4]; for (int i=1;i<chunk;i++) cs[1]+=input[4+i]; cs[2] = input[8]; for (int i=1;i<chunk;i++) cs[2]+=input[8+i]; cs[3] = input[12];for (int i=1;i<chunk;i++) cs[3]+=input[12+i]; printf("\n chunk_sums before prefix: [%d, %d, %d, %d]\n",
cs[0], cs[1], cs[2], cs[3]);
cs[1]+=cs[0]; cs[2]+=cs[1]; cs[3]+=cs[2];
printf(" chunk_sums after prefix: [%d, %d, %d, %d]\n", cs[0], cs[1], cs[2], cs[3]);
printf("\n Phase 2 Each thread adds preceding chunk total to its slice\n"); printf("	Thread 0 adds 0	| Thread 1 adds %-3d\n", cs[0]);
printf("	Thread 2 adds %-3d | Thread 3 adds %-3d\n", cs[1], cs[2]);

/* ── OpenMP directives used ── */

printf("\n────────────────────────────────────────────────────
─\n");
printf(" OpenMP Directives Used\n");

printf("─────────────────────────────────────────────────────\ n");
printf(" #pragma omp parallel num_threads(T)\n");
printf("	→ creates T threads; each gets its own tid via
omp_get_thread_num()\n\n"); printf(" #pragma omp single\n");
printf("	→ only ONE thread executes the block (prefix of chunk sums)\n"); printf("		implicit barrier at end so others wait\n\n");
printf(" #pragma omp barrier\n");
printf("	→ all threads wait here until every local scan is done\n"); printf("		before Phase 2 can begin\n");

printf("─────────────────────────────────────────────────────\ n\n");

return 0;
}

------------------------------ end ----------------------------------
Implementation of Matrix-Vector Multiplication.

#include <stdio.h> #include <stdlib.h> #include <string.h> #include <math.h> #include <omp.h>

#define ROWS 4
#define COLS 4

void printMatrix(int A[ROWS][COLS], const char *label) { printf("\n %s:\n", label);
printf(" ┌");
for (int j = 0; j < COLS; j++) printf("────────"); printf("┐\n");
for (int i = 0; i < ROWS; i++) {
printf(" │");
for (int j = 0; j < COLS; j++) printf(" %4d ", A[i][j]);
printf("│\n");
}
printf(" └");
for (int j = 0; j < COLS; j++) printf("────────"); printf("┘\n");
}

void printVector(int *v, int len, const char *label) { printf("\n %-28s: [", label);
for (int i = 0; i < len; i++)
printf("%4d%s", v[i], i < len - 1 ? "," : ""); printf(" ]\n");
}

void matVecSeq(int A[ROWS][COLS], int *x, int *result) { for (int i = 0; i < ROWS; i++) {
int sum = 0;
for (int j = 0; j < COLS; j++) sum += A[i][j] * x[j];
result[i] = sum;
}
}

void matVecParallel(int A[ROWS][COLS], int *x, int *result) { int i, j, sum;

#pragma omp parallel for \ shared(A, x, result) \ private(i, j, sum) \ schedule(static)
for (i = 0; i < ROWS; i++) { sum = 0;
for (j = 0; j < COLS; j++) sum += A[i][j] * x[j];
result[i] = sum;
}
}

void matVecReduction(int A[ROWS][COLS], int *x, int *result) { for (int i = 0; i < ROWS; i++) {
int sum = 0;

#pragma omp parallel for reduction(+:sum) for (int j = 0; j < COLS; j++)
sum += A[i][j] * x[j];

result[i] = sum;
}
}

void traceComputation(int A[ROWS][COLS], int *x, int *result) { printf("\n Row-by-row dot-product trace:\n");
printf("
─────────────────────────────────────────────────\n"); for (int i = 0; i < ROWS; i++) {
printf(" result[%d] = ", i); int sum = 0;
for (int j = 0; j < COLS; j++) { printf("%d×%d", A[i][j], x[j]); sum += A[i][j] * x[j];
if (j < COLS - 1) printf(" + ");
}
printf(" = %d\n", sum);
}
printf("
─────────────────────────────────────────────────\n");
}

/* ── Main ── */
int main() {
int A[ROWS][COLS] = {
{1, 2, 3, 4},
{5, 6, 7, 8},
{9, 10, 11, 12},
{13, 14, 15, 16}
};

int x[COLS]    = {1, 2, 3, 4};
int resultSeq[ROWS] = {0}; int resultPar[ROWS] = {0}; int resultRed[ROWS] = {0};


printf("╔═════════════════════════════════════════════════
═══╗\n");
printf("║	Matrix-Vector Multiplication using OpenMP	║\n");

printf("╚═════════════════════════════════════════════════
═══╝\n");

printMatrix(A, "Matrix A (4×4)"); printVector(x, COLS, "Vector x");

double t1 = omp_get_wtime(); matVecSeq(A, x, resultSeq); double t2 = omp_get_wtime();
printVector(resultSeq, ROWS, "Result (Sequential)"); printf(" Sequential time	: %.6f sec\n", t2 - t1);

int nthreads; #pragma omp parallel
{ #pragma omp single
nthreads = omp_get_num_threads(); }
printf("\n OpenMP threads available : %d\n", nthreads);
double t3 = omp_get_wtime(); matVecParallel(A, x, resultPar); double t4 = omp_get_wtime();
printVector(resultPar, ROWS, "Result (Parallel — rows)"); printf(" Parallel time	: %.6f sec\n", t4 - t3);

/* ── Parallel (reduction on inner loop) ── */ double t5 = omp_get_wtime(); matVecReduction(A, x, resultRed);
double t6 = omp_get_wtime();
printVector(resultRed, ROWS, "Result (Parallel — reduction)"); printf(" Reduction time	: %.6f sec\n", t6 - t5);

int ok1 = 1, ok2 = 1;
for (int i = 0; i < ROWS; i++) {
if (resultSeq[i] != resultPar[i]) ok1 = 0; if (resultSeq[i] != resultRed[i]) ok2 = 0;
}
printf("\n Seq == Parallel (rows)	: %s\n", ok1 ? "✔ CORRECT" : "✘ MISMATCH");
printf(" Seq == Parallel (reduc.) : %s\n", ok2 ? "✔ CORRECT" : "✘ MISMATCH");

traceComputation(A, x, resultSeq);


printf("\n Thread → Row assignment (schedule static, %d threads):\n", nthreads); printf("
─────────────────────────────────────────────────\n"); int chunk = ROWS / nthreads;
for (int t = 0; t < nthreads; t++) { int start = t * chunk;
int end = (t == nthreads - 1) ? ROWS - 1 : start + chunk - 1;
printf(" Thread %d → row(s) %d", t, start);
if (end != start) printf("..%d", end); printf(" result[%d]", start);
if (end != start) printf("..result[%d]", end); printf("\n");
}
printf("
─────────────────────────────────────────────────\n");
printf("\n OpenMP Directives Used\n"); printf("
─────────────────────────────────────────────────\n"); printf(" #pragma omp parallel for shared(A,x,result) private(i,j,sum)
schedule(static)\n");
printf("	parallel for → outer row loop split across threads\n"); printf("	shared	→ A and x are read-only for all threads\n"); printf("	private(j,sum) → each thread has its own column index &
accumulator\n");
printf("	schedule(static) → rows divided into equal contiguous blocks\n\n"); printf(" #pragma omp parallel for reduction(+:sum)\n");
printf(" reduction(+:sum) → inner column loop parallelised;\n"); printf("	private partial sums auto-combined with +\n"); printf("
─────────────────────────────────────────────────\n\n");

return 0;
}

---------------------------------- end -------------------------------------
