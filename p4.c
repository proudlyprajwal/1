Write an OpenMP program such that, it should print the name of your family members

#include <stdio.h> 
#include <omp.h>
int main()
{
char *family[] = {"Father", "Mother", "Brother", "Sister"}; int n = 4;
double start, end;
start = omp_get_wtime();
#pragma omp parallel num_threads(n)
{
int id = omp_get_thread_num();
printf("Family Member: %s executed by Thread ID: %d\n", family[id], id);
}
end = omp_get_wtime();
printf("\nStart Time : %f\n", start); printf("End Time : %f\n", end); printf("Execution Time : %f\n", end-start);
return 0;
} 

---------------------- end -----------------------------
Write an OpenMP program such that, it should print the sum of square of the thread id’s

#include <stdio.h> 
#include <omp.h>
int main()
{
int sum = 0;
double start, end;
start = omp_get_wtime();
#pragma omp parallel
{
int id = omp_get_thread_num(); int square = id * id;
printf("Thread ID: %d, Square: %d\n", id, square);
#pragma omp atomic sum += square;
}
end = omp_get_wtime(); 
printf("\nSum of squares of thread IDs = %d\n", sum);
printf("\nStart Time : %f\n", start); printf("End Time : %f\n", end); printf("Execution Time : %f\n", end-start);
return 0;
} 

------------------------------------ end ------------------------------------

Write an OpenMP program that calculates the partial sum of the first 20 natural numbers using parallelism

#include <stdio.h>
#include <omp.h>
int main()
{
int i;
int partial_sum = 0; int total_sum = 0;
double start, end;
start = omp_get_wtime();
#pragma omp parallel for private(partial_sum) lastprivate(total_sum) for(i = 1; i <= 20; i++)
{
partial_sum += i; total_sum = partial_sum;
printf("Thread %d adding %d, Partial Sum = %d\n", omp_get_thread_num(), i, partial_sum);
}
end = omp_get_wtime();
printf("\nFinal Total Sum = %d\n", total_sum);
printf("\nStart Time : %f\n", start); printf("End Time : %f\n", end); printf("Execution Time : %f\n", end-start);
return 0;
}
 
-------------------------------------- end -------------------------------

implement parallelization using both static and dynamic scheduling, and compare the execution time of each approach.

#include <stdio.h> 
#include <omp.h>

#define MAX 300
int main()
{
int A[MAX][MAX], B[MAX][MAX], C[MAX][MAX];
int i,j,k,n;
double start,end;
printf("Enter matrix size: "); scanf("%d",&n);
/* Initialize matrices */ for(i=0;i<n;i++)
for(j=0;j<n;j++)
{
A[i][j] = 1;
B[i][j] = 1;
}
/* -------- SERIAL MATRIX MULTIPLICATION	*/
start = omp_get_wtime();
for(i=0;i<n;i++)
{
 for(j=0;j<n;j++)
{
C[i][j] = 0;
for(k=0;k<n;k++)
{
C[i][j] += A[i][k] * B[k][j];
}
}
}
end = omp_get_wtime();
printf("\n===== SERIAL EXECUTION =====\n");
printf("Start Time : %f\n",start); printf("End Time : %f\n",end);
printf("Execution Time : %f seconds\n",end-start);
/* -------- PARALLEL STATIC SCHEDULING	*/
start = omp_get_wtime();
#pragma omp parallel for private(j,k) schedule(static) for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{
C[i][j] = 0;
 for(k=0;k<n;k++)
{
C[i][j] += A[i][k] * B[k][j];
}
}
}
end = omp_get_wtime();
printf("\n===== PARALLEL STATIC SCHEDULING =====\n");
printf("Start Time : %f\n",start); printf("End Time : %f\n",end);
printf("Execution Time : %f seconds\n",end-start);
/* -------- PARALLEL DYNAMIC SCHEDULING	*/
start = omp_get_wtime();
#pragma omp parallel for private(j,k) schedule(dynamic) for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{
C[i][j] = 0;
for(k=0;k<n;k++)
{
C[i][j] += A[i][k] * B[k][j];
}
}
}
end = omp_get_wtime();
printf("\n===== PARALLEL DYNAMIC SCHEDULING =====\n");
printf("Start Time : %f\n",start); printf("End Time : %f\n",end);
printf("Execution Time : %f seconds\n",end-start);
return 0;
}
