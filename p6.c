Implement a simple hello world program by setting number of processes equal to 4

#include <stdio.h> 
#include <mpi.h>
int main(int argc, char* argv[]) {
MPI_Init(&argc, &argv);
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
printf("Hello World from process %d of %d\n", rank, size);
MPI_Finalize();
return 0;
}

----------------- end -------------------------

Implement a program to display rank and communicator group of five processes

#include <stdio.h> #include <mpi.h>
int main(int argc, char* argv[]) { MPI_Init(&argc, &argv);
int rank, size;
MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
printf("Process Rank: %d | Communicator Group Size: %d | Communicator: MPI_COMM_WORLD\n", rank, size);
MPI_Finalize(); return 0;

-------------- end --------------------------

Implement a MPI program to give an example of Deadlock. 

#include <stdio.h> #include <mpi.h>

int main(int argc, char* argv[]) { MPI_Init(&argc, &argv);

int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

if (size < 2) {
printf("This program requires at least 2 processes.\n"); MPI_Finalize();
return 1;
}


int send_data, recv_data;
MPI_Status status;
 

if (rank == 0) { send_data = 100;
printf("Process 0: Trying to SEND %d to Process 1 (blocking)...\n", send_data);
MPI_Send(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); // Blocks until
Process 1 receives


printf("Process 0: Trying to RECEIVE from Process 1 (blocking)...\n");
MPI_Recv(&recv_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status); // Never
reached → DEADLOCK
printf("Process 0: Received %d from Process 1\n", recv_data);
}
else if (rank == 1) { send_data = 200;
printf("Process 1: Trying to SEND %d to Process 0 (blocking)...\n", send_data);
MPI_Send(&send_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); // Blocks until
Process 0 receives


printf("Process 1: Trying to RECEIVE from Process 0 (blocking)...\n");
MPI_Recv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status); // Never
reached → DEADLOCK
printf("Process 1: Received %d from Process 0\n", recv_data);
}


MPI_Finalize(); return 0;
}
 
-------------------------- end ------------------------

Implement blocking MPI send & receive to demonstrate Nearest neighbor exchange of data in a ring topology

#include <stdio.h> #include <mpi.h>

int main(int argc, char* argv[]) { MPI_Init(&argc, &argv);
int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
if (size < 2) {
printf("This program requires at least 2 processes.\n"); MPI_Finalize();
return 1;
}
int right_neighbor = (rank + 1) % size;
int left_neighbor = (rank - 1 + size) % size;
nt send_data = rank * 10; int recv_data = -1;
MPI_Status status;
printf("Process %d: Sending %d to Process %d (right) | Expecting from Process %d (left)\n",
rank, send_data, right_neighbor, left_neighbor);
if (rank % 2 == 0) {
// Even ranks: Send first, then Receive
MPI_Send(&send_data, 1, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD);
MPI_Recv(&recv_data, 1, MPI_INT, left_neighbor, 0, MPI_COMM_WORLD, &status);
} else {
MPI_Recv(&recv_data, 1, MPI_INT, left_neighbor, 0, MPI_COMM_WORLD, &status);
MPI_Send(&send_data, 1, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD);
}
printf("Process %d: Received %d from Process %d (left neighbor)\n", rank, recv_data, left_neighbor);
MPI_Finalize(); return 0;
}

----------------------------- end -------------------------------------

Write a MPI program to find the sum of all the elements of an array A of size n

#include <stdio.h> #include <mpi.h>

int main(int argc, char* argv[]) {
 
MPI_Init(&argc, &argv);


int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

if (size != 2) { if (rank == 0)
printf("This program requires exactly 2 processes.\n"); MPI_Finalize();
return 1;
}


int n = 10;
int A[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};


int half = n / 2; int local_sum = 0; int final_sum = 0;

if (rank == 0) {
// P0: sum first half [0 .. n/2-1]
printf("Process 0: Computing sum of first %d elements: ", half); for (int i = 0; i < half; i++) {
printf("%d ", A[i]); local_sum += A[i];
}
printf("\nProcess 0: Partial Sum = %d\n", local_sum);


// P0 sends second half data to P1
 
MPI_Send(&A[half], half, MPI_INT, 1, 0, MPI_COMM_WORLD);


// P0 receives partial sum from P1 int recv_sum = 0;
MPI_Recv(&recv_sum,	1,	MPI_INT,	1,	1,	MPI_COMM_WORLD, MPI_STATUS_IGNORE);


// P0 computes final sum
final_sum = local_sum + recv_sum;
printf("\nProcess 0: Received Partial Sum from Process 1 = %d\n", recv_sum); printf("─────────────────────────────────────────\n"); printf("Final Sum = %d + %d = %d\n", local_sum, recv_sum, final_sum);
}
else if (rank == 1) {
// P1: receive second half data from P0 int recv_data[half];
MPI_Recv(recv_data,	half,	MPI_INT,	0,	0,	MPI_COMM_WORLD, MPI_STATUS_IGNORE);


// P1: sum second half [n/2 .. n-1]
printf("Process 1: Computing sum of last %d elements: ", half); for (int i = 0; i < half; i++) {
printf("%d ", recv_data[i]); local_sum += recv_data[i];
}
printf("\nProcess 1: Partial Sum = %d\n", local_sum);


// P1 sends its partial sum back to P0
MPI_Send(&local_sum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
}
 
MPI_Finalize(); return 0;
}

----------------------- end ----------------------

