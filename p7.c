Execute the following program and check the properties of your GPGPU

#include <stdio.h> #include <stdlib.h> int main()
{
int deviceCount; cudaGetDeviceCount(&deviceCount); if (deviceCount == 0)
{
printf("There is no device supporting CUDA\n");
}
int dev;
for (dev = 0; dev < deviceCount; ++dev)
{
cudaGetDeviceProperties(&deviceProp, dev); if (dev == 0)
{
if (deviceProp.major < 1)
{ 
printf("There is no device supporting CUDA.\n");
}
else if (deviceCount == 1)
{
printf("There is 1 device supporting CUDA\n");
}
else
{
printf("There are %d devices supporting CUDA\n",
deviceCount);
}
}
printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name); printf(" Major revision number:	%d\n",
deviceProp.major);
printf(" Minor revision number:	%d\n", deviceProp.minor);
printf(" Total amount of global memory:	%d bytes\n", deviceProp.totalGlobalMem);
printf(" Total amount of constant memory:	%d bytes\n", deviceProp.totalConstMem);
printf(" Total amount of shared memory per block:	%d bytes\n", deviceProp.sharedMemPerBlock);
printf(" Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
printf(" Warp size:	%d\n", deviceProp.warpSize); printf(" Multiprocessor count:
%d\n",deviceProp.multiProcessorCount );
 
printf(" Maximum number of threads per block:	%d\n", deviceProp.maxThreadsPerBlock);
printf(" Maximum sizes of each dimension of a block:	%d x %d x
%d\n", deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
printf(" Maximum sizes of each dimension of a grid:	%d x %d x
%d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
printf(" Maximum memory pitch:	%d bytes\n", deviceProp.memPitch);
printf(" Texture alignment:	%d bytes\n", deviceProp.textureAlignment);
printf(" Clock rate:	%d kilohertz\n", deviceProp.clockRate);
}
}

------------------------------- end ---------------------------------

Write a program to where each thread prints its thread ID along with hello world. Launch the kernel with one block and multiple threads.

#include <stdio.h>
global	void helloWorld() { int threadId = threadIdx.x;
printf("Hello World from Thread %d!\n", threadId);
}
int main() {
int numThreads = 16;
printf("Launching kernel with 1 block and %d threads...\n\n", numThreads);
helloWorld<<<1, numThreads>>>();
// Wait for GPU to finish
printf("\nKernel execution complete.\n");
return 0;
}
 
----------------------------------- end ------------------------------

Write a program to where each thread prints its thread ID along with hello world. Launch the kernel with multiple blocks and multiple threads.

#include <stdio.h>
	global	void helloWorld() { int blockId = blockIdx.x;
int threadId = threadIdx.x;
// Global thread ID across all blocks
printf("Hello World from Block %d, Thread %d (Global Thread ID: %d)!\n", blockId, threadId, globalId);
}
int main() {
int numBlocks = 4;
int numThreads = 8; // threads per block
printf("Launching kernel with %d blocks and %d threads per block (%d total threads)...\n\n",
numBlocks, numThreads, numBlocks * numThreads);
helloWorld<<<numBlocks, numThreads>>>();
// Wait for GPU to finish cudaDeviceSynchronize();
printf("\nKernel execution complete.\n");
return 0;
}

---------------------- end --------------------------------


Write a program to where each thread prints its thread ID along with hello world. Launch the kernel with 2D blocks and 2D threads.

#include <stdio.h>
	global	void helloWorld() {
// 2D Block ID
int blockX = blockIdx.x; int blockY = blockIdx.y;
// 2D Thread ID within the block int threadX = threadIdx.x;
int threadY = threadIdx.y;
// Global 2D thread ID
int globalX = blockX * blockDim.x + threadX; int globalY = blockY * blockDim.y + threadY;
// Flattened global thread ID (useful for array indexing) int globalCols = gridDim.x * blockDim.x;
int globalId	= globalY * globalCols + globalX;
printf("Hello World from Block(%d,%d) Thread(%d,%d) -> Global(%d,%d) [GlobalID: %d]\n",
blockX, blockY, threadX, threadY, globalX, globalY, globalId); 
}
int main() {
// 2D grid: 2 blocks in X, 2 blocks in Y => 2x2 = 4 blocks total dim3 numBlocks(2, 2);

// 2D block: 4 threads in X, 4 threads in Y => 4x4 = 16 threads per block dim3 numThreads(4, 4);

int totalThreads = numBlocks.x * numBlocks.y * numThreads.x * numThreads.y;
printf("Grid : %d x %d blocks\n",	numBlocks.x, numBlocks.y); printf("Block : %d x %d threads\n", numThreads.x, numThreads.y); printf("Total : %d threads\n\n",	totalThreads);
helloWorld<<<numBlocks, numThreads>>>();
// Wait for GPU to finish cudaDeviceSynchronize();
printf("\nKernel execution complete.\n");
return 0;
}

-------------------------------------- end -------------------------------

