
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "..\hiperformancetimer\highperformancetimer.h"
#include <string>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>
#include <iostream>
using namespace std;

bool setUpArrays(int **a, int **b, int **c, int size);
cudaError_t computeOnGPU(int* c, int* a, int* b, unsigned int size, cudaDeviceProp devProp);


__global__ void addKernel(int *c, const int *a, const int *b, int size)
{
    int i = blockDim.x * blockIdx.x +  threadIdx.x;
	if (i < size)
	{
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char* argv[])
{
	srand(time(NULL)); // seed rand num generator
	cudaDeviceProp devProp; // declare device prop variable for later retrieval of properties
	cudaError_t cudaStatus = cudaSetDevice(0); // check for device
	
	if (cudaStatus != cudaSuccess)
	{
		cout << "No CUDA-compatible GPU detected." << endl;
		exit(1);
	}
	cudaGetDeviceProperties(&devProp, 0); // get device properties for later use
	int arraySize = 1020;
	int reps = 100;
	// if cmdline args were provided, use them for array size and repetitions
	if (argc > 1)
		arraySize = stoi(argv[1]);
	if (argc > 2)
		reps = stoi(argv[2]);

	cout << "Array size: " << arraySize << endl << "Repetitions: " << reps << endl;

	int retVal = 0;

	int *a, *b, *c; // declare CPU arrays
	a = b = c = nullptr; // nullptrs so they're not misused later
	bool success = setUpArrays(&a, &b, &c, arraySize); // allocs the arrays
	for (int i = 0; i < arraySize; i++)
	{
		// move this loop into a function later
		a[i] = rand() % 69 + 1;
		b[i] = rand() % 420 + 1;
		c[i] = 0;
	}

    // Add vectors in parallel.
	try
	{
		cudaError_t cudaStatus = computeOnGPU(c, a, b, arraySize, devProp);
		if (cudaStatus != cudaSuccess) {
			throw "computeOnGPU failed";
		}

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			throw "cudaDeviceReset failed";
		}
	}
	catch (char* err_message)
	{
		cerr << err_message;
		retVal = 1;
	}

    return retVal;
}

bool setUpArrays(int **a, int **b, int **c, int size)
{
	bool retVal = true;
	*a = (int*)malloc(size * sizeof(int));
	*b = (int*)malloc(size * sizeof(int));
	*c = (int*)malloc(size * sizeof(int));

	if (*a == nullptr || *b == nullptr || *c == nullptr)
	{
		// if we managed to mess that up
		retVal = false;
		cerr << "Failed on ";
		if (*a == nullptr)
			cerr << "a, ";
		if (*b == nullptr)
			cerr << "b, ";
		if (*c == nullptr)
			cerr << "c";
		cerr << endl;
	}
	cout << "Memory successfully allocated on CPU." << endl;
	return retVal;
}
cudaError_t computeOnGPU(int* c, int* a, int* b, unsigned int size, cudaDeviceProp devProp)
{

	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;
	// allocate memory on GPU
	try
	{
		cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			throw "cudaMalloc failed on c!";
		}
		cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			throw "cudaMalloc failed on b!";
		}
		cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			throw "cudaMalloc failed on a!";
		}
	}
	catch(char* err_mess)
	{
		cerr << err_mess;
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		exit(1);

	}
	cout << "Arrays set up on GPU." << endl;
	//Copy the CPU values to the GPU
	try
	{
		cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "cudaMemcpy failed on a!";
		}
		cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "cudaMemcpy failed on b!";
		}
		cudaStatus = cudaMemcpy(dev_c, c, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "cudaMemcpy failed on c!";
		}
	}
	catch (char* error_copy)
	{
		cerr << error_copy;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_c);
	}
	cout << "Array values copied to GPU." << endl;
	
	int blocks_needed = (size + devProp.maxThreadsPerBlock - 1) / devProp.maxThreadsPerBlock;
	cout << "There will be " << blocks_needed << " blocks with " << devProp.maxThreadsPerBlock << " threads each." << endl;

	addKernel <<<blocks_needed, size >>>(dev_c, dev_a, dev_b, size);

	try
	{
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			throw "addKernel launch failed!";
			
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			throw "cudaDeviceSync Failed!";
		}
		cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw "cudaMemcpy failed!";
		}
	}
	catch (char* message)
	{
		cerr << message;
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_c);
		exit(1);
	}
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cout << "Array copied back to host." << endl;
	return cudaStatus;
}

