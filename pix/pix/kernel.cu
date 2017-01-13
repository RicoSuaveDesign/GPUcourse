#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

using namespace std;
using namespace cv;

void threshold(unsigned char threshold, int width, int height, unsigned char* data);
void threshold(unsigned char threshold, Mat &image);
cudaError_t thresholdGPU(unsigned char threshold, Mat &image);

__global__ void threshKernel(unsigned char * image, unsigned char* moddedimage, int size, unsigned char threshold)
{
	int i = blockDim.x * gridDim.x + threadIdx.x;
	if (i < size)
	{
		if (image[i] > threshold)
		{
			moddedimage[i] = 255;
		}
		else
		{
			moddedimage[i] = 0;
		}
	}
}


int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}
	//set up cuda stuff so it only needs to happen once 
	cudaDeviceProp devProp;
	cudaSetDevice(0);
	cudaGetDeviceProperties(&devProp, 0);

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR); // read the file

	cout << "Number of channels: " << image.channels() << endl;
	if (!image.data)
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	cvtColor(image, image, cv::COLOR_RGB2GRAY);
	unsigned char Threshold = 128;
	//threshold(Threshold, image);
	cudaError_t cudaStatus;
	cudaStatus = thresholdGPU(Threshold, image);
	if (cudaStatus != cudaSuccess)
		cout << "Failed to apply threshold filter" << endl;
	//else
	//{

	//	namedWindow("Display window", WINDOW_NORMAL); //create window for display
	//	imshow("Display window", image); // show image inside it
	//}
	waitKey(0); // wait for keystroke in window
	return 0;


}

void threshold(unsigned char threshold, int width, int height, unsigned char* data)
{
	unsigned char* end_data = (data + (width * height) + width);
	for (unsigned char* p = data; p < end_data; p++)
	{
		if (*p > threshold)
		{
			*p = 255;
		}
		else
		{
			*p = 0;
		}
	}
}

void threshold(unsigned char threshold, Mat &image)
{
	unsigned char* end_data = (image.data + (image.cols * image.rows) + image.cols);
	for (unsigned char* p = image.data; p < end_data; p++)
	{
		if (*p > threshold)
		{
			*p = 255;
		}
		else
		{
			*p = 0;
		}
	}
}

cudaError_t thresholdGPU(unsigned char threshold, Mat &image)
{

	cudaError_t cudaStatus;
	int size = image.rows * image.cols *sizeof(unsigned char);

	// allocates on cpu as an unsigned char
	unsigned char* imagedata = image.data;

	// declare and then allocate GPU memory
	unsigned char* dev_image = nullptr;
	unsigned char* dev_moddedimage = nullptr;
	try
	{
		cudaStatus = cudaMalloc((void**)&dev_image, (size * sizeof(unsigned char)));
		if (cudaStatus != cudaSuccess)
		{
			throw "cudaMalloc failed on dev_image!";
		}
		cudaStatus = cudaMalloc((void**)&dev_moddedimage, (size * sizeof(unsigned char)));
		if (cudaStatus != cudaSuccess)
		{
			throw "cudaMalloc failed on dev_moddedimage!";
		}
	}
	catch (char* message)
	{
		cerr << message << endl;
		if (dev_image != nullptr)
			cudaFree(dev_image);
		if (dev_moddedimage != nullptr)
			cudaFree(dev_moddedimage);
		cerr << "Freed Cuda Memory" << endl;
	}

	//copy orig image to GPU
	cudaStatus = cudaMemcpy(dev_image, imagedata, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cerr << "Memcpy from CPU to GPU failed!" << endl;
		cudaFree(dev_image);
		cudaFree(dev_moddedimage);
	}

	int blocks_needed = size / 1024;
	cout << "There will be " << blocks_needed << " blocks with 1024 threads each." << endl;

	threshKernel << <blocks_needed, 1024 >> > (dev_image, dev_moddedimage, size, threshold);

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
		cudaStatus = cudaMemcpy((unsigned char*)image.data, dev_moddedimage, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw "cudaMemcpy failed!";
		}
	}
	catch (char* err_mess)
	{
		// just cout the error message for now cause we gon free the memory anyway
		cerr << err_mess;
	}
	cudaFree(dev_image);
	cudaFree(dev_moddedimage);

	if (cudaStatus == cudaSuccess)
	{
		namedWindow("Display window", WINDOW_NORMAL); //create window for display
		imshow("Display window", image); // show image inside it
	}

	return cudaStatus;





	
}


