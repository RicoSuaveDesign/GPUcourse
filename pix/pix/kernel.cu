#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "highperformancetimer.h"

#include <stdio.h>

using namespace std;
using namespace cv;

typedef unsigned char ubyte;

void threshold(unsigned char threshold, int width, int height, unsigned char* data);
void threshold(unsigned char threshold, Mat &image);
cudaError_t thresholdGPU(unsigned char threshold, Mat &image);
void BoxFilter(ubyte *s, ubyte *d, int w, int h, int* k, int kw, int kh, ubyte *temp);


// shitty goddamned bad global variables
HighPrecisionTime hpt;
unsigned char* dev_image = nullptr;
unsigned char* dev_moddedimage = nullptr; 
Mat image;
int Threshold_slider = 128;
int Box_Slider = 1;
float totalTime = 0.0; // remember to reset this every time a new timer is called
int timesCalled = 0;
int ke[9] = { -1,0,1,-2,0,2,-1,0,1 };
int k2[9] = { -1, 2,-1, 0,0,0, 1,2,1 };
int boxk[25];
int kh = 5;
int kw = 5;


__global__ void threshKernel(unsigned char * image, unsigned char* moddedimage, int size, int threshold)
{
	// multiply by blockdimx because it just werks i guess 
	int i = blockIdx.x * blockDim.x + threadIdx.x;
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

void on_trackbar(int, void*)
{
	//cudaError_t cudaStatus;
	//int blocks_needed = (1023 + image.rows * image.cols) / 1024;
	//// call the kernel on the now global device variables
	//threshKernel <<<blocks_needed, 1024 >>> (dev_image, dev_moddedimage, (image.rows * image.cols), Threshold_slider);
	//cudaStatus = cudaDeviceSynchronize();
	//
	//cudaStatus = cudaMemcpy(image.data, dev_moddedimage, (image.rows * image.cols), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess)
	//{
	//	cerr << "Memcpy from GPU to CPU failed!" << endl;
	//	cudaFree(dev_image);
	//	cudaFree(dev_moddedimage);
	//}
	//cout << Threshold_slider << endl;
	//BoxFilter(s, d, image.cols, image.rows, k, 3, 3, temp);
	cout << hpt.TimeSinceLastCall();
	imshow("Display window", image);

}
void box_trackbar(int, void*);


int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}
	for (int i = 0; i < 25; i++)
	{
		boxk[i] = 1;
	}
	//set up cuda stuff so it only needs to happen once 
	cudaDeviceProp devProp;
	cudaSetDevice(0);
	cudaGetDeviceProperties(&devProp, 0);

	//Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR); // read the file

	cout << "Number of channels: " << image.channels() << endl;
	if (!image.data)
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	
	cvtColor(image, image, cv::COLOR_RGB2GRAY);
	ubyte *src = image.data;
	ubyte *dst = image.data;
	ubyte *temp = image.data;
	
	namedWindow("Display window", WINDOW_NORMAL);
	imshow("Display window", image);
	waitKey(0);
	hpt.TimeSinceLastCall();
	BoxFilter(src, dst, image.cols, image.rows,  boxk, kh, kw, temp);
	cout << "The box filter took " << hpt.TimeSinceLastCall() << " seconds." << endl;
	//threshold(Threshold, image);
	//cudaError_t cudaStatus;
	//cudaStatus = thresholdGPU(Threshold_slider, image);
	//if (cudaStatus != cudaSuccess)
	//	cout << "Failed to apply threshold filter" << endl;
	//else
	//{

	//	namedWindow("Display window", WINDOW_NORMAL); //create window for display
	//	imshow("Display window", image); // show image inside it
	//}
	createTrackbar("Threshold", "Display window", &Box_Slider, 10, box_trackbar);
	imshow("Display window", image);
	box_trackbar(Box_Slider, 0);

	waitKey(0); // wait for keystroke in window

	cout << endl << "Final average: " << totalTime / timesCalled << " seconds" << endl;
	cout << "image size: " << image.cols << " x " << image.rows << endl;
	cout << "kernel size: 3 x 3" << endl;

#ifdef _WIN32 || _WIN64
	system("pause");
#endif

	//cudaFree(dev_image); // and here we are freein the memory on gpu
	//cudaFree(dev_moddedimage);
	return 0;


}

void box_trackbar(int, void*)
{
	int *p_k = boxk;

	ubyte *s = image.data;
	Mat d;
	image.copyTo(d);
	ubyte *tempo = image.data;
	

	hpt.TimeSinceLastCall();
	BoxFilter(s, d.data, image.cols, image.rows, p_k, kh, kw, tempo);
	float currentTime = hpt.TimeSinceLastCall();
	totalTime += currentTime;
	timesCalled++;

	cout << "Time this run: " << currentTime << " seconds" << endl;
	cout << "Current average: " << totalTime / timesCalled << endl;

	imshow("Display window", image);
	
}

void BoxFilter(ubyte *s, ubyte *d, int w, int h, int *k, int kw, int kh, ubyte *temp)
{
	
	// later on we divide by the sum of all the values in the box kernel -- so calculate it now
	int kernelSum = 0;
	for (int i = 0; i < kw*kh; i++)
	{
		kernelSum += k[i];
	}


	// this makes calculating relative indices (e.g. what is one value of "up" to a 1D array?) a one time task, or at the very least a much more readable operation
	//int indices[9] = { -(w + 1),  -w, -(w - 1), -1, 0,  +1,	w - 1, w,  w + 1 };
	
	// calculates our image edges -- wedge is width edge, hedge is height edge

	int kwedge = kw / 2;
	int khedge = kh / 2;

	int indexOffset;
	for (int i = khedge; i < h - khedge; i++)
	{
		for (int j = kwedge; j < w - kwedge; j++)
		{
			// first we start with current, which starts at 0.0. Then we calculate the relative ups, downs, etc with indexoffset.
			float current = 0.0f;
			indexOffset = (i*w) + j;
			for (int ki = -khedge; ki <= khedge; ki++)
			{
				for (int kj = -kwedge; kj <= kwedge; kj++)
				{
					// relative pixel is found by multiplying the current vertical kernel pixel by image width, and then adding the current kernel horizontal index
					int relativepixel = ki * w + kj;
					// kernel pixel is current kernel height plus vertical edge, then multiplied  by kernel hiehgt, which then current kernel width is added to horizontal edge
					int kernelpix = (ki + khedge) * kw + kj + kwedge;
					// current gets the value of the current pixel and multiplies by the value in the current index of the kernel
					current += float(s[indexOffset + relativepixel]) * float(k[kernelpix]);
				}
			}
			if (kernelSum != 0)
			{
				// output image pixels all are divided by kernel sum which is 9
				d[indexOffset] = int(current / (float)kernelSum);
			}
			else
			{
				d[indexOffset] = int(current / 1.0f);
			}
		}
	}
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

	// declare and then allocate GPU memory
	//unsigned char* dev_image = nullptr;
	//unsigned char* dev_moddedimage = nullptr;
	try
	{
		cudaStatus = cudaMalloc((void**)&dev_image, (size));
		if (cudaStatus != cudaSuccess)
		{
			throw "cudaMalloc failed on dev_image!";
		}
		cudaStatus = cudaMalloc((void**)&dev_moddedimage, (size));
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
	cudaStatus = cudaMemcpy(dev_image, image.data, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cerr << "Memcpy from CPU to GPU failed!" << endl;
		cudaFree(dev_image);
		cudaFree(dev_moddedimage);
	}


	int blocks_needed = (image.rows * image.cols + 1023) / 1024;
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
		cudaStatus = cudaMemcpy((unsigned char*)image.data, dev_moddedimage, size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw "cudaMemcpy failed!";
		}
	}
	catch (char* err_mess)
	{
		// just cout the error message for now cause we gon free the memory anyway
		cerr << err_mess;
	}


	return cudaStatus;
}


