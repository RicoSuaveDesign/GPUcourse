#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

using namespace std;
using namespace cv;

void threshold(int threshold, int width, int height, unsigned char* data);
void threshold(int threshold, Mat &image);

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR); // read the file

	cout << "Number of channels: " << image.channels();
	if (!image.data)
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	cvtColor(image, image, cv::COLOR_RGB2GRAY);
	unsigned char Threshold = 128;
	threshold(Threshold, image);

	namedWindow("Display window", WINDOW_NORMAL); //create window for display
	imshow("Display window", image); // show image inside it

	waitKey(0); // wait for keystroke in window
	return 0;


}

void threshold(int threshold, int width, int height, unsigned char* data)
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

void threshold(int threshold, Mat &image)
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


