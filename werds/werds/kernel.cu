
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../timertest/hiperformancetimer/highperformancetimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

int main()
{
	//declare file stuff
	ifstream ctfile;
	int giga = 1 << 30;

	ctfile.open("C:/Users/educ/Documents/enwiki-latest-abstract.xml");
	if (!ctfile.is_open())
	{
		cout << "Failed to open file." << endl;
		return(1);
	}


	// allocate ALL the memory.... jk its only like a billion bytes
	char* letters = (char*)malloc(giga);

	//add timer
	HighPrecisionTime hpt;
	hpt.TimeSinceLastCall();

	// load a gb of the file into the array
	ctfile.read(letters, giga);
	if (ctfile.fail())
	{
		cout << "Did not read 1 GB of the wiki file" << endl;
		return(1);
	}

	// time and print time
	cout << "Loading took " << hpt.TimeSinceLastCall() << " seconds." << endl;

	byte* bitmap = (byte*)calloc(giga/8, sizeof(byte));



	//manage the memory
	if (ctfile.is_open())
		ctfile.close();
	if(letters != nullptr)
		free(letters);
	if (bitmap != nullptr)
		free(bitmap);

	system("pause");
	return 0;
}
