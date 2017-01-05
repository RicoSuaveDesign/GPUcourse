
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <stdlib.h>
#include <ctime>
using namespace std;

void assignRandNum(int* a, int* b, int * c, int size);
bool allocMem(int** a, int** b, int** c, int size);
void freeMem(int* a, int* b, int* c);

int main(int argc, char* argv[])
{
	srand(time(NULL));

	int size = 69;
	cout << argc << endl;
	cout << argv[0] << endl;
	if (argc > 1)
	{
		size = stoi(argv[1]);
		cout << argv[1] << endl;

	}
	else
		cout << "Usage: whateveryouwant (arraySize)" << endl << "Size is now " << size << ". Donut forget the arraysize." << endl;
	// make arrays
	int *a, *b, *c;
	a = b = c = nullptr; // set to null so we can tell whether it has failed or not

	try
	{
		bool success = allocMem(&a, &b, &c, size);
		if (success)
		{
			cout << "All arrays successfully allocated on CPU." << endl;
		}
		else
		{
			throw("Arrays failed to allocate");
		}
	}
	catch (char* err_message)
	{
		cerr << "There was an error. Here is the message: " << endl;
		cerr << err_message;
	}

	// assign the variables
	assignRandNum(a, b, c, size);

	//lets see if they werk
	for (int i = 0; i < size; i++)
	{
		cout << a[i] << endl;
		cout << b[i] << endl;
		cout << c[i] << endl;
	}

	//free them
	freeMem(a, b, c);



#ifdef _WIN32 || _WIN64
	system("pause");
#endif

}

bool allocMem(int** a, int** b, int** c, int size)
{
	bool retVal = true; // assume we can do basic allocation

	// allocate memory
	*a = (int*)malloc(size * sizeof(int));
	*b = (int*)malloc(size * sizeof(int));
	*c = (int*)malloc(size * sizeof(int));

	if (*a == nullptr || *b == nullptr || *c == nullptr)
	{
		// if we managed to mess that up
		retVal = false;
		cout << "Failed on ";
		if (*a == nullptr)
			cout << "a, ";
		if (*b == nullptr)
			cout << "b, ";
		if (*c == nullptr)
			cout << "c";
		cout << endl;
	}

	return retVal;
}
void freeMem(int* a, int* b, int* c)
{
	if (a != nullptr)
	{
		free(a);
		cout << "a freed" << endl;
	}
	if (b != nullptr)
	{
		free(b);
		cout << "b freed" << endl;
	}
	if (c != nullptr)
	{
		free(c);
		cout << "c freed" << endl;
	}

}
void assignRandNum(int* a, int* b, int * c, int size)
{
	for (int i = 0; i < size; i++)
	{
		//assign rands to a and b
		a[i] = rand() % 12 + 1;
		b[i] = rand() % 10 + 1;
		// assign 0s to c
		c[i] = 0;
	}

}