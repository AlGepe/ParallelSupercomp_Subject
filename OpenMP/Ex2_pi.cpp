#include <omp.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>

#define STEPS 40960
#define THREADS 16 //you can also use the OMP_NUM_THREADS environmental variable

double power(double x, long n) {
	if (n == 0) {
		return 1;
	}

	return x * power(x, n - 1);
}


double powerParallelReduction(double x, long n) {
	double cumulative = 1;
#pragma omp parallel for reduction(*:cumulative)
	for (int i = 1; i <= n; i++)
	{
		cumulative *= x;
	}

	return cumulative; //cumulative;
}

double powerParallelCritical(double x, long n) {
	double cumulative = 1;
#pragma omp parallel for
	for (int i = 1; i <= n; i++)
	{
#pragma omp atomic update
		cumulative *= x;
	}

	return cumulative; //cumulative;
}

double calcPi(long n) {
	if (n < 0) {
		return 0;
	}

	return 1.0 / power(16, n)
		* (4.0 / (8 * n + 1.0)
				- 2.0 / (8 * n + 4.0)
				- 1.0 / (8 * n + 5.0)
				- 1.0/(8 * n + 6.0))
		+ calcPi(n - 1);
}

double calcPiParallelReduction(long n) {
	double pi = 0;
	double firstRatio = 0,
				 secondRatio = 0;
#pragma omp parallel for reduction(+:pi) 
	for(int i = 0; i <= n; i++)
	{
		firstRatio = (1. / powerParallelReduction(16, i));	
		secondRatio = (4./(8 * i + 1.)) - (2. / (8. * i + 4.)) - (1. / (8. * i + 5.)) - (1. / (8. * i + 6.));
		pi += firstRatio * secondRatio;
	}	
	return pi;

}

double calcPiParallelCritical(long n) {
	double pi = 0,
				 theRest = 0;
	int i = 0;
  #pragma omp parallel for private(i)
	for(i = n; i >= 0; i--)
	{
		theRest = (1. / powerParallelCritical(16., i)) 
			* (4. / (8 * i + 1.) 
				- (2. / (8 * i + 4.)) 
				- (1. / (8 * i + 5.)) 
				- (1. / (8 * i + 6.)));
    #pragma omp atomic update
		pi += theRest
	}	
	return pi;
}


int main(int argc, char *argv[]) {
	// Modifications here for file output so that I can make proper graphs
	// Calculate for different step number
	double sequential[3],
				 reduction[3],
				 critical[3],
				 stepVector[3];
	double t0 = 0,
				 t1 = 0;
	long steps = 0;
	for(int i = 2; i <= 5; i++)
	{
		steps = pow(10, i);
		t0 = omp_get_wtime();
		calcPi(steps);
		t1 = omp_get_wtime();
		sequential[i-2] = t1-t0;
		t0 = omp_get_wtime();
		calcPiParallelReduction(steps);
		t1 = omp_get_wtime();
		reduction[i-2] = t1-t0;
		t0 = omp_get_wtime();
		calcPiParallelCritical(steps);
		t1 = omp_get_wtime();
		critical[i-2] = t1-t0;
		stepVector[i-2] = steps; 
		std::cout << "Finished step " << i-2 << std::endl;
	}
	std::string filename("Data_steps.csv");
	std::ofstream outFile(filename);
	outFile << "STEPS , SEQUENTIAL , REDUCTION , CRITICAL" << std::endl;
	for(int i = 0; i <= 3; i++)
	{
		outFile << stepVector[i] << " , " << sequential[i] << " , " << reduction[i] << " , " << critical[i] << std::endl;
	}




	 


}
