#include <time.h>
#include <stdio.h>

#define RADIUS        3
#define NUM_ELEMENTS  1000 

static void handleError(cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))

__global__ void stencil_1d(int *in, int *out) {
	//PUT YOUR CODE HERE
	int array_length = sizeof(*in)/sizeof(in[0]);
	// Run whole array
	for(int i = 0; i < array_length; i++)
	{
		// Calculate for all neighbours and check
		for(int j = -RADIUS; j <= RADIUS; j++)
		{
			if(i + j < 0)
			{
				j += array_length;
			}
			else if (i+j > array_length)
			{
				j -= array_length;
			}
			out[i] += in[i+j];

		}

		void cpu_stencil_1d(int *in, int *out) {
			//PUT YOUR CODE HERE

		}

		int main() {
			//PUT YOUR CODE HERE - INPUT AND OUTPUT ARRAYS 
			float *input_array,
			      *output_array;

			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord( start, 0 );

			//PUT YOUR CODE HERE - DEVICE MEMORY ALLOCATION
			cudaCheck(cudaMalloc((void**)&input_array, NUM_ELEMENTS*sizeof(float)));
			cudaCheck(cudaMalloc((void**)&output_array, NUM_ELEMENTS*sizeof(float)));

			//PUT YOUR CODE HERE - KERNEL EXECUTION

			cudaCheck(cudaPeekAtLastError());

			//PUT YOUR CODE HERE - COPY RESULT FROM DEVICE TO HOST

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			float elapsedTime;
			cudaEventElapsedTime( &elapsedTime, start, stop);
			printf("Total GPU execution time:  %3.1f ms\n", elapsedTime);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			//PUT YOUR CODE HERE - FREE DEVICE MEMORY  
			cudaCheck(cudaFree(input_array));

			struct timespec cpu_start, cpu_stop;
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);

			cpu_stencil_1d(in, out);

			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
			double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
			printf( "CPU execution time:  %3.1f ms\n", result);

			return 0;
		}


