/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 */
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char * argv[])
{
	MPI_Init(&argc, &argv);
	double startTime,
				 endTime;
	startTime = MPI_Wtime();
	int numProcesses;
	int64_t item = rand();
	MPI_Comm_rank(MPI_COMM_WORLD, &numProcesses);
	for(int i = 1; i < numProcesses; i++)
	{
  printf("Hello world from %d/%d (slept %u s)!\n", myRank, numProcesses, t);
	/*
		MPI_Recv(&item,
				nItems,
				MPI_LONG_LONG,
				i-1,
				MPI_TAG_ANY,
				MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);

		MPI_Send(buf,
				nItems,
				MPI_LONG_LONG,
				i,
				MPI_TAG_ANY,
				MPI_COMM_WORLD);
*/
	}
	endTime = MPI_Wtime();
	MPI_Finalize();

  printf("It took: %d s\n", endTime-startTime);
  return 0;
}
