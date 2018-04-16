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
  unsigned t = rand() % 2;
  int64_t i=13;
  sleep(t);
	int myRank,
			numProcesses;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  printf("Hello world from %d/%d (slept %u s)!\n", myRank, numProcesses, t);
	MPI_Finalize();
  return 0;
}
