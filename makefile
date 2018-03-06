all: Exercise1 Exercise2

Exercise1: Ex1_matMult.cpp 
	g++ -fopenmp -std=c++11 -o Exercise1 Ex1_matMult.cpp
Exercise2: Ex2_pi.cpp 
	g++ -fopenmp -std=c++11 -o Exercise2 Ex2_pi.cpp

clean: 
	rm -rf Exercise1 Exercise2
