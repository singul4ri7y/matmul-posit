main: 
	g++ -ffast-math -O2 -lm -o main -lsoftposit -L. -Wl,-rpath=. -I../include main.cpp
