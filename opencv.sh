g++ -o $1 $1.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv`
