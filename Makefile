CC=g++
PROJECT = test_slic
DEPS = slic.h
OBJ = slic.o test_slic.o
CFLAGS = -Wall -pedantic `pkg-config --cflags opencv` -I.
LDFLAGS = `pkg-config --libs opencv`

all: $(PROJECT)

%.o: %.cpp $(DEPS)

test_slic: $(OBJ)
	g++ $(CFLAGS) $(LDFLAGS) -o test_slic $(OBJ)