# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lm

# Source files
MATRIX_SRC = matrix.c
MATRIX_HDR = matrix.h

# Targets
all: main

# Compile main program
main: main.c $(MATRIX_SRC) $(MATRIX_HDR)
	$(CC) $(CFLAGS) main.c $(MATRIX_SRC) -o main $(LDFLAGS)

# Compile and run tests
test: comprehensivetesting.c $(MATRIX_SRC) $(MATRIX_HDR)
	$(CC) $(CFLAGS) comprehensivetesting.c $(MATRIX_SRC) -o test $(LDFLAGS)
	./test

# Compile tests with valgrind
memcheck: comprehensivetesting.c $(MATRIX_SRC) $(MATRIX_HDR)
	$(CC) $(CFLAGS) comprehensivetesting.c $(MATRIX_SRC) -o test $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./test

# Clean compiled files
clean:
	rm -f main test *.o

# Phony targets (not actual files)
.PHONY: all test memcheck clean