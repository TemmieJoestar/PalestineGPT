# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lm

# Source files
MATRIX_SRC = matrix.c
MATRIX_HDR = matrix.h

GRADIENT_SRC = gradient_descent.c
GRADIENT_HDR = gradient_descent.h

# Grouping sources
ALL_SOURCES = $(MATRIX_SRC) $(GRADIENT_SRC)
ALL_HEADERS = $(MATRIX_HDR) $(GRADIENT_HDR)

# Targets
all: main

# Standard main target rule
main: main.c $(ALL_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) main.c $(ALL_SOURCES) -o main $(LDFLAGS)

# Custom FILE.c target rule
%.run: %.c $(ALL_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(ALL_SOURCES) -o $* $(LDFLAGS) 
	./$*

# Compile and run tests
test: comprehensivetesting.c $(ALL_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) comprehensivetesting.c $(ALL_SOURCES) -o test $(LDFLAGS)
	./test

# Dynamic Memcheck Rule
%.memcheck: %.c $(ALL_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(ALL_SOURCES) -o $*.mem $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$*.mem

memcheck: comprehensivetesting.c $(ALL_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) comprehensivetesting.c $(ALL_SOURCES) -o test $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./test

help:
	@echo "Available targets:"
	@printf "  %-20s - %s\n" "make" "Build main program"
	@printf "  %-20s - %s\n" "make FILE.run" "Compile and run 'FILE.c'"
	@printf "  %-20s - %s\n" "make test" "Compile and run the 'comprehensivetesting.c' suite"
	@printf "  %-20s - %s\n" "make memcheck" "Run valgrind on 'comprehensivetesting.c'"
	@printf "  %-20s - %s\n" "make FILE.memcheck" "Run valgrind on 'FILE.c'"
	@printf "  %-20s - %s\n" "make clean" "Remove executables files"

# Clean rule
clean:
	@rm -f main comprehensivetesting gradient_descent test *.o *.mem
	@echo "Cleaned up executables and object files."

.PHONY: all test help clean