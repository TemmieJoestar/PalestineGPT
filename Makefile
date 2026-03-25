CYAN    = \033[36m
GREEN   = \033[32m
MAGENTA = \033[35m
YELLOW  = \033[33m
RESET   = \033[0m

CC = gcc
CFLAGS = -Wall -Wextra -g -Iinclude
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
TEST_DIR = tests

MATRIX_SRC = $(SRC_DIR)/matrix.c
MATRIX_HDR = $(INC_DIR)/matrix.h

GRADIENT_SRC = $(SRC_DIR)/gradient_descent.c
GRADIENT_HDR = $(INC_DIR)/gradient_descent.h

ALL_SOURCES = $(MATRIX_SRC) $(GRADIENT_SRC)
ALL_HEADERS = $(MATRIX_HDR) $(GRADIENT_HDR)

all: main

main: $(SRC_DIR)/main.c $(ALL_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(ALL_SOURCES) -o main $(LDFLAGS)

%.run: %.c $(ALL_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(ALL_SOURCES) -o $* $(LDFLAGS) 
	./$*

test: $(TEST_DIR)/comprehensivetesting.c $(ALL_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(ALL_SOURCES) -o test $(LDFLAGS)
	./test

%.memcheck: %.c $(ALL_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(ALL_SOURCES) -o $*.mem $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$*.mem

memcheck: $(TEST_DIR)/comprehensivetesting.c $(ALL_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(ALL_SOURCES) -o test $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./test

help:
	@echo -e "$(MAGENTA)Available targets:$(RESET)"
	@printf "  $(CYAN)%-20s$(RESET) - %s\n" "make" "Build main program"
	@printf "  $(CYAN)%-20s$(RESET) - %s\n" "make FILE.run" "Compile and run 'FILE.c'"
	@printf "  $(CYAN)%-20s$(RESET) - %s\n" "make test" "Compile and run the 'comprehensivetesting.c' suite"
	@printf "  $(CYAN)%-20s$(RESET) - %s\n" "make memcheck" "Run valgrind on 'comprehensivetesting.c'"
	@printf "  $(CYAN)%-20s$(RESET) - %s\n" "make FILE.memcheck" "Run valgrind on 'FILE.c'"
	@printf "  $(YELLOW)%-20s$(RESET) - %s\n" "make clean" "Remove executables files"

clean:
	@rm -f main test *.mem *.o
	@echo -e "$(GREEN)Cleaned up executables and object files.$(RESET)"

.PHONY: all test help clean