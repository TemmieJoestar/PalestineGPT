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

LIB_SOURCES = $(SRC_DIR)/matrix.c $(SRC_DIR)/gradient_descent.c $(SRC_DIR)/data_loader.c
ALL_HEADERS = $(INC_DIR)/matrix.h $(INC_DIR)/gradient_descent.h $(INC_DIR)/data_loader.h

all: main

main: $(SRC_DIR)/main.c $(LIB_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $(SRC_DIR)/main.c $(LIB_SOURCES) -o main $(LDFLAGS)

%.run: $(SRC_DIR)/%.c $(LIB_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o $* $(LDFLAGS) 
	./$*

test: $(TEST_DIR)/comprehensivetesting.c $(LIB_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o test $(LDFLAGS)
	./test

%.memcheck: $(SRC_DIR)/%.c $(LIB_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o $*.mem $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$*.mem

memcheck: $(TEST_DIR)/comprehensivetesting.c $(LIB_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o test $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./test

help:
	@echo -e "$(MAGENTA)Available targets:$(RESET)"
	@printf "  $(CYAN)%-20s$(RESET) - %s\n" "make" "Build main program"
	@printf "  $(CYAN)%-20s$(RESET) - %s\n" "make main.run" "Compile and run 'src/main.c'"
	@printf "  $(CYAN)%-20s$(RESET) - %s\n" "make test" "Compile and run the 'tests/comprehensivetesting.c' suite"
	@printf "  $(CYAN)%-20s$(RESET) - %s\n" "make memcheck" "Run valgrind on 'tests/comprehensivetesting.c'"
	@printf "  $(CYAN)%-20s$(RESET) - %s\n" "make main.memcheck" "Run valgrind on 'src/main.c'"
	@printf "  $(YELLOW)%-20s$(RESET) - %s\n" "make clean" "Remove executables files"

clean:
	@rm -f main test *.mem *.o
	@echo -e "$(GREEN)Cleaned up executables and object files.$(RESET)"

.PHONY: all test help clean