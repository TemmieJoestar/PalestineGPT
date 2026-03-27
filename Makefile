CYAN    = \033[36m
GREEN   = \033[32m
MAGENTA = \033[35m
YELLOW  = \033[33m
RESET   = \033[0m

CC = gcc
CFLAGS = -Wall -Wextra -g -Iinclude
LDFLAGS = -lm

SRC_DIR = src
BONUS_DIR = $(SRC_DIR)/bonus
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

%.bonus: $(BONUS_DIR)/%.c $(LIB_SOURCES) $(ALL_HEADERS)
	@echo -e "$(MAGENTA)Compiling bonus: $< $(RESET)"
	$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o $* $(LDFLAGS)
	./$*

test: $(TEST_DIR)/comprehensivetesting.c $(LIB_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o test $(LDFLAGS)
	./test

%.bmem: $(BONUS_DIR)/%.c $(LIB_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o $*.mem $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$*.mem

memcheck: $(TEST_DIR)/comprehensivetesting.c $(LIB_SOURCES) $(ALL_HEADERS)
	$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o test $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./test

help:
	@echo -e "$(MAGENTA)Available targets:$(RESET)"
	@printf "  $(CYAN)%-25s$(RESET) - %s\n" "make" "Build main program"
	@printf "  $(CYAN)%-25s$(RESET) - %s\n" "make test" "Run comprehensive tests"
	@printf "  $(CYAN)%-25s$(RESET) - %s\n" "make main.run" "Run specific file in src/"
	@printf "  $(CYAN)%-25s$(RESET) - %s\n" "make <file>.bonus" "Run specific file in bonus/"
	@printf "  $(CYAN)%-25s$(RESET) - %s\n" "make memcheck" "Check tests for leaks"
	@printf "  $(CYAN)%-25s$(RESET) - %s\n" "make <file>.bmem" "Check bonus for leaks"
	@printf "  $(YELLOW)%-25s$(RESET) - %s\n" "make clean" "Remove executable files"

clean:
	@rm -f main test xor_problem iris_dataset *.mem *.o
	@echo -e "$(GREEN)Cleaned up executables and object files.$(RESET)"

.PHONY: all test help clean