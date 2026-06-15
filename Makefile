CYAN    = \033[36m
GREEN   = \033[32m
MAGENTA = \033[35m
YELLOW  = \033[33m
RED     = \033[31m
BOLD    = \033[1m
RESET   = \033[0m


CC = gcc
CFLAGS = -Wall -Wextra -g -Iinclude
LDFLAGS = -lm

SRC_DIR = src
BONUS_DIR = $(SRC_DIR)/bonus
INC_DIR = include
TEST_DIR = tests

LIB_SOURCES = $(SRC_DIR)/matrix.c \
              $(SRC_DIR)/gradient_descent.c \
              $(SRC_DIR)/data_loader.c \
              $(SRC_DIR)/vocabulary.c \
              $(SRC_DIR)/embedding.c \
              $(SRC_DIR)/tokenizer.c \
			  $(SRC_DIR)/ngram.c \
			  $(SRC_DIR)/attention.c

ALL_HEADERS = $(INC_DIR)/matrix.h \
              $(INC_DIR)/gradient_descent.h \
              $(INC_DIR)/data_loader.h \
              $(INC_DIR)/vocabulary.h \
              $(INC_DIR)/embedding.h \
              $(INC_DIR)/tokenizer.h \
              $(INC_DIR)/ngram.h \
			  $(INC_DIR)/attention.h
all: main

main: $(SRC_DIR)/main.c $(LIB_SOURCES) $(ALL_HEADERS)
	@echo -e "$(CYAN)$(BOLD)Building main program...$(RESET)"
	@$(CC) $(CFLAGS) $(SRC_DIR)/main.c $(LIB_SOURCES) -o main $(LDFLAGS)

%.run: $(SRC_DIR)/%.c $(LIB_SOURCES) $(ALL_HEADERS)
	@echo -e "$(CYAN)Running: $< $(RESET)"
	@$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o $* $(LDFLAGS) 
	@./$*

%.bonus: $(BONUS_DIR)/%.c $(LIB_SOURCES) $(ALL_HEADERS)
	@echo -e "$(MAGENTA)$(BOLD)Compiling bonus: $< $(RESET)"
	@$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o $* $(LDFLAGS)
	@./$*

%.test: $(TEST_DIR)/%.c $(LIB_SOURCES) $(ALL_HEADERS)
	@echo -e "$(MAGENTA)$(BOLD)Running test: $< $(RESET)"
	@$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o $* $(LDFLAGS)
	@./$*

%.mem: $(TEST_DIR)/%.c $(LIB_SOURCES) $(ALL_HEADERS)
	@echo -e "$(YELLOW)$(BOLD)Memory check: $< $(RESET)"
	@$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o $*.memcheck $(LDFLAGS)
	@valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$*.memcheck

%.bmem: $(BONUS_DIR)/%.c $(LIB_SOURCES) $(ALL_HEADERS)
	@echo -e "$(YELLOW)$(BOLD)Bonus memory check: $< $(RESET)"
	@$(CC) $(CFLAGS) $< $(LIB_SOURCES) -o $*.mem $(LDFLAGS)
	@valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$*.mem

help:
	@echo -e "$(MAGENTA)$(BOLD)Available targets:$(RESET)"
	@printf "  $(CYAN)$(BOLD)%-25s$(RESET) $(BOLD)- %s$(RESET)\n" "make" "Build main program"
	@printf "  $(CYAN)$(BOLD)%-25s$(RESET) $(BOLD)- %s$(RESET)\n" "make <file>.test" "Run specific test in tests/"
	@printf "  $(CYAN)$(BOLD)%-25s$(RESET) $(BOLD)- %s$(RESET)\n" "make <file>.mem" "Check specific test for leaks"
	@printf "  $(CYAN)$(BOLD)%-25s$(RESET) $(BOLD)- %s$(RESET)\n" "make <file>.run" "Run specific file in src/"
	@printf "  $(CYAN)$(BOLD)%-25s$(RESET) $(BOLD)- %s$(RESET)\n" "make <file>.bonus" "Run specific file in bonus/"
	@printf "  $(CYAN)$(BOLD)%-25s$(RESET) $(BOLD)- %s$(RESET)\n" "make <file>.bmem" "Check specific bonus for leaks"
	@printf "  $(YELLOW)$(BOLD)%-25s$(RESET) $(BOLD)- %s$(RESET)\n" "make clean" "Remove executable files"

clean:
	@rm -f main xor_problem iris_dataset *.mem *.o *.memcheck test_vocab test_attention comprehensive_testing test_embedding test_full_pipeline test_ngram
	@echo -e "$(GREEN)$(BOLD)Cleaned up executables and object files.$(RESET)"

.PHONY: all test help clean