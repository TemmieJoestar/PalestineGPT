#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vocabulary.h"

/*
Comprehensive Vocabulary Testing File
Matches the style of comprehensive_testing.c
*/

// Testing Macros
#define EXPECT_EQ_INT(actual, expected) \
    do { \
        if ((actual) != (expected)) { \
            printf(RED_TEXT("\nFAILED: Integer mismatch\n")); \
            printf("  Expected: %d\n  Actual:   %d\n", (expected), (actual)); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_EQ_STR(actual, expected) \
    do { \
        if (actual == NULL || expected == NULL || strcmp((actual), (expected)) != 0) { \
            printf(RED_TEXT("\nFAILED: String mismatch\n")); \
            printf("  Expected: %s\n  Actual:   %s\n", \
                   (expected) ? (expected) : "NULL", \
                   (actual) ? (actual) : "NULL"); \
            exit(1); \
        } \
    } while (0)

// Test Function Prototypes
void test_create_vocab();
void test_get_token_id();
void test_get_token_string();
void test_null_and_edge_cases();

int main() {
    printf(BOLD("=== Vocabulary Library Tests ===\n"));

    printf(MAGENTA_TEXT("\n--- Initialization ---\n"));
    test_create_vocab();

    printf(MAGENTA_TEXT("\n--- Lookup Operations ---\n"));
    test_get_token_id();
    test_get_token_string();

    printf(MAGENTA_TEXT("\n--- Robustness & Edge Cases ---\n"));
    test_null_and_edge_cases();

    printf(GREEN_TEXT("\n=== All Tests Passed! ===\n"));
    return 0;
}

// Test 1: Creation
void test_create_vocab() {
    printf(BOLD("Testing create_vocab... "));
    
    const char* tokens[] = {"[PAD]", "hello", "world"};
    Vocabulary* vocab = create_vocab(tokens, 3);
    
    if (vocab == NULL) {
        printf(RED_TEXT("FAILED: vocab is NULL\n"));
        exit(1);
    }
    
    EXPECT_EQ_INT(vocab->vocab_size, 3);
    
    free_vocab(vocab);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 2: Word to ID
void test_get_token_id() {
    printf(BOLD("Testing get_token_id... "));
    
    const char* tokens[] = {"#", "a", "b", "c"};
    Vocabulary* vocab = create_vocab(tokens, 4);
    
    // Check valid tokens
    EXPECT_EQ_INT(get_token_id(vocab, "#"), 0);
    EXPECT_EQ_INT(get_token_id(vocab, "b"), 2);
    EXPECT_EQ_INT(get_token_id(vocab, "c"), 3);
    
    // Check non-existent token
    EXPECT_EQ_INT(get_token_id(vocab, "xyz"), -1);
    
    free_vocab(vocab);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 3: ID to Word
void test_get_token_string() {
    printf(BOLD("Testing get_token_string... "));
    
    const char* tokens[] = {"apple", "banana", "cherry"};
    Vocabulary* vocab = create_vocab(tokens, 3);
    
    // Test valid IDs (Note: get_token_string returns a malloc'd copy)
    char* s0 = get_token_string(vocab, 0);
    EXPECT_EQ_STR(s0, "apple");
    free(s0);
    
    char* s2 = get_token_string(vocab, 2);
    EXPECT_EQ_STR(s2, "cherry");
    free(s2);
    
    // Test out of bounds
    char* s_invalid = get_token_string(vocab, 99);
    if (s_invalid != NULL) {
        printf(RED_TEXT("FAILED: Expected NULL for out of bounds ID\n"));
        exit(1);
    }
    
    free_vocab(vocab);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 4: Handling NULLs and Edges
void test_null_and_edge_cases() {
    printf(BOLD("Testing edge cases... "));
    
    const char* tokens[] = {"test"};
    Vocabulary* vocab = create_vocab(tokens, 1);
    
    // Test NULL string input for get_token_id
    EXPECT_EQ_INT(get_token_id(vocab, NULL), -1);
    
    // Test negative ID for get_token_string
    if (get_token_string(vocab, -1) != NULL) {
        printf(RED_TEXT("FAILED: Expected NULL for negative ID\n"));
        exit(1);
    }
    
    // Test NULL vocab pointer
    EXPECT_EQ_INT(get_token_id(NULL, "test"), -1);
    
    free_vocab(vocab);
    printf(GREEN_TEXT("PASSED\n"));
}