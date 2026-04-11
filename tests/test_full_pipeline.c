#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"
#include "vocabulary.h"
#include "tokenizer.h"
#include "embedding.h"

/*
Comprehensive Embedding Testing File
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

#define EXPECT_NEAR_FLOAT(actual, expected) \
    do { \
        if (fabsf((actual) - (expected)) > 0.0001f) { \
            printf(RED_TEXT("\nFAILED: Float mismatch\n")); \
            printf("  Expected: %f\n  Actual:   %f\n", (expected), (actual)); \
            exit(1); \
        } \
    } while (0)

// Test Function Prototypes
void test_full_pipeline();

int main() {
    printf(BOLD("=== Embedding Layer Tests ===\n"));

    printf(MAGENTA_TEXT("\n--- Integration ---\n"));
    test_full_pipeline();

    printf(GREEN_TEXT("\n=== All Tests Passed! ===\n"));
    return 0;
}

// Test 1: Full Pipeline
void test_full_pipeline() {
    printf(BOLD("Testing full_pipeline... "));
    
    const char* tokens[] = {"#", "a", "b", "c", "hello"};
    Vocabulary* vocab = create_vocab(tokens, 5);
    EmbeddingLayer* embed_layer = create_embedding_layer(5, 8);
    
    int len;
    int* token_ids = encode("abc", &len);
    
    Matrix embedded = embed_sequence(embed_layer, token_ids, len);
    
    EXPECT_EQ_INT(embedded.rows, 3);
    EXPECT_EQ_INT(embedded.cols, 8);
    
    char* decoded = decode(token_ids, len);
    if (strcmp(decoded, "abc") != 0) {
        printf(RED_TEXT("\nFAILED: String mismatch\n"));
        exit(1);
    }
    
    free(token_ids);
    free(decoded);
    free_matrix(embedded);
    free_embedding_layer(embed_layer);
    free_vocab(vocab);
    
    printf(GREEN_TEXT("PASSED\n"));
}