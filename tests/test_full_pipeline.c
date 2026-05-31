#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"
#include "vocabulary.h"
#include "tokenizer.h"
#include "embedding.h"

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
    printf("Test 1: Full tokenization pipeline\n");
    
    // Original text
    const char* text = "hello world";
    
    // 1. Encode to tokens
    int len;
    int* tokens = encode(text, &len);
    
    // 2. Decode back
    char* decoded = decode(tokens, len);
    
    // Should match
    assert(strcmp(text, decoded) == 0);
    
    // 3. Embed the tokens
    EmbeddingLayer* layer = create_embedding_layer(70, 128);
    Matrix embedded = embed_sequence(layer, tokens, len);
    
    // Verify dimensions
    assert(embedded.rows == len);
    assert(embedded.cols == 128);
    
    printf("Text: '%s'\n", text);
    printf("Tokens: %d\n", len);
    printf("Embedded: %dx%d matrix\n", embedded.rows, embedded.cols);
    
    free(tokens);
    free(decoded);
    free_matrix(embedded);
    free_embedding_layer(layer);
    
    printf("✓ Passed\n\n");
}