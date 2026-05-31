#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
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
void test_embedding_creation();
void test_single_embedding();
void test_sequence_embedding();
void test_random_initialization();

int main() {
    printf(BOLD("=== Embedding Layer Tests ===\n"));

    printf(MAGENTA_TEXT("\n--- Initialization ---\n"));
    test_embedding_creation();
    test_random_initialization();

    printf(MAGENTA_TEXT("\n--- Lookup Operations ---\n"));
    test_single_embedding();
    test_sequence_embedding();

    printf(GREEN_TEXT("\n=== All Tests Passed! ===\n"));
    return 0;
}

// Test 1: Creation
void test_embedding_creation() {
    printf(BOLD("Testing embedding_creation... "));
    
    EmbeddingLayer* layer = create_embedding_layer(10, 8);
    
    if (layer == NULL) {
        printf(RED_TEXT("FAILED: layer is NULL\n"));
        exit(1);
    }
    
    EXPECT_EQ_INT(layer->vocab_size, 10);
    EXPECT_EQ_INT(layer->embedding_dim, 8);
    EXPECT_EQ_INT(layer->embeddings.rows, 10);
    EXPECT_EQ_INT(layer->embeddings.cols, 8);
    
    free_embedding_layer(layer);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 2: Single Lookup
void test_single_embedding() {
    printf(BOLD("Testing get_embedding... "));
    
    EmbeddingLayer* layer = create_embedding_layer(5, 4);
    Matrix emb = get_embedding(layer, 2);
    
    EXPECT_EQ_INT(emb.rows, 1);
    EXPECT_EQ_INT(emb.cols, 4);
    
    // Verify it matches row 2 of embeddings matrix
    for (int j = 0; j < 4; j++) {
        EXPECT_NEAR_FLOAT(get_value(emb, 0, j), get_value(layer->embeddings, 2, j));
    }
    
    free_matrix(emb);
    free_embedding_layer(layer);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 3: Sequence Embedding
void test_sequence_embedding() {
    printf(BOLD("Testing embed_sequence... "));
    
    EmbeddingLayer* layer = create_embedding_layer(10, 8);
    int token_ids[] = {0, 5, 9};
    Matrix seq_emb = embed_sequence(layer, token_ids, 3);
    
    EXPECT_EQ_INT(seq_emb.rows, 3);
    EXPECT_EQ_INT(seq_emb.cols, 8);
    
    // Row 0 should match embedding of token 0
    for (int j = 0; j < 8; j++) {
        EXPECT_NEAR_FLOAT(get_value(seq_emb, 0, j), get_value(layer->embeddings, 0, j));
    }
    
    free_matrix(seq_emb);
    free_embedding_layer(layer);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 4: Initialization Check
void test_random_initialization() {
    printf(BOLD("Testing random_initialization... "));
    
    EmbeddingLayer* layer = create_embedding_layer(100, 128);
    
    // Check all values are in range [-0.5, 0.5]
    int total = layer->embeddings.rows * layer->embeddings.cols;
    for (int i = 0; i < total; i++) {
        float val = layer->embeddings.data[i];
        if (val < -0.5f || val > 0.5f) {
            printf(RED_TEXT("\nFAILED: Value %f out of range\n"), val);
            exit(1);
        }
    }
    
    free_embedding_layer(layer);
    printf(GREEN_TEXT("PASSED\n"));
}