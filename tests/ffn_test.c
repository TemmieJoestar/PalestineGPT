#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "ffn.h"
#include "error.h"

#define EXPECT_NEAR(actual, expected, tol) \
    do { \
        if (fabsf((actual) - (expected)) > (tol)) { \
            printf(RED_TEXT("\nFAILED: Value mismatch\n")); \
            printf("  Expected: %f\n  Actual:   %f\n", (double)(expected), (double)(actual)); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_EQ_INT(actual, expected) \
    do { \
        if ((actual) != (expected)) { \
            printf(RED_TEXT("\nFAILED: Integer mismatch\n")); \
            printf("  Expected: %d\n  Actual:   %d\n", (expected), (actual)); \
            exit(1); \
        } \
    } while (0)

// Test Function Prototypes (Set 1)
void test_ffn_initialization();
void test_ffn_forward();
void test_ffn_backward();

int main() {
    printf(BOLD("=== Feed-Forward Network Tests ===\n"));
    
    printf(MAGENTA_TEXT("\n--- Initialization & Memory ---\n"));
    test_ffn_initialization();
    
    printf(MAGENTA_TEXT("\n--- Forward & Backward Passes ---\n"));
    test_ffn_forward();
    test_ffn_backward();
    
    printf(GREEN_TEXT("\n=== All FFN Tests Passed! ===\n"));
    return 0;
}

// Test 1: Memory Allocation and Shapes (Set 1)
void test_ffn_initialization() {
    printf(BOLD("Testing FFN layer and cache initialization... "));
    
    int d_model = 4;
    int d_ff = 16;
    int batch_size = 2;
    
    // 1. Layer Init
    FFN_Layer layer = init_ffn_layer(d_model, d_ff);
    
    EXPECT_EQ_INT(layer.W1->rows, d_model);
    EXPECT_EQ_INT(layer.W1->cols, d_ff);
    EXPECT_EQ_INT(layer.W2->rows, d_ff);
    EXPECT_EQ_INT(layer.W2->cols, d_model);
    EXPECT_EQ_INT(layer.b1->rows, 1);
    EXPECT_EQ_INT(layer.b1->cols, d_ff);
    EXPECT_EQ_INT(layer.b2->rows, 1);
    EXPECT_EQ_INT(layer.b2->cols, d_model);
    
    // 2. Cache Init
    Matrix* X = create_matrix(batch_size, d_model, false);
    FFN_Cache cache = init_ffn_cache(X, d_model, d_ff);
    
    EXPECT_EQ_INT(cache.x_up->rows, batch_size);
    EXPECT_EQ_INT(cache.x_up->cols, d_ff);
    EXPECT_EQ_INT(cache.x_activated->rows, batch_size);
    EXPECT_EQ_INT(cache.x_activated->cols, d_ff);
    EXPECT_EQ_INT(cache.x_down->rows, batch_size);
    EXPECT_EQ_INT(cache.x_down->cols, d_model);
    
    // Clean up
    free_ffn_layer(&layer);
    free_ffn_cache(&cache);
    free_matrix(X);
    
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 2: Forward Pass Execution and Math Verification (Set 1)
void test_ffn_forward() {
    printf(BOLD("Testing FFN_Forward... "));
    
    int d_model = 2;
    int d_ff = 2;
    
    FFN_Layer layer = init_ffn_layer(d_model, d_ff);
    
    set_value(layer.W1, 0, 0,  1.0f); set_value(layer.W1, 0, 1, 1.0f);
    set_value(layer.W1, 1, 0, -1.0f); set_value(layer.W1, 1, 1, 1.0f);
    
    set_value(layer.b1, 0, 0, 0.0f); set_value(layer.b1, 0, 1, 0.0f);
    
    set_value(layer.W2, 0, 0, 1.0f); set_value(layer.W2, 0, 1, 2.0f);
    set_value(layer.W2, 1, 0, 3.0f); set_value(layer.W2, 1, 1, 4.0f);
    
    set_value(layer.b2, 0, 0, 1.0f); set_value(layer.b2, 0, 1, -1.0f);
    
    Matrix* X = create_matrix(1, d_model, false);
    set_value(X, 0, 0, 2.0f); set_value(X, 0, 1, 3.0f);
    
    FFN_Cache cache = init_ffn_cache(X, d_model, d_ff);
    
    FFN_Forward(&layer, &cache, X);
    
    EXPECT_NEAR(get_value(cache.x_down, 0, 0), 16.0f, 0.0001f);
    EXPECT_NEAR(get_value(cache.x_down, 0, 1), 19.0f, 0.0001f);
    
    free_ffn_layer(&layer);
    free_ffn_cache(&cache);
    free_matrix(X);
    
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 3: Backward Pass Shapes and Execution (Set 1)
void test_ffn_backward() {
    printf(BOLD("Testing FFN_Backward... "));
    
    int d_model = 2;
    int d_ff = 3;
    int batch_size = 2;
    
    FFN_Layer layer = init_ffn_layer(d_model, d_ff);
    Matrix* X = create_matrix(batch_size, d_model, false);
    FFN_Cache cache = init_ffn_cache(X, d_model, d_ff);
    
    FFN_Gradients grads = init_ffn_grads(d_model, d_ff);
    
    Matrix* d_output = create_matrix(batch_size, d_model, false);
    Matrix* dX = create_matrix(batch_size, d_model, false);
    
    FFN_Forward(&layer, &cache, X);
    FFN_Backward(&layer, &cache, &grads, X, d_output, dX);
    
    EXPECT_EQ_INT(grads.dW1->rows, d_model);
    EXPECT_EQ_INT(grads.dW1->cols, d_ff);
    EXPECT_EQ_INT(grads.db1->cols, d_ff);
    
    EXPECT_EQ_INT(grads.dW2->rows, d_ff);
    EXPECT_EQ_INT(grads.dW2->cols, d_model);
    EXPECT_EQ_INT(grads.db2->cols, d_model);
    
    EXPECT_EQ_INT(dX->rows, batch_size);
    EXPECT_EQ_INT(dX->cols, d_model);
    
    free_ffn_layer(&layer);
    free_ffn_cache(&cache);
    free_ffn_grads(&grads);
    free_matrix(X);
    free_matrix(d_output);
    free_matrix(dX);
    
    printf(GREEN_TEXT("PASSED\n"));
}