#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "layernorm.h"
#include "error.h"

// Testing Macros
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

// Test Function Prototypes
void test_layernorm_initialization();
void test_layernorm_forward();
void test_layernorm_backward();

int main() {
    printf(BOLD("=== LayerNorm Tests ===\n"));
    
    printf(MAGENTA_TEXT("\n--- Initialization & Memory ---\n"));
    test_layernorm_initialization();
    
    printf(MAGENTA_TEXT("\n--- Forward & Backward Passes ---\n"));
    test_layernorm_forward();
    test_layernorm_backward();
    
    printf(GREEN_TEXT("\n=== All LayerNorm Tests Passed! ===\n"));
    return 0;
}

// Test 1: Memory Allocation and Shapes
void test_layernorm_initialization() {
    printf(BOLD("Testing LayerNorm initialization and memory... "));
    
    int d_model = 4;
    int batch_size = 2;
    
    // 1. Layer Init
    LayerNorm_Layer layer = init_layer_norm_layer(d_model);
    
    EXPECT_EQ_INT(layer.Scale->rows, 1);
    EXPECT_EQ_INT(layer.Scale->cols, d_model);
    EXPECT_EQ_INT(layer.Bias->rows, 1);
    EXPECT_EQ_INT(layer.Bias->cols, d_model);
    
    // Default scale should be 1.0, bias should be 0.0
    EXPECT_NEAR(get_value(layer.Scale, 0, 0), 1.0f, 0.0001f);
    EXPECT_NEAR(get_value(layer.Bias, 0, 0), 0.0f, 0.0001f);
    
    // 2. Cache Init
    Matrix* X = create_matrix(batch_size, d_model, false);
    LayerNorm_Cache cache = init_layer_norm_cache(X);
    
    EXPECT_EQ_INT(cache.Mean->rows, batch_size);
    EXPECT_EQ_INT(cache.Mean->cols, 1);
    EXPECT_EQ_INT(cache.Variance->rows, batch_size);
    EXPECT_EQ_INT(cache.Variance->cols, 1);
    EXPECT_EQ_INT(cache.NormInput->rows, batch_size);
    EXPECT_EQ_INT(cache.NormInput->cols, d_model);
    EXPECT_EQ_INT(cache.Y->rows, batch_size);
    EXPECT_EQ_INT(cache.Y->cols, d_model);
    
    // 3. Gradients Init
    LayerNorm_Grads grads = init_layer_norm_grads(X);
    
    EXPECT_EQ_INT(grads.dX->rows, batch_size);
    EXPECT_EQ_INT(grads.dX->cols, d_model);
    EXPECT_EQ_INT(grads.dScale->rows, 1);
    EXPECT_EQ_INT(grads.dScale->cols, d_model);
    EXPECT_EQ_INT(grads.dBias->rows, 1);
    EXPECT_EQ_INT(grads.dBias->cols, d_model);
    
    // Clean up
    free_layer_norm_layer(&layer);
    free_layer_norm_cache(&cache);
    free_layer_norm_grads(&grads);
    free_matrix(X);
    
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 2: Forward Pass Execution and Math Verification
void test_layernorm_forward() {
    printf(BOLD("Testing LayerNorm_Forward... "));
    
    int d_model = 3;
    int batch_size = 2;
    float epsilon = 1e-5f;
    
    LayerNorm_Layer layer = init_layer_norm_layer(d_model);
    Matrix* X = create_matrix(batch_size, d_model, false);
    LayerNorm_Cache cache = init_layer_norm_cache(X);
    
    // Set Input Data __x)
    // Row 0: [1.0, 2.0, 3.0] -> Mean: 2.0, Var: ~0.6667
    set_value(X, 0, 0, 1.0f); set_value(X, 0, 1, 2.0f); set_value(X, 0, 2, 3.0f);
    // Row 1: [4.0, 5.0, 6.0] -> Mean: 5.0, Var: ~0.6667
    set_value(X, 1, 0, 4.0f); set_value(X, 1, 1, 5.0f); set_value(X, 1, 2, 6.0f);
    
    LayerNorm_Forward(X, &cache, &layer, epsilon);
    
    // Verify Mean
    EXPECT_NEAR(get_value(cache.Mean, 0, 0), 2.0f, 0.0001f);
    EXPECT_NEAR(get_value(cache.Mean, 1, 0), 5.0f, 0.0001f);
    
    // Verify Variance
    float expected_var = 0.6666667f; // (1 + 0 + 1) / 3
    EXPECT_NEAR(get_value(cache.Variance, 0, 0), expected_var, 0.001f);
    EXPECT_NEAR(get_value(cache.Variance, 1, 0), expected_var, 0.001f);
    
    // Verify Normalized Output (Y)
    // -1 / sqrt(0.6666667 + 1e-5) approx -1.2247
    EXPECT_NEAR(get_value(cache.Y, 0, 0), -1.2247f, 0.001f); 
    EXPECT_NEAR(get_value(cache.Y, 0, 1), 0.0f, 0.001f);
    EXPECT_NEAR(get_value(cache.Y, 0, 2), 1.2247f, 0.001f);
    
    free_layer_norm_layer(&layer);
    free_layer_norm_cache(&cache);
    free_matrix(X);
    
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 3: Backward Pass Shapes and Execution
void test_layernorm_backward() {
    printf(BOLD("Testing LayerNorm_Backward... "));
    
    int d_model = 3;
    int batch_size = 2;
    float epsilon = 1e-5f;
    
    LayerNorm_Layer layer = init_layer_norm_layer(d_model);
    Matrix* X = create_matrix(batch_size, d_model, false);
    LayerNorm_Cache cache = init_layer_norm_cache(X);
    LayerNorm_Grads grads = init_layer_norm_grads(X);
    Matrix* dY = create_matrix(batch_size, d_model, false);
    
    // Populate X with varying valid values
    set_value(X, 0, 0, 1.0f); set_value(X, 0, 1, 2.0f); set_value(X, 0, 2, 3.0f);
    set_value(X, 1, 0, -1.0f); set_value(X, 1, 1, 0.0f); set_value(X, 1, 2, 5.0f);
    
    // Set dY to a constant uniform gradient
    for (int i = 0; i < batch_size * d_model; i++) {
        dY->data[i] = 0.5f;
    }
    
    LayerNorm_Forward(X, &cache, &layer, epsilon);
    LayerNorm_Backward(dY, X, &cache, &layer, &grads, epsilon);
    
    // 1. Verify shapes of computed gradients
    EXPECT_EQ_INT(grads.dX->rows, batch_size);
    EXPECT_EQ_INT(grads.dX->cols, d_model);
    
    EXPECT_EQ_INT(grads.dScale->rows, 1);
    EXPECT_EQ_INT(grads.dScale->cols, d_model);
    
    EXPECT_EQ_INT(grads.dBias->rows, 1);
    EXPECT_EQ_INT(grads.dBias->cols, d_model);
    
    // 2. Numerical Check: The sum of dX for each independent row must be exactly 0
    for (int i = 0; i < batch_size; i++) {
        float dx_sum = 0.0f;
        for (int j = 0; j < d_model; j++) {
            dx_sum += get_value(grads.dX, i, j);
        }
        // Use a small tolerance for floating point accumulation
        EXPECT_NEAR(dx_sum, 0.0f, 1e-5f);
    }
    
    // Free resources
    free_layer_norm_layer(&layer);
    free_layer_norm_cache(&cache);
    free_layer_norm_grads(&grads);
    free_matrix(X);
    free_matrix(dY);
    
    printf(GREEN_TEXT("PASSED\n"));
}