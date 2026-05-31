#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"
#include "attention.h"
#include "error.h"

// Testing Macro
#define EXPECT_NEAR(actual, expected, tol) \
    do { \
        if (fabsf((actual) - (expected)) > (tol)) { \
            printf(RED_TEXT("\nFAILED: Value mismatch\n")); \
            printf("  Expected: %f\n  Actual:   %f\n", (double)(expected), (double)(actual)); \
            exit(1); \
        } \
    } while (0)

// Function Prototypes
void test_compute_attention_scores();
void test_compute_attention_weights();
void test_apply_attention_weights();
void test_single_attention_forward();
void test_multihead_attention_forward();

int main() {
    printf(BOLD("=== Attention Mechanism Tests ===\n"));
    
    printf(MAGENTA_TEXT("\n--- Component Logic ---\n"));
    test_compute_attention_scores();
    test_compute_attention_weights();
    test_apply_attention_weights();
    
    printf(MAGENTA_TEXT("\n--- Forward Pass Integration ---\n"));
    test_single_attention_forward();
    test_multihead_attention_forward();
    
    printf(GREEN_TEXT("\n=== All Attention Tests Passed! ===\n"));
    return 0;
}

// Test 1: Scaled Dot-Product Scores
void test_compute_attention_scores() {
    printf(BOLD("Testing compute_attention_scores... "));
    
    int d_k = 4; // sqrt(4) = 2.0
    Matrix Q = create_matrix(1, 4);
    Matrix K = create_matrix(1, 4);
    
    // Set Q and K to identity-like for 1.0 dot product
    set_value(Q, 0, 0, 1.0f);
    set_value(K, 0, 0, 1.0f);
    
    // Result = (Q * K^T) / sqrt(d_k) = 1.0 / 2.0 = 0.5
    Matrix scores = compute_attention_scores(Q, K, d_k);
    
    EXPECT_NEAR(get_value(scores, 0, 0), 0.5f, 0.0001f);
    
    free_matrix(Q);
    free_matrix(K);
    free_matrix(scores);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 2: Softmax Weights
void test_compute_attention_weights() {
    printf(BOLD("Testing compute_attention_weights... "));
    
    Matrix scores = create_matrix(1, 2);
    set_value(scores, 0, 0, 2.0f);
    set_value(scores, 0, 1, 2.0f);
    
    // Softmax of [2, 2] is [0.5, 0.5]
    Matrix weights = compute_attention_weights(scores);
    
    EXPECT_NEAR(get_value(weights, 0, 0), 0.5f, 0.0001f);
    EXPECT_NEAR(get_value(weights, 0, 1), 0.5f, 0.0001f);
    
    free_matrix(scores);
    free_matrix(weights);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 3: Value Application
void test_apply_attention_weights() {
    printf(BOLD("Testing apply_attention_weights... "));
    
    Matrix weights = create_matrix(1, 2);
    Matrix V = create_matrix(2, 2);
    
    set_value(weights, 0, 0, 1.0f); // Attend fully to first value
    set_value(V, 0, 0, 10.0f);
    set_value(V, 1, 0, 20.0f);
    
    Matrix result = apply_attention_weights(weights, V);
    
    EXPECT_NEAR(get_value(result, 0, 0), 10.0f, 0.0001f);
    
    free_matrix(weights);
    free_matrix(V);
    free_matrix(result);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 4: Single Head Integration (with Residual)
void test_single_attention_forward() {
    printf(BOLD("Testing single_attention_forward... "));
    
    AttentionHead head;
    head.d_model = 2;
    head.d_k = 2;
    head.Q_weights = create_matrix(2, 2); // Identity
    head.K_weights = create_matrix(2, 2); // Identity
    head.V_weights = create_matrix(2, 2); // Identity
    
    for(int i=0; i<2; i++) {
        set_value(head.Q_weights, i, i, 1.0f);
        set_value(head.K_weights, i, i, 1.0f);
        set_value(head.V_weights, i, i, 1.0f);
    }
    
    Matrix input = create_matrix(1, 2);
    set_value(input, 0, 0, 1.0f);
    
    Matrix output = single_attention_forward(&head, input);
    
    // Output should be Input + AttentionResult
    // For identity weights and single input row, output will be Input + Input = 2*Input
    EXPECT_NEAR(get_value(output, 0, 0), 2.0f, 0.001f);
    
    free_matrix(input);
    free_matrix(output);
    free_matrix(head.Q_weights);
    free_matrix(head.K_weights);
    free_matrix(head.V_weights);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 5: Full Multi-Head Attention Logic
void test_multihead_attention_forward() {
    printf(BOLD("Testing multihead_attention_forward... "));

    int d_model = 4;
    int num_heads = 2;
    int d_k = d_model / num_heads;
    int seq_len = 2;

    MultiHeadAttention mha;
    mha.num_heads = num_heads;
    mha.Heads = malloc(num_heads * sizeof(AttentionHead));
    mha.Output_Weights = create_matrix(d_model, d_model);

    // Initialize Identity-like weights
    for (int i = 0; i < num_heads; i++) {
        mha.Heads[i].d_model = d_model;
        mha.Heads[i].d_k = d_k;
        mha.Heads[i].Q_weights = create_matrix(d_model, d_k);
        mha.Heads[i].K_weights = create_matrix(d_model, d_k);
        mha.Heads[i].V_weights = create_matrix(d_model, d_k);
        
        for (int j = 0; j < d_k; j++) {
            set_value(mha.Heads[i].Q_weights, j, j, 10.0f); // High energy for focus
            set_value(mha.Heads[i].K_weights, j, j, 10.0f);
            set_value(mha.Heads[i].V_weights, j, j, 1.0f);
        }
    }

    for (int i = 0; i < d_model; i++) set_value(mha.Output_Weights, i, i, 1.0f);

    Matrix input = create_matrix(seq_len, d_model);
    set_value(input, 0, 0, 1.0f); 
    set_value(input, 1, 1, 1.0f); 

    Matrix output = multihead_attention_forward(&mha, input);

    // Verify Dimensions
    if (output.rows != seq_len || output.cols != d_model) {
        printf(RED_TEXT("FAILED: Dimension mismatch\n"));
        exit(1);
    }

    // Verify Residual Logic: Input[0,0] is 1.0, Attention contributes ~1.0, Total ~2.0
    EXPECT_NEAR(get_value(output, 0, 0), 2.0f, 0.01f);
    EXPECT_NEAR(get_value(output, 1, 1), 2.0f, 0.01f);

    // Cleanup
    free_matrix(input);
    free_matrix(output);
    for (int i = 0; i < num_heads; i++) {
        free_matrix(mha.Heads[i].Q_weights);
        free_matrix(mha.Heads[i].K_weights);
        free_matrix(mha.Heads[i].V_weights);
    }
    free(mha.Heads);
    free_matrix(mha.Output_Weights);

    printf(GREEN_TEXT("PASSED\n"));
}