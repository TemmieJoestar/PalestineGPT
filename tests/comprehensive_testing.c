#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "matrix.h"

// Testing Macro
#define EXPECT_NEAR(actual, expected, tol) \
    do { \
        if (fabsf((actual) - (expected)) > (tol)) { \
            printf(RED_TEXT("\nFAILED: Value mismatch\n")); \
            printf("  Expected: %f\n  Actual:   %f\n", (double)(expected), (double)(actual)); \
            exit(1); \
        } \
    } while (0)

int float_equals(float a, float b);

void test_matrix_multiply();
void test_matrix_addition();
void test_matrix_subtraction();
void test_matrix_hadamard();
void test_matrix_relu();
void test_matrix_softmax();
void test_matrix_scalar_multiply(); 
void test_matrix_scalar_addition();
void test_matrix_scalar_subtraction();
void test_matrix_copy();
void test_matrix_sigmoid();
void test_matrix_normalize();

int main() {
    printf(BOLD("=== Matrix Library Tests ===\n"));
    
    printf(MAGENTA_TEXT("\n--- Core Matrix Arithmetic ---\n"));
    test_matrix_multiply();
    test_matrix_addition();
    test_matrix_subtraction();
    test_matrix_hadamard();
    
    printf(MAGENTA_TEXT("\n--- Transformations & Scalars ---\n"));
    test_matrix_scalar_multiply();
    test_matrix_scalar_addition();
    test_matrix_scalar_subtraction();
    test_matrix_copy();
    
    printf(MAGENTA_TEXT("\n--- Neural Network Activations ---\n"));
    test_matrix_relu();
    test_matrix_sigmoid();
    test_matrix_softmax();
    
    printf(MAGENTA_TEXT("\n--- Data Preprocessing ---\n"));
    test_matrix_normalize();
    
    printf(GREEN_TEXT("\n=== All Tests Passed! ===\n"));
    return 0;
}

// Helper function to compare floats
int float_equals(float a, float b) {
    return fabsf(a - b) < 0.0001f;
}

// Test 1: Matrix Addition
void test_matrix_addition() {
    printf(BOLD("Testing matrix_addition... "));
    
    // Create Test Matrices
    Matrix A = create_matrix(2, 2);
    Matrix B = create_matrix(2, 2);
    
    // Fill A: [1 2]
    //           [3 4]
    set_value(A, 0, 0, 1.0f); set_value(A, 0, 1, 2.0f);
    set_value(A, 1, 0, 3.0f); set_value(A, 1, 1, 4.0f);
    
    // Fill B: [5 6]
    //           [7 8]
    set_value(B, 0, 0, 5.0f); set_value(B, 0, 1, 6.0f);
    set_value(B, 1, 0, 7.0f); set_value(B, 1, 1, 8.0f);
    
    // Create Testing Matrix
    Matrix C = matrix_addition(A, B);
    
    // Check Values
    // Addition:  [6  8]
    //            [10 12]
    EXPECT_NEAR(get_value(C, 0, 0), 6.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 8.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 0), 10.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 1), 12.0f, 0.0001f);
    
    // Clean Up
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 2: Matrix Subtraction
void test_matrix_subtraction(){
    printf(BOLD("Testing matrix_subtraction... "));

    // Create Tested Matrices
    Matrix A = create_matrix(2,2);
    Matrix B = create_matrix(2,2);

    // Fill A : [10 11]
    //            [12 13]
    set_value(A, 0, 0, 10.0f); set_value(A, 0, 1, 11.0f);
    set_value(A, 1, 0, 12.0f); set_value(A,1, 1, 13.0f);

    // Fill B [5 6]
    //          [7 8]
    set_value(B, 0, 0, 5.0f); set_value(B, 0, 1, 6.0f);
    set_value(B, 1, 0, 7.0f); set_value(B, 1, 1, 8.0f);

    // Create Testing Matrix
    Matrix C = matrix_subtraction(A,B);

    // Check Values
    // Subtracted : [5 5]
    //              [5 5]
    EXPECT_NEAR(get_value(C, 0, 0), 5.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 5.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 0), 5.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 1), 5.0f, 0.0001f);

    // Clean Up
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    printf(GREEN_TEXT("PASSED\n"));
}

// Test 3: Matrix Multiplication
void test_matrix_multiply() {
    printf(BOLD("Testing matrix_multiply... "));

    // Create Test Matrices
    Matrix A = create_matrix(2, 3);
    Matrix B = create_matrix(3, 2);
    
    // Fill A: [1 2 3]
    //           [4 5 6]
    set_value(A, 0, 0, 1.0f); set_value(A, 0, 1, 2.0f); set_value(A, 0, 2, 3.0f);
    set_value(A, 1, 0, 4.0f); set_value(A, 1, 1, 5.0f); set_value(A, 1, 2, 6.0f);
    
    // Fill B: [7 8]
    //           [9 10]
    //           [11 12]
    set_value(B, 0, 0, 7.0f); set_value(B, 0, 1, 8.0f);
    set_value(B, 1, 0, 9.0f); set_value(B, 1, 1, 10.0f);
    set_value(B, 2, 0, 11.0f); set_value(B, 2, 1, 12.0f);
    
    // Create Testing Matrix
    Matrix C = create_matrix(2, 2);
    matrix_multiply(A, B, C, false, false);

    // Check Values
    // Multiplied: [58 64]
    //             [139 154]
    EXPECT_NEAR(get_value(C, 0, 0), 58.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 64.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 0), 139.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 1), 154.0f, 0.0001f);
    
    // Clean Up
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    printf(GREEN_TEXT("PASSED\n"));
}

// Test 4: Matrix Hadamard
void test_matrix_hadamard(){
    printf(BOLD("Testing matrix_hadamard... "));

    // Create Tested Matrices
    Matrix A = create_matrix(2,2);
    Matrix B = create_matrix(2,2);

    // Fill A: [1 2]
    //           [3 4]
    set_value(A, 0, 0, 1.0f); set_value(A, 0, 1, 2.0f);
    set_value(A, 1, 0, 3.0f); set_value(A, 1, 1, 4.0f);
    
    // Fill B: [5 6]
    //           [7 8]
    set_value(B, 0, 0, 5.0f); set_value(B, 0, 1, 6.0f);
    set_value(B, 1, 0, 7.0f); set_value(B, 1, 1, 8.0f);
    
    // Testing Function
    Matrix C = matrix_hadamard(A, B);

    // Check Values
    // Hadamard: [5 12]
    //           [21 32]
    EXPECT_NEAR(get_value(C, 0, 0), 5.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 12.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 0), 21.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 1), 32.0f, 0.0001f);

    // Clean Up
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    printf(GREEN_TEXT("PASSED\n"));
}

// Test 5: Matrix ReLu
void test_matrix_relu(){
    printf(BOLD("Testing matrix_relu... "));

    // Create Tested Matrix
     Matrix A = create_matrix(2,2);
     
     // Fill A: [-1 2]
     //           [3 -4]
    set_value(A, 0, 0, -1.0f); set_value(A, 0, 1, 2.0f);
    set_value(A, 1, 0, 3.0f); set_value(A, 1, 1, -4.0f);

    // Create Testing Matrix
    Matrix C = matrix_relu(A);

    // Check Values
    // ReLu: [0 2]
    //       [3 0]
    EXPECT_NEAR(get_value(C, 0, 0), 0.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 2.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 0), 3.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 1), 0.0f, 0.0001f);

    // Clean Up
    free_matrix(A);
    free_matrix(C);

    printf(GREEN_TEXT("PASSED\n"));
}

// Test 6: Softmax
void test_matrix_softmax() {
    printf(BOLD("Testing matrix_softmax... "));

    // Create Tested Matrix
    Matrix A = create_matrix(1, 3);

     // Fill A: [1 2 3]
    set_value(A, 0, 0, 1.0f); set_value(A, 0, 1, 2.0f); set_value(A, 0, 2, 3.0f);

    // Create Testing Matrix
    Matrix C = matrix_softmax(A);

    // Test 6-1: Check specific probabilities
    // Expected for [1, 2, 3]: [0.0900, 0.2447, 0.6652]
    EXPECT_NEAR(get_value(C, 0, 0), 0.090031f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 0.244728f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 2), 0.665241f, 0.0001f);

    // Test 6-2: Sum should equal 1.0
    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        sum += C.data[i];
    }
    EXPECT_NEAR(sum, 1.0f, 0.0001f);

    // Clean Up
    free_matrix(A);
    free_matrix(C);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 7: Scalar Multiplication
void test_matrix_scalar_multiply() {
    printf(BOLD("Testing matrix_scalar_multiply... "));

    // Create test matrix
    Matrix A = create_matrix(2, 2);
    
    // Fill A: [1 2]
    //         [3 4]
    set_value(A, 0, 0, 1.0f); set_value(A, 0, 1, 2.0f);
    set_value(A, 1, 0, 3.0f); set_value(A, 1, 1, 4.0f);

    // Multiply by scalar 2.5
    Matrix C = matrix_scalar_multiply(A, 2.5f);

    // Check values
    // Expected: [2.5  5.0]
    //           [7.5 10.0]
    EXPECT_NEAR(get_value(C, 0, 0), 2.5f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 5.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 0), 7.5f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 1), 10.0f, 0.0001f);

    // Clean up
    free_matrix(A);
    free_matrix(C);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 8: Matrix Copy
void test_matrix_copy() {
    printf(BOLD("Testing matrix_copy... "));
    
    Matrix A = create_matrix(2, 2);
    set_value(A, 0, 0, 1.0f); set_value(A, 0, 1, 2.0f);
    set_value(A, 1, 0, 3.0f); set_value(A, 1, 1, 4.0f);
    
    Matrix B = matrix_copy(A);
    
    // Check values match
    EXPECT_NEAR(get_value(B, 0, 0), 1.0f, 0.0001f);
    EXPECT_NEAR(get_value(B, 1, 1), 4.0f, 0.0001f);
    
    // Check it's a DEEP copy (different memory)
    set_value(B, 0, 0, 99.0f);
    if (float_equals(get_value(A, 0, 0), 99.0f)) {
        printf(RED_TEXT("FAILED: Not a deep copy!\n"));
        exit(1);
    }
    
    free_matrix(A);
    free_matrix(B);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 9: Matrix Sigmoid
void test_matrix_sigmoid(){
    printf(BOLD("Testing matrix_sigmoid... "));

    // Create tested matrix
    Matrix A = create_matrix(1, 3);
    
    // Fill A: [-5 0 5]
    set_value(A, 0, 0, -5.0f);
    set_value(A, 0, 1, 0.0f);
    set_value(A, 0, 2, 5.0f);
    
    // Create testing matrix
    Matrix C = matrix_sigmoid(A);

    // Check values
    // Expected: [0.00669, 0.5000, 0.9933]
    EXPECT_NEAR(get_value(C, 0, 0), 0.006693f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 0.500000f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 2), 0.993307f, 0.0001f);

    // Clean up
    free_matrix(A);
    free_matrix(C);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 10: Scalar Addition
void test_matrix_scalar_addition() {
    printf(BOLD("Testing matrix_scalar_addition... "));

    Matrix A = create_matrix(2, 2);
    // Fill A: [1.0, 2.0]
    //         [3.0, 4.0]
    set_value(A, 0, 0, 1.0f); set_value(A, 0, 1, 2.0f);
    set_value(A, 1, 0, 3.0f); set_value(A, 1, 1, 4.0f);

    Matrix C = matrix_scalar_addition(A, 10.5f);

    // Expected: [11.5, 12.5]
    //           [13.5, 14.5]
    EXPECT_NEAR(get_value(C, 0, 0), 11.5f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 12.5f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 0), 13.5f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 1), 14.5f, 0.0001f);

    free_matrix(A);
    free_matrix(C);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 11: Scalar Subtraction
void test_matrix_scalar_subtraction() {
    printf(BOLD("Testing matrix_scalar_subtraction... "));

    Matrix A = create_matrix(2, 2);
    // Fill A: [10.0, 20.0]
    //         [30.0, 40.0]
    set_value(A, 0, 0, 10.0f); set_value(A, 0, 1, 20.0f);
    set_value(A, 1, 0, 30.0f); set_value(A, 1, 1, 40.0f);

    Matrix C = matrix_scalar_subtraction(A, 5.0f);

    // Expected: [5.0, 15.0]
    //           [25.0, 35.0]
    EXPECT_NEAR(get_value(C, 0, 0), 5.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 15.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 0), 25.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 1, 1), 35.0f, 0.0001f);

    free_matrix(A);
    free_matrix(C);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 12: Matrix Normalization
void test_matrix_normalize() {
    printf(BOLD("Testing matrix_normalize... "));

    // Scenario 1: Standard Normalization
    // Data: [10.0, 30.0, 50.0] -> Range is 40.0
    Matrix A = create_matrix(1, 3);
    set_value(A, 0, 0, 10.0f); 
    set_value(A, 0, 1, 30.0f); 
    set_value(A, 0, 2, 50.0f); 

    Matrix C = matrix_normalize(A);

    // Expected: [(10-10)/40, (30-10)/40, (50-10)/40] -> [0.0, 0.5, 1.0]
    EXPECT_NEAR(get_value(C, 0, 0), 0.0f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 1), 0.5f, 0.0001f);
    EXPECT_NEAR(get_value(C, 0, 2), 1.0f, 0.0001f);

    // Scenario 2: Zero Range Edge Case
    // Data: [7.0, 7.0] -> Should print error and return copy
    Matrix B = create_matrix(1, 2);
    set_value(B, 0, 0, 7.0f);
    set_value(B, 0, 1, 7.0f);
    
    printf(CYAN_TEXT("\n(Next warning message is expected behavior)\n"));
    Matrix D = matrix_normalize(B);
    
    EXPECT_NEAR(get_value(D, 0, 0), 7.0f, 0.0001f);
    EXPECT_NEAR(get_value(D, 0, 1), 7.0f, 0.0001f);

    // Clean Up
    free_matrix(A);
    free_matrix(C);
    free_matrix(B);
    free_matrix(D);

    printf(GREEN_TEXT("PASSED\n"));
}