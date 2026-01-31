#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

/*
Comprehensive Testing File

Will be used for :
        >Verify correct behavior (Does the function work and do it's job proprely)
        >Catch edges cases (Boundary Condition)
        >Will return clear feedback if something fails
*/

int float_equals(float a, float b);

void test_matrix_multiply();
void test_matrix_addition();
void test_matrix_subtraction();
void test_matrix_transpose();
void test_matrix_hadamard();
void test_matrix_relu();

int main() {
    printf(BOLD("=== Matrix Library Tests ===\n\n"));
    
    test_matrix_multiply();
    test_matrix_addition();
    test_matrix_subtraction();
    test_matrix_transpose();
    test_matrix_hadamard();
    test_matrix_relu();
    
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
    //                  [10 12]
    if (!float_equals(get_value(C, 0, 0), 6.0f) ||
        !float_equals(get_value(C, 0, 1), 8.0f) ||
        !float_equals(get_value(C, 1, 0), 10.0f) ||
        !float_equals(get_value(C, 1, 1), 12.0f)) {
        printf(RED_TEXT("FAILED: Incorrect values\n"));
        exit(1);
    }
    
    // Clean Up
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 2: Matrix Subtraction
void test_matrix_subtraction(){
    printf(BOLD("Testing matrix_subtraction..."));

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
    //                       [5 5]
    if (!float_equals(get_value(C, 0, 0), 5.0f) ||
        !float_equals(get_value(C, 0, 1), 5.0f) ||
        !float_equals(get_value(C, 1, 0), 5.0f) ||
        !float_equals(get_value(C, 1, 1), 5.0f)) {
        printf(RED_TEXT("FAILED: Incorrect values\n"));
        exit(1);
    }

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
    Matrix C = matrix_multiply(A, B);

    // Check Values
    // Multiplied: [58 64]
    //                  [139 154]
    if (!float_equals(get_value(C, 0, 0), 58.0f) ||
        !float_equals(get_value(C, 0, 1), 64.0f) ||
        !float_equals(get_value(C, 1, 0), 139.0f) ||
        !float_equals(get_value(C, 1, 1), 154.0f)) {
        printf(("FAILED: Incorrect values\n"));
        exit(1);
    }
    
    // Clean Up
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    printf(GREEN_TEXT("PASSED\n"));
}

// Test 4: Matrix Tranpose
void test_matrix_transpose() {
    printf(BOLD("Testing matrix_transpose... "));
    
    // Create Tested Matrix
    // Fill A : [1 2 3]
    //            [4 5 6]
    Matrix A = create_matrix(2, 3);
    set_value(A, 0, 0, 1.0f); set_value(A, 0, 1, 2.0f); set_value(A, 0, 2, 3.0f);
    set_value(A, 1, 0, 4.0f); set_value(A, 1, 1, 5.0f); set_value(A, 1, 2, 6.0f);
    
    // Create Testing Matrix
    Matrix C = matrix_transpose(A);
    
    // Check dimensions
    if (C.rows != 3 || C.cols != 2) {
        printf(RED_TEXT("FAILED: Wrong dimensions (expected 3×2, got %d×%d)\n"), C.rows, C.cols);
        exit(1);
    }
    
    // Check Values
    //Transposed: [1 4]
    //                     [2 5]
    //                     [3 6]
    if (!float_equals(get_value(C, 0, 0), 1.0f) ||
        !float_equals(get_value(C, 0, 1), 4.0f) ||
        !float_equals(get_value(C, 1, 0), 2.0f) ||
        !float_equals(get_value(C, 1, 1), 5.0f) ||
        !float_equals(get_value(C, 2, 0), 3.0f) ||
        !float_equals(get_value(C, 2, 1), 6.0f)) {
        printf(("FAILED: Incorrect values after transpose\n"));
        exit(1);
    }
    
    // Clean Up
    free_matrix(A);
    free_matrix(C);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 5: Matrix Hadamard
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
    //                     [21 32]
    if (!float_equals(get_value(C, 0, 0), 5.0f) ||
        !float_equals(get_value(C, 0, 1), 12.0f) ||
        !float_equals(get_value(C, 1, 0), 21.0f) ||
        !float_equals(get_value(C, 1, 1), 32.0f)) {
        printf(RED_TEXT("FAILED: Incorrect values\n"));
        exit(1);
    }

    // Clean Up
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    printf(GREEN_TEXT("PASSED\n"));

}

// Test 6: Matrix ReLu
void test_matrix_relu(){
    printf(BOLD("Testing matrix_relu..."));

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
    //            [3 0]
    if (!float_equals(get_value(C, 0, 0), 0.0f) ||
        !float_equals(get_value(C, 0, 1), 2.0f) ||
        !float_equals(get_value(C, 1, 0), 3.0f) ||
        !float_equals(get_value(C, 1, 1), 0.0f)) {
        printf(RED_TEXT("FAILED: Incorrect values\n"));
        exit(1);
    }

    // Clean Up
    free_matrix(A);
    free_matrix(C);

    printf(GREEN_TEXT("PASSED\n"));
}