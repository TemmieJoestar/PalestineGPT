#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "positional_encoding.h"
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
        if ((int)(actual) != (int)(expected)) { \
            printf(RED_TEXT("\nFAILED: Integer mismatch\n")); \
            printf("  Expected: %d\n  Actual:   %d\n", (int)(expected), (int)(actual)); \
            exit(1); \
        } \
    } while (0)

// Test Function Prototypes
void test_pos_encoding_initialization();
void test_pos_encoding_values();

int main() {
    printf(BOLD("=== Positional Encoding Tests ===\n"));
    
    printf(MAGENTA_TEXT("\n--- Initialization & Memory ---\n"));
    test_pos_encoding_initialization();
    
    printf(MAGENTA_TEXT("\n--- Value Verification ---\n"));
    test_pos_encoding_values();
    
    printf(GREEN_TEXT("\n=== All Positional Encoding Tests Passed! ===\n"));
    return 0;
}

// Test 1: Memory Allocation and Shapes
void test_pos_encoding_initialization() {
    printf(BOLD("Testing PosEncoding initialization and memory... "));
    
    int d_model = 8;
    int max_seq_len = 100;
    
    // Module Init
    PosEncoding pe = init_pos_encoding(d_model, max_seq_len);
    
    // Ensure Matrix was created with correct dimensions
    EXPECT_EQ_INT(pe.Matrix->rows, max_seq_len);
    EXPECT_EQ_INT(pe.Matrix->cols, d_model);
    
    // Clean up
    free_pos_encoding(&pe);
    
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 2: Mathematical Execution
void test_pos_encoding_values() {
    printf(BOLD("Testing PosEncoding values... "));
    
    int d_model = 4;
    int max_seq_len = 2;
    
    PosEncoding pe = init_pos_encoding(d_model, max_seq_len);
    
    // POSITION 0
    // angle = 0 / 10000^(2*i/d_model) = 0
    // sin(0) = 0.0, cos(0) = 1.0
    EXPECT_NEAR(get_value(pe.Matrix, 0, 0), 0.0f, 0.0001f); // i=0 (sin)
    EXPECT_NEAR(get_value(pe.Matrix, 0, 1), 1.0f, 0.0001f); // i=0 (cos)
    
    EXPECT_NEAR(get_value(pe.Matrix, 0, 2), 0.0f, 0.0001f); // i=2 (sin)
    EXPECT_NEAR(get_value(pe.Matrix, 0, 3), 1.0f, 0.0001f); // i=2 (cos)
    
    // POSITION 1
    // i = 0 -> angle = 1 / 10000^(0/4) = 1.0
    // sin(1.0) = ~0.84147, cos(1.0) = ~0.54030
    EXPECT_NEAR(get_value(pe.Matrix, 1, 0), 0.84147f, 0.0001f); 
    EXPECT_NEAR(get_value(pe.Matrix, 1, 1), 0.54030f, 0.0001f); 
    
    // i = 2 -> angle = 1 / 10000^(4/4) = 0.0001
    // sin(0.0001) = ~0.00010, cos(0.0001) = ~1.00000
    EXPECT_NEAR(get_value(pe.Matrix, 1, 2), 0.00010f, 0.0001f); 
    EXPECT_NEAR(get_value(pe.Matrix, 1, 3), 1.00000f, 0.0001f); 
    
    // Clean up
    free_pos_encoding(&pe);
    
    printf(GREEN_TEXT("PASSED\n"));
}