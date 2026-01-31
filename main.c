// main.c
#include <stdio.h>
#include "matrix.h"

int main() {
    printf("=== Matrix Library Demo ===\n\n");
    
    // Example: Create and print a matrix
    Matrix A = create_matrix(3, 3);
    set_value(A, 0, 0, 1.0f);
    set_value(A, 1, 1, 5.0f);
    set_value(A, 2, 2, 9.0f);
    
    printf("Sample matrix:\n");
    print_matrix(A);
    
    free_matrix(A);
    return 0;
}