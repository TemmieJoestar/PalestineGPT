// matrix.h
#ifndef MATRIX_H
#define MATRIX_H

// Text formatting
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[92;1m" x "\033[0m"
#define BOLD(x) "\033[1m" x "\033[0m" 

// Matrix struct
typedef struct {
    int rows;
    int cols;
    float* data;
} Matrix;

// Operations on two matrices
Matrix matrix_multiply(Matrix a, Matrix b);        // Linear algebra multiplication (dot products)
Matrix matrix_addition(Matrix a, Matrix b);        
Matrix matrix_subtraction(Matrix a, Matrix b);     
Matrix matrix_hadamard(Matrix a, Matrix b);        // Element-wise multiplication

// Operations on one matrix
Matrix matrix_transpose(Matrix m);                 // Swap rows and columns
Matrix matrix_relu(Matrix m);                      // ReLU activation (negative → 0)
Matrix matrix_softmax(Matrix m);                   // Convert to probability distribution (sum = 1.0)
Matrix matrix_scalar_multiply(Matrix m, float s);  // Multiply all elements by a number

// Helper functions
Matrix create_matrix(int rows, int cols);          // Allocate memory for matrix
void free_matrix(Matrix m);                        // Free allocated memory
void print_matrix(Matrix m);                       
void set_value(Matrix m, int r, int c, float v);  
float get_value(Matrix m, int r, int c);

#endif