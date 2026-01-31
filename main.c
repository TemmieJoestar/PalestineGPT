#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Text formating 
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[92;1m" x "\033[0m"   
#define BOLD(x) "\033[1m" x "\033[0m" 

// Matrix struct 
typedef struct 
{
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

// Will be used to test function
int main() {
    Matrix A = create_matrix(1, 2);
    printf("DEBUG: A.rows = %d, A.cols = %d\n", A.rows, A.cols);
    return 0;
}

Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols; 
    
    m.data = calloc(rows * cols, sizeof(float));
    return m;
}

void set_value(Matrix m, int r, int c, float value) {
    if (r < 0 || r >= m.rows || c < 0 || c >= m.cols) {
        fprintf(stderr, RED_TEXT("Error: Index out of bounds!\n"));
        fprintf(stderr, "Tried to access (%d, %d) in a %dx%d matrix\n", 
                r, c, m.rows, m.cols);
        exit(1);
    }
    
    int index = (r * m.cols) + c;
    m.data[index] = value;
}

float get_value(Matrix m, int r, int c) {
    if (r < 0 || r >= m.rows || c < 0 || c >= m.cols) {
        fprintf(stderr, RED_TEXT("Error: Index out of bounds\n"));
        fprintf(stderr, "Tried to access (%d, %d) in a %dx%d matrix\n",
                r, c, m.rows, m.cols);
        exit(1);
    }
    
    int index = (r * m.cols) + c;
    return m.data[index];
}

void free_matrix(Matrix m) { 
    free(m.data);
}

Matrix matrix_multiply(Matrix a, Matrix b) {  
    if (a.cols != b.rows) {  
        fprintf(stderr, RED_TEXT("Error: Inner dimensions must match (A.cols == B.rows)!\n"));
        exit(1);
    }
    
    Matrix c = create_matrix(a.rows, b.cols);
    
    for (int i = 0; i < c.rows; i++) {
        for (int j = 0; j < c.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a.cols; k++) {  
                float a_val = a.data[i * a.cols + k];
                float b_val = b.data[k * b.cols + j];
                sum += a_val * b_val;
            }
            set_value(c, i, j, sum);
        }
    }
    return c;
}

void print_matrix(Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            int index = ((i * m.cols) + j); 
            printf("%.2f ", m.data[index]);
        }
        printf("\n");
    }
}

Matrix matrix_relu(Matrix m) { 
    int total = m.rows * m.cols;
    Matrix c = create_matrix(m.rows, m.cols);

    for (int i = 0; i < total; i++) {
        if (m.data[i] < 0.0f) {
            c.data[i] = 0.0f;
        } else {
            c.data[i] = m.data[i];
        }
    }
    return c;
}

Matrix matrix_softmax(Matrix m) { 
    float max = m.data[0];
    int total = m.rows * m.cols;
    
    for (int i = 1; i < total; i++) {
        if (m.data[i] > max) {
            max = m.data[i];
        }
    }
    
    Matrix c = create_matrix(m.rows, m.cols);
    float total_sum = 0.0f;
    
    for (int i = 0; i < total; i++) {
        float e = expf(m.data[i] - max);
        c.data[i] = e;
        total_sum += e;
    }
    
    for (int i = 0; i < total; i++) {
        c.data[i] = c.data[i] / total_sum;
    }
    
    return c;
}

Matrix matrix_addition(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, RED_TEXT("Error: Matrices must be the same size for addition!\n"));
        exit(1);
    }
    
    Matrix c = create_matrix(a.rows, a.cols);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            int index = (i * a.cols) + j;
            float result = a.data[index] + b.data[index];
            set_value(c, i, j, result);
        }
    }
    return c;
}

Matrix matrix_subtraction(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, RED_TEXT("Error: Matrices must be the same size for subtraction!\n"));
        exit(1);
    }

    Matrix c = create_matrix(a.rows, a.cols);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            int index = (i * a.cols) + j;
            float result = a.data[index] - b.data[index];
            set_value(c, i, j, result);
        }
    }
    return c;
}

Matrix matrix_transpose(Matrix m) {
    Matrix c = create_matrix(m.cols, m.rows);

    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            int index = (i * m.cols) + j;
            float value = m.data[index];
            set_value(c, j, i, value);
        }
    }
    return c;
}

Matrix matrix_scalar_multiply(Matrix m, float scalar) {  
    Matrix c = create_matrix(m.rows, m.cols);

    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            int index = (i * m.cols) + j;
            float result = m.data[index] * scalar; 
            set_value(c, i, j, result);
        }
    }
    return c;
}

Matrix matrix_hadamard(Matrix a, Matrix b) { 
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, RED_TEXT("Error: Matrices must be the same size for element-wise multiplication!\n"));
        exit(1);
    }
    
    Matrix c = create_matrix(a.rows, a.cols);
    
    for (int i = 0; i < c.rows; i++) {
        for (int j = 0; j < c.cols; j++) {
            int index = (i * c.cols) + j;
            float result = a.data[index] * b.data[index];
            set_value(c, i, j, result);
        }
    }
    return c;
}