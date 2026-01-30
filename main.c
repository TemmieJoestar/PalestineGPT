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

// Return Matrix 
Matrix create_matrix(int rows,int cols);
Matrix multiply_matrices(Matrix a, Matrix b);
Matrix matrix_addition(Matrix a, Matrix b);
Matrix transpose(Matrix m);
Matrix relu_matrix(Matrix m);
Matrix sofmax_matrix(Matrix m);

// Return Void -- Will be change

void set_value(Matrix m, int r, int c, float value);
void free_matrix(Matrix m); 
void print_matrix(Matrix m);


// Others
float get_value(Matrix m, int r, int c);

// Will be used to test function
int main() {
    Matrix A = create_matrix(2, 2);
    
    set_value(A, 0, 0, 1.0f);
    set_value(A, 0, 1, 2.0f);
    set_value(A, 1, 0, 3.0f);
    set_value(A, 1, 1, 4.0f);
    
    printf("Matrix A:\n");
    print_matrix(A);
    
    Matrix B = sofmax_matrix(A);
    
    printf("\nAfter Softmax (Matrix B):\n");
    print_matrix(B);
    
    printf("\nMatrix A (should be unchanged):\n");
    print_matrix(A);
    
    float sum = 0.0f;
    for (int i = 0; i < B.rows * B.cols; i++) {
        sum += B.data[i];
    }
    printf("\nSum of softmax values: %.6f \n", sum);
    
    free_matrix(A);
    free_matrix(B);
    return 0;
}

Matrix create_matrix(int rows,int cols) {
    // Linking matrix to the params
    Matrix m;
    m.rows = rows;
    m.cols = cols; 
    
    m.data = calloc(rows * cols, sizeof(float)); // (rows * cols) -> SUM of every elements of the matrix. Will be used to allocate the right size of memory.
    return m;
}


void set_value(Matrix m, int r, int c, float value) {
    // Check if indices are valid
    if (r < 0 || r >= m.rows || c < 0 || c >= m.cols) {
        fprintf(stderr, RED_TEXT("Error: Index out of bounds!\n"));
        fprintf(stderr, "Tried to access (%d, %d) in a %dx%d matrix\n", 
                r, c, m.rows, m.cols);
        exit(1);
    }
    
    int index = (r * m.cols) + c;
    m.data[index] = value;
}

float get_value(Matrix m, int r, int c){
    // Check if indices are valid
    if ( r < 0 || r >= m.rows || c < 0 || c >= m.cols) {
      fprintf(stderr, RED_TEXT("Error: Index out of bounds\n"));
      fprintf(stderr, "Tried to acces (%d, %d) in a %dx%d matrix\n",
              r, c,m.rows, m.cols);
      exit(1);
    }
    
    int index = (r * m.cols) + c;

    return m.data[index];
}

void free_matrix(Matrix m) { 
    free(m.data);
}

Matrix multiply_matrices(Matrix a, Matrix b) {
    if (a.cols != b.rows) {
        printf(RED_TEXT("Error: Inner dimensions must match (a.cols == b.rows)!\n"));
        exit(1);
    }
    
    // Creating the result matrix
    Matrix c = create_matrix(a.rows , b.cols); 
    // Loop 'i' through every ROW of A
    for (int i = 0; i < c.rows; i++){
        // Loop 'j' through every COLUMN of B
        for (int j = 0; j < c.cols; j++){
            float sum = 0.0f;
            for (int k = 0; k < a.cols; k++){
                float a_val = a.data[i * a.cols + k];
                float b_val = b.data[k * b.cols + j];

                sum += a_val * b_val;
            }
            set_value(c, i, j, sum);
        }
    }
    return c;
}

void print_matrix(Matrix m){
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            int index = ((i * m.cols) + j); 
            printf("%.2f ",m.data[index]);
        }
        printf("\n");
    }
}

Matrix relu_matrix(Matrix m) {
    int total = m.rows * m.cols;
    Matrix c = create_matrix(m.rows,m.cols);

    for (int i = 0; i < total; i++) {
        if (m.data[i] < 0.0f) {
            c.data[i] = 0.0f; // If negative store 
        } else {
            c.data[i] = m.data[i]; // If not, copy original
        }
    }
    return c;
}


Matrix sofmax_matrix(Matrix m) {
    // Find max value from original "Matrix m"
    float max = m.data[0];
    int total = m.rows * m.cols;
    
    for (int i = 1; i < total; i++) {  // Start from 1 since we initialized with data[0]
        if (m.data[i] > max) {
            max = m.data[i];
        }
    }
    
    // Create new "Matrix c" and apply exp(x - max)
    Matrix c = create_matrix(m.rows, m.cols);
    float total_sum = 0.0f;
    
    for (int i = 0; i < total; i++) {
        float e = expf(m.data[i] - max);
        c.data[i] = e;
        total_sum += e;
    }
    
    // Normalize by dividing by sum
    for (int i = 0; i < total; i++) {
        c.data[i] = c.data[i] / total_sum;
    }
    
    return c;
}

Matrix matrix_addition(Matrix a, Matrix b){
    if (a.rows != b.rows || a.cols != b.cols) {
        printf(RED_TEXT("Error: Matrices must be the same size for addition!\n"));
        exit(1);
    }
    Matrix c =create_matrix(a.rows,a.cols);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            int index = (i * a.cols) + j;
            
            float result = a.data[index] + b.data[index];
            set_value(c, i, j, result);
        }
    }
    return c;
}

Matrix transpose(Matrix m){
    for (int i = 0; i > m.cols; i++){
        for (int j =0; j > m.rows; j++){
            int index = (i * m.cols) + j;
            float tmp = m.data[index];



        } 
    }

}