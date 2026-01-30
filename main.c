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

// Return Void -- Will be change

void set_value(Matrix m, int r, int c, float value);
void free_matrix(Matrix m); 
void print_matrix(Matrix m);
Matrix relu_matrix(Matrix m);
void sofmax_matrix(Matrix m);

// Others
float get_value(Matrix m, int r, int c);

// Will be used to test function
int main() {
    Matrix A = create_matrix(2, 2);
    
    // Set some values (including negatives)
    set_value(A, 0, 0, 5.0f);
    set_value(A, 0, 1, -3.0f);
    set_value(A, 1, 0, -7.0f);
    set_value(A, 1, 1, 2.0f);
    
    printf("Original Matrix A:\n");
    print_matrix(A);
    
    Matrix B = relu_matrix(A);
    
    printf("\nAfter ReLU (Matrix B):\n");
    print_matrix(B);
    
    printf("\nOriginal Matrix A (should be unchanged):\n");
    print_matrix(A);
    
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


void sofmax_matrix(Matrix m){
    float total_sum = 0.0f;
    // Find the maximum value
    float max = m.data[0];
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            float current = m.data[i * m.cols + j];
            if (current > max) {
                max = current; // Update max if current is greater
            }
        }
    }

    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++){
            int index = ((i * m.cols) + j); 
            float e = expf(m.data[index] - max);
            total_sum += e;
            m.data[index] = e;
        }
    }

    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++){
            int index = ((i * m.cols) + j);
            m.data[index] = m.data[index] / total_sum;
        }
    }
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