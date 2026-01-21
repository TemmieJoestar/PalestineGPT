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

// Used functions !
Matrix create_matrix(int rows,int cols);
Matrix multiply_matrices(Matrix a, Matrix b);
Matrix matrix_addition(Matrix a, Matrix b);

void set_value(Matrix m, int r, int c, float value);
void free_matrix(Matrix m); 
void print_matrix(Matrix m); // Will be used to easily print any matrix
void relu_matrix(Matrix m); // Convert any negative number to 0
void sofmax_matrix(Matrix m);

int main()
{
    
    Matrix A = create_matrix(2,2);
    Matrix B = create_matrix(2,2);

    set_value(A, 0, 0 , 1.0f); set_value(A, 0, 1, 2.0f);
    set_value(A, 1, 0, 3.0f); set_value(A,1, 1, 4.0f);


    set_value(B, 0, 0 , 5.0f); set_value(B, 0, 1, 6.0f);
    set_value(B, 1, 0, 7.0f); set_value(B,1, 1, 8.0f);

    printf(BOLD("Matrix A -> \n"));

    print_matrix(A);

    printf(BOLD("Matrix B -> \n"));
    print_matrix(B);
    
    Matrix C = matrix_addition(A,B);

    printf(BOLD("Result Matrix C -> \n"));
    print_matrix(C);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    

    /*
    Matrix C = create_matrix(2,2);
    set_value(C, 0, 0, -10.5f); 

    printf("Matrix C before ReLU:\n");
    print_matrix(C);

    sofmax_matrix(C);
    printf("\n");
    print_matrix(C);
    */
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
    // We need to calculate the 1D index using r, c, and m.cols
    int index = ((r * m.cols) + c); // r -> Current Rows, m.cols -> Total columns, c -> Current cols
    m.data[index] = value;
}

void free_matrix(Matrix m) { 
    free(m.data);
}

Matrix multiply_matrices(Matrix a, Matrix b) {
    if (a.cols != b.rows) {
        printf(RED_TEXT("Error: Inner dimensions must match (A.cols == B.rows)!\n"));
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

void relu_matrix(Matrix m) {
    int total = m.rows * m.cols;

    for (int i = 0; i < total; i++) {
        if (m.data[i] < 0.0f) {
            m.data[i] = 0.0f;
        }
    }
}

void sofmax_matrix(Matrix m) { 
    float total_sum = 0.0f; 

    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++){
            int index = ((i * m.cols) + j); 
            float e = expf(m.data[index]);
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