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
void set_value(Matrix m, int r, int c, float value);
void free_matrix(Matrix m);
Matrix multiply_matrices(Matrix a, Matrix b);
void print_matrix(Matrix m);

int main()
{
    Matrix A = create_matrix(2,2);
    Matrix I = create_matrix(2,2);

    set_value(A, 0, 0 , 1.0f); set_value(A, 0, 1, 2.0f);
    set_value(A, 1, 0, 3.0f); set_value(A,1, 1, 4.0f);


    set_value(I, 0, 0 , 1.0f); set_value(I, 0, 1, 0.0f);
    set_value(I, 1, 0, 0.0f); set_value(I,1, 1, 1.0f);

    Matrix C = multiply_matrices(A,I);

    printf("Result Matrix C -> \n");
    print_matrix(C);

    free_matrix(A);
    free_matrix(I);
    free_matrix(C);
    
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
    // We need to calculate the 1D index using r, c, and m.cols
    int index = ((r * m.cols) + c); // r -> Current Rows, m.cols -> Total columns, c -> Current cols
    m.data[index] = value;
}

void free_matrix(Matrix m) { 
    free(m.data);
}

Matrix multiply_matrices(Matrix a, Matrix b) {
    // Check if the "Golden Rule" is not violated, if so print an error message and exit the program.
    if (a.cols != b.rows){
        printf(RED_TEXT("Golden Rule broken !"));
        exit(1); // Stop the program
    }
    
    // Creating the result matrix
    Matrix c = create_matrix(a.rows , b.cols);

    // The Triple Loop 
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