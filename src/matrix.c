#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "error.h"

Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols; 
    
    m.data = calloc(rows * cols, sizeof(float));
    return m;
}

void set_value(Matrix m, int r, int c, float value) {
    if (r < 0 || r >= m.rows || c < 0 || c >= m.cols) {
        FATAL_ERROR("Index out of bounds. Tried accessing (%d, %d) in a %dx%d matrix. Exiting...", r, c, m.rows, m.cols);
    }
    int index = (r * m.cols) + c;
    m.data[index] = value;
}

float get_value(Matrix m, int r, int c) {
    if (r < 0 || r >= m.rows || c < 0 || c >= m.cols) {
        FATAL_ERROR("Index out of bounds. Tried to access (%d, %d) in a %dx%d matrix. Exiting...",r, c, m.rows, m.cols);
    }
    
    int index = (r * m.cols) + c;
    return m.data[index];
}

void free_matrix(Matrix m) { 
    if (m.data != NULL) {
            free(m.data);
        }
}

Matrix matrix_multiply(Matrix a, Matrix b) {  
    if (a.cols != b.rows) {  
        FATAL_ERROR("Inner dimensions must match (A.cols == B.rows). Exiting...");
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

void print_matrix(Matrix Input) {
    int total = Input.rows * Input.cols;
    
    for (int i = 0; i < total; i++) {
        printf("%.2f ", Input.data[i]);
        if ((i + 1) % Input.cols == 0) {
            printf("\n");
        }
    }
}

Matrix matrix_relu(Matrix Input) { 
    int total = Input.rows * Input.cols;
    Matrix Result = create_matrix(Input.rows, Input.cols);

    for (int i = 0; i < total; i++) {
        if (Input.data[i] < 0.0f) {
            Result.data[i] = 0.0f;
        } else {
            Result.data[i] = Input.data[i];
        }
    }
    return Result;
}

Matrix matrix_relu_derivative(Matrix Hidden_raw, Matrix Gradient_Hidden){
    if (Hidden_raw.rows != Gradient_Hidden.rows || Hidden_raw.cols != Gradient_Hidden.cols) {
        FATAL_ERROR("Matrices must be the same size. Exiting...");
    }

    int total = Hidden_raw.rows * Hidden_raw.cols;
    Matrix output = create_matrix(Hidden_raw.rows,Hidden_raw.cols);

    for (int i = 0; i < total; i++){
        if (Hidden_raw.data[i] > 0) {
            output.data[i] = Gradient_Hidden.data[i];
        } else {
            output.data[i] = 0.0f;
        }
    }
    return output;
}


Matrix matrix_sigmoid(Matrix Input){
    int total = Input.rows * Input.cols;
    Matrix Result = create_matrix(Input.rows,Input.cols);

    for (int i = 0; i < total; i++){
        float sigmoid_equation = 1.0f / (1.0f + expf(-Input.data[i]));
        Result.data[i] = sigmoid_equation;
    }
    return Result;
}

Matrix matrix_softmax(Matrix Input) { 
    float max = Input.data[0];
    int total = Input.rows * Input.cols;
    
    for (int i = 1; i < total; i++) {
        if (Input.data[i] > max) {
            max = Input.data[i];
        }
    }
    
    Matrix Result = create_matrix(Input.rows, Input.cols);
    float total_sum = 0.0f;
    
    for (int i = 0; i < total; i++) {
        float e = expf(Input.data[i] - max);
        Result.data[i] = e;
        total_sum += e;
    }
    
    for (int i = 0; i < total; i++) {
        Result.data[i] = Result.data[i] / total_sum;
    }
    
    return Result;
}

Matrix matrix_addition(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        FATAL_ERROR("Matrices must be the same size for addition. Exiting...");
    }
    
    Matrix Result = create_matrix(a.rows, a.cols);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            int index = (i * a.cols) + j;
            float output = a.data[index] + b.data[index];
            set_value(Result, i, j, output);
        }
    }
    return Result;
}

Matrix matrix_subtraction(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        FATAL_ERROR("Matrices must be the same size for subtraction. Exiting...");
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

Matrix matrix_scalar_multiply(Matrix Input, float scalar) {  
    Matrix Result = create_matrix(Input.rows, Input.cols);
    int total = Input.rows * Input.cols;

    for (int i = 0; i < total; i++){
        Result.data[i] = Input.data[i] * scalar;
    }
    return Result;
}

Matrix matrix_scalar_subtraction(Matrix Input, float scalar){
    Matrix Result = create_matrix(Input.rows, Input.cols);
    int total = Input.rows * Input.cols;

    for (int i = 0; i < total; i++){
        Result.data[i] = Input.data[i] - scalar;
    }
    return Result;
}

Matrix matrix_scalar_addition(Matrix Input, float scalar){
    Matrix Result = create_matrix(Input.rows, Input.cols);
    int total = Input.rows * Input.cols;

    for (int i = 0; i < total; i++){
        Result.data[i] = Input.data[i] + scalar;
    }
    return Result;
}


Matrix matrix_hadamard(Matrix a, Matrix b) { 
    if (a.rows != b.rows || a.cols != b.cols) {
        FATAL_ERROR("Matrices must be the same size for element-wise multiplication. Exiting...");
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

Matrix matrix_copy(Matrix Input) {
    Matrix copy = create_matrix(Input.rows, Input.cols);
    for (int i = 0; i < copy.rows; i++) {
        for (int j = 0; j < copy.cols; j++) {
            int index = (i * copy.cols) + j;
            float result = Input.data[index];
            set_value(copy, i, j, result);
        }
    }
    return copy;
}

Matrix matrix_normalize(Matrix Dataset) {
    int total = Dataset.rows * Dataset.cols;
    if (total <= 0) return create_matrix(0, 0);

    float max = Dataset.data[0];
    float min = Dataset.data[0];

    for (int i = 1; i < total; i++) {
        if (Dataset.data[i] > max) max = Dataset.data[i];
        if (Dataset.data[i] < min) min = Dataset.data[i];
    }

    float range = max - min;

    if (range == 0.0f) {
        ERROR("Range is 0! Returning a copy of Dataset");
        return matrix_copy(Dataset);
    }

    Matrix Temp = matrix_scalar_subtraction(Dataset, min);
    Matrix Result = matrix_scalar_multiply(Temp, (1.0f / range));

    free_matrix(Temp);
    
    return Result;
}

Matrix get_row(Matrix Input, int row_index) {
    if (row_index < 0 || row_index >= Input.rows) {
        FATAL_ERROR("Row index %d out of bounds", row_index);
    }

    Matrix Output = create_matrix(1, Input.cols);

    for (int i = 0; i < Input.cols; i++) {
        float val = get_value(Input, row_index, i);
        set_value(Output, 0, i, val);
    }

    return Output;
}

void matrix_swap_rows(Matrix Input, int row1, int row2) {
    if (row1 == row2) return;
    for (int col = 0; col < Input.cols; col++) {
        float temp = get_value(Input, row1, col);
        set_value(Input, row1, col, get_value(Input, row2, col));
        set_value(Input, row2, col, temp);
    }
}