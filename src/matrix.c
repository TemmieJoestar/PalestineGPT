#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "matrix.h"
#include "error.h"

Matrix create_matrix(int rows, int cols, bool requires_grad) {
    Matrix m;
    m.rows = rows;
    m.cols = cols; 
    
    m.data = calloc(rows * cols, sizeof(float));
    if (m.data == NULL){
        FATAL_ERROR("Failed to allocate memory for DATA");
    }

    if (requires_grad){
        m.grad = calloc(rows * cols, sizeof(float));
        if (m.grad == NULL){
            FATAL_ERROR("Failed to allocate memory for GRAD");
        }
    } else {
        m.grad = NULL;
    }

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
    if (m.data != NULL) free(m.data);
    if (m.grad != NULL) free(m.grad);
}

void matrix_multiply(Matrix A, Matrix B, Matrix Result, bool is_transA, bool is_transB){
    int inner_A = is_transA ? A.rows : A.cols;
    int inner_B = is_transB ? B.cols : B.rows;

    if (inner_A != inner_B) {
        FATAL_ERROR("Inner dimensions must match for multiplication. Exiting...");
    }
    
    int inner_dim = inner_A; 

    if (!is_transA && !is_transB){
        // Branch 1: Standard Multiplication
        for (int i = 0; i < A.rows; i++){
            for (int j = 0; j < B.cols; j++){
                float sum = 0.0f;
                for (int k = 0; k < inner_dim; k++){
                    sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
                }
                Result.data[i * Result.cols + j] = sum;
            }
        } 
    }
    else if (is_transA && !is_transB){
        // Branch 2: A is Transposed, B is Normal
        for (int i = 0; i < A.cols; i++){
            for (int j = 0; j < B.cols; j++){
                float sum = 0.0f;
                for (int k = 0; k < inner_dim; k++){
                    sum += A.data[k * A.cols + i] * B.data[k * B.cols + j];
                }
                Result.data[i * Result.cols + j] = sum;
            }
        }
    }
    else if (!is_transA && is_transB){
        // Branch 3: A is Normal, B is Transposed
        for (int i = 0; i < A.rows; i++){
            for (int j = 0; j < B.rows; j++){
                float sum = 0.0f;
                for (int k = 0; k < inner_dim; k++){
                    sum += A.data[i * A.cols + k] * B.data[j * B.cols + k];
                }
                Result.data[i * Result.cols + j] = sum;
            }
        }
    }
    else if (is_transA && is_transB){
        // Branch 4: A is Transposed, B is Transposed
        for (int i = 0; i < A.cols; i++){
            for (int j = 0; j < B.rows; j++){
                float sum = 0.0f;
                for (int k = 0; k < inner_dim; k++){
                    sum += A.data[k * A.cols + i] * B.data[j * B.cols + k];
                }
                Result.data[i * Result.cols + j] = sum;
            }
        }
    }
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

void matrix_relu(Matrix Input, Matrix Output) { 
    int total = Input.rows * Input.cols;

    for (int i = 0; i < total; i++) {
        if (Input.data[i] < 0.0f) {
            Output.data[i] = 0.0f;
        } else {
            Output.data[i] = Input.data[i];
        }
    }
}

void matrix_relu_derivative(Matrix Hidden_raw, Matrix Gradient_Hidden, Matrix Output) {
    if (Hidden_raw.rows != Gradient_Hidden.rows || 
        Hidden_raw.cols != Gradient_Hidden.cols ||
        Hidden_raw.rows != Output.rows || 
        Hidden_raw.cols != Output.cols) {
        FATAL_ERROR("Matrices must have identical dimensions for ReLU derivative.");
    }

    int total = Hidden_raw.rows * Hidden_raw.cols;
    for (int i = 0; i < total; i++) {
        if (Hidden_raw.data[i] > 0.0f) {
            Output.data[i] = Gradient_Hidden.data[i];
        } else {
            Output.data[i] = 0.0f;
        }
    }
}


Matrix matrix_sigmoid(Matrix Input){
    int total = Input.rows * Input.cols;
    Matrix Result = create_matrix(Input.rows,Input.cols, false);

    for (int i = 0; i < total; i++){
        float sigmoid_equation = 1.0f / (1.0f + expf(-Input.data[i]));
        Result.data[i] = sigmoid_equation;
    }
    return Result;
}

Matrix matrix_softmax(Matrix Input) { 
    Matrix Result = create_matrix(Input.rows, Input.cols, false);
    
    for (int i = 0; i < Input.rows; i++) {
        float row_max = get_value(Input, i, 0);
        for (int j = 1; j < Input.cols; j++) {
            float val = get_value(Input, i, j);
            if (val > row_max) row_max = val;
        }
        
        float row_sum = 0.0f;
        for (int j = 0; j < Input.cols; j++) {
            float e = expf(get_value(Input, i, j) - row_max);
            set_value(Result, i, j, e);
            row_sum += e;
        }
        
        for (int j = 0; j < Input.cols; j++) {
            float normalized_val = get_value(Result, i, j) / row_sum;
            set_value(Result, i, j, normalized_val);
        }
    }
    
    return Result;
}

Matrix matrix_addition(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        FATAL_ERROR("Matrices must be the same size for addition. Exiting...");
    }
    
    Matrix Result = create_matrix(a.rows, a.cols, false);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            int index = (i * a.cols) + j;
            float output = a.data[index] + b.data[index];
            set_value(Result, i, j, output);
        }
    }
    return Result;
}

void matrix_sum_rows(Matrix Input, Matrix Output){
    if (Output.rows != 1 || Output.cols != Input.cols) {
        FATAL_ERROR("Output must be shape (1, Input.cols)");
    }
    for (int j = 0; j < Output.cols; j++){
        Output.data[j] = 0.0f;
    }

    for (int i = 0; i < Input.rows; i++){
        for (int j = 0; j < Input.cols; j++){
            Output.data[j] += Input.data[i * Input.cols + j];
        }
    }
}

void matrix_add_bias(Matrix Input, Matrix Bias, Matrix Output){
    if (Bias.rows != 1 || Bias.cols != Input.cols) {
        FATAL_ERROR("Bias must be a (1, cols) matrix matching the Input's columns!");
    }
    for (int i = 0; i < Input.rows; i++){ 
        for (int j = 0; j < Input.cols; j++){
            int index = (i * Input.cols) + j;
            Output.data[index] = Input.data[index] + Bias.data[j];
        }
    }
}


Matrix matrix_subtraction(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        FATAL_ERROR("Matrices must be the same size for subtraction. Exiting...");
    }

    Matrix c = create_matrix(a.rows, a.cols, false);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            int index = (i * a.cols) + j;
            float result = a.data[index] - b.data[index];
            set_value(c, i, j, result);
        }
    }
    return c;
}

Matrix matrix_scalar_multiply(Matrix Input, float scalar) {  
    Matrix Result = create_matrix(Input.rows, Input.cols, false);
    int total = Input.rows * Input.cols;

    for (int i = 0; i < total; i++){
        Result.data[i] = Input.data[i] * scalar;
    }
    return Result;
}

Matrix matrix_scalar_subtraction(Matrix Input, float scalar){
    Matrix Result = create_matrix(Input.rows, Input.cols, false);
    int total = Input.rows * Input.cols;

    for (int i = 0; i < total; i++){
        Result.data[i] = Input.data[i] - scalar;
    }
    return Result;
}

Matrix matrix_scalar_addition(Matrix Input, float scalar){
    Matrix Result = create_matrix(Input.rows, Input.cols, false);
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
    
    Matrix c = create_matrix(a.rows, a.cols, false);
    
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
    Matrix copy = create_matrix(Input.rows, Input.cols, false);
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
    if (total <= 0) return create_matrix(0, 0, false);

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

    Matrix Output = create_matrix(1, Input.cols, false);

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

void matrix_reset(Matrix Input) {
    if (Input.data == NULL) {
        return; 
    }
    memset(Input.data, 0, Input.rows * Input.cols * sizeof(float));
    
    if (Input.grad != NULL) {
        memset(Input.grad, 0, Input.rows * Input.cols * sizeof(float));
    }
}