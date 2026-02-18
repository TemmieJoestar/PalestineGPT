#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

Matrix forward_pass(Matrix Weights, Matrix Input) {
    Matrix Prediction = matrix_multiply(Input, Weights);
    return Prediction;
}


Matrix backward_pass(Matrix Prediction, Matrix Input, Matrix Target) {
    float diff = get_value(Prediction, 0, 0) - get_value(Target, 0, 0);
    
    Matrix Input_T = matrix_transpose(Input);
    Matrix Gradient = matrix_scalar_multiply(Input_T, 2.0f * diff);
    
    free_matrix(Input_T);
    return Gradient; 
}

float calculate_error(Matrix Prediction, Matrix Target) {
    if (Prediction.rows != Target.rows || Prediction.cols != Target.cols) {
        fprintf(stderr,RED_TEXT("Error: Matrices must be the same size to calculate error !\n"));
        exit(1);
    }

    int total = Prediction.rows * Prediction.cols;
    float sum = 0.0f;
    for (int i = 0; i < total; i++){
            float pred = Prediction.data[i];
            float targ = Target.data[i];
            float diff = (pred - targ) * (pred - targ);
            sum += diff;
    }
    return (sum/total);
}

Matrix update_weights(Matrix Weights, Matrix Gradient, float learning_rate) {
    Matrix Scaled_Grad = matrix_scalar_multiply(Gradient, learning_rate);
    Matrix New_Weights = matrix_subtraction(Weights, Scaled_Grad);
    
    free_matrix(Scaled_Grad);
    return New_Weights;
}

void train(Matrix Input, Matrix Weights, Matrix Target, int steps, float learning_rate) {
    
    
    Matrix current = matrix_copy(Weights);
    
    for (int i = 0; i < steps; i++) {
        Matrix Prediction = forward_pass(current, Input);
        float error = calculate_error(Prediction, Target);
        Matrix Gradient = backward_pass(Prediction, Input, Target);
        Matrix New_weights = update_weights(current, Gradient, learning_rate);
        
        printf("\nStep %d: Error = %.4f\n", i+1, error);
        
        free_matrix(Prediction);
        free_matrix(Gradient);
        free_matrix(current);
        current = New_weights;
    }
    free_matrix(current);
}

int main() {

Matrix Input = create_matrix(1, 3);
set_value(Input, 0, 0, 1.0f);
set_value(Input, 0, 1, 2.0f);
set_value(Input, 0, 2, 3.0f);


Matrix Initial_weight = create_matrix(3, 1);
set_value(Initial_weight, 0, 0, 0.1f);
set_value(Initial_weight, 1, 0, 0.1f);
set_value(Initial_weight, 2, 0, 0.1f);

Matrix Target = create_matrix(1, 1);
set_value(Target, 0, 0, 14.0f);
    
train(Input, Initial_weight, Target, 10, 0.01f);
//Matrix Prediction = matrix_multiply(Input, Initial_weight);
//calculate_error(Prediction,Target);
    
free_matrix(Input);
free_matrix(Initial_weight);
free_matrix(Target);
}