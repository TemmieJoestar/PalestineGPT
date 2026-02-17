#include <stdio.h>
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
    float pred = get_value(Prediction, 0, 0);
    float targ = get_value(Target, 0, 0);
    
    float diff = pred - targ;
    return diff * diff;
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
        
        printf("\nStep %d: Error = %.4f", i+1, error);
        
        free_matrix(Prediction);
        free_matrix(Gradient);
        free_matrix(current);
        current = New_weights;
    }
    free_matrix(current);
}