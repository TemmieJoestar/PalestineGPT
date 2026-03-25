#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

#define EPSILON 1e-7f // Value close to 0 

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

void backward_pass_2layer(
    Matrix Output, Matrix Hidden, Matrix Hidden_raw, Matrix Input, Matrix Target, Matrix Weights_2, 
    Matrix *Grad_W1_out, Matrix *Grad_W2_out, Matrix *Grad_B1_out, Matrix *Grad_B2_out){   
        // Compute error
        float diff = get_value(Output,0,0) - get_value(Target,0,0);
        Matrix Gradient_B2 = create_matrix(1, 1);
        set_value(Gradient_B2, 0, 0, 2.0f * diff);

        
        // Gradient for Weights_2
        Matrix Hidden_T = matrix_transpose(Hidden);
        Matrix Gradient_W2 = matrix_scalar_multiply(Hidden_T, (2.0f*diff));
        free_matrix(Hidden_T);
        
        // Flow gradient back through Weights_2
        Matrix Weights_2_T = matrix_transpose(Weights_2);
        Matrix Gradient_Hidden = matrix_scalar_multiply(Weights_2_T, 2.0f*diff);
        Matrix Gradient_Hidden_masked = matrix_relu_derivative(Hidden_raw,Gradient_Hidden);
        Matrix Gradient_B1 = matrix_copy(Gradient_Hidden_masked);
        free_matrix(Weights_2_T);
        free_matrix(Gradient_Hidden);
        
        // Gradient for Weights_1
        Matrix Input_T = matrix_transpose(Input);
        Matrix Gradient_W1 = matrix_multiply(Input_T, Gradient_Hidden_masked);
        free_matrix(Input_T);
        free_matrix(Gradient_Hidden_masked);
        
        // Return via pointers
        *Grad_W1_out = Gradient_W1; 
        *Grad_W2_out = Gradient_W2;
        *Grad_B1_out = Gradient_B1;
        *Grad_B2_out = Gradient_B2;
        *Grad_B1_out = Gradient_B1;
        *Grad_B2_out = Gradient_B2;
    }

float mean_squared_error(Matrix Prediction, Matrix Target) {
    if (Prediction.rows != Target.rows || Prediction.cols != Target.cols) {
        fprintf(stderr,RED_TEXT("Error: Matrices must be the same size to Mean Squared Error (M.E.S) !\n"));
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

float cross_entropy_loss(Matrix Prediction, Matrix Target){
    if (Prediction.rows != Target.rows || Prediction.cols != Target.cols) {
        fprintf(stderr,RED_TEXT("Error: Matrices must be the same size for Cross Entropy Loss !\n"));
        exit(1);
    }
    int total = Prediction.rows * Prediction.cols;
    float sum = 0.0f;
    for (int i = 0; i < total; i++){
        float pred = Prediction.data[i];
        float targ = Target.data[i];
        float diff = (targ * logf(pred + EPSILON));
        sum += diff;
    }
    return (-sum/total);
}

Matrix update_weights(Matrix Weights, Matrix Gradient, float learning_rate) {
    Matrix Scaled_Grad = matrix_scalar_multiply(Gradient, learning_rate);
    Matrix New_Weights = matrix_subtraction(Weights, Scaled_Grad);
    
    free_matrix(Scaled_Grad);
    return New_Weights;
}

Matrix update_weights_momentum(Matrix Weight, Matrix Gradient, Matrix *Velocity, float learning_rate, float momentum) {
    Matrix Scaled_Velocity = matrix_scalar_multiply(*Velocity, momentum);
    Matrix New_Velocity = matrix_addition(Scaled_Velocity, Gradient);
    
    Matrix Update = matrix_scalar_multiply(New_Velocity, learning_rate);
    Matrix New_Weights = matrix_subtraction(Weight, Update);
    
    free_matrix(Scaled_Velocity);
    free_matrix(Update);
    free_matrix(*Velocity);
    
    *Velocity = New_Velocity;
    return New_Weights;
}


Matrix forward_pass_2layer(Matrix Input, Matrix Weights_1, Matrix Weights_2, Matrix Bias_1, Matrix Bias_2, Matrix *Hidden_out, Matrix *Hidden_raw_out){
    Matrix Hidden_raw_temp = matrix_multiply(Input, Weights_1);
    Matrix Hidden_raw = matrix_addition(Hidden_raw_temp,Bias_1);
    Matrix Hidden = matrix_relu(Hidden_raw);
    free_matrix(Hidden_raw_temp);
    
    Matrix Output_raw_temp = matrix_multiply(Hidden, Weights_2); 
    Matrix Output_raw = matrix_addition(Output_raw_temp,Bias_2);
    Matrix Output = matrix_sigmoid(Output_raw);
    free_matrix(Output_raw_temp);
    free_matrix(Output_raw);
    
    *Hidden_out = Hidden;
    *Hidden_raw_out = Hidden_raw;
    
    return Output;
}



void train(Matrix Input, Matrix Weights, Matrix Target, int steps, float learning_rate) {
    
    
    Matrix current = matrix_copy(Weights);
    
    for (int i = 0; i < steps; i++) {
        Matrix Prediction = forward_pass(current, Input);
        float error = mean_squared_error(Prediction, Target);
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