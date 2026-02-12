#include <stdio.h>
#include <math.h>
#include "matrix.h"

void gradient_descent(float input, int steps);

int main() {
    gradient_descent(5.0, 10); 
    return 0;
}

void gradient_descent(float input, int steps) {
    float weight = 3.75;
    float target = 40.0; 
    float learning_rate = 0.01;
    
    printf(BOLD("\nGradient Descent Training\n"));
    printf("Starting weight: %.2f\n", weight);
    printf("Target: %.2f\n", target);
    printf("Input: %.2f\n", input);
    printf("Learning rate: %.2f\n\n", learning_rate);
    
    for (int step = 0; step < steps; step++) {
        printf(BOLD("Step: %d\n"), step + 1);
        
        // Forward pass
        float prediction = weight * input;
        float error = (prediction - target, 2) * (prediction - target, 2);
        
        printf("Prediction: %.2f\n", prediction);
        printf("Error: %.2f\n", error);
        
        // Backward pass 
        float gradient = 2.0 * (weight * input - target) * input;
        
        printf("Gradient (dError/dWeight): %.2f\n", gradient);
        
        // Initalize the new weight
        float weight_new = weight - learning_rate * gradient;
        
        printf("Weight: %.2f -> %.2f\n", weight, weight_new);
        
        // Update 'weight' for next loop
        weight = weight_new;
        
        printf("\n");
    }
    
    
    float final_prediction = weight * input;
    printf(GREEN_TEXT("Training Complete\n"));
    printf("Final weight: %.2f\n", weight);
    printf("Final prediction: %.2f (target: %.2f)\n", final_prediction, target);
    printf("Final error: %.2f\n", (final_prediction - target, 2) * (final_prediction - target, 2));
}