#include <stdio.h>
#include <math.h>
#include "matrix.h"

typedef struct {
    float weight;
    float prediction;
    float error;
    float gradient;
} TrainingStep;

TrainingStep forward_pass(float weight, float input, float target) {
    TrainingStep step;
    step.weight = weight;
    step.prediction = weight * input;
    step.error = (step.prediction - target) * (step.prediction - target);
    return step;
}

float backward_pass(TrainingStep step, float input, float target) {
     return 2.0f * (step.prediction - target) * input;
}

float update_weight(float weight, float gradient, float learning_rate) {
    float weight_new = weight - learning_rate * gradient; 
    return weight_new;
}

void train(float initial_weight, float input, float target, int steps, float learning_rate) {
    
    float weight = initial_weight;

    printf(BOLD("\nTraining Started \n"));
    printf("Initial weight: %.2f\n", weight);
    printf("Target: %.2f, Input: %.2f\n", target, input);
    printf("Learning rate: %.2f\n\n", learning_rate);

    for (int i = 0; i < steps; i++){
        TrainingStep step = forward_pass(weight,input,target);

        float gradient = backward_pass(step,input,target);

        float new_weight = update_weight(weight,gradient,learning_rate);
        
        // Check if it's converged, if so program exit the loop, print which step converged.
        if (step.error < 0.0001f) {
            printf(GREEN_TEXT("\nConverged at step %d!\n"), i + 1);
            break;
        }

        // Print the value of "Weight" "Prediction" "Error" and "New Weight" after each step.
        printf(BOLD("\nStep %d: \n"),i+1);
        printf(BOLD("Weight -> %.2f\n"),weight);
        printf(BOLD("Prediction -> %.2f\n"),step.prediction);
        printf(BOLD("Error -> %.2f\n"),step.error);
        printf(BOLD("New Weight -> %.2f\n"),new_weight);

        weight = new_weight; // Initialize the new value of weight
    }
    printf(GREEN_TEXT("\nTraining Complete \n"));
    printf("Final weight: %.2f\n", weight);
    printf("Final prediction: %.2f (target: %.2f)\n", weight * input, target);
    printf("Converged: %s\n", (fabs(weight * input - target) < 0.01) ? "YES" : "NO");

}

// Testing
int main() {
    train(3.75, 5.0, 20.0, 10, 0.01); 
    return 0;
}