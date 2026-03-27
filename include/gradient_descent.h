// gradient_descent.h
#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include "matrix.h"

/* ============================================================================
 * NEURAL NETWORK OPERATIONS (TRAINING & INFERENCE)
 * ============================================================================ 
*/

/**
 * forward_pass - Compute the linear transformation (Inference)
 * @Weights: Matrix of model weights (N x 1)
 * @Input: Matrix of input features (1 x N)
 * 
 *  Returns: Prediction matrix result of (Input × Weights)
 * Note: Returns a new matrix that requires freeing.
 */
Matrix forward_pass(Matrix Weights, Matrix Input);

/**
 * forward_pass_2layer - Compute a 2-layer Neural Network inference
 * @Input: Matrix of input features
 * @Weights_1: Weights for the hidden layer
 * @Weights_2: Weights for the output layer
 * @Bias_1: Prevent 0's values for Hidden_raw
 * @Bias_2: Prevent 0's values for Output_raw
 * @*Hidden_out: Store the Hidden_out value
 * @*Hidden_raw_out: Sotre the Hidden_raw value
 * 
 *  Returns: Final output matrix after ReLU (hidden) and Sigmoid (output)
 * Note: Frees internal intermediate matrices; caller must free the returned result.
 */
Matrix forward_pass_2layer(Matrix Input, Matrix Weights_1, Matrix Weights_2, Matrix Bias_1, Matrix Bias_2, Matrix *Hidden_out, Matrix *Hidden_raw_out );

/**
 * forward_pass_2layer - Compute a 2-layer Neural Network inference (Multi-class)
 * @Input: Matrix of input features (1 x N)
 * @Weights_1: Weights for the hidden layer
 * @Weights_2: Weights for the output layer
 * @Bias_1: Bias vector for the hidden layer
 * @Bias_2: Bias vector for the output layer
 * @*Hidden_out: OUTPUT - Store the activated Hidden layer (caller must free)
 * @*Hidden_raw_out: OUTPUT - Store the pre-activation Hidden layer (caller must free)
 *  Returns: Final output matrix after ReLU (hidden) and Softmax (output)
 * Note: Uses Softmax to ensure output values represent a probability 
 * distribution (sum to 1.0). Frees internal intermediate matrices; 
 * caller must free the returned result and the two output pointers.
 */
Matrix forward_pass_2layer_softmax(Matrix Input, Matrix Weights_1, Matrix Weights_2, Matrix Bias_1, Matrix Bias_2, Matrix *Hidden_out, Matrix *Hidden_raw_out
);
/**
 * backward_pass - Calculate the gradient of the MSE loss
 * @Prediction: The output from forward_pass
 * @Input: The original input matrix used for the pass
 * @Target: The ground truth labels/values
 * 
 *  Returns: Gradient matrix used to update weights
 * Formula: 2 -> (Prediction - Target) * Input^T
 * Note: Must free the returned gradient matrix.
 */
Matrix backward_pass(Matrix Prediction, Matrix Input, Matrix Target);

/**
 * backward_pass_2layer - Compute gradients for a 2-layer neural network
 * @Output: Final prediction (after sigmoid)
 * @Hidden: Hidden layer values (after ReLU)
 * @Hidden_raw: Hidden layer values (before ReLU)
 * @Input: Original input data
 * @Target: Ground truth labels
 * @Weights_2: Output layer weights
 * @Grad_W1_out: OUTPUT - Gradient for Weights_1 (caller must free)
 * @Grad_W2_out: OUTPUT - Gradient for Weights_2 (caller must free)
 * 
 * Computes gradients using backpropagation through both layers.
 * Uses chain rule to flow error from output back through ReLU to input layer.
 */
void backward_pass_2layer(
    Matrix Output, Matrix Hidden, Matrix Hidden_raw, Matrix Input,
    Matrix Target, Matrix Weights_2, 
    Matrix *Grad_W1_out, Matrix *Grad_W2_out,
    Matrix *Grad_B1_out, Matrix *Grad_B2_out  // NEW!
);

/**
 * mean_squared_error - Compute the Mean Squared Error (MSE)
 * @Prediction: The predicted matrix
 * @Target: The target matrix
 * 
 *  Returns: Float representing (1/n) × Σ(prediction[i] - target[i])²
 */
float mean_squared_error(Matrix Prediction, Matrix Target);

/**
 * update_weights - Adjust weights using Gradient Descent
 * @Weights: Current weight matrix
 * @Gradient: Gradient calculated from backward_pass
 * @learning_rate: Step size for the update (alpha)
 * 
 *  Returns: New Matrix with updated weights (Weights - (learning_rate * Gradient))
 * Note: Does not modify original; returns a new matrix that requires freeing.
 */
Matrix update_weights(Matrix Weights, Matrix Gradient, float learning_rate);


/**
 * update_weights_momentum - Adjust weights using Gradient Descent with Momentum
 * @Weight: Current weight matrix
 * @Gradient: Gradient calculated from backward_pass
 * @Velocity: Pointer to the velocity matrix (updated in-place)
 * @learning_rate: Step size for the update (alpha)
 * @momentum: Friction/decay constant for velocity, usually 0.9
 *
 * Return: New Matrix with updated weights.
 * Note: Does not modify original Weights matrix; returns a new matrix 
 * that requires freeing. Updates *Velocity in-place.
 */
Matrix update_weights_momentum(Matrix Weight, Matrix Gradient, Matrix *Velocity, float learning_rate, float momentum);


/**
 * cross_entropy_loss - Measure classification performance
 * @Prediction: The predicted Matrix (probabilities)
 * @Target: The target Matrix (one-hot or labels)
 * 
 *  Returns: Float representing -(1/n) × Σ(target[i] × log(prediction[i] + ε))
 * Note: Inputs must be normalized (e.g., via Softmax).
 */
float cross_entropy_loss(Matrix Prediction, Matrix Target);

/**
 * train - Execute the training loop over multiple iterations
 * @Input: Training data matrix
 * @Weights: Initialized weight matrix
 * @Target: Target labels matrix
 * @steps: Number of iterations (epochs)
 * @learning_rate: Step size for weight updates
 *  Logic:
 *   1. Performs forward pass
 *   2. Calculates and prints error
 *   3. Performs backward pass for gradient
 *   4. Updates weights and cleans up all intermediate memory
 */
void train(Matrix Input, Matrix Weights, Matrix Target, int steps, float learning_rate);


#endif // GRADIENT_DESCENT_H