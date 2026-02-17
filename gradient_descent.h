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
 * Returns: Prediction matrix result of (Input × Weights)
 */
Matrix forward_pass(Matrix Weights, Matrix Input);

/**
 * backward_pass - Calculate the gradient of the MSE loss
 * @Prediction: The output from forward_pass
 * @Input: The original input matrix used for the pass
 * @Target: The ground truth labels/values
 * 
 * Returns: Gradient matrix used to update weights
 * Formula: 2 * (Prediction - Target) * Input^T
 * Note: Handles internal transposition and scaling; must free the result.
 */
Matrix backward_pass(Matrix Prediction, Matrix Input, Matrix Target);

/**
 * calculate_error - Compute the Mean Squared Error (MSE)
 * @Prediction: The predicted matrix
 * @Target: The target matrix
 * 
 * Returns: float representing the squared difference (Error = (pred - targ)^2)
 * Note: Currently optimized for single-value (1x1) output comparisons.
 */
float calculate_error(Matrix Prediction, Matrix Target);

/**
 * update_weights - Adjust weights using Gradient Descent
 * @Weights: Current weight matrix
 * @Gradient: Gradient calculated from backward_pass
 * @learning_rate: Step size for the update (alpha)
 * 
 * Returns: New Matrix with updated weights (Weights - (learning_rate * Gradient))
 * Note: Does not modify original; returns a new matrix that requires freeing.
 */
Matrix update_weights(Matrix Weights, Matrix Gradient, float learning_rate);

/**
 * train - Execute the training loop over multiple iterations
 * @Input: Training data matrix
 * @Weights: Initialized weight matrix
 * @Target: Target labels matrix
 * @steps: Number of iterations (epochs) to run
 * @learning_rate: Step size for weight updates
 * 
 * Logic:
 *  1. Performs forward pass to get prediction
 *  2. Calculates and prints MSE error
 *  3. Performs backward pass to find gradient
 *  4. Updates weights and cleans up intermediate memory
 * Note: Uses standard stdout to report progress.
 */
void train(Matrix Input, Matrix Weights, Matrix Target, int steps, float learning_rate);

#endif // GRADIENT_DESCENT_H