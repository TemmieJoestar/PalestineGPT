#ifndef FFN_H
#define FFN_H

#include "matrix.h"
/**
 * FFN - Layer Structure
 * Stores the learnable parameters for a Feed-Forward Network.
 *
 * Fields:
 * W1 - Shape (d_model, d_ff)    // Expansion weights
 * b1 - Shape (1, d_ff)          // Bias for expansion
 * W2 - Shape (d_ff, d_model)    // Contraction weights
 * b2 - Shape (1, d_model)       // Bias for output projection
 */
typedef struct {
    Matrix* W1; 
    Matrix* b1;
    Matrix* W2;
    Matrix* b2;
} FFN_Layer;

/**
 * FFN - Cache Structure
 * Stores intermediate activations for backpropagation.
 *
 * Fields:
 * x_up        - Shape (N, d_ff)    // Z_1: Linear projection (Expansion)
 * x_activated - Shape (N, d_ff)    // R_1: Post-activation (e.g., ReLU(Z_1))
 * x_down      - Shape (N, d_model) // R_2: Output projection (Contraction)
 */
typedef struct {
    Matrix* x_up;        
    Matrix* x_activated; 
    Matrix* x_down;      
} FFN_Cache;

/**
 * FFN - Gradients Structure
 * Stores computed gradients for the backpropagation pass.
 *
 * Fields:
 * dW1 - Shape (d_model, d_ff)    // Gradient of W1
 * db1 - Shape (1, d_ff)          // Gradient of b1
 * dW2 - Shape (d_ff, d_model)    // Gradient of W2
 * db2 - Shape (1, d_model)       // Gradient of b2
 *
 * Note: Must call free_ffn_gradients() when done to avoid memory leaks.
 */
typedef struct {
    Matrix* dW1;
    Matrix* db1;
    Matrix* dW2;
    Matrix* db2;
} FFN_Gradients;

/**
 * rand_float - Generate a random float
 * Returns: A random float value between -1.0 and 1.0
 */
float rand_float();

/* ==========================================
 * FFN LAYER FUNCTIONS
 * ========================================== */

/**
 * init_ffn_layer - Allocate and initialize weights and biases
 * * @d_model: Dimension of the input and output
 * @d_ff:    Dimension of the hidden expansion layer
 * Returns: An FFN_Layer with weights initialized
 * Note: Must call 'free_ffn_layer()' when done to avoid memory leaks
 */
FFN_Layer init_ffn_layer(int d_model, int d_ff);

/**
 * free_ffn_layer - Free the allocated memory for an FFN_Layer
 * * @Layer: Pointer to the FFN_Layer struct
 */
void free_ffn_layer(FFN_Layer *Layer);

/* ==========================================
 * FFN CACHE FUNCTIONS
 * ========================================== */

/**
 * init_ffn_cache - Allocate memory for FFN intermediate computations
 * @X:       The input matrix
 * @d_model: Dimension of the input (needed for X_down size)
 * @d_ff:    Dimension of the hidden expansion (needed for X_up size)
 * Returns: An FFN_Cache with matrices sized for the forward pass
 * Note: Must call 'free_ffn_cache()' when done to avoid memory leaks
 */
FFN_Cache init_ffn_cache(Matrix* X, int d_model, int d_ff);

/**
 * free_ffn_cache - Free the allocated memory for an FFN_Cache
 * * @Cache: Pointer to the FFN_Cache struct
 */
void free_ffn_cache(FFN_Cache *Cache);

/* ==========================================
 * FFN GRADIENTS FUNCTIONS
 * ========================================== */

/**
 * init_ffn_gradients - Allocate memory for FFN gradients
 * * Returns: An FFN_Gradients struct with empty (allocated) matrices
 * Note: Must call 'free_ffn_gradients()' when done to avoid memory leaks
 */
FFN_Gradients init_ffn_grads(int d_model, int d_ff);

/**
 * free_ffn_gradients - Free the allocated memory for FFN_Gradients
 * * @Grads: Pointer to the FFN_Gradients struct
 */
void free_ffn_grads(FFN_Gradients *Grads);

/**
 * FFN_Forward - Perform the forward pass of the FFN
 * * @Layer: The layer containing weights and biases
 * @Cache: The cache to store intermediate X_up, X_activated, and X_down
 * @X:     The input matrix
 */
void FFN_Forward(FFN_Layer *Layer, FFN_Cache *Cache, Matrix* X);

/**
 * FFN_Backward - Compute gradients for the FFN layer
 * * @Layer:    The layer containing weights
 * @Cache:    The cache containing intermediate forward pass results
 * @X:        The original input matrix
 * @d_output: The gradient of the loss with respect to the output (dL/dR2)
 * @dX:       Output matrix to store the gradient of the loss w.r.t input (dL/dX)
 * Note: Updates the weight gradients and computes dX for the previous layer
 */
void FFN_Backward(FFN_Layer *Layer, FFN_Cache *Cache, FFN_Gradients *Grads, Matrix* X, Matrix* d_output, Matrix* dX);

#endif