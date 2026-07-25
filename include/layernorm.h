#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "matrix.h"

/**
 * LayerNorm - Layer Structure
 * 
 *   Fields:
 * Scale  - Shape (1, d_model) // Learnable Scale
 * Bias   - Shape (1, d_model) // Learnable Bias
 */
typedef struct {
    Matrix* Scale; 
    Matrix* Bias;
} LayerNorm_Layer;

/**
 * LayerNorm - Cache Structure
 * 
 *   Fields:
 * Mean       - Shape (N, 1)       // Mean for each independent sample
 * Variance   - Shape (N, 1)       // Variance for each independent sample
 * NormInput  - Shape (N, d_model) // Normalized input (zero mean, unit variance)
 * Y          - Shape (N, d_model) // Scaled and Shifter output
 */
typedef struct {
    Matrix* Mean;
    Matrix* Variance;
    Matrix* NormInput;
    Matrix* Y;
} LayerNorm_Cache;

/**
 * LayerNorm - Gradients Structure
 * 
 *   Fields:
 * dX     - Shape (N, d_model) // Gradient flowing back to the input layer
 * dScale - Shape (1, d_model) // Gradient for Scale
 * dBias  - Shape (1, d_model) // Gradient for Bias
 */
typedef struct {
    Matrix* dX;
    Matrix* dScale;
    Matrix* dBias;
} LayerNorm_Grads;


/* ==========================================
 * LAYER NORM LAYER FUNCTIONS
 * ========================================== */

/**
 * init_layer_norm_layer - Allocate memory for the LayerNorm layer
 * 
 * @d_model: Dimension of the input and output
 * 
 *   Returns an initialized LayerNorm_Layer 
 * Note: Caller must call 'free_layer_norm_layer()' when done to avoid memory leaks
 */
LayerNorm_Layer init_layer_norm_layer(int d_model);

/**
 * free_layer_norm_layer - Free the allocated memory for LayerNorm_Layer
 * 
 * @Layer - Pointer to the LayerNorm_Layer struct
 */
void free_layer_norm_layer(LayerNorm_Layer *Layer);


/* ==========================================
 * LAYER NORM CACHE FUNCTIONS
 * ========================================== */

/**
 * init_layer_norm_cache - Allocate memory for the LayerNorm cache
 * 
 * @X: The input matrix (Shape: N, d_model)
 * 
 *   Returns an initialized LayerNorm_Cache
 * Note: Caller must call 'free_layer_norm_cache()' when done to avoid memory leaks
 */
LayerNorm_Cache init_layer_norm_cache(Matrix *X);

/**
 * free_layer_norm_cache - Free the allocated memory for LayerNorm_Cache
 * 
 * @Cache - Pointer to the LayerNorm_Cache struct
 */
void free_layer_norm_cache(LayerNorm_Cache *Cache);


/* ==========================================
 * LAYER NORM GRADIENTS FUNCTIONS
 * ========================================== */

/**
 * init_layer_norm_grads - Allocate memory for the LayerNorm gradients
 * 
 * @X: The input matrix (Shape: N, d_model)
 * 
 *   Returns an initialized LayerNorm_Grads
 * Note: Caller must call 'free_layer_norm_grads()' when done to avoid memory leaks
 */
LayerNorm_Grads init_layer_norm_grads(Matrix *X);

/**
 * free_layer_norm_grads - Free the allocated memory for LayerNorm_Grads
 *  
 * @Grads - Pointer to the LayerNorm_Grads struct
 */
void free_layer_norm_grads(LayerNorm_Grads *Grads);

void LayerNorm_Forward(Matrix *X, LayerNorm_Cache *Cache, LayerNorm_Layer *Layer, float epsilon);

void LayerNorm_Backward(Matrix *dY, Matrix *X, LayerNorm_Cache *Cache, LayerNorm_Layer *Layer, LayerNorm_Grads *Grads, float epsilon);

#endif