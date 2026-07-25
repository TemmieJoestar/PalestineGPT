#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "layernorm.h"
#include "error.h"

LayerNorm_Layer init_layer_norm_layer(int d_model) {
    LayerNorm_Layer Layer;

    Layer.Scale = create_matrix(1, d_model, true);
    Layer.Bias = create_matrix(1, d_model, true);

    // Initialize scale parameters (gamma) to 1.0f
    for (int i = 0; i < d_model; i++) {
        Layer.Scale->data[i] = 1.0f;
    }

    return Layer;
}

void free_layer_norm_layer(LayerNorm_Layer *Layer) {
    free_matrix(Layer->Scale);
    free_matrix(Layer->Bias);
}

LayerNorm_Cache init_layer_norm_cache(Matrix *X) {
    LayerNorm_Cache Cache;

    Cache.Mean = create_matrix(X->rows, 1, false);
    Cache.Variance = create_matrix(X->rows, 1, false);     
    Cache.NormInput = create_matrix(X->rows, X->cols, false); 
    Cache.Y = create_matrix(X->rows, X->cols, false);

    return Cache;
}

void free_layer_norm_cache(LayerNorm_Cache *Cache) {
    free_matrix(Cache->Mean);
    free_matrix(Cache->Variance);
    free_matrix(Cache->NormInput);
    free_matrix(Cache->Y);
}

LayerNorm_Grads init_layer_norm_grads(Matrix *X) {
    LayerNorm_Grads Grads;

    Grads.dX = create_matrix(X->rows, X->cols, false);
    Grads.dScale = create_matrix(1, X->cols, false);
    Grads.dBias = create_matrix(1, X->cols, false);

    return Grads;
}

void free_layer_norm_grads(LayerNorm_Grads *Grads) {
    free_matrix(Grads->dX);
    free_matrix(Grads->dScale);
    free_matrix(Grads->dBias);
}

void LayerNorm_Forward(Matrix* X, LayerNorm_Cache *Cache, LayerNorm_Layer *Layer, float epsilon) {
    // 1. Compute per-row mean: mu = (1 / d_model) * sum(x_i)
    for (int i = 0; i < X->rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < X->cols; j++) {
            int index = (i * X->cols + j);
            sum += X->data[index];   
        }
        Cache->Mean->data[i] = sum / X->cols;
    }

    // 2. Compute per-row variance: var = (1 / d_model) * sum((x_i - mu)^2)
    for (int i = 0; i < X->rows; i++) {
        float mean = Cache->Mean->data[i];
        float sum = 0.0f;

        for (int j = 0; j < X->cols; j++) {
            int index = (i * X->cols + j);
            float diff = X->data[index] - mean;
            sum += diff * diff;
        }
        Cache->Variance->data[i] = sum / X->cols;
    }

    // 3. Normalize inputs and apply learnable scale (gamma) and bias (beta)
    for (int i = 0; i < X->rows; i++) {
        float variance = Cache->Variance->data[i];
        float inv_deno = 1.0f / sqrtf(variance + epsilon);
        float mean = Cache->Mean->data[i];

        for (int j = 0; j < X->cols; j++) {
            int index = (i * X->cols + j);
            float numerator = X->data[index] - mean;
            float normalized = numerator * inv_deno;
            
            Cache->NormInput->data[index] = normalized;

            // Y = gamma * x_hat + beta
            float y = (Layer->Scale->data[j] * normalized) + Layer->Bias->data[j];
            Cache->Y->data[index] = y;
        }
    }
}

void LayerNorm_Backward(Matrix *dY, Matrix *X, LayerNorm_Cache *Cache, LayerNorm_Layer *Layer, LayerNorm_Grads *Grads, float epsilon) {
    // Zero out initial accumulated gradients
    matrix_reset(Grads->dBias);
    matrix_reset(Grads->dScale);
    matrix_reset(Grads->dX);
    
    for (int i = 0; i < X->rows; i++) {
        float variance = Cache->Variance->data[i];
        float mean = Cache->Mean->data[i];
        
        float inv_std = 1.0f / sqrtf(variance + epsilon);
        int row_offset = i * X->cols;
        
        float sum_dL_dxhat = 0.0f;       
        float dL_dvar = 0.0f;         
        
        // Accumulate row terms for partial derivatives wrt mean and variance
        for (int k = 0; k < X->cols; k++) {
            int index = row_offset + k;

            // dL/dxhat = dL/dY * gamma
            float dL_dxhat = dY->data[index] * Layer->Scale->data[k]; 
            
            // dxhat/dvar = -0.5 * (x - mean) * (var + eps)^(-1.5)
            float dxhat_dvar_k = -0.5f * (X->data[index] - mean) * inv_std * inv_std * inv_std;

            sum_dL_dxhat += dL_dxhat;
            dL_dvar += dL_dxhat * dxhat_dvar_k;
        }
        
        // Partial derivative wrt mean component: sum(dL/dxhat * dxhat/dmu)
        float dL_dmu_part1 = sum_dL_dxhat * -inv_std; 
        float dmu_dx = 1.0f / (float)X->cols;
        float dvar_scale = 2.0f / (float)X->cols;

        // Compute input gradient (dX) and parameter gradients (dScale, dBias)
        for (int j = 0; j < X->cols; j++) {
            int index = row_offset + j;

            float dvar_dx_j = (X->data[index] - mean) * dvar_scale;
            float dL_dxhat = dY->data[index] * Layer->Scale->data[j];
            
            // Combine direct, mean, and variance gradient paths via chain rule
            float grad_direct = dL_dxhat * inv_std;
            float final_grad_mean = dL_dmu_part1 * dmu_dx;
            float final_grad_var = dL_dvar * dvar_dx_j;
            
            Grads->dX->data[index] += grad_direct + final_grad_mean + final_grad_var;

            // Accumulate parameter gradients across all samples
            Grads->dBias->data[j] += dY->data[index];
            Grads->dScale->data[j] += dY->data[index] * Cache->NormInput->data[index];
        }
    }
}