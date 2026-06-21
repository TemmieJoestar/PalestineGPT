#include <math.h>
#include <stdlib.h>
#include "matrix.h"
#include "ffn.h"

float rand_float() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

FFN_Layer init_ffn_layer(int d_model, int d_ff) {
    FFN_Layer layer;

    layer.W1 = create_matrix(d_model, d_ff, true);
    layer.W2 = create_matrix(d_ff,d_model, true);
    layer.b1 = create_matrix(1, d_ff, true);
    layer.b2 = create_matrix(1, d_model, true);

    float scale_w1 = sqrt(2/(float)d_model);
    for (int i = 0; i < (d_model * d_ff); i++){
        layer.W1.data[i] = rand_float() * scale_w1;
    }

    float scale_w2 = sqrt(2/(float)d_ff);
    for (int i = 0; i < (d_ff * d_model); i++){
        layer.W2.data[i] = rand_float() * scale_w2;
    }

    for (int i = 0; i < d_ff; i++) layer.b1.data[i] =  0.0f;
    for (int i = 0; i < d_model; i++) layer.b2.data[i] = 0.0f;

    return layer;
}

void free_ffn_layer(FFN_Layer *Layer){
    free_matrix(Layer->W1);
    free_matrix(Layer->b1);
    free_matrix(Layer->W2);
    free_matrix(Layer->b2);
}

FFN_Cache init_ffn_cache(Matrix X, int d_model, int d_ff) {
    FFN_Cache Cache;
    
    Cache.x_up        = create_matrix(X.rows, d_ff, false);
    Cache.x_activated = create_matrix(X.rows, d_ff, false);
    Cache.x_down      = create_matrix(X.rows, d_model, false);

    return Cache;
}

void free_ffn_cache(FFN_Cache *Cache){
    free_matrix(Cache->x_up);
    free_matrix(Cache->x_activated);
    free_matrix(Cache->x_down);
}

FFN_Gradients init_ffn_grads(int d_model, int d_ff){
    FFN_Gradients Grads;
    Grads.dW1 = create_matrix(d_model, d_ff, false);
    Grads.db1 = create_matrix(1, d_ff, false);
    Grads.dW2 = create_matrix(d_ff, d_model, false);
    Grads.db2 = create_matrix(1, d_model, false);

    return Grads;
}

void free_ffn_grads(FFN_Gradients *Grads){
    free_matrix(Grads->dW1);
    free_matrix(Grads->db1);
    free_matrix(Grads->dW2);
    free_matrix(Grads->db2);
}

void FFN_Forward(FFN_Layer *layer, FFN_Cache *cache, Matrix X) {
    matrix_multiply(X, layer->W1, cache->x_up, false, false);
    matrix_add_bias(cache->x_up, layer->b1, cache->x_up);

    matrix_relu(cache->x_up, cache->x_activated);

    matrix_multiply(cache->x_activated, layer->W2, cache->x_down, false, false);
    matrix_add_bias(cache->x_down, layer->b2, cache->x_down);
}

void FFN_Backward(FFN_Layer *Layer, FFN_Cache *Cache, FFN_Gradients *Grads, Matrix X, Matrix d_output, Matrix dX){
    matrix_reset(Grads->dW1);
    matrix_reset(Grads->db1);
    matrix_reset(Grads->dW2);
    matrix_reset(Grads->db2);
    matrix_reset(dX);


    // Computing the gradients for the second layer
    matrix_multiply(Cache->x_activated, d_output, Grads->dW2, true, false); // dW_2 = R_1^T dZ_2
    matrix_sum_rows(d_output, Grads->db2); // db_2 = Sum of dZ_2

    Matrix dR1 = create_matrix(Cache->x_activated.rows, Cache->x_activated.cols, false);
    matrix_multiply(d_output, Layer->W2, dR1, false, true); // dR_1 = dZ_2 W_2^T
    
    // Computing the activation gradient
    matrix_relu_derivative(Cache->x_up, dR1, dR1);// dR_1 ReLU'(Z_1) // dR1 is used as dZ1 to save memory

    // Computing the gradients for the first layer
    matrix_multiply(X, dR1, Grads->dW1, true, false);// X^T dZ_1 // dR1 is used as dZ1 to save memory 
    matrix_sum_rows(dR1, Grads->db1); // db_1 = Sum of dZ_1
    
    matrix_multiply(dR1, Layer->W1, dX, false, true); //dZ_1 W_1^T
    free_matrix(dR1);
}