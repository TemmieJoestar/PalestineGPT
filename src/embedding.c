#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"
#include "embedding.h"
#include "error.h"

EmbeddingLayer* create_embedding_layer(int vocab_size, int embedding_dim){
    EmbeddingLayer* layer = (EmbeddingLayer*)malloc(sizeof(EmbeddingLayer));

    layer->vocab_size = vocab_size;
    layer->embedding_dim = embedding_dim;
    layer->embeddings = create_matrix(vocab_size, embedding_dim, true);

    int total = (vocab_size * embedding_dim);

    for (int i = 0; i < total; i++){
        float random_val = ((float)rand() / (float)RAND_MAX) - 0.5f;
        layer->embeddings.data[i] = random_val;
    }
    return layer;
}

Matrix get_embedding(EmbeddingLayer* layer, int token_id) {
    if (token_id < 0 || token_id >= layer->vocab_size) {
        FATAL_ERROR("Token_id out of bounds.");
        WARNING("Returning error_matrix.");
        Matrix error_matrix = {0}; 
        return error_matrix;
    }
    return get_row(layer->embeddings, token_id);
}

Matrix embed_sequence(EmbeddingLayer* layer, int* token_ids, int length) {
    Matrix Seq_Matrix = create_matrix(length, layer->embedding_dim, false);

    for (int i = 0; i < length; i++) {
        Matrix Vector = get_embedding(layer, token_ids[i]);

        if (Vector.data == NULL) {
            ERROR("Token_id %d out of bounds at sequence index %d. Skipping.", token_ids[i],i);
            WARNING("Returning error_matrix.");
            free_matrix(Seq_Matrix);
            Matrix error_matrix = {0}; 
            return error_matrix;
        }
        memcpy(Seq_Matrix.data + (i * Seq_Matrix.cols), Vector.data, Vector.cols * sizeof(float));
        free_matrix(Vector);
    }
    return Seq_Matrix;
}

void free_embedding_layer(EmbeddingLayer* layer){
    if (layer == NULL){
        return;
    }
    free_matrix(layer->embeddings);
    free(layer);
}