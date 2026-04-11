#ifndef EMBEDDING_H
#define EMBEDDING_H

typedef struct {
    Matrix embeddings;  // vocab_size x embedding_dim
    int vocab_size;     // 'rows'
    int embedding_dim;  // 'cols'
} EmbeddingLayer;

/**
 * create_embedding_layer - Initialize embeddings with random values
 * @vocab_size: Number of tokens in vocabulary
 * @embedding_dim: Size of each embedding vector
 * 
 * Returns: Pointer to EmbeddingLayer
 * 
 * Initializes embeddings to small random values [-0.5, 0.5]
 * These will be trained later to capture meaning
 */
EmbeddingLayer* create_embedding_layer(int vocab_size, int embedding_dim);

/**
 * get_embedding - Extract embedding vector for one token
 * @layer: The embedding layer
 * @token_id: ID of token to lookup
 * 
 * Returns: 1 × embedding_dim matrix (row vector)
 * 
 * Example: get_embedding(layer, 5) returns row 5 of embeddings matrix
 */
Matrix get_embedding(EmbeddingLayer* layer, int token_id);

/**
 * embed_sequence - Convert array of token IDs to matrix of vectors
 * @layer: The embedding layer
 * @token_ids: Array of token IDs
 * @length: Number of tokens
 * 
 * Returns: length × embedding_dim matrix
 * 
 * Example:
 *   token_ids = [5, 12, 33]
 *   Returns 3×128 matrix where:
 *     Row 0 = embedding of token 5
 *     Row 1 = embedding of token 12
 *     Row 2 = embedding of token 33
 */
Matrix embed_sequence(EmbeddingLayer* layer, int* token_ids, int length);

/**
 * free_embedding_layer - Clean up all memory
 * @layer: Layer to free
 */
void free_embedding_layer(EmbeddingLayer* layer);

#endif 