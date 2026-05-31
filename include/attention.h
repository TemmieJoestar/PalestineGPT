#ifndef ATTENTION_H
#define ATTENTION_H

#include "matrix.h"

// Text formatting macros for terminal output
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[92;1m" x "\033[0m"
#define YELLOW_TEXT(x) "\033[33;1m" x "\033[0m"
#define CYAN_TEXT(x) "\033[36;1m" x "\033[0m"
#define MAGENTA_TEXT(x) "\033[35;1m" x "\033[0m"
#define BOLD(x) "\033[1m" x "\033[0m"

typedef struct {
    Matrix Q_weights;    // Query projection [d_model x d_k]
    Matrix K_weights;    // Key projection   [d_model x d_k]
    Matrix V_weights;    // Value projection [d_model x d_k]
    int d_model;         // Input embedding dimension
    int d_k;             // Dimension per head (d_model / num_heads)
} AttentionHead;

typedef struct {
    AttentionHead* Heads;  // Array of attention heads
    int num_heads;
    Matrix Output_Weights; // Final projection [ (num_heads * d_k) x d_model ]
} MultiHeadAttention;

/**
 * single_attention_forward - Full pass for one head including Residual Connection.
 * @Head: Pointer to the AttentionHead configuration.
 * @Input: Input embeddings [seq_len x d_model].
 * 
 * NOTE: Returns (Input + Attention_Output). Caller must free the returned Matrix.
 */
Matrix single_attention_forward(AttentionHead* Head, Matrix Input);

/**
 * multihead_attention_forward - Full pass for multiple heads including Residual Connection.
 * @MHA: Pointer to the MultiHeadAttention configuration.
 * @Input: Input embeddings [seq_len x d_model].
 * 
 * NOTE: Concatenates all heads, projects via Output_Weights, and adds residual. 
 * Caller must free the returned Matrix.
 */
Matrix multihead_attention_forward(MultiHeadAttention* MHA, Matrix Input);

/**
 * compute_attention_scores - Calculate scaled dot-product: (Q · K^T) / sqrt(d_k).
 * @Q: Query matrix [seq_len x d_k].
 * @K: Key matrix [seq_len x d_k].
 * @d_k: Scaling factor (dimension of keys).
 * 
 * Returns: Raw attention scores [seq_len x seq_len].
 */
Matrix compute_attention_scores(Matrix Q, Matrix K, int d_k);

/**
 * compute_attention_weights - Applies Softmax normalization to attention scores.
 * @AttentionScores: Raw scores [seq_len x seq_len].
 * 
 * Returns: Probability distribution weights [seq_len x seq_len] where rows sum to 1.
 */
Matrix compute_attention_weights(Matrix AttentionScores);

/**
 * apply_attention_weights - Weighted sum of values: Weights · V.
 * @AttentionWeights: Normalized weights [seq_len x seq_len].
 * @V: Value matrix [seq_len x d_k].
 * 
 * Returns: Contextualized embeddings [seq_len x d_k].
 */
Matrix apply_attention_weights(Matrix AttentionWeights, Matrix V);

#endif