#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "attention.h"
#include "error.h"


int main() {
    int seq_len = 3;
    int d_k = 4;

    Matrix Q = create_matrix(seq_len, d_k);
    Matrix K = create_matrix(seq_len, d_k);

    for (int r = 0; r < seq_len; r++) {
        for (int c = 0; c < d_k; c++) {
            set_value(Q, r, c, (float)(r + c));
        }
    }

    for (int r = 0; r < seq_len; r++) {
        for (int c = 0; c < d_k; c++) {
            set_value(K, r, c, (float)(r == c ? 1.0 : 0.0));
        }
    }
    Matrix scores = compute_attention_scores(Q, K, d_k);

    printf("Scaled Attention Scores Matrix (%d x %d):\n", seq_len, seq_len);
    print_matrix(scores);


    free_matrix(Q);
    free_matrix(K);
    free_matrix(scores);

    return 0;
}

Matrix compute_attention_scores(Matrix Q, Matrix K, int d_k){
    if (Q.data == NULL || K.data == NULL){
        FATAL_ERROR("Q or K data is NULL! Exiting...");
    }

    if (Q.rows == 0 || Q.cols == 0 || K.rows == 0 || K.cols == 0){
        FATAL_ERROR("Matrix dimensions cannot be 0. Exiting...");
    }

    if (Q.cols != K.cols){
        FATAL_ERROR("Dimension mismatch for dot product. Exiting...");
    }

    if (d_k <= 0){
        FATAL_ERROR("d_k must be greater than 0. Exiting...");
    }

    float scale = 1.0f / sqrtf((float)d_k);

    Matrix K_t = matrix_transpose(K);
    Matrix Temp = matrix_multiply(Q,K_t);
    Matrix Result = matrix_scalar_multiply(Temp,scale);

    free_matrix(K_t);
    free_matrix(Temp);
    return Result;
}