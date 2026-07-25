#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix.h"
#include "attention.h"
#include "error.h"

Matrix* compute_attention_scores(Matrix* Q, Matrix* K, int d_k){
    if (Q->data == NULL || K->data == NULL){
        FATAL_ERROR("Q or K data is NULL! Exiting...");
    }

    if (Q->rows == 0 || Q->cols == 0 || K->rows == 0 || K->cols == 0){
        FATAL_ERROR("Matrix dimensions cannot be 0. Exiting...");
    }

    if (Q->cols != K->cols){
        FATAL_ERROR("Dimension mismatch for dot product. Exiting...");
    }

    if (d_k <= 0){
        FATAL_ERROR("d_k must be greater than 0. Exiting...");
    }

    float scale = 1.0f / sqrtf((float)d_k);

    Matrix* Temp = create_matrix(Q->rows, K->rows, false);
    matrix_multiply(Q, K, Temp, false, true);
    
    Matrix* Result = create_matrix(Temp->rows, Temp->cols, false);
    matrix_scalar_multiply(Temp, scale, Result);

    free_matrix(Temp);
    return Result;
}

Matrix* compute_attention_weights(Matrix* AttentionScores){
    if (AttentionScores->data == NULL){
        FATAL_ERROR("Attention_Score is NULL. Exiting...");
    }

    if (AttentionScores->rows == 0 || AttentionScores->cols == 0){
        FATAL_ERROR("Matrix dimensions cannot be 0. Exiting...");
    }

    Matrix* AttentionWeights = create_matrix(AttentionScores->rows, AttentionScores->cols, false);
    matrix_softmax(AttentionScores, AttentionWeights);

    return AttentionWeights;
}

Matrix* apply_attention_weights(Matrix* AttentionWeights, Matrix* V){
    if (AttentionWeights->data == NULL || V->data == NULL){
        FATAL_ERROR("AttentionWeights or V data is NULL. Exiting...");
    }

    if (AttentionWeights->rows == 0 || AttentionWeights->cols == 0 || V->rows == 0 || V->cols == 0){
        FATAL_ERROR("Matrix dimensions cannot be 0. Exiting...");
    }

    Matrix* Result = create_matrix(AttentionWeights->rows, V->cols, false);
    matrix_multiply(AttentionWeights, V, Result, false, false);
    return Result;
}

Matrix* single_attention_forward(AttentionHead* Head, Matrix* Input){
    Matrix* Q = create_matrix(Input->rows, Head->Q_weights->cols, false);
    matrix_multiply(Input, Head->Q_weights, Q, false, false);

    Matrix* K = create_matrix(Input->rows, Head->K_weights->cols, false);
    matrix_multiply(Input, Head->K_weights, K, false, false);

    Matrix* V = create_matrix(Input->rows, Head->V_weights->cols, false);
    matrix_multiply(Input, Head->V_weights, V, false, false);

    Matrix* AttentionScores = compute_attention_scores(Q, K, Head->d_k);

    if (AttentionScores->data == NULL){
        FATAL_ERROR("Score data is NULL. Exiting...");
    }
    
    if (AttentionScores->rows == 0 || AttentionScores->cols == 0){
        FATAL_ERROR("Score dimensions cannot be 0. Exiting...");
    }

    Matrix* AttentionWeights = compute_attention_weights(AttentionScores);

    if (AttentionWeights->data == NULL){
        FATAL_ERROR("NormalizedAttentionScores data is NULL. Exiting...");
    }

    if (AttentionWeights->rows == 0 || AttentionWeights->cols == 0){
        FATAL_ERROR("AttentionWeights dimensions cannot be 0. Exiting...");
    }

    Matrix* ContextualMeaning = apply_attention_weights(AttentionWeights, V);

    if (ContextualMeaning->data == NULL){
        FATAL_ERROR("ContextualMeaning data is NULL. Exiting...");
    }

    if (ContextualMeaning->rows == 0 || ContextualMeaning->cols == 0){
        FATAL_ERROR("ContextualMeaning dimensions cannot be 0. Exiting...");
    }

    Matrix* UpdatedEmbedding = create_matrix(Input->rows, Input->cols, false);
    matrix_addition(Input, ContextualMeaning, UpdatedEmbedding);

    free_matrix(Q);
    free_matrix(K);
    free_matrix(V);
    free_matrix(AttentionScores);
    free_matrix(AttentionWeights);
    free_matrix(ContextualMeaning);

    return UpdatedEmbedding;
}

Matrix* multihead_attention_forward(MultiHeadAttention* MHA, Matrix* Input){
    if (Input->data == NULL){
        FATAL_ERROR("Input data is NULL. Exiting...");
    }
    Matrix* MultiHead = create_matrix(Input->rows, MHA->num_heads * MHA->Heads[0].d_k, false);

    for (int count = 0; count < MHA->num_heads; count++){
        Matrix* Q = create_matrix(Input->rows, MHA->Heads[count].Q_weights->cols, false);
        matrix_multiply(Input, MHA->Heads[count].Q_weights, Q, false, false);

        Matrix* K = create_matrix(Input->rows, MHA->Heads[count].K_weights->cols, false);
        matrix_multiply(Input, MHA->Heads[count].K_weights, K, false, false);

        Matrix* V = create_matrix(Input->rows, MHA->Heads[count].V_weights->cols, false);
        matrix_multiply(Input, MHA->Heads[count].V_weights, V, false, false);

        Matrix* AttentionScores = compute_attention_scores(Q, K, MHA->Heads[count].d_k);
        
        if (AttentionScores->data == NULL){
            FATAL_ERROR("AttentionScores data is NULL. Exiting...");
        }
        
        if (AttentionScores->rows == 0 || AttentionScores->cols == 0){
            FATAL_ERROR("AttentionScores dimensions cannot be 0. Exiting...");
        }
        
        Matrix* AttentionsWeights = compute_attention_weights(AttentionScores);

        if (AttentionsWeights->data == NULL){
            FATAL_ERROR("AttentionWeights data is NULL. Exiting...");
        }

        if (AttentionsWeights->rows == 0 || AttentionsWeights->cols == 0){
            FATAL_ERROR("AttentionWeights dimension cannot be 0. Exiting...");
        }

        Matrix* ContextualMeaning = apply_attention_weights(AttentionsWeights, V);

        if (ContextualMeaning->data == NULL){
            FATAL_ERROR("ContextualMeaning data is NULL. Exiting...");
        }

        if (ContextualMeaning->rows == 0 || ContextualMeaning->cols == 0){
            FATAL_ERROR("ContextualMeaning dimensions cannot be 0. Exiting...");
        }

        for (int i = 0; i < ContextualMeaning->rows; i++){
            for (int j = 0; j < ContextualMeaning->cols; j++){
                int target_col = (count * ContextualMeaning->cols) + j;
                set_value(MultiHead, i, target_col, get_value(ContextualMeaning, i, j));
            }
        }

        free_matrix(Q);
        free_matrix(K);
        free_matrix(V);
        free_matrix(AttentionScores);
        free_matrix(AttentionsWeights);
        free_matrix(ContextualMeaning);
    }

    Matrix* ProjectedOutput = create_matrix(MultiHead->rows, MHA->Output_Weights->cols, false);
    matrix_multiply(MultiHead, MHA->Output_Weights, ProjectedOutput, false, false);
    free_matrix(MultiHead);

    Matrix* FinalOutput = create_matrix(Input->rows, Input->cols, false);
    matrix_addition(Input, ProjectedOutput, FinalOutput);
    
    free_matrix(ProjectedOutput);
    return FinalOutput;
}