#include <stdlib.h>
#include <string.h>
#include "vocabulary.h"

Vocabulary* create_vocab(const char** token_list, int size) {
    Vocabulary* vocab = malloc(sizeof(Vocabulary));
    if (vocab == NULL) return NULL;

    vocab->vocab_size = size;

    vocab->tokens = malloc(size * sizeof(char*));
    if (vocab->tokens == NULL) {
        free(vocab); 
        return NULL;
    }

    for (int i = 0; i < size; i++) {
        vocab->tokens[i] = strdup(token_list[i]);
        
        if (vocab->tokens[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(vocab->tokens[j]);
            }
            free(vocab->tokens);
            free(vocab);
            return NULL;
        }
    }
    return vocab;
}

int get_token_id(Vocabulary* vocab, const char* token) {
    if (vocab == NULL || vocab->tokens == NULL || token == NULL) {
        return -1; 
    }

    for (int i = 0; i < vocab->vocab_size; i++) {
        if (vocab->tokens[i] != NULL) {
            if (strcmp(vocab->tokens[i], token) == 0) {
                return i;
            }
        }
    }
    return -1;
}

char* get_token_string(Vocabulary* vocab, int token_id) {
    if (vocab == NULL || vocab->tokens == NULL || token_id < 0 || token_id >= vocab->vocab_size) {
        return NULL;
    }

    if (vocab->tokens[token_id] == NULL) {
        return NULL;
    }

    char* copy = malloc(strlen(vocab->tokens[token_id]) + 1);
    if (copy == NULL) {
        return NULL;
    }

    strcpy(copy, vocab->tokens[token_id]);
    
    return copy;
}

void free_vocab(Vocabulary* vocab){
    if (vocab == NULL) {
        return;
    }
    // We free each individual token string
    for (int i = 0; i < vocab->vocab_size; i++){
        free(vocab->tokens[i]);
    }
    // We free the tokens array
    free(vocab->tokens);
    // We free the Vocabulary struct itself
    free(vocab);
}