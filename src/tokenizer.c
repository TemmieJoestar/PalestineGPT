#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tokenizer.h"
#include "error.h"

const char* VOCAB = "#"
                    "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789"
                    " .,!?;:'-\n";


int* encode(const char* text, int* output_length) {
    int len = strlen(text);
    *output_length = len;

    int* tokens = malloc(len * sizeof(int));
    if (tokens == NULL) {
        return NULL;
    }

    for (int i = 0; i < len; i++) {
        const char* pos = strchr(VOCAB, text[i]);
        
        if (pos != NULL) {
            int index = (int)(pos - VOCAB);
            tokens[i] = index;
        } 
        else {
            tokens[i] = 0;

            static int error_count = 0;
            if (error_count < 10) {
                WARNING("Character '%c' not found, replaced by '#' (0)", text[i]);
                error_count++;
            }
        }
    }

    return tokens;
}

char* decode(int* token_ids, int length) {
    char* result = malloc((length + 1) * sizeof(char));
    if (result == NULL) {
        return NULL;
    }

    result[length] = '\0';

    int vocab_size = strlen(VOCAB);

    for (int i = 0; i < length; i++) {
        
        if (token_ids[i] >= 0 && token_ids[i] < vocab_size) {
            result[i] = VOCAB[token_ids[i]];
        } 
        else {
            result[i] = '#';
        }
    }
    return result;
}