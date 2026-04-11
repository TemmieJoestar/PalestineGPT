#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tokenizer.h"


const char* VOCAB = "#"
                    "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789"
                    " .,!?;:'-\n";


int* encode(const char* text, int* output_length) {
    int len = strlen(text);
    *output_length = len;

    // Allocate memory for the token array
    int* tokens = malloc(len * sizeof(int));
    if (tokens == NULL) {
        return NULL;
    }

    for (int i = 0; i < len; i++) {
        // Locate the character within the predefined VOCAB string
        const char* pos = strchr(VOCAB, text[i]);
        
        if (pos != NULL) {
            // Calculate the relative index using pointer arithmetic
            int index = (int)(pos - VOCAB);
            tokens[i] = index;
        } 
        else {
            // Fallback for characters not found in VOCAB:
            // Map to index 0 (assumed to be the '#' unknown token)
            tokens[i] = 0;

            // Log a limited number of errors to prevent console pollution during large-scale processing
            static int error_count = 0;
            if (error_count < 10) {
                printf(RED_TEXT("Character '%c' not found, replaced by '#' (0)\n"), text[i]);
                error_count++;
            }
        }
    }

    return tokens;
}

char* decode(int* token_ids, int length) {
    // Allocate memory for the string plus the null terminator
    char* result = malloc((length + 1) * sizeof(char));
    if (result == NULL) {
        return NULL;
    }

    // Set the null terminator at the end of the string
    result[length] = '\0';

    int vocab_size = strlen(VOCAB);

    // Iterate through the token array to map each ID back to its character
    for (int i = 0; i < length; i++) {
        
        // Defensive check: ensure the token ID is within the valid range of the VOCAB array
        if (token_ids[i] >= 0 && token_ids[i] < vocab_size) {
            // Map the token ID directly to the character at that index in VOCAB
            result[i] = VOCAB[token_ids[i]];
        } 
        else {
            // Fallback for invalid or out-of-bounds token IDs
            result[i] = '#';
        }
    }
    return result;
}