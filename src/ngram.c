#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "matrix.h"
#include "ngram.h"

BigramModel* train_bigram_model(const char* text) {
    BigramModel* Model = calloc(1, sizeof(BigramModel));
    char* copy = strdup(text);
    char* current_token = strtok(copy, " ");
    char* next_token = strtok(NULL, " ");

    // Error handling
    if (Model == NULL){
        fprintf(stderr,("ERROR: Model is NULL."));
        fprintf(stderr, YELLOW_TEXT("WARNING: Exiting..."));
        free(copy);
        exit(1);
    }

    while (current_token != NULL){
        BigramEntry* Entry = find_entry(Model, current_token);
        
        if (Entry == NULL){
            Model->num_entries++;
            Model->entries = realloc(Model->entries, Model->num_entries * sizeof(BigramEntry));

            Entry = &Model->entries[Model->num_entries - 1];
            Entry->word = strdup(current_token);
            Entry->next_words = NULL;
            Entry->counts = NULL;
            Entry->num_next = 0;
        }
        if (next_token != NULL) {
            add_next_word(Entry, next_token);
        }

        current_token = next_token;
        next_token = strtok(NULL, " ");
    }

    free(copy);
    return Model;
}

char* generate_text(BigramModel* Model, const char* start_word, int num_words) {
    BigramEntry* current_entry = find_entry(Model, start_word);
    if (!current_entry) return NULL;

    char* generated_string = strdup(start_word);
    int words_count = 1;

    while (words_count < num_words) {
        const char* next_word = pick_next_word(current_entry);
        
        if (next_word == NULL) break;

        size_t new_size = strlen(generated_string) + strlen(next_word) + 2;
        char* temp = realloc(generated_string, new_size);

        if (!temp) break;
        
        generated_string = temp;
        strcat(generated_string, " ");
        strcat(generated_string, next_word);
        
        words_count++;
        
        current_entry = find_entry(Model, next_word);
        if (!current_entry) break;
    }

    return generated_string;
}

BigramEntry* find_entry(BigramModel* Model, const char* word) {
    // Error handling: Check for NULL pointers
    if (Model == NULL || word == NULL) return NULL;
    
    // Check if the model is empty or invalid
    if (Model->num_entries <= 0){
        return NULL;
    }

    // Searching Loop
    for (int i = 0; i < Model->num_entries; i++){
        if (strcmp(Model->entries[i].word, word) == 0){
            return &Model->entries[i]; // "word" has been found
        }
    } 
    return NULL; // "word" hasn't been found
}

void add_next_word(BigramEntry* Entry, const char* next_word) {
    // Error handling: Check for NULL pointers
    if (Entry == NULL || next_word == NULL) return;

    // Check if next_word already exists
    for (int i = 0; i < Entry->num_next; i++){
        if (strcmp(Entry->next_words[i], next_word) == 0){
            Entry->counts[i]++; // Increment count
            return;
        }
    }
    
    // New word: expand arrays
    Entry->num_next++;
    Entry->next_words = realloc(Entry->next_words, Entry->num_next * sizeof(char*));
    Entry->counts = realloc(Entry->counts, Entry->num_next * sizeof(int));
    Entry->next_words[Entry->num_next-1] = strdup(next_word);
    Entry->counts[Entry->num_next-1] = 1;
}

const char* pick_next_word(BigramEntry* Entry) {
    if (Entry == NULL || Entry->num_next == 0 || Entry->counts == NULL) {
        return NULL;
    }

    int total_sum = 0;
    for (int i = 0; i < Entry->num_next; i++) {
        total_sum += Entry->counts[i];
    }

    if (total_sum == 0) return NULL;

    int random = rand() % total_sum;

    for (int i = 0; i < Entry->num_next; i++) {
        if (random < Entry->counts[i]) {
            return Entry->next_words[i];
        }
        random -= Entry->counts[i];
    }
    
    return Entry->next_words[0];
}

void free_bigram_model(BigramModel* Model) {
    if (Model == NULL) return;
    for (int i = 0; i < Model->num_entries; i++) {
        free(Model->entries[i].word);
        for (int j = 0; j < Model->entries[i].num_next; j++) {
            free(Model->entries[i].next_words[j]);
        }
        free(Model->entries[i].next_words);
        free(Model->entries[i].counts);
    }
    free(Model->entries);
    free(Model);
}