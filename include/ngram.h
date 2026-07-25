#ifndef NGRAM_H
#define NGRAM_H

typedef struct {
    char* word;           // Current word
    char** next_words;    // Array of words that can follow
    int* counts;          // How many times each next word appears
    int num_next;         // Number of unique next words
} BigramEntry;

typedef struct {
    BigramEntry* entries;  // Array of all bigrams
    int num_entries;       // Number of unique starting words
} BigramModel;

/**
 * train_bigram_model - Build bigram statistics from text
 * @text: Training text (space-separated words)
 * 
 * Returns: Trained BigramModel
 * 
 * Process:
 *   - Split text into words
 *   - For each pair (word[i], word[i+1]):
 *     - Find or create entry for word[i]
 *     - Add word[i+1] to its next_words
 *     - Increment count
 */
BigramModel* train_bigram_model(const char* text);

/**
 * generate_text - Generate sequence using bigram model
 * @model: Trained bigram model
 * @start_word: First word of sequence
 * @num_words: How many words to generate
 * 
 * Returns: Generated text string (caller must free)
 * 
 * Algorithm:
 *   1. Start with start_word
 *   2. Look up possible next words
 *   3. Pick one randomly (weighted by count)
 *   4. Repeat
 */
char* generate_text(BigramModel* model, const char* start_word, int num_words);

/**
 * find_entry - Locate a word in the bigram model
 * @Model: Bigram model to search
 * @word: Word to find
 * 
 * Returns: Pointer to entry if found, NULL otherwise
 */
BigramEntry* find_entry(BigramModel* Model, const char* word);

/**
 * add_next_word - Associate a succeeding word with a BigramEntry
 * @Entry: The entry for the current word
 * @next_word: The word found following the current word in the text
 * 
 * Process:
 *   - Check if next_word already exists in Entry->next_words
 *   - If exists: Increment corresponding count
 *   - If new: Reallocate arrays and add word with count 1
 */
void add_next_word(BigramEntry* Entry, const char* next_word);

/**
 * pick_next_word - Select a following word based on probability
 * @Entry: The entry containing possible next words
 * 
 * Returns: A pointer to one of the strings in next_words
 * 
 * Algorithm:
 *   - Calculate total sum of counts
 *   - Generate random number between 0 and sum
 *   - Return the word where the cumulative count exceeds the random number
 */
const char* pick_next_word(BigramEntry* Entry);

/**
 * free_bigram_model - Clean up all memory
 * @model: Model to free
 * 
 * Note: Recursively frees entries, their word lists, and the model struct itself.
 */
void free_bigram_model(BigramModel* model);


#endif