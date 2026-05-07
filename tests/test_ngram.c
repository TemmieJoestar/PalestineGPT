#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "ngram.h"

// Testing Macros
#define EXPECT_EQ_INT(actual, expected) \
    do { \
        if ((actual) != (expected)) { \
            printf(RED_TEXT("\nFAILED: Integer mismatch\n")); \
            printf("  Expected: %d\n  Actual:   %d\n", (expected), (actual)); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_STR_EQ(actual, expected) \
    do { \
        if (strcmp((actual), (expected)) != 0) { \
            printf(RED_TEXT("\nFAILED: String mismatch\n")); \
            printf("  Expected: \"%s\"\n  Actual:   \"%s\"\n", (expected), (actual)); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_NOT_NULL(ptr) \
    do { \
        if ((ptr) == NULL) { \
            printf(RED_TEXT("\nFAILED: Pointer is NULL\n")); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_NULL(ptr) \
    do { \
        if ((ptr) != NULL) { \
            printf(RED_TEXT("\nFAILED: Pointer is not NULL\n")); \
            exit(1); \
        } \
    } while (0)

// Test Function Prototypes
void test_bigram_training();
void test_add_next_word();
void test_pick_next_word();
void test_text_generation();
void test_unknown_word_handling();
void test_empty_text();

int main() {
    // Seed for reproducibility in pick_next_word tests if needed
    srand(time(NULL));

    printf(BOLD("=== Bigram Model Tests ===\n"));

    printf(MAGENTA_TEXT("\n--- Training Logic ---\n"));
    test_bigram_training();
    test_add_next_word();

    printf(MAGENTA_TEXT("\n--- Selection & Generation ---\n"));
    test_pick_next_word();
    test_text_generation();
    test_unknown_word_handling();
    test_empty_text();

    printf(GREEN_TEXT("\n=== All Bigram Tests Passed! ===\n"));
    return 0;
}

// Test 1
void test_bigram_training() {
    printf(BOLD("Testing train_bigram_model and find_entry... "));
    
    const char* text = "the cat sat on the mat";
    BigramModel* model = train_bigram_model(text);
    
    EXPECT_NOT_NULL(model);
    // Unique words: "the", "cat", "sat", "on", "mat"
    EXPECT_EQ_INT(model->num_entries, 5);
    
    // Test find_entry (now using the header's version)
    BigramEntry* entry = find_entry(model, "the");
    EXPECT_NOT_NULL(entry);
    EXPECT_STR_EQ(entry->word, "the");
    
    // "the" should have 2 unique following words: "cat" and "mat"
    EXPECT_EQ_INT(entry->num_next, 2);
    
    free_bigram_model(model);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 2
void test_add_next_word() {
    printf(BOLD("Testing add_next_word... "));

    // Create a manual entry to test logic
    BigramEntry entry;
    entry.word = strdup("hello");
    entry.next_words = NULL;
    entry.counts = NULL;
    entry.num_next = 0;

    add_next_word(&entry, "world");
    add_next_word(&entry, "world");
    add_next_word(&entry, "friend");

    // "world" (count 2), "friend" (count 1) = 2 unique next words
    EXPECT_EQ_INT(entry.num_next, 2);
    EXPECT_STR_EQ(entry.next_words[0], "world");
    EXPECT_EQ_INT(entry.counts[0], 2);
    EXPECT_STR_EQ(entry.next_words[1], "friend");
    EXPECT_EQ_INT(entry.counts[1], 1);

    // Manual cleanup for this isolated test
    free(entry.word);
    for(int i = 0; i < entry.num_next; i++) free(entry.next_words[i]);
    free(entry.next_words);
    free(entry.counts);

    printf(GREEN_TEXT("PASSED\n"));
}

// Test 3
void test_pick_next_word() {
    printf(BOLD("Testing pick_next_word... "));

    BigramEntry entry;
    entry.word = strdup("test");
    entry.num_next = 1;
    entry.next_words = malloc(sizeof(char*));
    entry.next_words[0] = strdup("only_option");
    entry.counts = malloc(sizeof(int));
    entry.counts[0] = 10;

    // If there is only one option, it MUST pick it
    const char* result = pick_next_word(&entry);
    EXPECT_NOT_NULL(result);
    EXPECT_STR_EQ(result, "only_option");

    free(entry.word);
    free(entry.next_words[0]);
    free(entry.next_words);
    free(entry.counts);

    printf(GREEN_TEXT("PASSED\n"));
}

// Test 4
void test_text_generation() {
    printf(BOLD("Testing generate_text... "));
    
    const char* text = "i think therefore i am";
    BigramModel* model = train_bigram_model(text);
    
    // Generate 3 words starting with "i"
    char* generated = generate_text(model, "i", 3);
    
    EXPECT_NOT_NULL(generated);
    
    // Verify start word is present
    if (strstr(generated, "i") == NULL) {
        printf(RED_TEXT("FAILED: Start word not found in output\n"));
        exit(1);
    }
    
    printf("\n  Debug Output: %s\n", generated);
    
    free(generated);
    free_bigram_model(model);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 5
void test_unknown_word_handling() {
    printf(BOLD("Testing unknown word handling... "));
    
    const char* text = "simple text";
    BigramModel* model = train_bigram_model(text);
    
    // "missing" is not in the model training data
    char* generated = generate_text(model, "missing", 5);
    
    if (generated != NULL) {
        printf(RED_TEXT("FAILED: Should return NULL for unknown start word\n"));
        exit(1);
    }
    
    free_bigram_model(model);
    printf(GREEN_TEXT("PASSED\n"));
}

// Test 6 
void test_empty_text() {
    BigramModel* model = train_bigram_model("");
    EXPECT_EQ_INT(model->num_entries, 0);
    
    char* gen = generate_text(model, "any", 5);
    EXPECT_NULL(gen);
    
    free_bigram_model(model);
}