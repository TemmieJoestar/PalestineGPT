// vocabulary.h
#ifndef VOCABULARY_H
#define VOCABULARY_H

// Text formatting macros
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[92;1m" x "\033[0m"
#define YELLOW_TEXT(x) "\033[33;1m" x "\033[0m"
#define CYAN_TEXT(x) "\033[36;1m" x "\033[0m"
#define MAGENTA_TEXT(x) "\033[35;1m" x "\033[0m"
#define BOLD(x) "\033[1m" x "\033[0m"

typedef struct {
    char** tokens;
    int vocab_size;
} Vocabulary;

/**
 * create_vocab - Create vocabulary from array of token strings
 * @token_list: Array of token strings (e.g., ["#", "a", "b", ...])
 * @size: Number of tokens in the list
 * 
 * Returns: Pointer to newly allocated Vocabulary structure
 * 
 * Memory:
 *   - Allocates Vocabulary struct
 *   - Allocates array of char* pointers
 *   - Allocates and COPIES each token string
 * 
 * Example:
 *   const char* tokens[] = {"#", "a", "b", "c"};
 *   Vocabulary* vocab = create_vocab(tokens, 4);
 */
Vocabulary* create_vocab(const char** token_list, int size);

/**
 * get_token_id - Find the ID of a given token string
 * @vocab: Vocabulary structure
 * @token: Token string to search for
 * 
 * Returns: Token ID (0-based index), or -1 if not found
 * 
 * Example:
 *   int id = get_token_id(vocab, "a");  // Returns 1
 *   int unknown = get_token_id(vocab, "xyz");  // Returns -1
 */
int get_token_id(Vocabulary* vocab, const char* token);

/**
 * get_token_string - Get token string from its ID
 * @vocab: Vocabulary structure
 * @token_id: ID to lookup
 * 
 * Returns: Token string, or NULL if ID is out of bounds
 * 
 * Example:
 *   char* token = get_token_string(vocab, 1);  // Returns "a"
 *   char* invalid = get_token_string(vocab, 999);  // Returns NULL
 */
char* get_token_string(Vocabulary* vocab, int token_id);

/**
 * free_vocab - Free all memory associated with vocabulary
 * @vocab: Vocabulary to free
 * 
 * Must free:
 *   - Each individual token string
 *   - The tokens array
 *   - The Vocabulary struct itself
 */
void free_vocab(Vocabulary* vocab);

#endif