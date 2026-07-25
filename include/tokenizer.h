#ifndef TOKENIZER_H
#define TOKENIZER_H

extern const char* VOCAB;

/**
 * encode - Convert text string to array of token IDs
 * @text: Input string to encode
 * @output_length: Pointer to store the length of output array
 * 
 * Returns: Dynamically allocated array of integers (token IDs)
 * 
 * Example:
 *   int len;
 *   int* tokens = encode("Hi!", &len);
 *   // tokens = [33, 40, 64]  (hypothetical IDs for 'H', 'i', '!')
 *   // len = 3
 */
int* encode(const char* text, int* output_length);


/**
 * decode - Convert array of token IDs back to text string
 * @token_ids: Array of token IDs
 * @length: Length of the token_ids array
 * 
 * Returns: Dynamically allocated string
 * 
 * Example:
 *   char* text = decode(tokens, 3);
 *   // text = "Hi!"
 */
char* decode(int* token_ids, int length);

#endif