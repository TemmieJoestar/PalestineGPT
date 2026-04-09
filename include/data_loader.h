#ifndef DATA_LOADER_H
#define DATA_LOADER_H

// Text formatting macros
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[92;1m" x "\033[0m"
#define YELLOW_TEXT(x) "\033[33;1m" x "\033[0m"
#define CYAN_TEXT(x) "\033[36;1m" x "\033[0m"
#define MAGENTA_TEXT(x) "\033[35;1m" x "\033[0m"
#define BOLD(x) "\033[1m" x "\033[0m"

/* ============================================================================
 * IRIS DATASET UTILITIES
 * ============================================================================ */

/**
 * load_iris_features - Load numerical features from an Iris CSV file
 * @filename: Path to the .csv file
 * 
 * Returns: A (150 x 4) Matrix containing sepal/petal measurements
 * 
 * Process:
 *   - Opens file and reads up to 150 rows.
 *   - Parses the first 4 comma-separated values of each line.
 *   - Converts strings to floats and stores them in the matrix.
 * 
 * Note: 
 *   - Exits program if the file cannot be opened.
 *   - Caller is responsible for calling free_matrix() on the result.
 */
Matrix load_iris_features(const char* filename);

/**
 * load_iris_labels - Load classification labels from an Iris CSV file
 * @filename: Path to the .csv file
 * 
 * Returns: A (150 x 1) Matrix containing the species category index
 * 
 * Process:
 *   - Skips the first 4 columns (features) of each row.
 *   - Extracts the 5th column value as a float label.
 *   - Stores labels in a column vector format.
 * 
 * Note: 
 *   - Expects labels to be numerical (e.g., 0, 1, 2) in the CSV.
 *   - Caller is responsible for calling free_matrix() on the result.
 */
Matrix load_iris_labels(const char* filename);

/**
 * label_to_onehot - Convert a single integer label to a one-hot row vector
 * @label: The class index to activate (0 to num_classes - 1)
 * @num_classes: The total number of possible categories
 * 
 * Returns: A (1 x num_classes) Matrix (row vector)
 * 
 * Example:
 *   label_to_onehot(1, 3) returns [[0.0, 1.0, 0.0]]
 * 
 * Note:
 *   - If label is out of bounds, returns a vector of all zeros and prints a warning.
 *   - Useful for preparing targets for Cross-Entropy loss or Softmax layers.
 *   - Caller is responsible for calling free_matrix() on the result.
 */
Matrix label_to_onehot(int label, int num_classes);

/* ============================================================================
 * DATASET MANIPULATION
 * ============================================================================ */

/**
 * shuffle_dataset - Synchronized shuffle of features and labels
 * @Features: Matrix of input data (N x M)
 * @Labels: Matrix of target labels (N x 1 or N x C)
 * 
 * Logic:
 *   - Uses the Fisher-Yates shuffle algorithm.
 *   - Swaps rows in both matrices simultaneously using the same random indices.
 *   - This ensures that Sample X in Features remains paired with Label X in Labels.
 * 
 * Note: 
 *   - Requires both matrices to have the same number of rows.
 *   - Operates in-place; modifies the original matrices.
 *   - Ensure srand() is called in main() for unique shuffles across runs.
 */
void shuffle_dataset(Matrix Features, Matrix Labels);

#endif 