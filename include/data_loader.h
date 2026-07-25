#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "matrix.h"

/* ============================================================================
 * IRIS DATASET UTILITIES
 * ============================================================================ */

/**
 * load_iris_features - Load numerical features from an Iris CSV file
 * @filename: Path to the .csv file
 * * Returns: A pointer to a (150 x 4) Matrix containing sepal/petal measurements
 * * Note: Caller is responsible for calling free_matrix() on the returned pointer.
 */
Matrix* load_iris_features(const char* filename);

/**
 * load_iris_labels - Load classification labels from an Iris CSV file
 * @filename: Path to the .csv file
 * * Returns: A pointer to a (150 x 1) Matrix containing the species category index
 * * Note: Caller is responsible for calling free_matrix() on the returned pointer.
 */
Matrix* load_iris_labels(const char* filename);

/**
 * label_to_onehot - Convert a single integer label to a one-hot row vector
 * @label: The class index to activate (0 to num_classes - 1)
 * @num_classes: The total number of possible categories
 * * Returns: A pointer to a (1 x num_classes) Matrix (row vector)
 * * Note: Caller is responsible for calling free_matrix() on the returned pointer.
 */
Matrix* label_to_onehot(int label, int num_classes);

/* ============================================================================
 * DATASET MANIPULATION
 * ============================================================================ */

/**
 * shuffle_dataset - Synchronized shuffle of features and labels
 * @Features: Pointer to Matrix of input data (N x M)
 * @Labels: Pointer to Matrix of target labels (N x 1 or N x C)
 * * Note: Operates in-place on the matrices pointed to.
 */
void shuffle_dataset(Matrix* Features, Matrix* Labels);

#endif