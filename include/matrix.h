// matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

/**
 * Matrix structure
 * * Represents a 2D matrix stored as a 1D array in row-major order.
 * * Fields:
 * rows - Number of rows
 * cols - Number of columns
 * data - Pointer to flat array of size (rows * cols) containing the matrix data
 * grad - Pointer to flat array of size (rows * cols) containing the matrix gradient
 */
typedef struct {
    int rows;
    int cols;
    float* data;
    float* grad;
} Matrix;

/* ============================================================================
 * CORE MATRIX OPERATIONS
 * ============================================================================ */

/**
 * create_matrix - Allocate memory for a new matrix
 * @rows: Number of rows
 * @cols: Number of columns
 * * Returns: Matrix initialized to zero (uses calloc)
 * Note: Must call free_matrix() when done to avoid memory leaks
 */
Matrix* create_matrix(int rows, int cols, bool requires_grad);

/**
 * free_matrix - Free memory allocated for a matrix
 * @m: Matrix to free
 * * Note: Always call this when done with a matrix to prevent memory leaks
 */
void free_matrix(Matrix* m);

/**
 * print_matrix - Display matrix to stdout
 * @Input: Matrix to print
 * * Prints matrix in readable format with 2 decimal places
 */
void print_matrix(Matrix* Input);

/**
 * set_value - Set element at position (r, c)
 * @m: Matrix to modify
 * @r: Row index (0-based)
 * @c: Column index (0-based)
 * @v: Value to set
 * * Exits with error if indices are out of bounds
 */
void set_value(Matrix* m, int r, int c, float v);

/**
 * get_value - Get element at position (r, c)
 * @m: Matrix to read from
 * @r: Row index (0-based)
 * @c: Column index (0-based)
 * * Returns: Value at position (r, c)
 * Exits with error if indices are out of bounds
 */
float get_value(Matrix* m, int r, int c);

/**
 * get_row - Extract a specific row from a matrix into the output matrix
 * @Input: The source matrix to extract from
 * @row_index: The 0-based index of the row to retrieve
 * @Output: The destination matrix to store the row data
 * * Note: 
 * - Exits with an error if row_index is out of bounds or dimensions do not match.
 */
void get_row(Matrix* Input, int row_index, Matrix* Output);

/**
 * matrix_swap_rows - Swap two rows in a matrix in-place
 * @M: The matrix to modify
 * @row1: Index of the first row (0-based)
 * @row2: Index of the second row (0-based)
 * * Note: 
 * - Performs the swap in-place; does not allocate new memory.
 * - If row1 == row2, the function returns immediately.
 * - Exits with an error if indices are out of bounds.
 */
void matrix_swap_rows(Matrix* M, int row1, int row2);

/**
 * matrix_sum_rows - Sums all rows of the input matrix into a single row vector
 * @Input:  The source matrix of shape (N, D)
 * @Output: The destination vector of shape (1, D)
 * Note: 
 * - Output.data will be zeroed out before summation.
 * - Exits with an error if Output is not shape (1, Input.cols).
 */
void matrix_sum_rows(Matrix* Input, Matrix* Output);

/**
 * matrix_reset - Sets all elements of the matrix data and grad to 0.0f
 * @m: The matrix to reset
 */
void matrix_reset(Matrix* input);

/* ============================================================================
 * MATRIX ARITHMETIC (TWO MATRICES)
 * ============================================================================ */

/**
 * matrix_multiply - Performs matrix multiplication with optional transposition
 * @A: Left input matrix
 * @B: Right input matrix
 * @Result: Output matrix where the product is stored
 * @is_transA: If true, matrix A is transposed before multiplication
 * @is_transB: If true, matrix B is transposed before multiplication
 * * Note: The dimensions must be compatible for multiplication after accounting 
 * for any transpositions:
 * - Let A_cols = is_transA ? A.rows : A.cols
 * - Let B_rows = is_transB ? B.rows : B.cols
 * - A_cols must equal B_rows, exits with error otherwise.
 * * Formula: 
 * Let A' = is_transA ? transpose(A) : A  (Size: M x N)
 * Let B' = is_transB ? transpose(B) : B  (Size: N x P)
 * Result[i][j] = sum(A'[i][k] * B'[k][j]) for k from 0 to N-1
 * The Result matrix must be of size M x P.
 */
void matrix_multiply(Matrix* A, Matrix* B, Matrix* Result, bool is_transA, bool is_transB);

/**
 * matrix_addition - Element-wise addition
 * @a: First matrix
 * @b: Second matrix
 * @Output: Destination matrix where result is stored
 * Note: Matrices must be same size, exits with error otherwise
 */
void matrix_addition(Matrix* a, Matrix* b, Matrix* Output);

/**
 * matrix_add_bias - Adds bias 
 * @Input: Input Matrix 
 * @Bias: Bias Matrix
 * @Output: Output Matrix
 * * Returns: Populated pre-created matrix, with Bias added to each values
 * Note: Bias matrix has to be (1,Cols), user must free the Input Matrix if not used later.
 */
void matrix_add_bias(Matrix* Input, Matrix* Bias, Matrix* Output);

/**
 * matrix_subtraction - Element-wise subtraction
 * @a: First matrix
 * @b: Second matrix
 * @Output: Destination matrix where result is stored
 * Note: Matrices must be same size, exits with error otherwise
 */
void matrix_subtraction(Matrix* a, Matrix* b, Matrix* Output);

/**
 * matrix_hadamard - Element-wise multiplication (Hadamard product)
 * @a: First matrix
 * @b: Second matrix
 * @Output: Destination matrix where result is stored
 * Note: Matrices must be same size, exits with error otherwise
 */
void matrix_hadamard(Matrix* a, Matrix* b, Matrix* Output);

/* ============================================================================
 * MATRIX TRANSFORMATIONS (ONE MATRIX)
 * ============================================================================ */

/**
 * matrix_scalar_multiply - Multiply all elements by a scalar
 * @Input: Matrix to scale
 * @scalar: Scalar value
 * @Output: Destination matrix where result is stored
 */
void matrix_scalar_multiply(Matrix* Input, float scalar, Matrix* Output);

/**
 * matrix_scalar_addition - Add a scalar to all elements
 * @Input: Matrix to modify
 * @scalar: Scalar value to add
 * @Output: Destination matrix where result is stored
 */
void matrix_scalar_addition(Matrix* Input, float scalar, Matrix* Output);

/**
 * matrix_scalar_subtraction - Subtract a scalar from all elements
 * @Input: Matrix to modify
 * @scalar: Scalar value to subtract
 * @Output: Destination matrix where result is stored
 */
void matrix_scalar_subtraction(Matrix* Input, float scalar, Matrix* Output);

/* ============================================================================
 * ACTIVATION FUNCTIONS (NEURAL NETWORKS)
 * ============================================================================ */

/**
 * matrix_relu - Apply ReLU activation function
 * @Input: Input matrix
 * @Output: Output Matrix
 * * Returns: Populated pre-created matrix where Output[i][j] = max(0, m[i][j])
 * Note: User has to free the Input matrix if not used anymore
 */
void matrix_relu(Matrix* Input, Matrix* Output);


/**
 * matrix_relu_derivative - Apply ReLU derivative during backpropagation
 * @Hidden_raw: Values before ReLU activation (pre-activation)
 * @Gradient_Hidden: Gradient flowing back from the next layer
 * @Output: Matrix to store the result of the masked gradient
 * * How it works:
 * - ReLU derivative is 1 for positive inputs, 0 for negative.
 * - This masks the incoming gradient element-wise based on the pre-activation state.
 * - Formula: Output[i] = (Hidden_raw[i] > 0) ? Gradient_Hidden[i] : 0
 * Note: 
 * - All three matrices must have the same dimensions.
 * - Performs the operation in-place if Gradient_Hidden and Output are the same matrix.
 */
void matrix_relu_derivative(Matrix* Hidden_raw, Matrix* Gradient_Hidden, Matrix* Output);

/**
 * matrix_sigmoid - Apply Sigmoid activation function
 * @Input: Input matrix
 * @Output: Destination matrix where result is stored
 * Sigmoid: Maps any real number to range [0,1]. Negative values → close to 0, positive values → close to 1, zero → exactly 0.5 
*/
void matrix_sigmoid(Matrix* Input, Matrix* Output);

/**
 * matrix_softmax - Apply softmax activation function
 * @Input: Input matrix
 * @Output: Destination matrix where result is stored
 * Properties:
 * - All values in range [0, 1]
 * - Sum of all values = 1.0
 * - Uses numerical stability trick (subtract max before exp)
 */
void matrix_softmax(Matrix* Input, Matrix* Output);

/**
 * matrix_copy - Create a deep copy of a matrix
 * @m: Matrix to be copied
 * * Returns: A new Matrix structure containing the same dimensions and data
 * * Note: 
 * - Allocates new memory for the data array.
 * - Must call free_matrix() on the returned matrix to prevent leaks.
 * - Useful for preserving an original matrix before performing in-place transformations or destructive operations.
 */
Matrix* matrix_copy(Matrix* Input);

/* ============================================================================
 * DATA PREPROCESSING
 * ============================================================================ */

/**
 * matrix_normalize - Scale all values to the range [0, 1]
 * @Dataset: Input matrix to be normalized
 * @Output: Destination matrix where result is stored
 * * Note: 
 * - Exits with error if range is 0 or matrix is empty.
 */
void matrix_normalize(Matrix* Dataset, Matrix* Output);
#endif // MATRIX_H