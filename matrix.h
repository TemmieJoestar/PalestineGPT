// matrix.h
#ifndef MATRIX_H
#define MATRIX_H

// Text formatting macros
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[92;1m" x "\033[0m"
#define BOLD(x) "\033[1m" x "\033[0m"

/**
 * Matrix structure
 * 
 * Represents a 2D matrix stored as a 1D array in row-major order.
 * 
 * Fields:
 *   rows - Number of rows
 *   cols - Number of columns
 *   data - Pointer to flat array of size (rows * cols)
 */
typedef struct {
    int rows;
    int cols;
    float* data;
} Matrix;

/* ============================================================================
 * CORE MATRIX OPERATIONS
 * ============================================================================ */

/**
 * create_matrix - Allocate memory for a new matrix
 * @rows: Number of rows
 * @cols: Number of columns
 * 
 * Returns: Matrix initialized to zero (uses calloc)
 * Note: Must call free_matrix() when done to avoid memory leaks
 */
Matrix create_matrix(int rows, int cols);

/**
 * free_matrix - Free memory allocated for a matrix
 * @m: Matrix to free
 * 
 * Note: Always call this when done with a matrix to prevent memory leaks
 */
void free_matrix(Matrix m);

/**
 * print_matrix - Display matrix to stdout
 * @m: Matrix to print
 * 
 * Prints matrix in readable format with 2 decimal places
 */
void print_matrix(Matrix m);

/**
 * set_value - Set element at position (r, c)
 * @m: Matrix to modify
 * @r: Row index (0-based)
 * @c: Column index (0-based)
 * @v: Value to set
 * 
 * Exits with error if indices are out of bounds
 */
void set_value(Matrix m, int r, int c, float v);

/**
 * get_value - Get element at position (r, c)
 * @m: Matrix to read from
 * @r: Row index (0-based)
 * @c: Column index (0-based)
 * 
 * Returns: Value at position (r, c)
 * Exits with error if indices are out of bounds
 */
float get_value(Matrix m, int r, int c);

/* ============================================================================
 * MATRIX ARITHMETIC (TWO MATRICES)
 * ============================================================================ */

/**
 * matrix_multiply - Matrix multiplication (linear algebra)
 * @a: Left matrix (M x N)
 * @b: Right matrix (N x P)
 * 
 * Returns: Result matrix (M x P)
 * Note: a.cols must equal b.rows, exits with error otherwise
 * Formula: C[i][j] = sum(A[i][k] * B[k][j]) for all k
 */
Matrix matrix_multiply(Matrix a, Matrix b);

/**
 * matrix_addition - Element-wise addition
 * @a: First matrix
 * @b: Second matrix
 * 
 * Returns: New matrix where result[i][j] = a[i][j] + b[i][j]
 * Note: Matrices must be same size, exits with error otherwise
 */
Matrix matrix_addition(Matrix a, Matrix b);

/**
 * matrix_subtraction - Element-wise subtraction
 * @a: First matrix
 * @b: Second matrix
 * 
 * Returns: New matrix where result[i][j] = a[i][j] - b[i][j]
 * Note: Matrices must be same size, exits with error otherwise
 */
Matrix matrix_subtraction(Matrix a, Matrix b);

/**
 * matrix_hadamard - Element-wise multiplication (Hadamard product)
 * @a: First matrix
 * @b: Second matrix
 * 
 * Returns: New matrix where result[i][j] = a[i][j] * b[i][j]
 * Note: Matrices must be same size, exits with error otherwise
 */
Matrix matrix_hadamard(Matrix a, Matrix b);

/* ============================================================================
 * MATRIX TRANSFORMATIONS (ONE MATRIX)
 * ============================================================================ */

/**
 * matrix_transpose - Swap rows and columns
 * @m: Matrix to transpose (M x N)
 * 
 * Returns: Transposed matrix (N x M)
 * Example: [[1,2,3],    becomes    [[1,4],
 *                  [4,5,6]]                         [2,5],
 *                                                       [3,6]]
 */
Matrix matrix_transpose(Matrix m);

/**
 * matrix_scalar_multiply - Multiply all elements by a scalar
 * @m: Matrix to scale
 * @s: Scalar value
 * 
 * Returns: New matrix where result[i][j] = m[i][j] * s
 */
Matrix matrix_scalar_multiply(Matrix m, float s);

/* ============================================================================
 * ACTIVATION FUNCTIONS (NEURAL NETWORKS)
 * ============================================================================ */

/**
 * matrix_relu - Apply ReLU activation function
 * @m: Input matrix
 * 
 * Returns: New matrix where result[i][j] = max(0, m[i][j])
 * ReLU (Rectified Linear Unit): negative values become 0, positive unchanged
 */
Matrix matrix_relu(Matrix m);

/**
 * matrix_softmax - Apply softmax activation function
 * @m: Input matrix
 * 
 * Returns: New matrix where values are converted to probability distribution
 * Properties:
 *   - All values in range [0, 1]
 *   - Sum of all values = 1.0
 *   - Uses numerical stability trick (subtract max before exp)
 */
Matrix matrix_softmax(Matrix m);

/**
 * matrix_copy - Create a deep copy of a matrix
 * @m: Matrix to be copied
 * 
 * Returns: A new Matrix structure containing the same dimensions and data
 * 
 * Note: 
 *  - Allocates new memory for the data array.
 *  - Must call free_matrix() on the returned matrix to prevent leaks.
 *  - Useful for preserving an original matrix before performing in-place transformations or destructive operations.
 */
Matrix matrix_copy(Matrix m);

#endif // MATRIX_H