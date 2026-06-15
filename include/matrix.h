// matrix.h
#ifndef MATRIX_H
#define MATRIX_H

// Text formatting macros
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[92;1m" x "\033[0m"
#define YELLOW_TEXT(x) "\033[33;1m" x "\033[0m"
#define CYAN_TEXT(x) "\033[36;1m" x "\033[0m"
#define MAGENTA_TEXT(x) "\033[35;1m" x "\033[0m"
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
 * @Input: Matrix to print
 * 
 * Prints matrix in readable format with 2 decimal places
 */
void print_matrix(Matrix Input);

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

/**
 * get_row - Extract a specific row from a matrix into a new matrix
 * @Input: The source matrix to extract from
 * @row_index: The 0-based index of the row to retrieve
 * 
 * Returns: A new Matrix of size (1 x Input.cols) containing a copy of the row data
 * 
 * How it works:
 *   - Allocates a new matrix with 1 row and the same number of columns as the input.
 *   - Copies the values from the specified row of the Input matrix into the new matrix.
 * 
 * Note: 
 *   - Exits with an error if row_index is out of bounds.
 *   - Allocates new memory; caller must call free_matrix() on the result.
 */
Matrix get_row(Matrix Input, int row_index);

/**
 * matrix_swap_rows - Swap two rows in a matrix in-place
 * @M: The matrix to modify
 * @row1: Index of the first row (0-based)
 * @row2: Index of the second row (0-based)
 * 
 * Note: 
 *   - Performs the swap in-place; does not allocate new memory.
 *   - If row1 == row2, the function returns immediately.
 *   - Exits with an error if indices are out of bounds.
 */
void matrix_swap_rows(Matrix M, int row1, int row2);

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
void matrix_multiply(Matrix A, Matrix B, Matrix Result, bool is_transA, bool is_transB);

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
 * matrix_scalar_multiply - Multiply all elements by a scalar
 * @m: Matrix to scale
 * @s: Scalar value
 * 
 * Returns: New matrix where result[i][j] = m[i][j] * s
 */
Matrix matrix_scalar_multiply(Matrix Input, float scalar);

/**
 * matrix_scalar_addition - Add a scalar to all elements
 * @m: Matrix to modify
 * @s: Scalar value to add
 * Returns: New matrix where result[i][j] = m[i][j] + s
 */
Matrix matrix_scalar_addition(Matrix Input, float scalar);

/**
 * matrix_scalar_subtraction - Subtract a scalar from all elements
 * @m: Matrix to modify
 * @s: Scalar value to subtract
 * Returns: New matrix where result[i][j] = m[i][j] - s
 */
Matrix matrix_scalar_subtraction(Matrix Input, float scalar);

/* ============================================================================
 * ACTIVATION FUNCTIONS (NEURAL NETWORKS)
 * ============================================================================ */

/**
 * matrix_relu - Apply ReLU activation function
 * @Input: Input matrix
 * 
 * Returns: New matrix where result[i][j] = max(0, m[i][j])
 * ReLU (Rectified Linear Unit): negative values become 0, positive unchanged
 */
Matrix matrix_relu(Matrix Input);


/**
 * matrix_relu_derivative - Apply ReLU derivative during backpropagation
 * @Hidden_raw: Values before ReLU activation (pre-activation)
 * @Gradient_Hidden: Gradient flowing back from the next layer
 * 
 * Returns: Masked gradient where values are kept only where Hidden_raw > 0
 * 
 * How it works:
 *   - ReLU derivative is 1 for positive inputs, 0 for negative
 *   - This masks the incoming gradient element-wise
 *   - Formula: output[i] = Gradient_Hidden[i] if Hidden_raw[i] > 0, else 0
 * 
 * Note: Both matrices must have the same dimensions.
 */
Matrix matrix_relu_derivative(Matrix Hidden_raw, Matrix Gradient_Hidden);

/**
 * matrix_sigmoid - Apply Sigmoid activation function
 * @Input: Input matrix
 * 
 * Returns: New matrix where result[i][j] = 1 / (1 + expf(-x))
 * Sigmoid: Maps any real number to range [0,1]. Negative values → close to 0, positive values → close to 1, zero → exactly 0.5 
*/
Matrix matrix_sigmoid(Matrix Input);

/**
 * matrix_softmax - Apply softmax activation function
 * @Input: Input matrix
 * 
 * Returns: New matrix where values are converted to probability distribution
 * Properties:
 *   - All values in range [0, 1]
 *   - Sum of all values = 1.0
 *   - Uses numerical stability trick (subtract max before exp)
 */
Matrix matrix_softmax(Matrix Input);

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
Matrix matrix_copy(Matrix Input);

/* ============================================================================
 * DATA PREPROCESSING
 * ============================================================================ */

/**
 * matrix_normalize - Scale all values to the range [0, 1]
 * @m: Input matrix to be normalized
 * * Returns: A new Matrix where: result = (x - min) / (max - min)
 * * Note: 
 * - If range is 0 (all values same), prints an error to stderr and returns a copy.
 * - Allocates new memory; caller must call free_matrix() on the result.
 */
Matrix matrix_normalize(Matrix m);
#endif // MATRIX_H