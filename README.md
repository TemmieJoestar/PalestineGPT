# PalestineGPT

Building a Large Language Model (LLM) with Transformer architecture from scratch in C.

## Project Status

**Matrix Library** DONE
- Custom matrix operations library
- Zero memory leaks (valgrind verified)
- Comprehensive test suite (9 tests)
- Professional build system

**Mathematical Foundations** IN PROGRESS
- Linear Algebra -> DONE.
- Calculus & Gradients -> DONE.
- Loss Functions & Multi-Layer Networks. -> IN PROGRESS...

## Current Features

### Matrix Operations
- Matrix multiplication (linear algebra).
- Element-wise operations (add, subtract, multiply).
- Matrix transpose.
- Scalar multiplication.
- Deep copy.

### Activation Functions
- ReLU (Rectified Linear Unit).
- Softmax (probability distribution).

### Neural Network Foundations
- Forward pass (prediction).
- Backward pass (gradient calculation).
- Gradient descent (weight updates).
- Training loop.

## Building
```bash
# Shows available Target, and how to use them.
make help

# Build main program.
make

# Compile and run 'comprehensivetesting.c'.
make test

# Compile and run 'FILE.c' (e.g, gradient_descent.run).
make FILE.run

# Run valgrind on 'comprehensivetest.c'.
make memcheck

# Run valgrind on 'FILE.c' (e.g, gradient_descent.memcheck).
make FILE.memcheck

# Remove compliled executable files. (Only -> 'main', 'comprehensivetesting', 'gradient_descent' and 'test).
```

## Testing

All functions are tested with comprehensive test suite:
- 8 test functions.
- 100% coverage.
- Zero memory leaks.

## Project Structure 
```
├── .gitignore              # Contains files that'll be ignored in commit.
├── comprehensivetesting.c  # Comprehensive Test suite.
├── gradient_descent        # Gradient Calculus with Matrix.
├── LICENSE                 # MIT license file.
├── main.c                  # Demo/Experiments.
├── Makefile                # Build automation (make).
├── matrix.c                # Matrix library implementation.
├── matrix.h                # Matrix library header (documented).
└── README.md               # README file.
```

## Requirements

- GCC compiler
- Math library (-lm)
- Valgrind (for memory checking)

## License

See LICENSE file.