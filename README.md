# PalestineGPT

Building a Large Language Model (LLM) with Transformer architecture from scratch in C.

## Project Status

**Phase 0: C Fundamentals** ✅ COMPLETE
- Matrix library with 17 operations
- Zero memory leaks (valgrind verified)
- Comprehensive test suite (13 tests, all passing)

**Phase 1: Mathematical Foundations** ✅ COMPLETE
- Linear Algebra ✓
- Calculus & Gradients ✓
- Loss Functions ✓

**Phase 2: Neural Network Basics** 🔄 IN PROGRESS
- 2-layer feedforward network ✓
- Backpropagation with bias terms ✓
- Momentum optimization ✓
- XOR problem solved ✓
- **Current:** Training on Iris dataset (classification)

---

## Current Features

### Matrix Operations (17 total)
**Core Operations:**
- Matrix multiplication (linear algebra)
- Element-wise operations (add, subtract, Hadamard product)
- Matrix transpose
- Deep copy

**Scalar Operations:**
- Scalar multiplication
- Scalar addition
- Scalar subtraction

**Data Preprocessing:**
- Feature normalization (min-max scaling to [0,1])

### Activation Functions
- ReLU (Rectified Linear Unit)
- Sigmoid (for binary classification)
- Softmax (for multi-class classification)

### Neural Network Components
- Forward propagation (2-layer with bias)
- Backpropagation (with ReLU derivative masking)
- Gradient descent optimizer
- Momentum-based gradient descent
- Loss functions:
  - Mean Squared Error (MSE)
  - Cross-Entropy Loss

---

## Building

**Requirements:**
- GCC compiler
- Make (Unix-style)
  - **Windows:** Use [Git Bash](https://git-scm.com/downloads) or WSL
  - **Linux/macOS:** Built-in
- Valgrind (optional, for memory checking)

### Commands
```bash
# Show all available commands
make help

# Build main program
make

# Run comprehensive test suite
make test

# Run specific file (e.g., gradient_descent.c)
make gradient_descent.run

# Check for memory leaks
make memcheck

# Check specific file for memory leaks
make gradient_descent.memcheck

# Clean up compiled files
make clean
```

---

## Testing

**Test Coverage:**
- 13 comprehensive tests
- All matrix operations verified
- Activation functions tested
- Normalization edge cases covered
- **Result:** 100% passing, zero memory leaks ✅

**Run tests:**
```bash
make test
```

**Sample output:**
```
=== Matrix Library Tests ===

--- Core Matrix Arithmetic ---
Testing matrix_multiply... PASSED
Testing matrix_addition... PASSED
Testing matrix_subtraction... PASSED
Testing matrix_hadamard... PASSED

--- Transformations & Scalars ---
Testing matrix_transpose... PASSED
Testing matrix_scalar_multiply... PASSED
Testing matrix_scalar_addition... PASSED
Testing matrix_scalar_subtraction... PASSED
Testing matrix_copy... PASSED

--- Neural Network Activations ---
Testing matrix_relu... PASSED
Testing matrix_sigmoid... PASSED
Testing matrix_softmax... PASSED

--- Data Preprocessing ---
Testing matrix_normalize... PASSED

=== All Tests Passed! ===
```

---

## Project Structure
```
PalestineGPT/
├── .gitignore                  # Git ignore rules
├── comprehensivetesting.c      # Full test suite (13 tests)
├── gradient_descent.c          # Neural network training functions
├── gradient_descent.h          # Training function declarations
├── LICENSE                     # MIT License
├── main.c                      # Experiments and demos
├── Makefile                    # Build automation
├── matrix.c                    # Matrix library implementation
├── matrix.h                    # Matrix library header (documented)
└── README.md                   # This file
```

---

## Milestones Achieved

- ✅ **Matrix library:** 17 operations, fully tested
- ✅ **XOR problem solved:** 2-layer network with 100% accuracy
- ✅ **Momentum optimization:** Consistent, stable training
- 🔄 **Iris classification:** In progress (Phase 2 final task)

---

## What's Next

**Phase 2 Completion:**
- Train on Iris flower dataset (multi-class classification)
- Implement validation split
- Compute classification accuracy

**Phase 3: NLP Basics** (Coming Soon)
- Tokenization
- Vocabulary building
- Word embeddings
- Text preprocessing

---

## Learning Resources

This project follows the **7-phase roadmap** for building LLMs from scratch:
1. ✅ C Fundamentals
2. ✅ Mathematical Foundations  
3. 🔄 Neural Network Basics (95% complete)
4. ⏳ NLP Basics
5. ⏳ Attention Mechanism
6. ⏳ Transformer Architecture
7. ⏳ Training Infrastructure
8. ⏳ Text Generation

---

## Requirements

- **Compiler:** GCC with C99 support
- **Platform:** Linux, macOS, or Windows (with Git Bash/WSL)
- **Memory:** 2GB RAM minimum (for training)
- **Math library:** `-lm` flag (standard)

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Author

Built from scratch by [TemmieChan](https://github.com/TemmieChan) as a deep learning journey in C.

**Goal:** Understand every detail of how LLMs work by implementing one from first principles.