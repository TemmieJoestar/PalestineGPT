# PalestineGPT

Building a Large Language Model (LLM) with Transformer architecture from scratch in C.

## Project Status

**Phase 0: Matrix Library** COMPLETE
- Custom matrix operations library
- Zero memory leaks (valgrind verified)
- Comprehensive test suite
- Professional build system

## Current Features

### Matrix Operations
- Matrix multiplication (linear algebra)
- Element-wise operations (add, subtract, multiply)
- Matrix transpose
- Scalar multiplication

### Activation Functions
- ReLU (Rectified Linear Unit)
- Softmax (probability distribution)

## Building
```bash
# Run tests
make test

# Build main program
make

# Check for memory leaks
make memcheck

# Clean
make clean
```

## Testing

All functions are tested with comprehensive test suite:
- 8 test functions
- 100% coverage
- Zero memory leaks

## Project Structure 
```
├── matrix.h              # Matrix library header (documented)
├── matrix.c              # Matrix library implementation
├── comprehensivetesting.c # Test suite
├── main.c                # Demo/experiments
└── Makefile              # Build automation
```

## Next Steps

- Phase 1: Mathematical Foundations
- Phase 2: Neural Network Basics
- Phase 3: Natural Language Processing
- Phase 4: Attention Mechanism
- Phase 5: Transformer Architecture
- Phase 6: Training Infrastructure
- Phase 7: Text Generation

## Requirements

- GCC compiler
- Math library (-lm)
- Valgrind (for memory checking)

## License

See LICENSE file.