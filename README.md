# PalestineGPT

Building a Large Language Model (LLM) with Transformer architecture from scratch in C.

> **Current Status:** Phase 3 - Working...

---

## Project Milestones

- ✅ **Phase 0:** C Fundamentals (Matrix library, memory management)
- ✅ **Phase 1:** Mathematical Foundations (Linear algebra, calculus, gradients)
- ✅ **Phase 2:** Neural Network Basics (2-layer networks, backpropagation, real datasets)
- ⏳ **Phase 3:** NLP Basics (Tokenization, embeddings, text processing)
- ⏳ **Phase 4:** Attention Mechanism
- ⏳ **Phase 5:** Transformer Architecture
- ⏳ **Phase 6:** Training Infrastructure
- ⏳ **Phase 7:** Text Generation

---

## Current Capabilities

### Successfully Trained On:
- **XOR Problem:** 100% accuracy (classic non-linear classification)
- **Iris Dataset:** 96-100% accuracy (multi-class flower classification)
  - 150 samples, 4 features, 3 classes
  - Proper train/validation split (80/20)
  - Data shuffling implemented

---

## Project Structure
```
PalestineGPT/
├── include/                   # Header files
│   ├── data_loader.h          # Dataset loading utilities
│   ├── embedding.h            # Embeddings function library
│   ├── gradient_descent.h     # Neural network training
│   ├── matrix.h               # Matrix operations library
│   ├── tokenizer.h            # Tokenizer functions library
│   └── vocabulary.h           # Vocabulary functions library
│
├── others/                    # Data files
│   └── iris.csv               # Iris flower dataset
│
├── src/                       # Source implementations
│   ├── bonus/                 # Example programs
│   │   ├── iris_dataset.c     # Iris classification demo
│   │   └── xor_problem.c      # XOR solver demo
│   ├── data_loader.c          # CSV loading & preprocessing
│   ├── embedding.c            # Embedding Library
│   ├── gradient_descent.c     # Training algorithms
│   ├── matrix.c               # Matrix library
│   ├── tokenizer.c            # Tokenizer library
│   └── vocabulary             # Vocabulary library
│
├── tests/                     # Test suite
│   ├── comprehensivetesting.c # 13 tests
│   ├── test_embedding.c       # 4 tests
│   └── test_vocab.c           # 4 tests
│
├── .gitignore
├── LICENSE                    # MIT License
├── Makefile                   # Build system
└── README.md                  # Project documentation
```

---

## Building & Running

### Requirements
- GCC compiler
- Make (Unix-style)
  - **Windows:** Use [Git Bash](https://git-scm.com/downloads) or WSL
  - **Linux/macOS:** Built-in
- Valgrind (optional, for memory checking)

### Commands
```bash
# Show all available commands
make help

# Build the main program
make

# Run the comprehensive test suite
make test

# Run a specific file in src/ (e.g., src/main.c)
make main.run

# Run a bonus file (e.g., src/bonus/iris_dataset.c)
make iris_dataset.bonus

# Check the test suite for memory leaks
make memcheck

# Check a bonus file for memory leaks
make iris_dataset.bmem

# Clean up compiled files and executables
make clean
```

---

## Testing

**Test Suite:** 13 comprehensive tests covering:
- Core matrix arithmetic
- Transformations & scalar operations
- Neural network activations
- Data preprocessing

---

## Learning Journey

**Completed:**
- Phase 0: C programming fundamentals
- Phase 1: Linear algebra, calculus, gradient descent
- Phase 2: Neural networks, real dataset training

**What's Next:**
- Phase 3: Text processing, tokenization, embeddings
- Phase 4: Attention mechanism (core of Transformers)
- Phase 5: Full Transformer architecture
- Phase 6: Training infrastructure at scale
- Phase 7: Text generation capabilities

**Timeline:** ~6-9 months total (3 months complete)

---

## Project Goals

1. **Deep Understanding:** Learn how LLMs work at the lowest level
2. **From Scratch:** No ML frameworks - pure C implementation
3. **Scalable Architecture:** Build foundations that scale to real models

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Author

**Repository:** [PalestineGPT on GitHub](https://github.com/TemmieJoestar/PalestineGPT)

**Goal:** Understand every detail of how LLMs work by implementing one from first principles.

Built from scratch by [TemmieJoestar](https://github.com/TemmieJoestar) as a deep learning journey in C.