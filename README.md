# PalestineGPT

Building a Large Language Model (LLM) with Transformer architecture from scratch in C.

> **Current Status:** Phase 4 - Working...

---

## Project Milestones

- ✅ **Phase 0:** C Fundamentals (Matrix library, memory management)
- ✅ **Phase 1:** Mathematical Foundations (Linear algebra, calculus, gradients)
- ✅ **Phase 2:** Neural Network Basics (2-layer networks, backpropagation, real datasets)
- ✅ **Phase 3:** NLP Basics (Tokenization, embeddings, text processing)
- ⏳ **Phase 4:** Attention Mechanism
- ⏳ **Phase 5:** Transformer Architecture
- ⏳ **Phase 6:** Training Infrastructure
- ⏳ **Phase 7:** Text Generation

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
│   ├── ngram.h                # Ngram functions library
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
│   ├── embedding.c            # Embedding Function
│   ├── gradient_descent.c     # Training algorithms
│   ├── main.h                 # Main test
│   ├── matrix.c               # Matrix Function
│   ├── ngram.c                # Ngram Function
│   ├── tokenizer.c            # Tokenizer Function
│   └── vocabulary             # Vocabulary Function
│
├── tests/                     # Test suite
│   ├── comprehensivetesting.c # 13 tests
│   ├── test_embedding.c       # 4 tests
│   ├── test_full_pipeline.c   # Tokenization Pipeline Test
│   ├── test_ngram             # 6 tests
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

**Test Suite:** comprehensive tests covering:
- Core matrix arithmetic
- Transformations & scalar operations
- Neural network activations
- Data preprocessing
- Ngram generation

---

## Learning Journey

**Completed:**
- Phase 0: C programming fundamentals
- Phase 1: Linear algebra, calculus, gradient descent
- Phase 2: Neural networks, real dataset training
- Phase 3: Text processing, tokenization, embeddings

**What's Next:**
- Phase 4: Attention mechanism (core of Transformers)
- Phase 5: Full Transformer architecture
- Phase 6: Training infrastructure at scale
- Phase 7: Text generation capabilities

---

## Project Goals

1. **Deep Understanding:** Learn how LLMs work at the lowest level
2. **From Scratch:** No ML frameworks - pure C implementation
3. **Scalable Architecture:** Build foundations that scale to real models (hopefully)

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Author

**Repository:** [PalestineGPT on GitHub](https://github.com/TemmieJoestar/PalestineGPT)

**Goal:** Understand every detail of how LLMs work by implementing one from first principles.

Built from scratch by [TemmieJoestar](https://github.com/TemmieJoestar) as a deep learning journey in C.