# PalestineGPT

Building a Large Language Model (LLM) with Transformer architecture from scratch in C.

> **Current Status:** Phase 2 Complete - Neural Network Basics ✅

---

## 🎯 Project Milestones

- ✅ **Phase 0:** C Fundamentals (Matrix library, memory management)
- ✅ **Phase 1:** Mathematical Foundations (Linear algebra, calculus, gradients)
- ✅ **Phase 2:** Neural Network Basics (2-layer networks, backpropagation, real datasets)
- ⏳ **Phase 3:** NLP Basics (Tokenization, embeddings, text processing)
- ⏳ **Phase 4:** Attention Mechanism
- ⏳ **Phase 5:** Transformer Architecture
- ⏳ **Phase 6:** Training Infrastructure
- ⏳ **Phase 7:** Text Generation

---

## 📊 Current Capabilities

### Successfully Trained On:
- **XOR Problem:** 100% accuracy (classic non-linear classification)
- **Iris Dataset:** 96-100% accuracy (multi-class flower classification)
  - 150 samples, 4 features, 3 classes
  - Proper train/validation split (80/20)
  - Data shuffling implemented

---

## 🏗️ Project Structure
```
PalestineGPT/
├── include/                   # Header files
│   ├── data_loader.h          # Dataset loading utilities
│   ├── gradient_descent.h     # Neural network training
│   └── matrix.h               # Matrix operations library
│
├── src/                       # Source implementations
│   ├── bonus/                 # Example programs
│   │   ├── iris_dataset.c     # Iris classification demo
│   │   └── xor_problem.c      # XOR solver demo
│   ├── data_loader.c          # CSV loading & preprocessing
│   ├── gradient_descent.c     # Training algorithms
│   └── matrix.c               # Matrix library
│
├── tests/                     # Test suite
│   └── comprehensivetesting.c # 13 tests, 100% passing
│
├── others/                    # Data files
│   └── iris.csv               # Iris flower dataset
│
├── .gitignore
├── LICENSE                    # MIT License
├── Makefile                   # Build system
└── README.md                  # Project documentation
```

---

## ⚡ Features

### Matrix Operations (20+ functions)
**Core Operations:**
- Matrix multiplication, addition, subtraction
- Hadamard product (element-wise multiplication)
- Matrix transpose
- Deep copy

**Scalar Operations:**
- Scalar multiplication, addition, subtraction

**Data Utilities:**
- Row extraction (`get_row`)
- Row swapping (`matrix_swap_rows`)
- Dataset shuffling (Fisher-Yates algorithm)

**Data Preprocessing:**
- Min-max normalization `[0, 1]`

### Neural Network Components
**Activations:**
- ReLU (hidden layers)
- Sigmoid (binary classification)
- Softmax (multi-class classification)

**Training:**
- Forward propagation (2-layer with bias)
- Backpropagation (universal matrix-based)
- Momentum-based gradient descent
- Loss functions: MSE, Cross-Entropy

**Data Handling:**
- CSV loading
- One-hot encoding
- Train/validation splitting
- Dataset shuffling

---

## 🚀 Building & Running

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

### Example Output

**Iris Classification:**
```
Epoch 0: Accuracy = 33.33%
Epoch 100: Accuracy = 66.67%
Epoch 200: Accuracy = 90.00%
Epoch 500: Accuracy = 96.67%
Epoch 1000: Accuracy = 100.00%

Final Validation Accuracy: 96.67%
```

**XOR Problem:**
```
Epoch 0: Average Loss = 0.2500
Epoch 10000: Average Loss = 0.0012
Epoch 20000: Average Loss = 0.0001

Final XOR Results:
(0,0) -> 0.0023 (Target: 0.0) ✓
(0,1) -> 0.9981 (Target: 1.0) ✓
(1,0) -> 0.9977 (Target: 1.0) ✓
(1,1) -> 0.0019 (Target: 0.0) ✓
```

---

## 🧪 Testing

**Test Suite:** 13 comprehensive tests covering:
- Core matrix arithmetic
- Transformations & scalar operations
- Neural network activations
- Data preprocessing

---

## 📚 Technical Achievements

### Phase 2 Highlights:
- ✅ **Multi-class classification** (Softmax + Cross-Entropy)
- ✅ **Data preprocessing** (Normalization, one-hot encoding)
- ✅ **Dataset shuffling** (Fisher-Yates algorithm)
- ✅ **Universal backpropagation** (Matrix-based gradients)
- ✅ **Momentum optimization** (Faster, more stable training)
- ✅ **Train/validation splitting** (Proper generalization testing)

### Key Algorithms Implemented:
- Backpropagation through ReLU activations
- Softmax gradient computation
- Momentum-based gradient descent
- Fisher-Yates shuffle
- Min-max feature scaling

---

## 📖 Learning Journey

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

## 🎯 Project Goals

1. **Deep Understanding:** Learn how LLMs work at the lowest level
2. **From Scratch:** No ML frameworks - pure C implementation
3. **Production Quality:** Professional code, documentation, testing
4. **Scalable Architecture:** Build foundations that scale to real models

---

## 🌟 Acknowledgments

- Fisher-Yates shuffle algorithm
- Iris flower dataset (UCI Machine Learning Repository)
- Momentum optimization (Polyak, 1964)

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

## 👤 Author

**Repository:** [PalestineGPT on GitHub](https://github.com/TemmieJoestar/PalestineGPT)

**Goal:** Understand every detail of how LLMs work by implementing one from first principles.

Built from scratch by [TemmieJoestar](https://github.com/TemmieJoestar) as a deep learning journey in C.