# PalestineGPT

Building a Large Language Model (LLM) with Transformer architecture from scratch in C.

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

## License

MIT License - See [LICENSE](LICENSE) file

---

## Author

**Repository:** [PalestineGPT on GitHub](https://github.com/TemmieJoestar/PalestineGPT)

**Goal:** Understand every detail of how LLMs work by implementing one from first principles.

Built from scratch by [TemmieJoestar](https://github.com/TemmieJoestar) as a deep learning journey in C.