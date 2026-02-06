# Contributing to CCC — Claude's C Compiler

Thank you for your interest in contributing to Claude's C Compiler! This document provides guidelines for contributing to the project.

## ⚠️ Important Disclaimer

As noted in the README, 100% of the code and documentation in this repository was written by Claude Opus 4.6. The code has **not been validated for correctness**. Contributions that improve correctness, add tests, or fix bugs are especially welcome.

## Getting Started

### Prerequisites

- **Rust** (stable, 2021 edition) — install via [rustup](https://rustup.rs/)
- **Linux host** — the compiler targets Linux ELF executables

### Building

```bash
cargo build --release
```

### Running Tests

```bash
# Unit tests
cargo test --release

# Integration tests are in the tests/ directory
```

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Include minimal reproduction steps
- Specify which architecture(s) are affected (x86-64, i686, AArch64, RISC-V)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Make your changes
4. Run tests to ensure nothing is broken
5. Commit your changes with clear commit messages
6. Push to your fork and submit a Pull Request

### Good First Contributions

- **Documentation improvements**: Fix typos, clarify explanations, add examples
- **Test cases**: Add new C programs to `tests/` that exercise compiler features
- **Bug fixes**: Check the `ideas/` folder for known issues and improvement ideas
- **Code quality**: Remove dead code, improve comments, fix clippy warnings

## Code Style

- Follow existing Rust conventions in the codebase
- Run `cargo fmt` before committing
- Run `cargo clippy` and address any warnings

## Project Structure

```
src/                Compiler source code (Rust)
  frontend/         C source -> typed AST
  ir/               Target-independent SSA IR
  passes/           SSA optimization passes
  backend/          IR -> assembly -> ELF
  common/           Shared types, diagnostics
  driver/           CLI parsing, pipeline

include/            Bundled C headers
tests/              Integration tests
ideas/              Future work proposals
```

Each `src/` subdirectory has its own `README.md` with detailed documentation.

## Questions?

- Open an issue for questions about the codebase
- See [DESIGN_DOC.md](DESIGN_DOC.md) for architecture details
- Check subdirectory `README.md` files for module-specific documentation

## License

This project is licensed under CC0 1.0 Universal. By contributing, you agree that your contributions will be licensed under the same terms.
