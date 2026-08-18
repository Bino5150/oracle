# Contributing

The project is currently architecture-first. Keep changes small, measurable, and reversible.

- Add tests for behavior changes.
- Preserve the CPU reference path.
- Avoid backend-specific types in public runtime interfaces.
- Include benchmark evidence with performance claims.
- Do not add a dependency when a small standard-library implementation is sufficient.
- Run \ctest --test-dir build --output-on-failure` before submitting a pull request.
