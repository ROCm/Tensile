# Contributing Code

We'd love your help, but...

1. Never check in a tab (\t); use 2 spaces.
2. Make pull requests against *develop* branch.
3. Rebase your develop branch against clMathLibrary's develop branch before pull requesting.
4. In your pull request, state what you tested to ensure that your changes don't break anything. Did you
  - build CobaltBenchmark?
  - run CobaltBenchmark in validation mode?
    - what parameter space did you cover (precisions, dimensions, transposes...)
  - build CobaltLib?
  - run your application with CobaltLib?
  - check performance (if change was for kernel code or kernel-calling code)
  - what is your system setup (GPU, OS, compiler, CMake options)
