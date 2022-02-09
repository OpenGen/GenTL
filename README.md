# Gen Inference and Learning Template Library (GenTL)
Template inference and learning library in C++ based on probabilistic programming  principles

[![status](https://github.com/OpenGen/GenTL/actions/workflows/test.yml/badge.svg)](https://github.com/OpenGen/GenTL/actions?query=workflow/test)

[**Documentation**](https://opengen.github.io/gentl-docs/latest/)

## Dependencies

### C++ compiler

Most of the library uses C++17 features, which are supported by recent clang++ and g++ versions.

Currently, `include/gentl/learning.h` uses C++20 features, which requires g++ version 11 or above, and clang++ version 11 or above.

### CMake

CMake 3.7 or above. See [Installing CMake](https://cmake.org/install/).

### Doxygen

This is only required for generating documentation.

See [Doxygen Installation](https://www.doxygen.nl/manual/install.html).

## Testing

```
cmake -S . -B build
cmake --build build
cmake --build --target test
```

## Generating documentation
```
cmake --build --target docs
```



