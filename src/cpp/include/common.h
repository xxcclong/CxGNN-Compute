#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#include "Yaml.hpp"
#include "spdlog/spdlog.h"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

using Index = int64_t;
#define ASSERTWITH(condition, args...) \
  if (unlikely(!(condition))) {        \
    SPDLOG_WARN(args);                 \
    exit(1);                           \
  }

#define ASSERT(condition)          \
  if (unlikely(!(condition))) {    \
    SPDLOG_WARN("ASSERT FAILURE"); \
    exit(1);                       \
  }

#define checkCudaErrors(status)                             \
  do {                                                      \
    if (status != 0) {                                      \
      fprintf(stderr, "CUDA failure at [%s] (%s:%d): %s\n", \
              __PRETTY_FUNCTION__, __FILE__, __LINE__,      \
              cudaGetErrorString(status));                  \
      cudaDeviceReset();                                    \
      abort();                                              \
    }                                                       \
  } while (0)
