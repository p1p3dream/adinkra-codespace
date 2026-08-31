#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace {

constexpr uint32_t kPrimeCount = 3;
constexpr uint32_t kPrimes[kPrimeCount] = {
    1073741783U, 1073741723U, 1073741719U};
constexpr uint32_t kMaximumRank = 320;
constexpr uint32_t kMaximumRightHandSides = 55;
constexpr uint32_t kTargetRows = 10560;

struct Fp2 {
  uint32_t real;
  uint32_t imaginary;
};

struct ThreePrimeFp2 {
  uint32_t lane[6];
};

struct ModularCooEntry {
  uint32_t row;
  uint32_t column;
  ThreePrimeFp2 value;
};

static_assert(sizeof(Fp2) == 8);
static_assert(sizeof(ThreePrimeFp2) == 24);
static_assert(sizeof(ModularCooEntry) == 32);

struct Context {
  uint32_t rank = 0;
  uint32_t right_hand_sides = 0;
  uint32_t width = 0;
  uint64_t resident_bytes = 0;
  uint64_t high_water_bytes = 0;
  uint64_t device_hard_cap = 0;
  Fp2 *augmented = nullptr;
  Fp2 *coefficients = nullptr;
  Fp2 *full_rhs = nullptr;
  uint32_t *row_offsets = nullptr;
  uint32_t *column_indices = nullptr;
  ThreePrimeFp2 *csr_values = nullptr;
  uint64_t csr_capacity = 0;
  uint32_t *invalid = nullptr;
  uint32_t *singular = nullptr;
  uint64_t *residual_counts = nullptr;
  uint64_t *first_residual_key = nullptr;
  ThreePrimeFp2 *first_residual_value = nullptr;
};

void set_error(char *error, uint64_t capacity, const char *message) {
  if (error == nullptr || capacity == 0) return;
  std::snprintf(error, static_cast<size_t>(capacity), "%s", message);
}

bool checked(cudaError_t status, char *error, uint64_t capacity,
             const char *action) {
  if (status == cudaSuccess) return true;
  char buffer[512];
  std::snprintf(buffer, sizeof(buffer), "%s: %s", action,
                cudaGetErrorString(status));
  set_error(error, capacity, buffer);
  return false;
}

__host__ __device__ constexpr uint32_t prime_at(uint32_t index) {
  return index == 0 ? 1073741783U
                    : (index == 1 ? 1073741723U : 1073741719U);
}

__device__ uint32_t add_mod(uint32_t left, uint32_t right, uint32_t prime) {
  const uint64_t sum = static_cast<uint64_t>(left) + right;
  return static_cast<uint32_t>(sum >= prime ? sum - prime : sum);
}

__device__ uint32_t subtract_mod(uint32_t left, uint32_t right,
                                 uint32_t prime) {
  return left >= right ? left - right : left + prime - right;
}

__device__ uint32_t multiply_mod(uint32_t left, uint32_t right,
                                 uint32_t prime) {
  return static_cast<uint32_t>(static_cast<uint64_t>(left) * right % prime);
}

__device__ uint32_t power_mod(uint32_t base, uint32_t exponent,
                              uint32_t prime) {
  uint32_t result = 1;
  while (exponent != 0) {
    if ((exponent & 1U) != 0) result = multiply_mod(result, base, prime);
    base = multiply_mod(base, base, prime);
    exponent >>= 1U;
  }
  return result;
}

__device__ Fp2 add_fp2(Fp2 left, Fp2 right, uint32_t prime) {
  return {add_mod(left.real, right.real, prime),
          add_mod(left.imaginary, right.imaginary, prime)};
}

__device__ Fp2 subtract_fp2(Fp2 left, Fp2 right, uint32_t prime) {
  return {subtract_mod(left.real, right.real, prime),
          subtract_mod(left.imaginary, right.imaginary, prime)};
}

__device__ Fp2 multiply_fp2(Fp2 left, Fp2 right, uint32_t prime) {
  const uint32_t real = subtract_mod(
      multiply_mod(left.real, right.real, prime),
      multiply_mod(left.imaginary, right.imaginary, prime), prime);
  const uint32_t imaginary = add_mod(
      multiply_mod(left.real, right.imaginary, prime),
      multiply_mod(left.imaginary, right.real, prime), prime);
  return {real, imaginary};
}

__device__ Fp2 inverse_fp2(Fp2 value, uint32_t prime) {
  const uint32_t norm = add_mod(
      multiply_mod(value.real, value.real, prime),
      multiply_mod(value.imaginary, value.imaginary, prime), prime);
  const uint32_t inverse_norm = power_mod(norm, prime - 2, prime);
  return {multiply_mod(value.real, inverse_norm, prime),
          value.imaginary == 0
              ? 0
              : multiply_mod(prime - value.imaginary, inverse_norm, prime)};
}

__device__ bool is_zero(Fp2 value) {
  return value.real == 0 && value.imaginary == 0;
}

__global__ void fill_augmented(
    const ModularCooEntry *matrix_entries, uint64_t matrix_count,
    const ModularCooEntry *rhs_entries, uint64_t rhs_count,
    const int32_t *row_to_pivot, const int32_t *column_to_basis,
    uint32_t rank, uint32_t right_hand_sides, uint32_t width,
    Fp2 *augmented, uint32_t *invalid) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t total = matrix_count + rhs_count;
  if (index >= total) return;
  const bool matrix = index < matrix_count;
  const ModularCooEntry entry =
      matrix ? matrix_entries[index] : rhs_entries[index - matrix_count];
  if (entry.row >= kTargetRows) {
    atomicExch(invalid, 1U);
    return;
  }
  const int32_t pivot_row = row_to_pivot[entry.row];
  if (pivot_row < 0) return;
  int32_t output_column;
  if (matrix) {
    if (entry.column >= 1760U) {
      atomicExch(invalid, 1U);
      return;
    }
    output_column = column_to_basis[entry.column];
    if (output_column < 0) return;
  } else {
    if (entry.column >= right_hand_sides) {
      atomicExch(invalid, 1U);
      return;
    }
    output_column = static_cast<int32_t>(rank + entry.column);
  }
  if (static_cast<uint32_t>(pivot_row) >= rank ||
      static_cast<uint32_t>(output_column) >= width) {
    atomicExch(invalid, 1U);
    return;
  }
  for (uint32_t prime_index = 0; prime_index < kPrimeCount; ++prime_index) {
    const uint64_t offset =
        (static_cast<uint64_t>(prime_index) * rank + pivot_row) * width +
        static_cast<uint32_t>(output_column);
    augmented[offset] = {entry.value.lane[2 * prime_index],
                         entry.value.lane[2 * prime_index + 1]};
  }
}

__global__ void solve_augmented(uint32_t rank, uint32_t right_hand_sides,
                                uint32_t width, Fp2 *augmented,
                                Fp2 *coefficients, uint32_t *singular) {
  const uint32_t prime_index = blockIdx.x;
  if (prime_index >= kPrimeCount) return;
  const uint32_t prime = prime_at(prime_index);
  Fp2 *matrix = augmented + static_cast<uint64_t>(prime_index) * rank * width;

  for (uint32_t pivot_column = 0; pivot_column < rank; ++pivot_column) {
    __shared__ uint32_t pivot_row;
    if (threadIdx.x == 0) {
      pivot_row = rank;
      for (uint32_t row = pivot_column; row < rank; ++row) {
        if (!is_zero(matrix[static_cast<uint64_t>(row) * width + pivot_column])) {
          pivot_row = row;
          break;
        }
      }
      if (pivot_row == rank) atomicExch(singular + prime_index, 1U);
    }
    __syncthreads();
    if (pivot_row == rank) return;

    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      if (pivot_row != pivot_column) {
        Fp2 &left = matrix[static_cast<uint64_t>(pivot_column) * width + column];
        Fp2 &right = matrix[static_cast<uint64_t>(pivot_row) * width + column];
        const Fp2 temporary = left;
        left = right;
        right = temporary;
      }
    }
    __syncthreads();

    __shared__ Fp2 inverse;
    if (threadIdx.x == 0) {
      inverse = inverse_fp2(
          matrix[static_cast<uint64_t>(pivot_column) * width + pivot_column],
          prime);
    }
    __syncthreads();
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      Fp2 &value = matrix[static_cast<uint64_t>(pivot_column) * width + column];
      value = multiply_fp2(value, inverse, prime);
    }
    __syncthreads();

    for (uint32_t row = threadIdx.x; row < rank; row += blockDim.x) {
      if (row == pivot_column) continue;
      const Fp2 factor =
          matrix[static_cast<uint64_t>(row) * width + pivot_column];
      if (is_zero(factor)) continue;
      for (uint32_t column = pivot_column; column < width; ++column) {
        Fp2 &value = matrix[static_cast<uint64_t>(row) * width + column];
        const Fp2 pivot_value =
            matrix[static_cast<uint64_t>(pivot_column) * width + column];
        value = subtract_fp2(value, multiply_fp2(factor, pivot_value, prime),
                             prime);
      }
    }
    __syncthreads();
  }

  const uint64_t count = static_cast<uint64_t>(rank) * right_hand_sides;
  for (uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
    const uint32_t row = static_cast<uint32_t>(index / right_hand_sides);
    const uint32_t rhs = static_cast<uint32_t>(index % right_hand_sides);
    coefficients[(index * kPrimeCount) + prime_index] =
        matrix[static_cast<uint64_t>(row) * width + rank + rhs];
  }
}

__global__ void fill_full_rhs(const ModularCooEntry *rhs_entries,
                              uint64_t rhs_count, uint32_t right_hand_sides,
                              Fp2 *full_rhs, uint32_t *invalid) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index >= rhs_count) return;
  const ModularCooEntry entry = rhs_entries[index];
  if (entry.row >= kTargetRows || entry.column >= right_hand_sides) {
    atomicExch(invalid, 1U);
    return;
  }
  for (uint32_t prime_index = 0; prime_index < kPrimeCount; ++prime_index) {
    const uint64_t offset =
        (static_cast<uint64_t>(prime_index) * kTargetRows + entry.row) *
            right_hand_sides +
        entry.column;
    full_rhs[offset] = {entry.value.lane[2 * prime_index],
                        entry.value.lane[2 * prime_index + 1]};
  }
}

__global__ void validate_full_residual(
    const uint32_t *row_offsets, const uint32_t *column_indices,
    const ThreePrimeFp2 *values, uint32_t rank, uint32_t right_hand_sides,
    const Fp2 *coefficients, const Fp2 *full_rhs, uint64_t *residual_counts,
    uint64_t *first_residual_key) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t total = static_cast<uint64_t>(kPrimeCount) * kTargetRows *
                         right_hand_sides;
  if (index >= total) return;
  const uint32_t rhs = static_cast<uint32_t>(index % right_hand_sides);
  const uint64_t quotient = index / right_hand_sides;
  const uint32_t row = static_cast<uint32_t>(quotient % kTargetRows);
  const uint32_t prime_index = static_cast<uint32_t>(quotient / kTargetRows);
  const uint32_t prime = prime_at(prime_index);
  Fp2 sum{};
  for (uint32_t offset = row_offsets[row]; offset < row_offsets[row + 1];
       ++offset) {
    const uint32_t column = column_indices[offset];
    if (column >= rank) continue;
    const ThreePrimeFp2 packed = values[offset];
    const Fp2 value = {packed.lane[2 * prime_index],
                       packed.lane[2 * prime_index + 1]};
    const Fp2 coefficient =
        coefficients[(static_cast<uint64_t>(column) * right_hand_sides + rhs) *
                         kPrimeCount +
                     prime_index];
    sum = add_fp2(sum, multiply_fp2(value, coefficient, prime), prime);
  }
  const Fp2 rhs_value = full_rhs[index];
  const Fp2 residual = subtract_fp2(sum, rhs_value, prime);
  if (is_zero(residual)) return;
  atomicAdd(reinterpret_cast<unsigned long long *>(residual_counts + prime_index),
            1ULL);
  const uint64_t key =
      (static_cast<uint64_t>(prime_index) * kTargetRows + row) *
          right_hand_sides +
      rhs;
  atomicMin(
      reinterpret_cast<unsigned long long *>(first_residual_key),
      static_cast<unsigned long long>(key));
}

__global__ void capture_first_residual(
    const uint32_t *row_offsets, const uint32_t *column_indices,
    const ThreePrimeFp2 *values, uint32_t rank, uint32_t right_hand_sides,
    const Fp2 *coefficients, const Fp2 *full_rhs,
    const uint64_t *first_residual_key, ThreePrimeFp2 *first_residual_value) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const uint64_t key = *first_residual_key;
  if (key == UINT64_MAX) return;
  const uint32_t rhs = static_cast<uint32_t>(key % right_hand_sides);
  const uint64_t quotient = key / right_hand_sides;
  const uint32_t row = static_cast<uint32_t>(quotient % kTargetRows);
  const uint32_t prime_index = static_cast<uint32_t>(quotient / kTargetRows);
  if (prime_index >= kPrimeCount) return;
  const uint32_t prime = prime_at(prime_index);
  Fp2 sum{};
  for (uint32_t offset = row_offsets[row]; offset < row_offsets[row + 1];
       ++offset) {
    const uint32_t column = column_indices[offset];
    const ThreePrimeFp2 packed = values[offset];
    const Fp2 value = {packed.lane[2 * prime_index],
                       packed.lane[2 * prime_index + 1]};
    const Fp2 coefficient =
        coefficients[(static_cast<uint64_t>(column) * right_hand_sides + rhs) *
                         kPrimeCount +
                     prime_index];
    sum = add_fp2(sum, multiply_fp2(value, coefficient, prime), prime);
  }
  const uint64_t dense =
      (static_cast<uint64_t>(prime_index) * kTargetRows + row) *
          right_hand_sides +
      rhs;
  const Fp2 residual = subtract_fp2(sum, full_rhs[dense], prime);
  ThreePrimeFp2 packed{};
  packed.lane[2 * prime_index] = residual.real;
  packed.lane[2 * prime_index + 1] = residual.imaginary;
  *first_residual_value = packed;
}

void destroy(Context *context) {
  if (context == nullptr) return;
  cudaFree(context->augmented);
  cudaFree(context->coefficients);
  cudaFree(context->full_rhs);
  cudaFree(context->row_offsets);
  cudaFree(context->column_indices);
  cudaFree(context->csr_values);
  cudaFree(context->invalid);
  cudaFree(context->singular);
  cudaFree(context->residual_counts);
  cudaFree(context->first_residual_key);
  cudaFree(context->first_residual_value);
  delete context;
}

}  // namespace

extern "C" void *adynkra_teleparallel_lorentz_descent_create(
    uint32_t rank, uint32_t right_hand_sides, uint64_t csr_capacity,
    uint64_t device_hard_cap, char *error, uint64_t error_capacity) {
  if (rank == 0 || rank > kMaximumRank || right_hand_sides == 0 ||
      right_hand_sides > kMaximumRightHandSides || csr_capacity == 0 ||
      csr_capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    set_error(error, error_capacity, "invalid Lorentz-descent dimensions");
    return nullptr;
  }
  Context *context = new (std::nothrow) Context();
  if (context == nullptr) {
    set_error(error, error_capacity, "allocate Lorentz-descent context");
    return nullptr;
  }
  context->rank = rank;
  context->right_hand_sides = right_hand_sides;
  context->width = rank + right_hand_sides;
  context->csr_capacity = csr_capacity;
  context->device_hard_cap = device_hard_cap;
  const uint64_t augmented_count =
      static_cast<uint64_t>(kPrimeCount) * rank * context->width;
  const uint64_t coefficient_count =
      static_cast<uint64_t>(rank) * right_hand_sides * kPrimeCount;
  const uint64_t full_rhs_count =
      static_cast<uint64_t>(kPrimeCount) * kTargetRows * right_hand_sides;
  context->resident_bytes =
      augmented_count * sizeof(Fp2) + coefficient_count * sizeof(Fp2) +
      full_rhs_count * sizeof(Fp2) + (kTargetRows + 1) * sizeof(uint32_t) +
      csr_capacity * (sizeof(uint32_t) + sizeof(ThreePrimeFp2)) +
      sizeof(uint32_t) * (1 + kPrimeCount) + sizeof(uint64_t) * 4 +
      sizeof(ThreePrimeFp2);
  context->high_water_bytes = context->resident_bytes;
  if (device_hard_cap != 0 && context->resident_bytes > device_hard_cap) {
    set_error(error, error_capacity, "Lorentz-descent buffers exceed device cap");
    destroy(context);
    return nullptr;
  }
  if (!checked(cudaMalloc(&context->augmented,
                          augmented_count * sizeof(Fp2)),
               error, error_capacity, "allocate descent augmented matrix") ||
      !checked(cudaMalloc(&context->coefficients,
                          coefficient_count * sizeof(Fp2)),
               error, error_capacity, "allocate descent coefficients") ||
      !checked(cudaMalloc(&context->full_rhs, full_rhs_count * sizeof(Fp2)),
               error, error_capacity, "allocate descent full RHS") ||
      !checked(cudaMalloc(&context->row_offsets,
                          (kTargetRows + 1) * sizeof(uint32_t)),
               error, error_capacity, "allocate descent row offsets") ||
      !checked(cudaMalloc(&context->column_indices,
                          csr_capacity * sizeof(uint32_t)),
               error, error_capacity, "allocate descent column indices") ||
      !checked(cudaMalloc(&context->csr_values,
                          csr_capacity * sizeof(ThreePrimeFp2)),
               error, error_capacity, "allocate descent CSR values") ||
      !checked(cudaMalloc(&context->invalid, sizeof(uint32_t)), error,
               error_capacity, "allocate descent invalid flag") ||
      !checked(cudaMalloc(&context->singular,
                          kPrimeCount * sizeof(uint32_t)),
               error, error_capacity, "allocate descent singular flags") ||
      !checked(cudaMalloc(&context->residual_counts,
                          kPrimeCount * sizeof(uint64_t)),
               error, error_capacity, "allocate descent residual counts") ||
      !checked(cudaMalloc(&context->first_residual_key, sizeof(uint64_t)),
               error, error_capacity, "allocate descent first residual key") ||
      !checked(cudaMalloc(&context->first_residual_value,
                          sizeof(ThreePrimeFp2)),
               error, error_capacity, "allocate descent first residual value")) {
    destroy(context);
    return nullptr;
  }
  return context;
}

extern "C" int32_t adynkra_teleparallel_lorentz_descent_solve(
    void *opaque, const ModularCooEntry *host_matrix_entries,
    uint64_t matrix_count, const ModularCooEntry *host_rhs_entries,
    uint64_t rhs_count, const int32_t *host_row_to_pivot,
    const int32_t *host_column_to_basis, const uint32_t *host_row_offsets,
    const uint32_t *host_column_indices,
    const ThreePrimeFp2 *host_csr_values, uint64_t csr_count,
    ThreePrimeFp2 *host_coefficients, uint64_t coefficient_capacity,
    uint64_t host_residual_counts[3], uint64_t *host_first_residual_key,
    ThreePrimeFp2 *host_first_residual_value, float *host_device_milliseconds,
    char *error, uint64_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || host_matrix_entries == nullptr ||
      host_rhs_entries == nullptr || host_row_to_pivot == nullptr ||
      host_column_to_basis == nullptr || host_row_offsets == nullptr ||
      host_column_indices == nullptr || host_csr_values == nullptr ||
      host_coefficients == nullptr || host_residual_counts == nullptr ||
      host_first_residual_key == nullptr || host_first_residual_value == nullptr ||
      host_device_milliseconds == nullptr) {
    set_error(error, error_capacity, "null Lorentz-descent solve argument");
    return 1;
  }
  const uint64_t coefficient_count =
      static_cast<uint64_t>(context->rank) * context->right_hand_sides;
  if (matrix_count == 0 || rhs_count == 0 || csr_count == 0 ||
      csr_count > context->csr_capacity ||
      coefficient_capacity < coefficient_count) {
    set_error(error, error_capacity, "Lorentz-descent solve capacity mismatch");
    return 2;
  }

  ModularCooEntry *matrix_entries = nullptr;
  ModularCooEntry *rhs_entries = nullptr;
  int32_t *row_to_pivot = nullptr;
  int32_t *column_to_basis = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  const uint64_t matrix_bytes = matrix_count * sizeof(ModularCooEntry);
  const uint64_t rhs_bytes = rhs_count * sizeof(ModularCooEntry);
  const uint64_t mapping_bytes =
      kTargetRows * sizeof(int32_t) + 1760 * sizeof(int32_t);
  context->high_water_bytes = context->resident_bytes + matrix_bytes + rhs_bytes +
                              mapping_bytes;
  if (context->device_hard_cap != 0 &&
      context->high_water_bytes > context->device_hard_cap) {
    set_error(error, error_capacity,
              "Lorentz-descent transient buffers exceed device cap");
    return 3;
  }
  if (!checked(cudaMalloc(&matrix_entries, matrix_bytes), error, error_capacity,
               "allocate descent matrix COO") ||
      !checked(cudaMalloc(&rhs_entries, rhs_bytes), error, error_capacity,
               "allocate descent RHS COO") ||
      !checked(cudaMalloc(&row_to_pivot, kTargetRows * sizeof(int32_t)), error,
               error_capacity, "allocate descent row map") ||
      !checked(cudaMalloc(&column_to_basis, 1760 * sizeof(int32_t)), error,
               error_capacity, "allocate descent column map") ||
      !checked(cudaEventCreate(&start), error, error_capacity,
               "create descent start event") ||
      !checked(cudaEventCreate(&stop), error, error_capacity,
               "create descent stop event")) {
    cudaFree(matrix_entries);
    cudaFree(rhs_entries);
    cudaFree(row_to_pivot);
    cudaFree(column_to_basis);
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    return 3;
  }

  bool ok =
      checked(cudaMemcpy(matrix_entries, host_matrix_entries, matrix_bytes,
                         cudaMemcpyHostToDevice),
              error, error_capacity, "upload descent matrix COO") &&
      checked(cudaMemcpy(rhs_entries, host_rhs_entries, rhs_bytes,
                         cudaMemcpyHostToDevice),
              error, error_capacity, "upload descent RHS COO") &&
      checked(cudaMemcpy(row_to_pivot, host_row_to_pivot,
                         kTargetRows * sizeof(int32_t), cudaMemcpyHostToDevice),
              error, error_capacity, "upload descent row map") &&
      checked(cudaMemcpy(column_to_basis, host_column_to_basis,
                         1760 * sizeof(int32_t), cudaMemcpyHostToDevice),
              error, error_capacity, "upload descent column map") &&
      checked(cudaMemcpy(context->row_offsets, host_row_offsets,
                         (kTargetRows + 1) * sizeof(uint32_t),
                         cudaMemcpyHostToDevice),
              error, error_capacity, "upload descent row offsets") &&
      checked(cudaMemcpy(context->column_indices, host_column_indices,
                         csr_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
              error, error_capacity, "upload descent column indices") &&
      checked(cudaMemcpy(context->csr_values, host_csr_values,
                         csr_count * sizeof(ThreePrimeFp2),
                         cudaMemcpyHostToDevice),
              error, error_capacity, "upload descent CSR values") &&
      checked(cudaMemset(context->augmented, 0,
                         static_cast<uint64_t>(kPrimeCount) * context->rank *
                             context->width * sizeof(Fp2)),
              error, error_capacity, "clear descent augmented matrix") &&
      checked(cudaMemset(context->full_rhs, 0,
                         static_cast<uint64_t>(kPrimeCount) * kTargetRows *
                             context->right_hand_sides * sizeof(Fp2)),
              error, error_capacity, "clear descent full RHS") &&
      checked(cudaMemset(context->invalid, 0, sizeof(uint32_t)), error,
              error_capacity, "clear descent invalid flag") &&
      checked(cudaMemset(context->singular, 0,
                         kPrimeCount * sizeof(uint32_t)),
              error, error_capacity, "clear descent singular flags") &&
      checked(cudaMemset(context->residual_counts, 0,
                         kPrimeCount * sizeof(uint64_t)),
              error, error_capacity, "clear descent residual counts") &&
      checked(cudaMemset(context->first_residual_value, 0,
                         sizeof(ThreePrimeFp2)),
              error, error_capacity, "clear descent first residual value");
  const uint64_t no_residual = std::numeric_limits<uint64_t>::max();
  if (ok) {
    ok = checked(cudaMemcpy(context->first_residual_key, &no_residual,
                            sizeof(uint64_t), cudaMemcpyHostToDevice),
                 error, error_capacity, "clear descent first residual key") &&
         checked(cudaEventRecord(start), error, error_capacity,
                 "record descent start");
  }
  if (ok) {
    const uint64_t total_entries = matrix_count + rhs_count;
    fill_augmented<<<static_cast<uint32_t>((total_entries + 255) / 256), 256>>>(
        matrix_entries, matrix_count, rhs_entries, rhs_count, row_to_pivot,
        column_to_basis, context->rank, context->right_hand_sides,
        context->width, context->augmented, context->invalid);
    fill_full_rhs<<<static_cast<uint32_t>((rhs_count + 255) / 256), 256>>>(
        rhs_entries, rhs_count, context->right_hand_sides, context->full_rhs,
        context->invalid);
    solve_augmented<<<kPrimeCount, 320>>>(
        context->rank, context->right_hand_sides, context->width,
        context->augmented, context->coefficients, context->singular);
    const uint64_t validation_count =
        static_cast<uint64_t>(kPrimeCount) * kTargetRows *
        context->right_hand_sides;
    validate_full_residual<<<
        static_cast<uint32_t>((validation_count + 255) / 256), 256>>>(
        context->row_offsets, context->column_indices, context->csr_values,
        context->rank, context->right_hand_sides, context->coefficients,
        context->full_rhs, context->residual_counts,
        context->first_residual_key);
    capture_first_residual<<<1, 1>>>(
        context->row_offsets, context->column_indices, context->csr_values,
        context->rank, context->right_hand_sides, context->coefficients,
        context->full_rhs, context->first_residual_key,
        context->first_residual_value);
    ok = checked(cudaGetLastError(), error, error_capacity,
                 "launch Lorentz-descent kernels") &&
         checked(cudaEventRecord(stop), error, error_capacity,
                 "record descent stop") &&
         checked(cudaEventSynchronize(stop), error, error_capacity,
                 "synchronize Lorentz descent");
  }

  uint32_t invalid = 0;
  uint32_t singular[kPrimeCount] = {};
  if (ok) {
    ok = checked(cudaMemcpy(&invalid, context->invalid, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost),
                 error, error_capacity, "download descent invalid flag") &&
         checked(cudaMemcpy(singular, context->singular,
                            sizeof(singular), cudaMemcpyDeviceToHost),
                 error, error_capacity, "download descent singular flags") &&
         checked(cudaMemcpy(host_coefficients, context->coefficients,
                            coefficient_count * kPrimeCount * sizeof(Fp2),
                            cudaMemcpyDeviceToHost),
                 error, error_capacity, "download descent coefficients") &&
         checked(cudaMemcpy(host_residual_counts, context->residual_counts,
                            kPrimeCount * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost),
                 error, error_capacity, "download descent residual counts") &&
         checked(cudaMemcpy(host_first_residual_key,
                            context->first_residual_key, sizeof(uint64_t),
                            cudaMemcpyDeviceToHost),
                 error, error_capacity, "download descent first residual key") &&
         checked(cudaMemcpy(host_first_residual_value,
                            context->first_residual_value,
                            sizeof(ThreePrimeFp2), cudaMemcpyDeviceToHost),
                 error, error_capacity, "download descent first residual value") &&
         checked(cudaEventElapsedTime(host_device_milliseconds, start, stop),
                 error, error_capacity, "measure Lorentz descent");
  }

  cudaFree(matrix_entries);
  cudaFree(rhs_entries);
  cudaFree(row_to_pivot);
  cudaFree(column_to_basis);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (!ok) return 4;
  if (invalid != 0) {
    set_error(error, error_capacity, "invalid Lorentz-descent COO input");
    return 5;
  }
  if (singular[0] != 0 || singular[1] != 0 || singular[2] != 0) {
    set_error(error, error_capacity,
              "Lorentz-descent pivot minor is singular at a pinned prime");
    return 6;
  }
  return 0;
}

extern "C" uint64_t adynkra_teleparallel_lorentz_descent_resident_bytes(
    const void *opaque) {
  const Context *context = static_cast<const Context *>(opaque);
  return context == nullptr ? 0 : context->resident_bytes;
}

extern "C" void adynkra_teleparallel_lorentz_descent_primes(
    uint32_t output[3]) {
  if (output == nullptr) return;
  output[0] = kPrimes[0];
  output[1] = kPrimes[1];
  output[2] = kPrimes[2];
}

extern "C" uint64_t adynkra_teleparallel_lorentz_descent_high_water_bytes(
    const void *opaque) {
  const Context *context = static_cast<const Context *>(opaque);
  return context == nullptr ? 0 : context->high_water_bytes;
}

extern "C" void adynkra_teleparallel_lorentz_descent_destroy(void *opaque) {
  destroy(static_cast<Context *>(opaque));
}
