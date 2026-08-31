#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <limits>

namespace {

constexpr uint32_t kColumns = 56;
constexpr uint64_t kRows21 = UINT64_C(18436915200);
constexpr uint64_t kRows02 = UINT64_C(223027200);
constexpr uint64_t kRows = kRows21 + kRows02;
constexpr uint32_t kPrimes[3] = {1073741783U, 1073741723U, 1073741719U};

__host__ __device__ constexpr uint32_t prime_at(uint32_t index) {
  return index == 0 ? 1073741783U
                    : (index == 1 ? 1073741723U : 1073741719U);
}

struct HostEntry {
  uint64_t row;
  uint32_t column;
  uint32_t reserved;
  int64_t real;
  int64_t imaginary;
};

struct PrimeValue {
  uint32_t value[6];
};

static_assert(sizeof(HostEntry) == 32);
static_assert(sizeof(PrimeValue) == 24);

struct Context {
  uint64_t capacity = 0;
  uint64_t resident_bytes = 0;
  uint64_t high_water_bytes = 0;
  uint64_t *keys_in = nullptr;
  uint64_t *keys_out = nullptr;
  PrimeValue *values_in = nullptr;
  PrimeValue *values_out = nullptr;
  uint32_t *invalid = nullptr;
  uint32_t *unique_count = nullptr;
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
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

uint64_t gcd_u64(uint64_t left, uint64_t right) {
  while (right != 0) {
    const uint64_t next = left % right;
    left = right;
    right = next;
  }
  return left;
}

uint32_t pow_mod(uint32_t base, uint32_t exponent, uint32_t prime) {
  uint64_t result = 1;
  uint64_t factor = base;
  while (exponent != 0) {
    if ((exponent & 1U) != 0) result = result * factor % prime;
    factor = factor * factor % prime;
    exponent >>= 1U;
  }
  return static_cast<uint32_t>(result);
}

__device__ uint32_t signed_mod(int64_t value, uint32_t prime) {
  const int64_t residue = value % static_cast<int64_t>(prime);
  return static_cast<uint32_t>(residue < 0 ? residue + prime : residue);
}

__global__ void encode_entries(const HostEntry *entries, uint64_t count,
                               const uint32_t *denominator_inverse,
                               uint64_t *keys, PrimeValue *values,
                               uint32_t *invalid) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index >= count) return;
  const HostEntry entry = entries[index];
  if (entry.row >= kRows || entry.column >= kColumns || entry.reserved != 0) {
    atomicExch(invalid, 1U);
    keys[index] = 0;
    values[index] = {};
    return;
  }
  keys[index] = entry.row * static_cast<uint64_t>(kColumns) + entry.column;
  PrimeValue output{};
#pragma unroll
  for (uint32_t prime_index = 0; prime_index < 3; ++prime_index) {
    const uint32_t prime = prime_at(prime_index);
    const uint64_t inverse = denominator_inverse[prime_index];
    output.value[2 * prime_index] = static_cast<uint32_t>(
        static_cast<uint64_t>(signed_mod(entry.real, prime)) * inverse % prime);
    output.value[2 * prime_index + 1] = static_cast<uint32_t>(
        static_cast<uint64_t>(signed_mod(entry.imaginary, prime)) * inverse %
        prime);
  }
  values[index] = output;
}

struct PrimeAdd {
  __host__ __device__ PrimeValue operator()(const PrimeValue &left,
                                             const PrimeValue &right) const {
    PrimeValue output{};
#pragma unroll
    for (uint32_t prime_index = 0; prime_index < 3; ++prime_index) {
      const uint32_t prime = prime_at(prime_index);
#pragma unroll
      for (uint32_t component = 0; component < 2; ++component) {
        const uint32_t lane = 2 * prime_index + component;
        const uint64_t sum = static_cast<uint64_t>(left.value[lane]) +
                             static_cast<uint64_t>(right.value[lane]);
        output.value[lane] =
            static_cast<uint32_t>(sum >= prime ? sum - prime : sum);
      }
    }
    return output;
  }
};

void destroy(Context *context) {
  if (context == nullptr) return;
  cudaFree(context->keys_in);
  cudaFree(context->keys_out);
  cudaFree(context->values_in);
  cudaFree(context->values_out);
  cudaFree(context->invalid);
  cudaFree(context->unique_count);
  cudaFree(context->temporary);
  delete context;
}

}  // namespace

extern "C" void *adynkra_four_form_56_create(uint64_t capacity,
                                               uint64_t device_hard_cap,
                                               char *error,
                                               uint64_t error_capacity) {
  if (capacity == 0) {
    set_error(error, error_capacity, "four-form capacity is zero");
    return nullptr;
  }
  if (capacity > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    set_error(error, error_capacity, "four-form capacity exceeds CUB item bound");
    return nullptr;
  }
  size_t sort_bytes = 0;
  size_t reduce_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, sort_bytes, static_cast<uint64_t *>(nullptr),
      static_cast<uint64_t *>(nullptr), static_cast<PrimeValue *>(nullptr),
      static_cast<PrimeValue *>(nullptr), static_cast<int>(capacity));
  cub::DeviceReduce::ReduceByKey(
      nullptr, reduce_bytes, static_cast<uint64_t *>(nullptr),
      static_cast<uint64_t *>(nullptr), static_cast<PrimeValue *>(nullptr),
      static_cast<PrimeValue *>(nullptr), static_cast<uint32_t *>(nullptr),
      PrimeAdd(), static_cast<int>(capacity));
  const size_t temporary_bytes = sort_bytes > reduce_bytes ? sort_bytes : reduce_bytes;
  const uint64_t resident = capacity *
      (2 * sizeof(uint64_t) + 2 * sizeof(PrimeValue)) + 2 * sizeof(uint32_t) +
      temporary_bytes;
  if (device_hard_cap != 0 && resident > device_hard_cap) {
    set_error(error, error_capacity, "four-form fixed buffers exceed device cap");
    return nullptr;
  }
  Context *context = new (std::nothrow) Context();
  if (context == nullptr) {
    set_error(error, error_capacity, "allocate four-form CUDA context");
    return nullptr;
  }
  context->capacity = capacity;
  context->resident_bytes = resident;
  context->high_water_bytes = resident;
  context->temporary_bytes = temporary_bytes;
  if (!checked(cudaMalloc(&context->keys_in, capacity * sizeof(uint64_t)), error,
               error_capacity, "allocate four-form input keys") ||
      !checked(cudaMalloc(&context->keys_out, capacity * sizeof(uint64_t)), error,
               error_capacity, "allocate four-form output keys") ||
      !checked(cudaMalloc(&context->values_in, capacity * sizeof(PrimeValue)),
               error, error_capacity, "allocate four-form input values") ||
      !checked(cudaMalloc(&context->values_out, capacity * sizeof(PrimeValue)),
               error, error_capacity, "allocate four-form output values") ||
      !checked(cudaMalloc(&context->invalid, sizeof(uint32_t)), error,
               error_capacity, "allocate four-form invalid flag") ||
      !checked(cudaMalloc(&context->unique_count, sizeof(uint32_t)), error,
               error_capacity, "allocate four-form unique count") ||
      !checked(cudaMalloc(&context->temporary, temporary_bytes), error,
               error_capacity, "allocate four-form CUB workspace")) {
    destroy(context);
    return nullptr;
  }
  return context;
}

extern "C" int32_t adynkra_four_form_56_reduce(
    void *opaque, const HostEntry *host_entries, uint64_t count,
    uint64_t common_denominator, uint64_t *host_keys,
    PrimeValue *host_values, uint64_t output_capacity, uint64_t *output_count,
    uint64_t *input_terms, char *error, uint64_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || host_entries == nullptr || host_keys == nullptr ||
      host_values == nullptr || output_count == nullptr || input_terms == nullptr) {
    set_error(error, error_capacity, "null four-form reduce argument");
    return 1;
  }
  if (count == 0 || count > context->capacity || output_capacity < count) {
    set_error(error, error_capacity, "four-form reduce capacity mismatch");
    return 2;
  }
  if (common_denominator == 0) {
    set_error(error, error_capacity, "four-form denominator is zero");
    return 3;
  }
  uint32_t inverse[3];
  for (uint32_t prime_index = 0; prime_index < 3; ++prime_index) {
    if (gcd_u64(common_denominator, kPrimes[prime_index]) != 1) {
      set_error(error, error_capacity,
                "four-form denominator is not invertible at a pinned prime");
      return 4;
    }
    const uint32_t residue =
        static_cast<uint32_t>(common_denominator % kPrimes[prime_index]);
    inverse[prime_index] = pow_mod(residue, kPrimes[prime_index] - 2,
                                   kPrimes[prime_index]);
  }
  HostEntry *device_entries = nullptr;
  uint32_t *device_inverse = nullptr;
  if (!checked(cudaMalloc(&device_entries, count * sizeof(HostEntry)), error,
               error_capacity, "allocate four-form entry batch") ||
      !checked(cudaMalloc(&device_inverse, sizeof(inverse)), error,
               error_capacity, "allocate four-form denominator inverse") ||
      !checked(cudaMemcpy(device_entries, host_entries, count * sizeof(HostEntry),
                          cudaMemcpyHostToDevice),
               error, error_capacity, "upload four-form entries") ||
      !checked(cudaMemcpy(device_inverse, inverse, sizeof(inverse),
                          cudaMemcpyHostToDevice),
               error, error_capacity, "upload four-form denominator inverse") ||
      !checked(cudaMemset(context->invalid, 0, sizeof(uint32_t)), error,
               error_capacity, "clear four-form invalid flag")) {
    cudaFree(device_entries);
    cudaFree(device_inverse);
    return 5;
  }
  context->high_water_bytes = context->resident_bytes +
      count * sizeof(HostEntry) + sizeof(inverse);
  const uint32_t threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((count + threads - 1) / threads);
  encode_entries<<<blocks, threads>>>(device_entries, count, device_inverse,
                                     context->keys_in, context->values_in,
                                     context->invalid);
  if (!checked(cudaGetLastError(), error, error_capacity,
               "launch four-form encoder") ||
      !checked(cudaDeviceSynchronize(), error, error_capacity,
               "synchronize four-form encoder")) {
    cudaFree(device_entries);
    cudaFree(device_inverse);
    return 6;
  }
  uint32_t invalid = 0;
  const bool downloaded_invalid =
      checked(cudaMemcpy(&invalid, context->invalid, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
              error, error_capacity, "download four-form invalid flag");
  cudaFree(device_entries);
  cudaFree(device_inverse);
  if (!downloaded_invalid) return 7;
  if (invalid != 0) {
    set_error(error, error_capacity, "invalid four-form row or column ordinal");
    return 8;
  }

  if (!checked(cub::DeviceRadixSort::SortPairs(
                   context->temporary, context->temporary_bytes,
                   context->keys_in, context->keys_out, context->values_in,
                   context->values_out, static_cast<int>(count)),
               error, error_capacity, "sort four-form COO") ||
      !checked(cub::DeviceReduce::ReduceByKey(
                   context->temporary, context->temporary_bytes,
                   context->keys_out, context->keys_in, context->values_out,
                   context->values_in, context->unique_count, PrimeAdd(),
                   static_cast<int>(count)),
               error, error_capacity, "reduce four-form COO") ||
      !checked(cudaDeviceSynchronize(), error, error_capacity,
               "synchronize four-form COO reduction")) {
    return 9;
  }
  uint32_t reduced_count = 0;
  if (!checked(cudaMemcpy(&reduced_count, context->unique_count, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download four-form unique count") ||
      !checked(cudaMemcpy(host_keys, context->keys_in,
                          static_cast<uint64_t>(reduced_count) * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download four-form keys") ||
      !checked(cudaMemcpy(host_values, context->values_in,
                          static_cast<uint64_t>(reduced_count) * sizeof(PrimeValue),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download four-form residues")) {
    return 10;
  }
  *output_count = reduced_count;
  *input_terms = count;
  return 0;
}

extern "C" uint64_t adynkra_four_form_56_resident_bytes(const void *opaque) {
  const Context *context = static_cast<const Context *>(opaque);
  return context == nullptr ? 0 : context->resident_bytes;
}

extern "C" uint64_t adynkra_four_form_56_high_water_bytes(const void *opaque) {
  const Context *context = static_cast<const Context *>(opaque);
  return context == nullptr ? 0 : context->high_water_bytes;
}

extern "C" void adynkra_four_form_56_destroy(void *opaque) {
  destroy(static_cast<Context *>(opaque));
}
