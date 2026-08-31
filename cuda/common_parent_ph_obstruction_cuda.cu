#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

constexpr uint32_t kPrimeCount = 3;
constexpr uint32_t kPrimes[kPrimeCount] = {1073741783U, 1073741723U,
                                           1073741719U};
constexpr uint32_t kMaximumColumns = 4096;

struct Fp2 {
  uint32_t real;
  uint32_t imaginary;
};

struct ThreePrimeFp2 {
  uint32_t lane[6];
};

struct PackedPhCooEntry {
  uint64_t row_key;
  uint32_t column;
  uint32_t reserved;
  ThreePrimeFp2 value;
};

struct SparseKey {
  uint64_t row_key;
  uint32_t column;
  uint32_t reserved;
};

struct Status {
  uint32_t ranks[3];
  uint32_t obstruction_prime;
  uint32_t obstruction_pivot_column;
  uint32_t stopped;
  uint32_t invalid;
  uint64_t obstruction_row_key;
  uint64_t batches_submitted;
  uint64_t input_entries;
  uint64_t reduced_entries;
  uint64_t rows_visited;
};

static_assert(sizeof(Fp2) == 8);
static_assert(sizeof(ThreePrimeFp2) == 24);
static_assert(sizeof(PackedPhCooEntry) == 40);
static_assert(sizeof(SparseKey) == 16);

struct Slot {
  PackedPhCooEntry *host_entries = nullptr;
  PackedPhCooEntry *device_entries = nullptr;
  SparseKey *keys = nullptr;
  SparseKey *reduced_keys = nullptr;
  ThreePrimeFp2 *values = nullptr;
  ThreePrimeFp2 *reduced_values = nullptr;
  uint64_t *reduced_count = nullptr;
  cudaStream_t upload_stream = nullptr;
  cudaEvent_t upload_done = nullptr;
  cudaEvent_t reusable = nullptr;
};

struct Context {
  uint32_t columns = 0;
  uint32_t expected_rank = 0;
  uint32_t maximum_rank = 0;
  uint64_t batch_capacity = 0;
  uint64_t device_hard_cap = 0;
  uint64_t resident_bytes = 0;
  uint64_t high_water_bytes = 0;
  uint64_t next_batch_ordinal = 0;
  uint64_t last_row_key = 0;
  bool has_last_row = false;
  Slot slots[2];
  cudaStream_t compute_stream = nullptr;
  Fp2 *basis = nullptr;
  int32_t *pivot_for_column = nullptr;
  uint32_t *pivot_columns = nullptr;
  uint64_t *pivot_row_keys = nullptr;
  Status *status = nullptr;
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

__host__ __device__ uint32_t add_mod(uint32_t left, uint32_t right,
                                     uint32_t prime) {
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

__device__ Fp2 subtract_fp2(Fp2 left, Fp2 right, uint32_t prime) {
  return {subtract_mod(left.real, right.real, prime),
          subtract_mod(left.imaginary, right.imaginary, prime)};
}

__device__ Fp2 multiply_fp2(Fp2 left, Fp2 right, uint32_t prime) {
  return {subtract_mod(multiply_mod(left.real, right.real, prime),
                       multiply_mod(left.imaginary, right.imaginary, prime),
                       prime),
          add_mod(multiply_mod(left.real, right.imaginary, prime),
                  multiply_mod(left.imaginary, right.real, prime), prime)};
}

__device__ Fp2 inverse_fp2(Fp2 value, uint32_t prime) {
  const uint32_t norm = add_mod(multiply_mod(value.real, value.real, prime),
                                multiply_mod(value.imaginary,
                                             value.imaginary, prime),
                                prime);
  const uint32_t inverse_norm = power_mod(norm, prime - 2, prime);
  return {multiply_mod(value.real, inverse_norm, prime),
          value.imaginary == 0
              ? 0
              : multiply_mod(prime - value.imaginary, inverse_norm, prime)};
}

__device__ bool is_zero(Fp2 value) {
  return value.real == 0 && value.imaginary == 0;
}

struct KeyLess {
  __host__ __device__ bool operator()(const SparseKey &left,
                                      const SparseKey &right) const {
    return left.row_key < right.row_key ||
           (left.row_key == right.row_key && left.column < right.column);
  }
};

struct KeyEqual {
  __host__ __device__ bool operator()(const SparseKey &left,
                                      const SparseKey &right) const {
    return left.row_key == right.row_key && left.column == right.column;
  }
};

struct AddThreePrime {
  __host__ __device__ ThreePrimeFp2 operator()(const ThreePrimeFp2 &left,
                                               const ThreePrimeFp2 &right) const {
    ThreePrimeFp2 output{};
    for (uint32_t prime_index = 0; prime_index < kPrimeCount; ++prime_index) {
      const uint32_t prime = prime_at(prime_index);
      output.lane[2 * prime_index] =
          add_mod(left.lane[2 * prime_index], right.lane[2 * prime_index],
                  prime);
      output.lane[2 * prime_index + 1] = add_mod(
          left.lane[2 * prime_index + 1],
          right.lane[2 * prime_index + 1], prime);
    }
    return output;
  }
};

__global__ void unpack_entries(const PackedPhCooEntry *entries,
                               uint64_t count, uint32_t columns,
                               SparseKey *keys, ThreePrimeFp2 *values,
                               Status *status) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index >= count) return;
  const PackedPhCooEntry entry = entries[index];
  if (entry.column >= columns || entry.reserved != 0) {
    atomicExch(&status->invalid, 1U);
    return;
  }
  for (uint32_t prime_index = 0; prime_index < kPrimeCount; ++prime_index) {
    const uint32_t prime = prime_at(prime_index);
    if (entry.value.lane[2 * prime_index] >= prime ||
        entry.value.lane[2 * prime_index + 1] >= prime) {
      atomicExch(&status->invalid, 1U);
      return;
    }
  }
  keys[index] = {entry.row_key, entry.column, 0};
  values[index] = entry.value;
}

__global__ void retained_rref(const SparseKey *keys,
                              const ThreePrimeFp2 *values,
                              const uint64_t *reduced_count,
                              uint32_t columns, uint32_t maximum_rank,
                              uint32_t expected_rank, Fp2 *basis,
                              int32_t *pivot_for_column,
                              uint32_t *pivot_columns,
                              uint64_t *pivot_row_keys, Status *status) {
  if (blockIdx.x != 0) return;
  extern __shared__ Fp2 row[];
  if (status->stopped != 0 || status->invalid != 0) return;
  const uint64_t count = *reduced_count;
  uint64_t offset = 0;
  while (offset < count && status->stopped == 0 && status->invalid == 0) {
    const uint64_t row_key = keys[offset].row_key;
    uint64_t end = offset + 1;
    while (end < count && keys[end].row_key == row_key) ++end;
    for (uint32_t prime_index = 0; prime_index < kPrimeCount;
         ++prime_index) {
      const uint32_t prime = prime_at(prime_index);
      Fp2 *prime_row = row + static_cast<uint64_t>(prime_index) * columns;
      for (uint32_t column = threadIdx.x; column < columns;
           column += blockDim.x) {
        prime_row[column] = {};
      }
      __syncthreads();
      for (uint64_t entry = offset + threadIdx.x; entry < end;
           entry += blockDim.x) {
        const uint32_t column = keys[entry].column;
        const ThreePrimeFp2 packed = values[entry];
        prime_row[column] = {packed.lane[2 * prime_index],
                             packed.lane[2 * prime_index + 1]};
      }
      __syncthreads();

      while (true) {
        __shared__ uint32_t leading_column;
        if (threadIdx.x == 0) {
          leading_column = columns;
          for (uint32_t column = 0; column < columns; ++column) {
            if (!is_zero(prime_row[column])) {
              leading_column = column;
              break;
            }
          }
        }
        __syncthreads();
        if (leading_column == columns) break;
        const int32_t basis_row =
            pivot_for_column[static_cast<uint64_t>(prime_index) * columns +
                             leading_column];
        if (basis_row < 0) break;
        const Fp2 factor = prime_row[leading_column];
        const Fp2 *stored =
            basis + (static_cast<uint64_t>(prime_index) * maximum_rank +
                     static_cast<uint32_t>(basis_row)) *
                        columns;
        for (uint32_t column = threadIdx.x; column < columns;
             column += blockDim.x) {
          prime_row[column] = subtract_fp2(
              prime_row[column], multiply_fp2(factor, stored[column], prime),
              prime);
        }
        __syncthreads();
      }

      __shared__ uint32_t new_pivot;
      __shared__ uint32_t new_rank;
      if (threadIdx.x == 0) {
        new_pivot = columns;
        for (uint32_t column = 0; column < columns; ++column) {
          if (!is_zero(prime_row[column])) {
            new_pivot = column;
            break;
          }
        }
        new_rank = status->ranks[prime_index];
        if (new_pivot != columns) {
          if (new_rank >= maximum_rank) {
            status->invalid = 1;
          } else {
            pivot_for_column[static_cast<uint64_t>(prime_index) * columns +
                             new_pivot] = static_cast<int32_t>(new_rank);
            pivot_columns[static_cast<uint64_t>(prime_index) * maximum_rank +
                          new_rank] = new_pivot;
            pivot_row_keys[static_cast<uint64_t>(prime_index) * maximum_rank +
                           new_rank] = row_key;
            status->ranks[prime_index] = new_rank + 1;
          }
        }
      }
      __syncthreads();
      if (status->invalid != 0) break;
      if (new_pivot != columns) {
        __shared__ Fp2 inverse;
        if (threadIdx.x == 0) inverse = inverse_fp2(prime_row[new_pivot], prime);
        __syncthreads();
        Fp2 *stored =
            basis + (static_cast<uint64_t>(prime_index) * maximum_rank +
                     new_rank) *
                        columns;
        for (uint32_t column = threadIdx.x; column < columns;
             column += blockDim.x) {
          stored[column] = multiply_fp2(prime_row[column], inverse, prime);
        }
        __syncthreads();
        if (threadIdx.x == 0 && new_rank + 1 > expected_rank &&
            atomicCAS(&status->stopped, 0U, 1U) == 0U) {
          status->obstruction_prime = prime_index;
          status->obstruction_pivot_column = new_pivot;
          status->obstruction_row_key = row_key;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0)
      atomicAdd(reinterpret_cast<unsigned long long *>(&status->rows_visited),
                1ULL);
    __syncthreads();
    offset = end;
  }
  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long *>(&status->reduced_entries),
              static_cast<unsigned long long>(count));
  }
}

void destroy(Context *context) {
  if (context == nullptr) return;
  for (Slot &slot : context->slots) {
    if (slot.reusable != nullptr) cudaEventSynchronize(slot.reusable);
  }
  if (context->compute_stream != nullptr)
    cudaStreamSynchronize(context->compute_stream);
  for (Slot &slot : context->slots) {
    cudaFreeHost(slot.host_entries);
    cudaFree(slot.device_entries);
    cudaFree(slot.keys);
    cudaFree(slot.reduced_keys);
    cudaFree(slot.values);
    cudaFree(slot.reduced_values);
    cudaFree(slot.reduced_count);
    if (slot.upload_done != nullptr) cudaEventDestroy(slot.upload_done);
    if (slot.reusable != nullptr) cudaEventDestroy(slot.reusable);
    if (slot.upload_stream != nullptr) cudaStreamDestroy(slot.upload_stream);
  }
  cudaFree(context->basis);
  cudaFree(context->pivot_for_column);
  cudaFree(context->pivot_columns);
  cudaFree(context->pivot_row_keys);
  cudaFree(context->status);
  if (context->compute_stream != nullptr) cudaStreamDestroy(context->compute_stream);
  delete context;
}

}  // namespace

extern "C" void *adynkra_common_parent_ph_obstruction_create(
    uint32_t columns, uint32_t expected_rank, uint64_t batch_capacity,
    uint64_t device_hard_cap, char *error, uint64_t error_capacity) {
  if (columns == 0 || columns > kMaximumColumns || expected_rank >= columns ||
      batch_capacity == 0 ||
      batch_capacity > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    set_error(error, error_capacity, "invalid P_H obstruction dimensions");
    return nullptr;
  }
  Context *context = new (std::nothrow) Context();
  if (context == nullptr) {
    set_error(error, error_capacity, "allocate P_H obstruction context");
    return nullptr;
  }
  context->columns = columns;
  context->expected_rank = expected_rank;
  context->maximum_rank = expected_rank + 1;
  context->batch_capacity = batch_capacity;
  context->device_hard_cap = device_hard_cap;
  const uint64_t per_slot =
      batch_capacity * (sizeof(PackedPhCooEntry) + 2 * sizeof(SparseKey) +
                        2 * sizeof(ThreePrimeFp2)) +
      sizeof(uint64_t);
  const uint64_t basis_count = static_cast<uint64_t>(kPrimeCount) *
                               context->maximum_rank * columns;
  context->resident_bytes =
      2 * per_slot + basis_count * sizeof(Fp2) +
      static_cast<uint64_t>(kPrimeCount) * columns * sizeof(int32_t) +
      static_cast<uint64_t>(kPrimeCount) * context->maximum_rank *
          (sizeof(uint32_t) + sizeof(uint64_t)) +
      sizeof(Status);
  // Thrust may reserve temporary sort/reduce storage.  Count a conservative
  // two-copy bound for one active key/value slot so the hard cap applies to
  // transient work as well as resident buffers.
  context->high_water_bytes =
      context->resident_bytes +
      2 * batch_capacity * (sizeof(SparseKey) + sizeof(ThreePrimeFp2));
  if (device_hard_cap != 0 && context->high_water_bytes > device_hard_cap) {
    set_error(error, error_capacity, "P_H obstruction buffers exceed device cap");
    destroy(context);
    return nullptr;
  }
  if (!checked(cudaStreamCreateWithFlags(&context->compute_stream,
                                         cudaStreamNonBlocking),
               error, error_capacity, "create P_H compute stream")) {
    destroy(context);
    return nullptr;
  }
  for (Slot &slot : context->slots) {
    if (!checked(cudaHostAlloc(&slot.host_entries,
                               batch_capacity * sizeof(PackedPhCooEntry),
                               cudaHostAllocPortable),
                 error, error_capacity, "allocate P_H pinned batch") ||
        !checked(cudaMalloc(&slot.device_entries,
                            batch_capacity * sizeof(PackedPhCooEntry)),
                 error, error_capacity, "allocate P_H device batch") ||
        !checked(cudaMalloc(&slot.keys, batch_capacity * sizeof(SparseKey)),
                 error, error_capacity, "allocate P_H sort keys") ||
        !checked(cudaMalloc(&slot.reduced_keys,
                            batch_capacity * sizeof(SparseKey)),
                 error, error_capacity, "allocate P_H reduced keys") ||
        !checked(cudaMalloc(&slot.values,
                            batch_capacity * sizeof(ThreePrimeFp2)),
                 error, error_capacity, "allocate P_H sort values") ||
        !checked(cudaMalloc(&slot.reduced_values,
                            batch_capacity * sizeof(ThreePrimeFp2)),
                 error, error_capacity, "allocate P_H reduced values") ||
        !checked(cudaMalloc(&slot.reduced_count, sizeof(uint64_t)), error,
                 error_capacity, "allocate P_H reduced count") ||
        !checked(cudaStreamCreateWithFlags(&slot.upload_stream,
                                           cudaStreamNonBlocking),
                 error, error_capacity, "create P_H upload stream") ||
        !checked(cudaEventCreateWithFlags(&slot.upload_done,
                                          cudaEventDisableTiming),
                 error, error_capacity, "create P_H upload event") ||
        !checked(cudaEventCreateWithFlags(&slot.reusable,
                                          cudaEventDisableTiming),
                 error, error_capacity, "create P_H reusable event") ||
        !checked(cudaEventRecord(slot.reusable, context->compute_stream), error,
                 error_capacity, "initialize P_H reusable event")) {
      destroy(context);
      return nullptr;
    }
  }
  if (!checked(cudaMalloc(&context->basis, basis_count * sizeof(Fp2)), error,
               error_capacity, "allocate P_H retained basis") ||
      !checked(cudaMalloc(&context->pivot_for_column,
                          static_cast<uint64_t>(kPrimeCount) * columns *
                              sizeof(int32_t)),
               error, error_capacity, "allocate P_H pivot map") ||
      !checked(cudaMalloc(&context->pivot_columns,
                          static_cast<uint64_t>(kPrimeCount) *
                              context->maximum_rank * sizeof(uint32_t)),
               error, error_capacity, "allocate P_H pivot columns") ||
      !checked(cudaMalloc(&context->pivot_row_keys,
                          static_cast<uint64_t>(kPrimeCount) *
                              context->maximum_rank * sizeof(uint64_t)),
               error, error_capacity, "allocate P_H pivot row keys") ||
      !checked(cudaMalloc(&context->status, sizeof(Status)), error,
               error_capacity, "allocate P_H status") ||
      !checked(cudaMemsetAsync(context->basis, 0,
                               basis_count * sizeof(Fp2),
                               context->compute_stream),
               error, error_capacity, "clear P_H retained basis") ||
      !checked(cudaMemsetAsync(
                   context->pivot_for_column, 0xff,
                   static_cast<uint64_t>(kPrimeCount) * columns *
                       sizeof(int32_t),
                   context->compute_stream),
               error, error_capacity, "clear P_H pivot map") ||
      !checked(cudaMemsetAsync(context->status, 0, sizeof(Status),
                               context->compute_stream),
               error, error_capacity, "clear P_H status") ||
      !checked(cudaStreamSynchronize(context->compute_stream), error,
               error_capacity, "initialize P_H context")) {
    destroy(context);
    return nullptr;
  }
  return context;
}

extern "C" int32_t adynkra_common_parent_ph_obstruction_submit(
    void *opaque, uint64_t batch_ordinal, const PackedPhCooEntry *entries,
    uint64_t entry_count, uint64_t first_row_key, uint64_t last_row_key,
    char *error, uint64_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || entries == nullptr || entry_count == 0 ||
      entry_count > context->batch_capacity ||
      batch_ordinal != context->next_batch_ordinal ||
      first_row_key > last_row_key ||
      (context->has_last_row && first_row_key <= context->last_row_key)) {
    set_error(error, error_capacity,
              "invalid or noncanonical P_H obstruction batch");
    return 1;
  }
  for (uint64_t index = 0; index < entry_count; ++index) {
    if (entries[index].row_key < first_row_key ||
        entries[index].row_key > last_row_key) {
      set_error(error, error_capacity, "P_H entry lies outside batch row range");
      return 2;
    }
  }
  Slot &slot = context->slots[batch_ordinal & 1U];
  if (!checked(cudaEventSynchronize(slot.reusable), error, error_capacity,
               "wait for reusable P_H batch slot"))
    return 3;
  std::memcpy(slot.host_entries, entries,
              static_cast<size_t>(entry_count) * sizeof(PackedPhCooEntry));
  if (!checked(cudaMemcpyAsync(slot.device_entries, slot.host_entries,
                               entry_count * sizeof(PackedPhCooEntry),
                               cudaMemcpyHostToDevice, slot.upload_stream),
               error, error_capacity, "upload P_H batch") ||
      !checked(cudaEventRecord(slot.upload_done, slot.upload_stream), error,
               error_capacity, "record P_H upload") ||
      !checked(cudaStreamWaitEvent(context->compute_stream, slot.upload_done,
                                   0),
               error, error_capacity, "join P_H upload"))
    return 4;
  const uint32_t threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((entry_count + threads - 1) /
                                                 threads);
  unpack_entries<<<blocks, threads, 0, context->compute_stream>>>(
      slot.device_entries, entry_count, context->columns, slot.keys,
      slot.values, context->status);
  try {
    auto policy = thrust::cuda::par.on(context->compute_stream);
    thrust::sort_by_key(policy, thrust::device_pointer_cast(slot.keys),
                        thrust::device_pointer_cast(slot.keys + entry_count),
                        thrust::device_pointer_cast(slot.values), KeyLess{});
    const auto end = thrust::reduce_by_key(
        policy, thrust::device_pointer_cast(slot.keys),
        thrust::device_pointer_cast(slot.keys + entry_count),
        thrust::device_pointer_cast(slot.values),
        thrust::device_pointer_cast(slot.reduced_keys),
        thrust::device_pointer_cast(slot.reduced_values), KeyEqual{},
        AddThreePrime{});
    const uint64_t reduced = static_cast<uint64_t>(end.first.get() -
                                                   slot.reduced_keys);
    if (!checked(cudaMemcpyAsync(slot.reduced_count, &reduced, sizeof(uint64_t),
                                 cudaMemcpyHostToDevice,
                                 context->compute_stream),
                 error, error_capacity, "publish P_H reduced count"))
      return 5;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    return 6;
  }
  const uint64_t shared_bytes =
      static_cast<uint64_t>(kPrimeCount) * context->columns * sizeof(Fp2);
  if (shared_bytes > 48 * 1024) {
    cudaFuncSetAttribute(retained_rref,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_bytes));
  }
  retained_rref<<<1, 256, shared_bytes, context->compute_stream>>>(
      slot.reduced_keys, slot.reduced_values, slot.reduced_count,
      context->columns, context->maximum_rank, context->expected_rank,
      context->basis, context->pivot_for_column, context->pivot_columns,
      context->pivot_row_keys, context->status);
  if (!checked(cudaEventRecord(slot.reusable, context->compute_stream), error,
               error_capacity, "record reusable P_H slot") ||
      !checked(cudaPeekAtLastError(), error, error_capacity,
               "enqueue P_H obstruction kernels"))
    return 7;
  context->next_batch_ordinal += 1;
  context->last_row_key = last_row_key;
  context->has_last_row = true;
  // The two-slot pipeline permits the next host pack/upload while this batch
  // is sorting and reducing.  Counters are finalized by poll/finalize.
  return 0;
}

extern "C" int32_t adynkra_common_parent_ph_obstruction_poll(
    void *opaque, Status *output, char *error, uint64_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || output == nullptr) {
    set_error(error, error_capacity, "invalid P_H poll arguments");
    return 1;
  }
  if (!checked(cudaStreamSynchronize(context->compute_stream), error,
               error_capacity, "synchronize P_H obstruction engine") ||
      !checked(cudaMemcpy(output, context->status, sizeof(Status),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download P_H status"))
    return 2;
  output->batches_submitted = context->next_batch_ordinal;
  // Input count is a host-side canonical counter to avoid one device atomic
  // per sparse term.  The Rust wrapper maintains and publishes the exact sum.
  return 0;
}

extern "C" int32_t adynkra_common_parent_ph_obstruction_checkpoint(
    void *opaque, Fp2 *basis, uint64_t basis_capacity,
    int32_t *pivot_for_column, uint64_t pivot_map_capacity,
    uint32_t *pivot_columns, uint64_t pivot_capacity,
    uint64_t *pivot_row_keys, uint64_t row_key_capacity, Status *output,
    char *error, uint64_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || basis == nullptr || pivot_for_column == nullptr ||
      pivot_columns == nullptr || pivot_row_keys == nullptr || output == nullptr)
    return 1;
  const uint64_t basis_count = static_cast<uint64_t>(kPrimeCount) *
                               context->maximum_rank * context->columns;
  const uint64_t pivot_map_count =
      static_cast<uint64_t>(kPrimeCount) * context->columns;
  const uint64_t pivot_count =
      static_cast<uint64_t>(kPrimeCount) * context->maximum_rank;
  if (basis_capacity < basis_count || pivot_map_capacity < pivot_map_count ||
      pivot_capacity < pivot_count || row_key_capacity < pivot_count) {
    set_error(error, error_capacity, "P_H checkpoint capacity is too small");
    return 2;
  }
  if (!checked(cudaStreamSynchronize(context->compute_stream), error,
               error_capacity, "synchronize P_H checkpoint") ||
      !checked(cudaMemcpy(basis, context->basis, basis_count * sizeof(Fp2),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download P_H basis") ||
      !checked(cudaMemcpy(pivot_for_column, context->pivot_for_column,
                          pivot_map_count * sizeof(int32_t),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download P_H pivot map") ||
      !checked(cudaMemcpy(pivot_columns, context->pivot_columns,
                          pivot_count * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download P_H pivots") ||
      !checked(cudaMemcpy(pivot_row_keys, context->pivot_row_keys,
                          pivot_count * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download P_H pivot rows") ||
      !checked(cudaMemcpy(output, context->status, sizeof(Status),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download P_H checkpoint status"))
    return 3;
  output->batches_submitted = context->next_batch_ordinal;
  return 0;
}

extern "C" int32_t adynkra_common_parent_ph_obstruction_restore(
    void *opaque, const Fp2 *basis, uint64_t basis_count,
    const int32_t *pivot_for_column, uint64_t pivot_map_count,
    const uint32_t *pivot_columns, uint64_t pivot_count,
    const uint64_t *pivot_row_keys, uint64_t row_key_count,
    const Status *status, uint64_t next_batch_ordinal, uint64_t last_row_key,
    uint32_t has_last_row, char *error, uint64_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || basis == nullptr || pivot_for_column == nullptr ||
      pivot_columns == nullptr || pivot_row_keys == nullptr || status == nullptr)
    return 1;
  const uint64_t required_basis = static_cast<uint64_t>(kPrimeCount) *
                                  context->maximum_rank * context->columns;
  const uint64_t required_map =
      static_cast<uint64_t>(kPrimeCount) * context->columns;
  const uint64_t required_pivots =
      static_cast<uint64_t>(kPrimeCount) * context->maximum_rank;
  if (basis_count != required_basis || pivot_map_count != required_map ||
      pivot_count != required_pivots || row_key_count != required_pivots ||
      status->invalid != 0 || status->batches_submitted != next_batch_ordinal)
    return 2;
  for (uint32_t prime_index = 0; prime_index < kPrimeCount; ++prime_index) {
    if (status->ranks[prime_index] > context->maximum_rank) return 3;
    for (uint32_t column = 0; column < context->columns; ++column) {
      const int32_t pivot =
          pivot_for_column[static_cast<uint64_t>(prime_index) *
                               context->columns +
                           column];
      if (pivot < -1 ||
          pivot >= static_cast<int32_t>(status->ranks[prime_index]))
        return 4;
    }
  }
  if (!checked(cudaStreamSynchronize(context->compute_stream), error,
               error_capacity, "synchronize P_H restore") ||
      !checked(cudaMemcpy(context->basis, basis,
                          required_basis * sizeof(Fp2),
                          cudaMemcpyHostToDevice),
               error, error_capacity, "restore P_H basis") ||
      !checked(cudaMemcpy(context->pivot_for_column, pivot_for_column,
                          required_map * sizeof(int32_t),
                          cudaMemcpyHostToDevice),
               error, error_capacity, "restore P_H pivot map") ||
      !checked(cudaMemcpy(context->pivot_columns, pivot_columns,
                          required_pivots * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               error, error_capacity, "restore P_H pivot columns") ||
      !checked(cudaMemcpy(context->pivot_row_keys, pivot_row_keys,
                          required_pivots * sizeof(uint64_t),
                          cudaMemcpyHostToDevice),
               error, error_capacity, "restore P_H pivot rows") ||
      !checked(cudaMemcpy(context->status, status, sizeof(Status),
                          cudaMemcpyHostToDevice),
               error, error_capacity, "restore P_H status"))
    return 5;
  context->next_batch_ordinal = next_batch_ordinal;
  context->last_row_key = last_row_key;
  context->has_last_row = has_last_row != 0;
  return 0;
}

extern "C" uint64_t adynkra_common_parent_ph_obstruction_resident_bytes(
    const void *opaque) {
  const Context *context = static_cast<const Context *>(opaque);
  return context == nullptr ? 0 : context->resident_bytes;
}

extern "C" uint64_t adynkra_common_parent_ph_obstruction_high_water_bytes(
    const void *opaque) {
  const Context *context = static_cast<const Context *>(opaque);
  return context == nullptr ? 0 : context->high_water_bytes;
}

extern "C" void adynkra_common_parent_ph_obstruction_primes(uint32_t *output) {
  if (output == nullptr) return;
  for (uint32_t index = 0; index < kPrimeCount; ++index)
    output[index] = kPrimes[index];
}

extern "C" void adynkra_common_parent_ph_obstruction_destroy(void *opaque) {
  destroy(static_cast<Context *>(opaque));
}

#ifdef ADYNKRA_COMMON_PARENT_PH_OBSTRUCTION_STANDALONE
int main(int argc, char **argv) {
  const uint64_t entries_requested =
      argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000ULL;
  constexpr uint32_t columns = 24;
  constexpr uint32_t expected_rank = 12;
  constexpr uint64_t rows = 50000;
  const uint64_t entries_per_row =
      std::max<uint64_t>(1, entries_requested / rows);
  std::vector<PackedPhCooEntry> entries;
  entries.reserve(entries_requested);
  for (uint64_t index = 0; index < entries_requested; ++index) {
    const uint64_t row = std::min<uint64_t>(rows - 1, index / entries_per_row);
    const uint32_t column = static_cast<uint32_t>(index % entries_per_row) % columns;
    PackedPhCooEntry entry{};
    entry.row_key = row;
    entry.column = column;
    for (uint32_t prime_index = 0; prime_index < kPrimeCount; ++prime_index) {
      const uint32_t prime = prime_at(prime_index);
      uint64_t value = 1;
      const uint64_t base = (row + 1) % prime;
      for (uint32_t exponent = 0; exponent < column; ++exponent)
        value = value * base % prime;
      entry.value.lane[2 * prime_index] = static_cast<uint32_t>(value);
    }
    entries.push_back(entry);
  }
  char error[1024]{};
  void *context = adynkra_common_parent_ph_obstruction_create(
      columns, expected_rank, entries.size() / 2 + 1, 2ULL << 30, error,
      sizeof(error));
  if (context == nullptr) {
    std::fprintf(stderr, "%s\n", error);
    return 2;
  }
  const auto start = std::chrono::steady_clock::now();
  const uint64_t split_row = rows / 2;
  const uint64_t split_entry = split_row * entries_per_row;
  const int32_t submit0 = adynkra_common_parent_ph_obstruction_submit(
      context, 0, entries.data(), split_entry, 0, split_row - 1, error,
      sizeof(error));
  const int32_t submit1 = adynkra_common_parent_ph_obstruction_submit(
      context, 1, entries.data() + split_entry, entries.size() - split_entry,
      split_row, rows - 1, error, sizeof(error));
  Status status{};
  const int32_t poll = adynkra_common_parent_ph_obstruction_poll(
      context, &status, error, sizeof(error));
  const double milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count();
  std::printf(
      "{\"input_entries\":%llu,\"milliseconds\":%.6f,"
      "\"entries_per_second\":%.3f,\"ranks\":[%u,%u,%u],"
      "\"stopped\":%u,\"witness_row\":%llu,"
      "\"resident_bytes\":%llu,\"high_water_bytes\":%llu}\n",
      static_cast<unsigned long long>(entries.size()), milliseconds,
      entries.size() / (milliseconds / 1000.0), status.ranks[0],
      status.ranks[1], status.ranks[2], status.stopped,
      static_cast<unsigned long long>(status.obstruction_row_key),
      static_cast<unsigned long long>(
          adynkra_common_parent_ph_obstruction_resident_bytes(context)),
      static_cast<unsigned long long>(
          adynkra_common_parent_ph_obstruction_high_water_bytes(context)));
  adynkra_common_parent_ph_obstruction_destroy(context);
  if (submit0 != 0 || submit1 != 0 || poll != 0) {
    std::fprintf(stderr, "status=%d/%d/%d %s\n", submit0, submit1, poll,
                 error);
    return 3;
  }
  return 0;
}
#endif
