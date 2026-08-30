#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

struct SparseEntry {
  uint32_t row;
  int64_t real;
  int64_t imaginary;
};

struct SparseInput {
  uint32_t lane;
  uint32_t column;
  int64_t real;
  int64_t imaginary;
};

struct SparseOutput {
  uint32_t lane;
  uint32_t row;
  int64_t real;
  int64_t imaginary;
};

static_assert(sizeof(SparseEntry) == 24);
static_assert(offsetof(SparseEntry, row) == 0);
static_assert(offsetof(SparseEntry, real) == 8);
static_assert(offsetof(SparseEntry, imaginary) == 16);
static_assert(sizeof(SparseInput) == 24);
static_assert(offsetof(SparseInput, lane) == 0);
static_assert(offsetof(SparseInput, column) == 4);
static_assert(offsetof(SparseInput, real) == 8);
static_assert(offsetof(SparseInput, imaginary) == 16);
static_assert(sizeof(SparseOutput) == 24);
static_assert(offsetof(SparseOutput, lane) == 0);
static_assert(offsetof(SparseOutput, row) == 4);
static_assert(offsetof(SparseOutput, real) == 8);
static_assert(offsetof(SparseOutput, imaginary) == 16);

struct Context {
  int device = 0;
  uint32_t input_dimension = 0;
  uint32_t output_dimension = 0;
  uint32_t entry_count = 0;
  uint32_t input_capacity = 0;
  uint32_t output_lane_capacity = 0;
  std::vector<uint32_t> host_offsets;
  uint32_t *offsets = nullptr;
  SparseEntry *entries = nullptr;
  SparseInput *inputs = nullptr;
  int64_t *output_real = nullptr;
  int64_t *output_imaginary = nullptr;
  uint64_t *expanded_products = nullptr;
  SparseOutput *compact_outputs = nullptr;
  uint64_t compact_capacity = 0;
  uint64_t *compact_count = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t started = nullptr;
  cudaEvent_t finished = nullptr;
  bool poisoned = false;
};

void set_error(char *error, size_t capacity, const char *message) {
  if (error == nullptr || capacity == 0) {
    return;
  }
  std::snprintf(error, capacity, "%s", message);
}

bool check_cuda(cudaError_t status, char *error, size_t capacity,
                const char *action) {
  if (status == cudaSuccess) {
    return true;
  }
  if (error != nullptr && capacity != 0) {
    std::snprintf(error, capacity, "%s: %s", action,
                  cudaGetErrorString(status));
  }
  return false;
}

int drain_failed_apply(Context *context, bool must_poison, char *error,
                       size_t capacity) {
  if (must_poison) {
    context->poisoned = true;
  }
  const cudaError_t drain_status = cudaStreamSynchronize(context->stream);
  if (drain_status == cudaSuccess) {
    return 1;
  }
  context->poisoned = true;
  if (error != nullptr && capacity != 0) {
    const size_t used = std::strlen(error);
    if (used < capacity) {
      std::snprintf(error + used, capacity - used,
                    "%scontext poisoned while draining stream: %s",
                    used == 0 ? "" : "; ", cudaGetErrorString(drain_status));
    }
  }
  return 1;
}

void release(Context *context) {
  if (context == nullptr) {
    return;
  }
  cudaSetDevice(context->device);
  if (context->stream != nullptr) {
    cudaStreamSynchronize(context->stream);
  }
  cudaFree(context->offsets);
  cudaFree(context->entries);
  cudaFree(context->inputs);
  cudaFree(context->output_real);
  cudaFree(context->output_imaginary);
  cudaFree(context->expanded_products);
  cudaFree(context->compact_outputs);
  cudaFree(context->compact_count);
  if (context->started != nullptr) {
    cudaEventDestroy(context->started);
  }
  if (context->finished != nullptr) {
    cudaEventDestroy(context->finished);
  }
  if (context->stream != nullptr) {
    cudaStreamDestroy(context->stream);
  }
  delete context;
}

__device__ inline void atomic_add_signed(int64_t *address, int64_t value) {
  atomicAdd(reinterpret_cast<unsigned long long *>(address),
            static_cast<unsigned long long>(value));
}

__global__ void dense_apply_kernel(const uint32_t *offsets,
                                   const SparseEntry *entries,
                                   const int64_t *input_real,
                                   const int64_t *input_imaginary,
                                   uint64_t dense_count,
                                   uint32_t input_dimension,
                                   uint32_t output_dimension,
                                   int64_t *output_real,
                                   int64_t *output_imaginary,
                                   uint64_t *expanded_products) {
  const uint64_t dense_index =
      uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (dense_index >= dense_count) {
    return;
  }
  const uint32_t column = uint32_t(dense_index % input_dimension);
  const uint32_t lane = uint32_t(dense_index / input_dimension);
  const int64_t source_real = input_real[dense_index];
  const int64_t source_imaginary = input_imaginary[dense_index];
  if (source_real == 0 && source_imaginary == 0) {
    return;
  }
  const uint32_t begin = offsets[column];
  const uint32_t end = offsets[column + 1];
  atomicAdd(reinterpret_cast<unsigned long long *>(expanded_products),
            static_cast<unsigned long long>(end - begin));
  const uint64_t output_base = uint64_t(lane) * output_dimension;
  for (uint32_t position = begin; position < end; ++position) {
    const SparseEntry entry = entries[position];
    const int64_t real = entry.real * source_real -
                         entry.imaginary * source_imaginary;
    const int64_t imaginary = entry.real * source_imaginary +
                              entry.imaginary * source_real;
    atomic_add_signed(&output_real[output_base + entry.row], real);
    atomic_add_signed(&output_imaginary[output_base + entry.row], imaginary);
  }
}

__global__ void sparse_apply_kernel(const uint32_t *offsets,
                                    const SparseEntry *entries,
                                    const SparseInput *inputs,
                                    uint32_t input_count,
                                    uint32_t output_dimension,
                                    int64_t *output_real,
                                    int64_t *output_imaginary) {
  const uint32_t input_index = blockIdx.x;
  if (input_index >= input_count) {
    return;
  }
  const SparseInput input = inputs[input_index];
  const uint64_t output_base = uint64_t(input.lane) * output_dimension;
  const uint32_t begin = offsets[input.column];
  const uint32_t end = offsets[input.column + 1];
  for (uint32_t position = begin + threadIdx.x; position < end;
       position += blockDim.x) {
    const SparseEntry entry = entries[position];
    const int64_t real = entry.real * input.real -
                         entry.imaginary * input.imaginary;
    const int64_t imaginary = entry.real * input.imaginary +
                              entry.imaginary * input.real;
    atomic_add_signed(&output_real[output_base + entry.row], real);
    atomic_add_signed(&output_imaginary[output_base + entry.row], imaginary);
  }
}

__global__ void compact_nonzero_kernel(const int64_t *output_real,
                                       const int64_t *output_imaginary,
                                       uint64_t output_count,
                                       uint32_t output_dimension,
                                       SparseOutput *compact_outputs,
                                       uint64_t compact_capacity,
                                       uint64_t *compact_count) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_count) {
    return;
  }
  const int64_t real = output_real[index];
  const int64_t imaginary = output_imaginary[index];
  if (real == 0 && imaginary == 0) {
    return;
  }
  const uint64_t position = atomicAdd(
      reinterpret_cast<unsigned long long *>(compact_count), 1ULL);
  if (position >= compact_capacity) {
    return;
  }
  compact_outputs[position] = SparseOutput{
      uint32_t(index / output_dimension), uint32_t(index % output_dimension),
      real, imaginary};
}

__global__ void count_nonzero_kernel(const int64_t *output_real,
                                     const int64_t *output_imaginary,
                                     uint64_t output_count,
                                     uint64_t *compact_count) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_count ||
      (output_real[index] == 0 && output_imaginary[index] == 0)) {
    return;
  }
  atomicAdd(reinterpret_cast<unsigned long long *>(compact_count), 1ULL);
}

bool reserve_outputs(Context *context, uint32_t lane_count, char *error,
                     size_t error_capacity) {
  if (lane_count <= context->output_lane_capacity) {
    return true;
  }
  const size_t count = size_t(lane_count) * context->output_dimension;
  if (context->output_dimension != 0 &&
      count / context->output_dimension != lane_count) {
    set_error(error, error_capacity,
              "complete-F batched output element count overflow");
    return false;
  }
  if (count > std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
    set_error(error, error_capacity,
              "complete-F batched output byte count overflow");
    return false;
  }
  int64_t *replacement_real = nullptr;
  int64_t *replacement_imaginary = nullptr;
  if (!check_cuda(cudaMalloc(&replacement_real, count * sizeof(int64_t)), error,
                  error_capacity, "allocate complete-F batched real output") ||
      !check_cuda(cudaMalloc(&replacement_imaginary, count * sizeof(int64_t)),
                  error, error_capacity,
                  "allocate complete-F batched imaginary output")) {
    cudaFree(replacement_real);
    cudaFree(replacement_imaginary);
    return false;
  }
  cudaFree(context->output_real);
  cudaFree(context->output_imaginary);
  context->output_real = replacement_real;
  context->output_imaginary = replacement_imaginary;
  context->output_lane_capacity = lane_count;
  return true;
}

bool reserve_inputs(Context *context, uint32_t count, char *error,
                    size_t error_capacity) {
  if (count <= context->input_capacity) {
    return true;
  }
  SparseInput *replacement = nullptr;
  if (size_t(count) >
      std::numeric_limits<size_t>::max() / sizeof(SparseInput)) {
    set_error(error, error_capacity,
              "complete-F sparse input byte count overflow");
    return false;
  }
  if (!check_cuda(cudaMalloc(&replacement, size_t(count) * sizeof(SparseInput)),
                  error, error_capacity, "allocate complete-F sparse inputs")) {
    return false;
  }
  cudaFree(context->inputs);
  context->inputs = replacement;
  context->input_capacity = count;
  return true;
}

bool reserve_compact_outputs(Context *context, uint64_t count, char *error,
                             size_t error_capacity) {
  if (count <= context->compact_capacity) {
    return true;
  }
  if (count > uint64_t(std::numeric_limits<size_t>::max() /
                       sizeof(SparseOutput))) {
    set_error(error, error_capacity,
              "complete-F compact output byte count overflow");
    return false;
  }
  SparseOutput *replacement = nullptr;
  if (!check_cuda(cudaMalloc(&replacement,
                             size_t(count) * sizeof(SparseOutput)),
                  error, error_capacity,
                  "allocate complete-F compact outputs")) {
    return false;
  }
  cudaFree(context->compact_outputs);
  context->compact_outputs = replacement;
  context->compact_capacity = count;
  return true;
}

} // namespace

extern "C" {

void *adynkra_complete_f_sparse_create(
    int device, uint32_t input_dimension, uint32_t output_dimension,
    const uint32_t *host_offsets, const SparseEntry *host_entries,
    uint32_t entry_count, char *error, size_t error_capacity) {
  if (input_dimension == 0 || output_dimension == 0 || host_offsets == nullptr ||
      (entry_count != 0 && host_entries == nullptr)) {
    set_error(error, error_capacity, "invalid complete-F sparse operator");
    return nullptr;
  }
  if (host_offsets[0] != 0 || host_offsets[input_dimension] != entry_count) {
    set_error(error, error_capacity, "invalid complete-F sparse offsets");
    return nullptr;
  }
  for (uint32_t column = 0; column < input_dimension; ++column) {
    if (host_offsets[column] > host_offsets[column + 1]) {
      set_error(error, error_capacity, "nonmonotone complete-F sparse offsets");
      return nullptr;
    }
  }
  for (uint32_t index = 0; index < entry_count; ++index) {
    if (host_entries[index].row >= output_dimension) {
      set_error(error, error_capacity, "complete-F sparse row is out of range");
      return nullptr;
    }
  }

  Context *context = new (std::nothrow) Context();
  if (context == nullptr) {
    set_error(error, error_capacity, "allocate complete-F sparse context");
    return nullptr;
  }
  context->device = device;
  context->input_dimension = input_dimension;
  context->output_dimension = output_dimension;
  context->entry_count = entry_count;
  context->host_offsets.assign(host_offsets, host_offsets + input_dimension + 1);

  if (!check_cuda(cudaSetDevice(device), error, error_capacity,
                  "select complete-F CUDA device") ||
      !check_cuda(cudaStreamCreateWithFlags(&context->stream,
                                            cudaStreamNonBlocking),
                  error, error_capacity, "create complete-F CUDA stream") ||
      !check_cuda(cudaEventCreate(&context->started), error, error_capacity,
                  "create complete-F start event") ||
      !check_cuda(cudaEventCreate(&context->finished), error, error_capacity,
                  "create complete-F finish event") ||
      !check_cuda(cudaMalloc(&context->expanded_products, sizeof(uint64_t)),
                  error, error_capacity,
                  "allocate complete-F expanded-products counter") ||
      !check_cuda(cudaMalloc(&context->compact_count, sizeof(uint64_t)),
                  error, error_capacity,
                  "allocate complete-F compact-output counter") ||
      !check_cuda(cudaMalloc(&context->offsets,
                             size_t(input_dimension + 1) * sizeof(uint32_t)),
                  error, error_capacity, "allocate complete-F offsets") ||
      (entry_count != 0 &&
       !check_cuda(cudaMalloc(&context->entries,
                              size_t(entry_count) * sizeof(SparseEntry)),
                   error, error_capacity, "allocate complete-F entries")) ||
      !check_cuda(cudaMemcpyAsync(context->offsets, host_offsets,
                                  size_t(input_dimension + 1) * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload complete-F offsets") ||
      (entry_count != 0 &&
       !check_cuda(cudaMemcpyAsync(context->entries, host_entries,
                                   size_t(entry_count) * sizeof(SparseEntry),
                                   cudaMemcpyHostToDevice, context->stream),
                   error, error_capacity, "upload complete-F entries")) ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish complete-F sparse upload")) {
    release(context);
    return nullptr;
  }
  if (!reserve_outputs(context, 1, error, error_capacity)) {
    release(context);
    return nullptr;
  }
  return context;
}

int adynkra_complete_f_sparse_apply_composed_batch(
    void *raw_first, void *raw_second, const SparseInput *host_inputs,
    uint32_t input_count, uint32_t lane_count, int64_t *host_output_real,
    int64_t *host_output_imaginary, uint64_t *expanded_products,
    float *elapsed_milliseconds, char *error, size_t error_capacity) {
  Context *first = static_cast<Context *>(raw_first);
  Context *second = static_cast<Context *>(raw_second);
  if (first == nullptr || second == nullptr || first == second ||
      lane_count == 0 ||
      first->device != second->device ||
      first->output_dimension != second->input_dimension ||
      (input_count != 0 && host_inputs == nullptr) ||
      host_output_real == nullptr || host_output_imaginary == nullptr) {
    set_error(error, error_capacity,
              "invalid complete-F composed sparse apply input");
    return 1;
  }
  if (first->poisoned || second->poisoned) {
    set_error(error, error_capacity,
              "complete-F composed sparse context is poisoned");
    return 1;
  }
  uint64_t first_expanded = 0;
  for (uint32_t index = 0; index < input_count; ++index) {
    if (host_inputs[index].lane >= lane_count ||
        host_inputs[index].column >= first->input_dimension) {
      set_error(error, error_capacity,
                "complete-F composed sparse input is out of range");
      return 1;
    }
    first_expanded +=
        uint64_t(first->host_offsets[host_inputs[index].column + 1] -
                 first->host_offsets[host_inputs[index].column]);
  }
  const size_t first_output_count =
      size_t(lane_count) * first->output_dimension;
  const size_t second_output_count =
      size_t(lane_count) * second->output_dimension;
  const uint64_t dense_count = uint64_t(lane_count) * second->input_dimension;
  const uint64_t dense_blocks = (dense_count + 255) / 256;
  int maximum_grid_x = 0;
  if (!check_cuda(cudaStreamSynchronize(second->stream), error, error_capacity,
                  "synchronize complete-F composed second stream") ||
      !check_cuda(cudaDeviceGetAttribute(&maximum_grid_x, cudaDevAttrMaxGridDimX,
                                         first->device),
                  error, error_capacity,
                  "query complete-F composed grid limit")) {
    return 1;
  }
  if (dense_blocks > uint64_t(maximum_grid_x)) {
    set_error(error, error_capacity,
              "complete-F composed dense grid exceeds device limit");
    return 1;
  }
  if (!reserve_inputs(first, input_count, error, error_capacity) ||
      !reserve_outputs(first, lane_count, error, error_capacity) ||
      !reserve_outputs(second, lane_count, error, error_capacity) ||
      (input_count != 0 &&
       !check_cuda(cudaMemcpyAsync(first->inputs, host_inputs,
                                   size_t(input_count) * sizeof(SparseInput),
                                   cudaMemcpyHostToDevice, first->stream),
                   error, error_capacity,
                   "upload complete-F composed sparse inputs")) ||
      !check_cuda(cudaMemsetAsync(first->output_real, 0,
                                  first_output_count * sizeof(int64_t),
                                  first->stream),
                  error, error_capacity,
                  "clear complete-F composed first real output") ||
      !check_cuda(cudaMemsetAsync(first->output_imaginary, 0,
                                  first_output_count * sizeof(int64_t),
                                  first->stream),
                  error, error_capacity,
                  "clear complete-F composed first imaginary output") ||
      !check_cuda(cudaMemsetAsync(second->output_real, 0,
                                  second_output_count * sizeof(int64_t),
                                  first->stream),
                  error, error_capacity,
                  "clear complete-F composed second real output") ||
      !check_cuda(cudaMemsetAsync(second->output_imaginary, 0,
                                  second_output_count * sizeof(int64_t),
                                  first->stream),
                  error, error_capacity,
                  "clear complete-F composed second imaginary output") ||
      !check_cuda(cudaMemcpyAsync(second->expanded_products, &first_expanded,
                                  sizeof(uint64_t), cudaMemcpyHostToDevice,
                                  first->stream),
                  error, error_capacity,
                  "initialize complete-F composed product count") ||
      !check_cuda(cudaEventRecord(first->started, first->stream), error,
                  error_capacity, "record complete-F composed start")) {
    return 1;
  }
  if (input_count != 0) {
    sparse_apply_kernel<<<input_count, 256, 0, first->stream>>>(
        first->offsets, first->entries, first->inputs, input_count,
        first->output_dimension, first->output_real, first->output_imaginary);
    if (!check_cuda(cudaGetLastError(), error, error_capacity,
                    "launch complete-F composed sparse kernel")) {
      return 1;
    }
  }
  if (dense_blocks != 0) {
    dense_apply_kernel<<<uint32_t(dense_blocks), 256, 0, first->stream>>>(
        second->offsets, second->entries, first->output_real,
        first->output_imaginary, dense_count, second->input_dimension,
        second->output_dimension, second->output_real,
        second->output_imaginary, second->expanded_products);
    if (!check_cuda(cudaGetLastError(), error, error_capacity,
                    "launch complete-F composed dense kernel")) {
      return 1;
    }
  }
  if (!check_cuda(cudaEventRecord(first->finished, first->stream), error,
                  error_capacity, "record complete-F composed finish") ||
      !check_cuda(cudaMemcpyAsync(host_output_real, second->output_real,
                                  second_output_count * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost, first->stream),
                  error, error_capacity,
                  "download complete-F composed real output") ||
      !check_cuda(cudaMemcpyAsync(host_output_imaginary,
                                  second->output_imaginary,
                                  second_output_count * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost, first->stream),
                  error, error_capacity,
                  "download complete-F composed imaginary output") ||
      !check_cuda(cudaMemcpyAsync(&first_expanded, second->expanded_products,
                                  sizeof(uint64_t), cudaMemcpyDeviceToHost,
                                  first->stream),
                  error, error_capacity,
                  "download complete-F composed product count") ||
      !check_cuda(cudaStreamSynchronize(first->stream), error, error_capacity,
                  "finish complete-F composed sparse apply")) {
    return 1;
  }
  float elapsed = 0.0F;
  if (!check_cuda(cudaEventElapsedTime(&elapsed, first->started,
                                       first->finished),
                  error, error_capacity,
                  "measure complete-F composed sparse kernels")) {
    return 1;
  }
  if (expanded_products != nullptr) {
    *expanded_products = first_expanded;
  }
  if (elapsed_milliseconds != nullptr) {
    *elapsed_milliseconds = elapsed;
  }
  return 0;
}

int adynkra_complete_f_sparse_apply_batch(
    void *raw_context, const SparseInput *host_inputs, uint32_t input_count,
    uint32_t lane_count,
    int64_t *host_output_real, int64_t *host_output_imaginary,
    uint64_t *expanded_products, float *elapsed_milliseconds, char *error,
    size_t error_capacity) {
  Context *context = static_cast<Context *>(raw_context);
  if (context == nullptr || lane_count == 0 ||
      (input_count != 0 && host_inputs == nullptr) ||
      host_output_real == nullptr || host_output_imaginary == nullptr) {
    set_error(error, error_capacity, "invalid complete-F sparse apply input");
    return 1;
  }
  if (context->poisoned) {
    set_error(error, error_capacity,
              "complete-F sparse context is poisoned");
    return 1;
  }
  uint64_t expanded = 0;
  for (uint32_t index = 0; index < input_count; ++index) {
    if (host_inputs[index].lane >= lane_count ||
        host_inputs[index].column >= context->input_dimension) {
      set_error(error, error_capacity, "complete-F sparse input is out of range");
      return 1;
    }
    const uint64_t column_products =
        uint64_t(context->host_offsets[host_inputs[index].column + 1] -
                 context->host_offsets[host_inputs[index].column]);
    if (expanded > std::numeric_limits<uint64_t>::max() - column_products) {
      set_error(error, error_capacity,
                "complete-F sparse expanded-product count overflow");
      return 1;
    }
    expanded += column_products;
  }
  const size_t output_count = size_t(lane_count) * context->output_dimension;
  if (!reserve_inputs(context, input_count, error, error_capacity) ||
      !reserve_outputs(context, lane_count, error, error_capacity) ||
      (input_count != 0 &&
       !check_cuda(cudaMemcpyAsync(context->inputs, host_inputs,
                                   size_t(input_count) * sizeof(SparseInput),
                                   cudaMemcpyHostToDevice, context->stream),
                   error, error_capacity, "upload complete-F sparse inputs")) ||
      !check_cuda(cudaMemsetAsync(context->output_real, 0,
                                  output_count * sizeof(int64_t),
                                  context->stream),
                  error, error_capacity, "clear complete-F real output") ||
      !check_cuda(cudaMemsetAsync(context->output_imaginary, 0,
                                  output_count * sizeof(int64_t),
                                  context->stream),
                  error, error_capacity, "clear complete-F imaginary output") ||
      !check_cuda(cudaEventRecord(context->started, context->stream), error,
                  error_capacity, "record complete-F start")) {
    return 1;
  }
  if (input_count != 0) {
    sparse_apply_kernel<<<input_count, 256, 0, context->stream>>>(
        context->offsets, context->entries, context->inputs, input_count,
        context->output_dimension, context->output_real,
        context->output_imaginary);
    if (!check_cuda(cudaGetLastError(), error, error_capacity,
                    "launch complete-F sparse kernel")) {
      return 1;
    }
  }
  if (!check_cuda(cudaEventRecord(context->finished, context->stream), error,
                  error_capacity, "record complete-F finish") ||
      !check_cuda(cudaMemcpyAsync(host_output_real, context->output_real,
                                  output_count * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity, "download complete-F real output") ||
      !check_cuda(cudaMemcpyAsync(host_output_imaginary,
                                  context->output_imaginary,
                                  output_count * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity,
                  "download complete-F imaginary output") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish complete-F sparse apply")) {
    return 1;
  }
  float elapsed = 0.0F;
  if (!check_cuda(cudaEventElapsedTime(&elapsed, context->started,
                                       context->finished),
                  error, error_capacity, "measure complete-F sparse kernel")) {
    return 1;
  }
  if (expanded_products != nullptr) {
    *expanded_products = expanded;
  }
  if (elapsed_milliseconds != nullptr) {
    *elapsed_milliseconds = elapsed;
  }
  return 0;
}

int adynkra_complete_f_sparse_apply_compact_batch(
    void *raw_context, const SparseInput *host_inputs, uint32_t input_count,
    uint32_t lane_count, SparseOutput *host_outputs,
    uint64_t host_output_capacity, uint64_t *host_output_count,
    uint64_t *expanded_products, float *elapsed_milliseconds, char *error,
    size_t error_capacity) {
  Context *context = static_cast<Context *>(raw_context);
  if (context == nullptr || lane_count == 0 || host_output_count == nullptr ||
      (input_count != 0 && host_inputs == nullptr)) {
    set_error(error, error_capacity,
              "invalid complete-F compact sparse apply input");
    return 1;
  }
  *host_output_count = 0;
  if (context->poisoned) {
    set_error(error, error_capacity,
              "complete-F compact sparse context is poisoned");
    return 1;
  }
  uint64_t expanded = 0;
  for (uint32_t index = 0; index < input_count; ++index) {
    if (host_inputs[index].lane >= lane_count ||
        host_inputs[index].column >= context->input_dimension) {
      set_error(error, error_capacity,
                "complete-F compact sparse input is out of range");
      return 1;
    }
    const uint64_t column_products =
        uint64_t(context->host_offsets[host_inputs[index].column + 1] -
                 context->host_offsets[host_inputs[index].column]);
    if (expanded > std::numeric_limits<uint64_t>::max() - column_products) {
      set_error(error, error_capacity,
                "complete-F compact expanded-product count overflow");
      return 1;
    }
    expanded += column_products;
  }
  const uint64_t output_count =
      uint64_t(lane_count) * context->output_dimension;
  const uint64_t maximum_nonzeros =
      output_count < expanded ? output_count : expanded;
  if (host_output_capacity < maximum_nonzeros ||
      (maximum_nonzeros != 0 && host_outputs == nullptr)) {
    set_error(error, error_capacity,
              "complete-F compact host output capacity is too small");
    return 1;
  }
  const uint64_t compact_blocks = (output_count + 255) / 256;
  int maximum_grid_x = 0;
  if (!check_cuda(cudaDeviceGetAttribute(&maximum_grid_x, cudaDevAttrMaxGridDimX,
                                         context->device),
                  error, error_capacity,
                  "query complete-F compact grid limit")) {
    return 1;
  }
  if (compact_blocks > uint64_t(maximum_grid_x)) {
    set_error(error, error_capacity,
              "complete-F compact grid exceeds device limit");
    return 1;
  }
  if (!reserve_inputs(context, input_count, error, error_capacity) ||
      !reserve_outputs(context, lane_count, error, error_capacity) ||
      (input_count != 0 &&
       !check_cuda(cudaMemcpyAsync(context->inputs, host_inputs,
                                   size_t(input_count) * sizeof(SparseInput),
                                   cudaMemcpyHostToDevice, context->stream),
                   error, error_capacity,
                   "upload complete-F compact sparse inputs")) ||
      !check_cuda(cudaMemsetAsync(context->output_real, 0,
                                  size_t(output_count) * sizeof(int64_t),
                                  context->stream),
                  error, error_capacity,
                  "clear complete-F compact real output") ||
      !check_cuda(cudaMemsetAsync(context->output_imaginary, 0,
                                  size_t(output_count) * sizeof(int64_t),
                                  context->stream),
                  error, error_capacity,
                  "clear complete-F compact imaginary output") ||
      !check_cuda(cudaMemsetAsync(context->compact_count, 0, sizeof(uint64_t),
                                  context->stream),
                  error, error_capacity,
                  "clear complete-F compact output count") ||
      !check_cuda(cudaEventRecord(context->started, context->stream), error,
                  error_capacity, "record complete-F compact start")) {
    return drain_failed_apply(context, false, error, error_capacity);
  }
  if (input_count != 0) {
    sparse_apply_kernel<<<input_count, 256, 0, context->stream>>>(
        context->offsets, context->entries, context->inputs, input_count,
        context->output_dimension, context->output_real,
        context->output_imaginary);
    if (!check_cuda(cudaGetLastError(), error, error_capacity,
                    "launch complete-F compact sparse kernel")) {
      return drain_failed_apply(context, false, error, error_capacity);
    }
  }
  if (compact_blocks != 0) {
    count_nonzero_kernel<<<uint32_t(compact_blocks), 256, 0,
                           context->stream>>>(
        context->output_real, context->output_imaginary, output_count,
        context->compact_count);
    if (!check_cuda(cudaGetLastError(), error, error_capacity,
                    "launch complete-F compact-output count kernel")) {
      return drain_failed_apply(context, false, error, error_capacity);
    }
  }
  uint64_t compact_count = 0;
  if (!check_cuda(cudaMemcpyAsync(&compact_count, context->compact_count,
                                  sizeof(uint64_t), cudaMemcpyDeviceToHost,
                                  context->stream),
                  error, error_capacity,
                  "download complete-F compact output count") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "count complete-F compact sparse output")) {
    return drain_failed_apply(context, true, error, error_capacity);
  }
  if (compact_count > maximum_nonzeros) {
    context->poisoned = true;
    set_error(error, error_capacity,
              "complete-F compact output exceeded proven capacity");
    return 1;
  }
  if (!reserve_compact_outputs(context, compact_count, error, error_capacity) ||
      !check_cuda(cudaMemsetAsync(context->compact_count, 0, sizeof(uint64_t),
                                  context->stream),
                  error, error_capacity,
                  "clear complete-F compact emission count")) {
    return drain_failed_apply(context, false, error, error_capacity);
  }
  if (compact_count != 0) {
    compact_nonzero_kernel<<<uint32_t(compact_blocks), 256, 0,
                             context->stream>>>(
        context->output_real, context->output_imaginary, output_count,
        context->output_dimension, context->compact_outputs, compact_count,
        context->compact_count);
    if (!check_cuda(cudaGetLastError(), error, error_capacity,
                    "launch complete-F compact-output emission kernel")) {
      return drain_failed_apply(context, false, error, error_capacity);
    }
  }
  uint64_t emitted_count = 0;
  if (!check_cuda(cudaEventRecord(context->finished, context->stream), error,
                  error_capacity, "record complete-F compact finish") ||
      (compact_count != 0 &&
       !check_cuda(cudaMemcpyAsync(host_outputs, context->compact_outputs,
                                   size_t(compact_count) * sizeof(SparseOutput),
                                   cudaMemcpyDeviceToHost, context->stream),
                   error, error_capacity,
                   "download complete-F compact outputs")) ||
      !check_cuda(cudaMemcpyAsync(&emitted_count, context->compact_count,
                                  sizeof(uint64_t), cudaMemcpyDeviceToHost,
                                  context->stream),
                  error, error_capacity,
                  "download complete-F compact emission count") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish complete-F compact sparse apply")) {
    return drain_failed_apply(context, true, error, error_capacity);
  }
  if (emitted_count != compact_count) {
    context->poisoned = true;
    set_error(error, error_capacity,
              "complete-F compact count changed between exact passes");
    return 1;
  }
  float elapsed = 0.0F;
  if (!check_cuda(cudaEventElapsedTime(&elapsed, context->started,
                                       context->finished),
                  error, error_capacity,
                  "measure complete-F compact sparse kernel")) {
    return 1;
  }
  *host_output_count = compact_count;
  if (expanded_products != nullptr) {
    *expanded_products = expanded;
  }
  if (elapsed_milliseconds != nullptr) {
    *elapsed_milliseconds = elapsed;
  }
  return 0;
}

uint64_t adynkra_complete_f_sparse_resident_bytes(const void *raw_context) {
  const Context *context = static_cast<const Context *>(raw_context);
  if (context == nullptr) {
    return 0;
  }
  return uint64_t(context->input_dimension + 1) * sizeof(uint32_t) +
         uint64_t(context->entry_count) * sizeof(SparseEntry) +
         uint64_t(context->input_capacity) * sizeof(SparseInput) +
         uint64_t(context->output_lane_capacity) *
             uint64_t(context->output_dimension) * 2 * sizeof(int64_t) +
         uint64_t(context->compact_capacity) * sizeof(SparseOutput) +
         2 * sizeof(uint64_t);
}

void adynkra_complete_f_sparse_destroy(void *raw_context) {
  release(static_cast<Context *>(raw_context));
}

} // extern "C"
