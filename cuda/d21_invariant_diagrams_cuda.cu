#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace {

constexpr uint32_t kSpin = 32;
constexpr uint32_t kMasks = 2048;
constexpr uint32_t kDiagrams = 400;

struct PackedDiagram {
  uint8_t outer_degree;
  uint8_t inner_degree;
  uint8_t cross;
  uint8_t outer_count;
  uint8_t inner_count;
  uint8_t metric_count;
  uint8_t reserved0;
  uint8_t reserved1;
  uint8_t outer_external[6];
  uint8_t inner_external[6];
  uint8_t metric_pairs[12];
};

struct Query {
  uint16_t diagram;
  uint8_t outer_left;
  uint8_t outer_right;
  uint8_t momentum;
  uint8_t h_vector;
  uint8_t output_axes[4];
  uint8_t input_spinor;
  uint8_t output_spinor;
  int16_t h_coefficient;
  uint16_t reserved;
};

static_assert(sizeof(PackedDiagram) == 32);
static_assert(sizeof(Query) == 16);

struct Context {
  uint64_t capacity = 0;
  PackedDiagram* diagrams = nullptr;
  Query* queries = nullptr;
  int64_t* output = nullptr;
  uint8_t* gamma_row = nullptr;
  int8_t* gamma_value = nullptr;
  uint8_t* charge_gamma_row = nullptr;
  int8_t* charge_gamma_value = nullptr;
};

void set_error(char* error, uint64_t capacity, const char* message) {
  if (error != nullptr && capacity != 0) {
    std::snprintf(error, static_cast<size_t>(capacity), "%s", message);
  }
}

bool checked(cudaError_t status, char* error, uint64_t capacity,
             const char* action) {
  if (status == cudaSuccess) return true;
  char buffer[512];
  std::snprintf(buffer, sizeof(buffer), "%s: %s", action,
                cudaGetErrorString(status));
  set_error(error, capacity, buffer);
  return false;
}

void destroy(Context* context) {
  if (context == nullptr) return;
  cudaFree(context->diagrams);
  cudaFree(context->queries);
  cudaFree(context->output);
  cudaFree(context->gamma_row);
  cudaFree(context->gamma_value);
  cudaFree(context->charge_gamma_row);
  cudaFree(context->charge_gamma_value);
  delete context;
}

__device__ __forceinline__ int metric(uint32_t axis) {
  return axis == 0 ? -1 : 1;
}

__device__ __forceinline__ uint32_t axis_of(uint8_t label, const Query& query) {
  if (label == 0) return query.momentum;
  if (label == 1) return query.h_vector;
  return query.output_axes[label - 2];
}

__device__ __forceinline__ int attachment_metric(uint8_t label,
                                                  uint32_t axis) {
  return label == 0 ? 1 : metric(axis);
}

__device__ __forceinline__ int metric_pair(uint8_t left, uint8_t right,
                                            const Query& query) {
  const uint32_t a = axis_of(left, query);
  const uint32_t b = axis_of(right, query);
  if (a != b) return 0;
  if (left == 0 || right == 0) {
    return 1;
  }
  return metric(a);
}

__device__ __forceinline__ bool append_axis(uint32_t axis, uint32_t& mask,
                                             int& sign) {
  const uint32_t bit = 1U << axis;
  if ((mask & bit) != 0) return false;
  sign *= (__popc(mask >> (axis + 1)) & 1) != 0 ? -1 : 1;
  mask |= bit;
  return true;
}

__device__ __forceinline__ int signed_permutation_entry(
    const uint8_t* rows, const int8_t* values, uint32_t mask, uint32_t row,
    uint32_t column) {
  const uint32_t ordinal = mask * kSpin + column;
  return rows[ordinal] == row ? static_cast<int>(values[ordinal]) : 0;
}

__global__ void evaluate_kernel(
    const PackedDiagram* __restrict__ diagrams,
    const Query* __restrict__ queries, uint64_t count,
    const uint8_t* __restrict__ gamma_row,
    const int8_t* __restrict__ gamma_value,
    const uint8_t* __restrict__ charge_gamma_row,
    const int8_t* __restrict__ charge_gamma_value,
    int64_t* __restrict__ output) {
  const uint64_t ordinal = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                           static_cast<uint64_t>(threadIdx.x);
  if (ordinal >= count) return;
  const Query query = queries[ordinal];
  const PackedDiagram diagram = diagrams[query.diagram];
  int64_t base = query.h_coefficient;

#pragma unroll
  for (uint32_t pair = 0; pair < 6; ++pair) {
    if (pair >= diagram.metric_count) break;
    base *= metric_pair(diagram.metric_pairs[2 * pair],
                        diagram.metric_pairs[2 * pair + 1], query);
  }

  uint32_t outer_base_mask = 0;
  uint32_t inner_base_mask = 0;
  int outer_base_sign = 1;
  int inner_base_sign = 1;
#pragma unroll
  for (uint32_t slot = 0; slot < 6; ++slot) {
    if (slot >= diagram.outer_count) break;
    const uint8_t label = diagram.outer_external[slot];
    const uint32_t axis = axis_of(label, query);
    base *= attachment_metric(label, axis);
    if (!append_axis(axis, outer_base_mask, outer_base_sign)) base = 0;
  }
#pragma unroll
  for (uint32_t slot = 0; slot < 6; ++slot) {
    if (slot >= diagram.inner_count) break;
    const uint8_t label = diagram.inner_external[slot];
    const uint32_t axis = axis_of(label, query);
    base *= attachment_metric(label, axis);
    if (!append_axis(axis, inner_base_mask, inner_base_sign)) base = 0;
  }

  int64_t sum = 0;
  if (base != 0) {
    const uint32_t limit0 = diagram.cross > 0 ? 11 : 1;
    const uint32_t limit1 = diagram.cross > 1 ? 11 : 1;
    const uint32_t limit2 = diagram.cross > 2 ? 11 : 1;
    const uint32_t limit3 = diagram.cross > 3 ? 11 : 1;
    for (uint32_t axis0 = 0; axis0 < limit0; ++axis0)
      for (uint32_t axis1 = 0; axis1 < limit1; ++axis1)
        for (uint32_t axis2 = 0; axis2 < limit2; ++axis2)
          for (uint32_t axis3 = 0; axis3 < limit3; ++axis3) {
            const uint32_t axes[4] = {axis0, axis1, axis2, axis3};
            bool increasing = true;
#pragma unroll
            for (uint32_t slot = 1; slot < 4; ++slot) {
              if (slot >= diagram.cross) break;
              increasing &= axes[slot - 1] < axes[slot];
            }
            if (!increasing) continue;
            uint32_t outer_mask = outer_base_mask;
            uint32_t inner_mask = inner_base_mask;
            int outer_sign = outer_base_sign;
            int inner_sign = inner_base_sign;
            int cross_metric = 1;
            bool valid = true;
#pragma unroll
            for (uint32_t slot = 0; slot < 4; ++slot) {
              if (slot >= diagram.cross) break;
              valid &= append_axis(axes[slot], outer_mask, outer_sign);
              valid &= append_axis(axes[slot], inner_mask, inner_sign);
              cross_metric *= metric(axes[slot]);
            }
            if (!valid) continue;
            const int left = signed_permutation_entry(
                charge_gamma_row, charge_gamma_value, outer_mask,
                query.outer_left, query.outer_right);
            const int right = signed_permutation_entry(
                gamma_row, gamma_value, inner_mask, query.output_spinor,
                query.input_spinor);
            sum += base * outer_sign * inner_sign * cross_metric * left * right;
          }
  }
  output[ordinal] = sum;
}

}  // namespace

extern "C" void* adynkra_d21_create(
    const PackedDiagram* host_diagrams, uint64_t diagram_count,
    const uint8_t* host_gamma_row, const int8_t* host_gamma_value,
    const uint8_t* host_charge_gamma_row,
    const int8_t* host_charge_gamma_value, uint64_t capacity, char* error,
    uint64_t error_capacity) {
  if (host_diagrams == nullptr || host_gamma_row == nullptr ||
      host_gamma_value == nullptr || host_charge_gamma_row == nullptr ||
      host_charge_gamma_value == nullptr || diagram_count != kDiagrams ||
      capacity == 0 ||
      capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) *
                     256U) {
    set_error(error, error_capacity, "invalid D21 CUDA context input");
    return nullptr;
  }
  Context* context = new (std::nothrow) Context();
  if (context == nullptr) {
    set_error(error, error_capacity, "allocate D21 CUDA context");
    return nullptr;
  }
  context->capacity = capacity;
  const size_t lookup_items = static_cast<size_t>(kMasks) * kSpin;
  if (!checked(cudaMalloc(&context->diagrams,
                          diagram_count * sizeof(PackedDiagram)),
               error, error_capacity, "allocate D21 diagrams") ||
      !checked(cudaMalloc(&context->queries, capacity * sizeof(Query)), error,
               error_capacity, "allocate D21 queries") ||
      !checked(cudaMalloc(&context->output, capacity * sizeof(int64_t)), error,
               error_capacity, "allocate D21 output") ||
      !checked(cudaMalloc(&context->gamma_row, lookup_items), error,
               error_capacity, "allocate D21 gamma rows") ||
      !checked(cudaMalloc(&context->gamma_value, lookup_items), error,
               error_capacity, "allocate D21 gamma values") ||
      !checked(cudaMalloc(&context->charge_gamma_row, lookup_items), error,
               error_capacity, "allocate D21 charge-gamma rows") ||
      !checked(cudaMalloc(&context->charge_gamma_value, lookup_items), error,
               error_capacity, "allocate D21 charge-gamma values") ||
      !checked(cudaMemcpy(context->diagrams, host_diagrams,
                          diagram_count * sizeof(PackedDiagram),
                          cudaMemcpyHostToDevice),
               error, error_capacity, "upload D21 diagrams") ||
      !checked(cudaMemcpy(context->gamma_row, host_gamma_row, lookup_items,
                          cudaMemcpyHostToDevice),
               error, error_capacity, "upload D21 gamma rows") ||
      !checked(cudaMemcpy(context->gamma_value, host_gamma_value, lookup_items,
                          cudaMemcpyHostToDevice),
               error, error_capacity, "upload D21 gamma values") ||
      !checked(cudaMemcpy(context->charge_gamma_row, host_charge_gamma_row,
                          lookup_items, cudaMemcpyHostToDevice),
               error, error_capacity, "upload D21 charge-gamma rows") ||
      !checked(cudaMemcpy(context->charge_gamma_value,
                          host_charge_gamma_value, lookup_items,
                          cudaMemcpyHostToDevice),
               error, error_capacity, "upload D21 charge-gamma values")) {
    destroy(context);
    return nullptr;
  }
  return context;
}

extern "C" int32_t adynkra_d21_evaluate(
    void* opaque, const Query* host_queries, uint64_t count,
    int64_t* host_output, float* kernel_milliseconds, char* error,
    uint64_t error_capacity) {
  Context* context = static_cast<Context*>(opaque);
  if (context == nullptr || host_queries == nullptr || host_output == nullptr ||
      count == 0 || count > context->capacity) {
    set_error(error, error_capacity, "invalid D21 CUDA evaluate input");
    return 1;
  }
  if (!checked(cudaMemcpy(context->queries, host_queries,
                          count * sizeof(Query), cudaMemcpyHostToDevice),
               error, error_capacity, "upload D21 query batch")) {
    return 2;
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!checked(cudaEventCreate(&start), error, error_capacity,
               "create D21 start event") ||
      !checked(cudaEventCreate(&stop), error, error_capacity,
               "create D21 stop event")) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 3;
  }
  const uint32_t threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((count + threads - 1) / threads);
  cudaEventRecord(start);
  evaluate_kernel<<<blocks, threads>>>(
      context->diagrams, context->queries, count, context->gamma_row,
      context->gamma_value, context->charge_gamma_row,
      context->charge_gamma_value, context->output);
  cudaEventRecord(stop);
  if (!checked(cudaGetLastError(), error, error_capacity,
               "launch D21 evaluator") ||
      !checked(cudaEventSynchronize(stop), error, error_capacity,
               "synchronize D21 evaluator") ||
      !checked(cudaMemcpy(host_output, context->output, count * sizeof(int64_t),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download D21 outputs")) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 4;
  }
  if (kernel_milliseconds != nullptr) {
    cudaEventElapsedTime(kernel_milliseconds, start, stop);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

extern "C" void adynkra_d21_destroy(void* opaque) {
  destroy(static_cast<Context*>(opaque));
}
