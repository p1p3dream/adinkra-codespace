#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <new>

namespace {

constexpr uint32_t kSpin = 32;
constexpr uint32_t kMasks = 2048;
constexpr uint32_t kForms = 330;
constexpr uint32_t kTarget = kSpin * kForms;
constexpr uint32_t kCasimirSupport = 29;
constexpr uint32_t kSectors = 5;
constexpr uint32_t kPrimesCount = 3;
constexpr uint32_t kMaximumRank = 14;
constexpr uint32_t kMaximumCandidates = 209;

__host__ __device__ constexpr uint32_t prime_at(uint32_t slot) {
  return slot == 0 ? 1073741783U
                   : (slot == 1 ? 1073741723U : 1073741719U);
}

struct PackedDiagram {
  uint8_t outer_degree, inner_degree, cross, outer_count, inner_count,
      metric_count, reserved0, reserved1;
  uint8_t outer_external[6];
  uint8_t inner_external[6];
  uint8_t metric_pairs[12];
};

struct CasimirEntry {
  uint16_t column;
  int16_t value;
};

struct HTerm {
  uint8_t input_spinor;
  uint8_t h_vector;
  int16_t coefficient;
};

static_assert(sizeof(PackedDiagram) == 32);
static_assert(sizeof(CasimirEntry) == 4);
static_assert(sizeof(HTerm) == 4);

struct Context {
  uint32_t candidate_count = 0;
  PackedDiagram* diagrams = nullptr;
  uint16_t* candidate_diagrams = nullptr;
  uint8_t* gamma_row = nullptr;
  int8_t* gamma_value = nullptr;
  uint8_t* charge_gamma_row = nullptr;
  int8_t* charge_gamma_value = nullptr;
  uint8_t* form_axes = nullptr;
  CasimirEntry* casimir_rows = nullptr;
  int64_t* raw = nullptr;
  uint32_t* modular_raw = nullptr;
  uint32_t* modular_a = nullptr;
  uint32_t* modular_b = nullptr;
  uint32_t* evaluated = nullptr;
  uint32_t* basis = nullptr;
  uint32_t* ranks = nullptr;
  uint16_t* pivots = nullptr;
  uint64_t* pivot_rows = nullptr;
  uint32_t* overflow = nullptr;
};

void set_error(char* error, uint64_t capacity, const char* message) {
  if (error && capacity) std::snprintf(error, size_t(capacity), "%s", message);
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
  if (!context) return;
  cudaFree(context->diagrams);
  cudaFree(context->candidate_diagrams);
  cudaFree(context->gamma_row);
  cudaFree(context->gamma_value);
  cudaFree(context->charge_gamma_row);
  cudaFree(context->charge_gamma_value);
  cudaFree(context->form_axes);
  cudaFree(context->casimir_rows);
  cudaFree(context->raw);
  cudaFree(context->modular_raw);
  cudaFree(context->modular_a);
  cudaFree(context->modular_b);
  cudaFree(context->evaluated);
  cudaFree(context->basis);
  cudaFree(context->ranks);
  cudaFree(context->pivots);
  cudaFree(context->pivot_rows);
  cudaFree(context->overflow);
  delete context;
}

__device__ __forceinline__ int metric(uint32_t axis) {
  return axis == 0 ? -1 : 1;
}

__device__ __forceinline__ uint32_t axis_of(uint8_t label, uint32_t momentum,
                                             uint32_t h_vector,
                                             const uint8_t* output_axes) {
  if (label == 0) return momentum;
  if (label == 1) return h_vector;
  return output_axes[label - 2];
}

__device__ __forceinline__ bool append_axis(uint32_t axis, uint32_t& mask,
                                             int& sign) {
  const uint32_t bit = 1U << axis;
  if (mask & bit) return false;
  sign *= (__popc(mask >> (axis + 1)) & 1) ? -1 : 1;
  mask |= bit;
  return true;
}

__device__ __forceinline__ int pair_factor(uint8_t left, uint8_t right,
                                            uint32_t momentum,
                                            uint32_t h_vector,
                                            const uint8_t* output_axes) {
  const uint32_t a = axis_of(left, momentum, h_vector, output_axes);
  const uint32_t b = axis_of(right, momentum, h_vector, output_axes);
  if (a != b) return 0;
  if (left == 0 || right == 0) return 1;
  return metric(a);
}

__global__ void emit_witness(
    const PackedDiagram* __restrict__ diagrams,
    const uint16_t* __restrict__ candidate_diagrams, uint32_t candidate_count,
    const HTerm* terms, uint32_t term_count, uint32_t outer_left,
    uint32_t outer_right, uint32_t momentum,
    const uint8_t* __restrict__ form_axes,
    const uint8_t* __restrict__ gamma_row,
    const int8_t* __restrict__ gamma_value,
    const uint8_t* __restrict__ charge_gamma_row,
    const int8_t* __restrict__ charge_gamma_value, int64_t* raw) {
  const uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total = candidate_count * kForms;
  if (ordinal >= total) return;
  const uint32_t candidate = ordinal % candidate_count;
  const uint32_t form = ordinal / candidate_count;
  const PackedDiagram diagram = diagrams[candidate_diagrams[candidate]];
  const uint8_t* canonical_outputs = form_axes + 4 * form;
  for (uint32_t p0 = 0; p0 < 4; ++p0)
    for (uint32_t p1 = 0; p1 < 4; ++p1)
      for (uint32_t p2 = 0; p2 < 4; ++p2)
        for (uint32_t p3 = 0; p3 < 4; ++p3) {
          if (p0 == p1 || p0 == p2 || p0 == p3 || p1 == p2 || p1 == p3 ||
              p2 == p3)
            continue;
          const uint32_t permutation[4] = {p0, p1, p2, p3};
          const uint8_t outputs[4] = {
              canonical_outputs[p0], canonical_outputs[p1],
              canonical_outputs[p2], canonical_outputs[p3]};
          uint32_t inversions = 0;
          for (uint32_t left = 0; left < 4; ++left)
            for (uint32_t right = left + 1; right < 4; ++right)
              inversions += permutation[left] > permutation[right];
          const int output_parity = inversions & 1U ? -1 : 1;
          for (uint32_t term_slot = 0; term_slot < term_count; ++term_slot) {
            const HTerm term = terms[term_slot];
            int64_t base = int64_t(output_parity) * term.coefficient;
            for (uint32_t pair = 0; pair < diagram.metric_count; ++pair) {
              base *= pair_factor(diagram.metric_pairs[2 * pair],
                                  diagram.metric_pairs[2 * pair + 1], momentum,
                                  term.h_vector, outputs);
            }
            uint32_t outer_base = 0, inner_base = 0;
            int outer_base_sign = 1, inner_base_sign = 1;
            for (uint32_t slot = 0; slot < diagram.outer_count; ++slot) {
              const uint8_t label = diagram.outer_external[slot];
              const uint32_t axis =
                  axis_of(label, momentum, term.h_vector, outputs);
              base *= label == 0 ? 1 : metric(axis);
              if (!append_axis(axis, outer_base, outer_base_sign)) base = 0;
            }
            for (uint32_t slot = 0; slot < diagram.inner_count; ++slot) {
              const uint8_t label = diagram.inner_external[slot];
              const uint32_t axis =
                  axis_of(label, momentum, term.h_vector, outputs);
              base *= label == 0 ? 1 : metric(axis);
              if (!append_axis(axis, inner_base, inner_base_sign)) base = 0;
            }
            if (base == 0) continue;
            const uint32_t limit0 = diagram.cross > 0 ? 11 : 1;
            const uint32_t limit1 = diagram.cross > 1 ? 11 : 1;
            const uint32_t limit2 = diagram.cross > 2 ? 11 : 1;
            const uint32_t limit3 = diagram.cross > 3 ? 11 : 1;
            for (uint32_t a0 = 0; a0 < limit0; ++a0)
              for (uint32_t a1 = 0; a1 < limit1; ++a1)
                for (uint32_t a2 = 0; a2 < limit2; ++a2)
                  for (uint32_t a3 = 0; a3 < limit3; ++a3) {
                    const uint32_t axes[4] = {a0, a1, a2, a3};
                    bool increasing = true;
                    for (uint32_t slot = 1; slot < diagram.cross; ++slot)
                      increasing &= axes[slot - 1] < axes[slot];
                    if (!increasing) continue;
                    uint32_t outer_mask = outer_base, inner_mask = inner_base;
                    int outer_sign = outer_base_sign,
                        inner_sign = inner_base_sign;
                    int cross_metric = 1;
                    bool valid = true;
                    for (uint32_t slot = 0; slot < diagram.cross; ++slot) {
                      valid &= append_axis(axes[slot], outer_mask, outer_sign);
                      valid &= append_axis(axes[slot], inner_mask, inner_sign);
                      cross_metric *= metric(axes[slot]);
                    }
                    if (!valid) continue;
                    const uint32_t outer_lookup =
                        outer_mask * kSpin + outer_right;
                    if (charge_gamma_row[outer_lookup] != outer_left) continue;
                    const int left = charge_gamma_value[outer_lookup];
                    const uint32_t inner_lookup =
                        inner_mask * kSpin + term.input_spinor;
                    const uint32_t output_spinor = gamma_row[inner_lookup];
                    const int right = gamma_value[inner_lookup];
                    const int64_t value = base * outer_sign * inner_sign *
                                          cross_metric * left * right;
                    atomicAdd(
                        reinterpret_cast<unsigned long long*>(
                            raw + (output_spinor * kForms + form) *
                                      candidate_count +
                                  candidate),
                        static_cast<unsigned long long>(value));
                  }
          }
        }
}

__device__ __forceinline__ uint32_t signed_mod(int64_t value,
                                                uint32_t prime) {
  const int64_t residue = value % static_cast<int64_t>(prime);
  return static_cast<uint32_t>(residue < 0 ? residue + prime : residue);
}

__global__ void reduce_raw(const int64_t* raw, uint32_t candidate_count,
                           uint32_t* modular) {
  const uint64_t ordinal = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t plane = uint64_t(kTarget) * candidate_count;
  if (ordinal >= plane) return;
  const int64_t value = raw[ordinal];
  for (uint32_t slot = 0; slot < kPrimesCount; ++slot)
    modular[uint64_t(slot) * plane + ordinal] = signed_mod(value, prime_at(slot));
}

__global__ void casimir_shift(
    const uint32_t* input, uint32_t* output,
    const CasimirEntry* __restrict__ casimir_rows, uint32_t candidate_count,
    int32_t shift) {
  const uint64_t ordinal = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t plane = uint64_t(kTarget) * candidate_count;
  const uint64_t total = plane * kPrimesCount;
  if (ordinal >= total) return;
  const uint32_t prime_slot = ordinal / plane;
  const uint64_t within = ordinal % plane;
  const uint32_t row = within / candidate_count;
  const uint32_t candidate = within % candidate_count;
  const uint32_t prime = prime_at(prime_slot);
  uint64_t sum = 0;
  const CasimirEntry* entries = casimir_rows + row * kCasimirSupport;
  for (uint32_t slot = 0; slot < kCasimirSupport; ++slot) {
    const int32_t coefficient = entries[slot].value;
    const uint32_t residue = coefficient < 0 ? prime - uint32_t(-coefficient)
                                             : uint32_t(coefficient);
    sum += uint64_t(residue) *
           input[uint64_t(prime_slot) * plane +
                 uint64_t(entries[slot].column) * candidate_count + candidate];
    sum %= prime;
  }
  const uint32_t diagonal = input[ordinal];
  const uint32_t shift_residue = shift < 0 ? prime - uint32_t(-shift)
                                           : uint32_t(shift);
  const uint32_t subtract =
      uint64_t(shift_residue) * diagonal % prime;
  output[ordinal] = sum >= subtract ? uint32_t(sum) - subtract
                                    : uint32_t(sum + prime) - subtract;
}

__device__ uint32_t pow_mod(uint32_t base, uint32_t exponent, uint32_t prime) {
  uint64_t result = 1, factor = base;
  while (exponent) {
    if (exponent & 1U) result = result * factor % prime;
    factor = factor * factor % prime;
    exponent >>= 1U;
  }
  return uint32_t(result);
}

__global__ void retained_rref(
    const uint32_t* projected, uint32_t candidate_count, uint32_t sector,
    uint32_t expected_rank, uint64_t source_row_base, uint32_t* basis,
    uint32_t* ranks, uint16_t* pivots, uint64_t* pivot_rows) {
  const uint32_t prime_slot = blockIdx.x;
  if (prime_slot >= kPrimesCount) return;
  const uint32_t state = sector * kPrimesCount + prime_slot;
  uint32_t rank = ranks[state];
  if (rank >= expected_rank) return;
  const uint32_t prime = prime_at(prime_slot);
  const uint64_t plane = uint64_t(kTarget) * candidate_count;
  extern __shared__ uint32_t row[];
  __shared__ uint32_t pivot;
  for (uint32_t target_row = 0; target_row < kTarget && rank < expected_rank;
       ++target_row) {
    for (uint32_t column = threadIdx.x; column < candidate_count;
         column += blockDim.x) {
      row[column] = projected[uint64_t(prime_slot) * plane +
                              uint64_t(target_row) * candidate_count + column];
    }
    __syncthreads();
    for (uint32_t existing = 0; existing < rank; ++existing) {
      const uint32_t existing_pivot = pivots[state * kMaximumRank + existing];
      const uint32_t factor = row[existing_pivot];
      for (uint32_t column = threadIdx.x; column < candidate_count;
           column += blockDim.x) {
        const uint32_t value =
            basis[(uint64_t(state) * kMaximumRank + existing) *
                      kMaximumCandidates +
                  column];
        const uint32_t product = uint64_t(factor) * value % prime;
        row[column] = row[column] >= product ? row[column] - product
                                            : row[column] + prime - product;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) pivot = candidate_count;
    __syncthreads();
    for (uint32_t column = threadIdx.x; column < candidate_count;
         column += blockDim.x)
      if (row[column] != 0) atomicMin(&pivot, column);
    __syncthreads();
    if (pivot == candidate_count) continue;
    const uint32_t inverse = pow_mod(row[pivot], prime - 2, prime);
    for (uint32_t column = threadIdx.x; column < candidate_count;
         column += blockDim.x) {
      const uint32_t normalized = uint64_t(row[column]) * inverse % prime;
      basis[(uint64_t(state) * kMaximumRank + rank) * kMaximumCandidates +
            column] = normalized;
    }
    if (threadIdx.x == 0) {
      pivots[state * kMaximumRank + rank] = uint16_t(pivot);
      pivot_rows[state * kMaximumRank + rank] = source_row_base + target_row;
      ++rank;
      ranks[state] = rank;
    }
    __syncthreads();
    rank = ranks[state];
    __syncthreads();
  }
}

__global__ void retain_candidate_sector(
    const uint32_t* projected, uint32_t* evaluated,
    const uint8_t* candidate_sectors, uint32_t candidate_count,
    uint32_t sector) {
  const uint64_t ordinal = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t plane = uint64_t(kTarget) * candidate_count;
  if (ordinal >= plane * kPrimesCount) return;
  const uint32_t candidate = ordinal % candidate_count;
  if (candidate_sectors[candidate] == sector) evaluated[ordinal] = projected[ordinal];
}

}  // namespace

extern "C" void* adynkra_d21_witness_create(
    const PackedDiagram* diagrams, uint64_t diagram_count,
    const uint16_t* candidate_diagrams, uint32_t candidate_count,
    const uint8_t* gamma_row, const int8_t* gamma_value,
    const uint8_t* charge_gamma_row, const int8_t* charge_gamma_value,
    const uint8_t* form_axes, const CasimirEntry* casimir_rows, char* error,
    uint64_t error_capacity) {
  if (!diagrams || diagram_count != 400 || !candidate_diagrams ||
      candidate_count == 0 || candidate_count > kMaximumCandidates ||
      !gamma_row || !gamma_value || !charge_gamma_row || !charge_gamma_value ||
      !form_axes || !casimir_rows) {
    set_error(error, error_capacity, "invalid D21 witness context input");
    return nullptr;
  }
  Context* context = new (std::nothrow) Context();
  if (!context) return nullptr;
  context->candidate_count = candidate_count;
  const size_t lookup = kMasks * kSpin;
  const size_t plane = size_t(kTarget) * candidate_count;
  const size_t states = kSectors * kPrimesCount;
#define ALLOC(field, bytes, action)                                             \
  if (!checked(cudaMalloc(&context->field, bytes), error, error_capacity,       \
               action)) {                                                       \
    destroy(context);                                                           \
    return nullptr;                                                             \
  }
  ALLOC(diagrams, diagram_count * sizeof(PackedDiagram), "allocate diagrams");
  ALLOC(candidate_diagrams, candidate_count * sizeof(uint16_t),
        "allocate candidates");
  ALLOC(gamma_row, lookup, "allocate gamma rows");
  ALLOC(gamma_value, lookup, "allocate gamma values");
  ALLOC(charge_gamma_row, lookup, "allocate charge rows");
  ALLOC(charge_gamma_value, lookup, "allocate charge values");
  ALLOC(form_axes, kForms * 4, "allocate form axes");
  ALLOC(casimir_rows, kTarget * kCasimirSupport * sizeof(CasimirEntry),
        "allocate Casimir rows");
  ALLOC(raw, plane * sizeof(int64_t), "allocate raw witness");
  ALLOC(modular_raw, 3 * plane * sizeof(uint32_t), "allocate modular raw");
  ALLOC(modular_a, 3 * plane * sizeof(uint32_t), "allocate modular A");
  ALLOC(modular_b, 3 * plane * sizeof(uint32_t), "allocate modular B");
  ALLOC(evaluated, 3 * plane * sizeof(uint32_t), "allocate evaluated witness");
  ALLOC(basis, states * kMaximumRank * kMaximumCandidates * sizeof(uint32_t),
        "allocate retained basis");
  ALLOC(ranks, states * sizeof(uint32_t), "allocate retained ranks");
  ALLOC(pivots, states * kMaximumRank * sizeof(uint16_t), "allocate pivots");
  ALLOC(pivot_rows, states * kMaximumRank * sizeof(uint64_t),
        "allocate pivot rows");
  ALLOC(overflow, sizeof(uint32_t), "allocate overflow flag");
#undef ALLOC
#define UPLOAD(field, source, bytes, action)                                    \
  if (!checked(cudaMemcpy(context->field, source, bytes, cudaMemcpyHostToDevice),\
               error, error_capacity, action)) {                                \
    destroy(context);                                                           \
    return nullptr;                                                             \
  }
  UPLOAD(diagrams, diagrams, diagram_count * sizeof(PackedDiagram),
         "upload diagrams");
  UPLOAD(candidate_diagrams, candidate_diagrams,
         candidate_count * sizeof(uint16_t), "upload candidates");
  UPLOAD(gamma_row, gamma_row, lookup, "upload gamma rows");
  UPLOAD(gamma_value, gamma_value, lookup, "upload gamma values");
  UPLOAD(charge_gamma_row, charge_gamma_row, lookup, "upload charge rows");
  UPLOAD(charge_gamma_value, charge_gamma_value, lookup,
         "upload charge values");
  UPLOAD(form_axes, form_axes, kForms * 4, "upload form axes");
  UPLOAD(casimir_rows, casimir_rows,
         kTarget * kCasimirSupport * sizeof(CasimirEntry),
         "upload Casimir rows");
#undef UPLOAD
  cudaMemset(context->basis, 0,
             states * kMaximumRank * kMaximumCandidates * sizeof(uint32_t));
  cudaMemset(context->ranks, 0, states * sizeof(uint32_t));
  cudaMemset(context->pivots, 0, states * kMaximumRank * sizeof(uint16_t));
  cudaMemset(context->pivot_rows, 0,
             states * kMaximumRank * sizeof(uint64_t));
  return context;
}

extern "C" int32_t adynkra_d21_witness_evaluate(
    void* opaque, uint8_t outer_left, uint8_t outer_right, uint8_t momentum,
    const HTerm* host_terms, uint32_t term_count,
    const uint8_t* host_candidate_sectors, uint32_t* host_evaluated,
    uint64_t host_evaluated_count, float* stage_milliseconds, char* error,
    uint64_t error_capacity) {
  Context* context = static_cast<Context*>(opaque);
  const uint64_t plane = context ? uint64_t(kTarget) * context->candidate_count : 0;
  if (!context || !host_terms || term_count == 0 || term_count > 2 ||
      outer_left >= outer_right || outer_right >= kSpin || momentum >= 11 ||
      !host_candidate_sectors || !host_evaluated ||
      host_evaluated_count != plane * kPrimesCount) {
    set_error(error, error_capacity, "invalid D21 evaluated-witness input");
    return 1;
  }
  HTerm* terms = nullptr;
  uint8_t* candidate_sectors = nullptr;
  if (!checked(cudaMalloc(&terms, term_count * sizeof(HTerm)), error,
               error_capacity, "allocate evaluated H terms") ||
      !checked(cudaMalloc(&candidate_sectors, context->candidate_count), error,
               error_capacity, "allocate candidate sectors") ||
      !checked(cudaMemcpy(terms, host_terms, term_count * sizeof(HTerm),
                          cudaMemcpyHostToDevice),
               error, error_capacity, "upload evaluated H terms") ||
      !checked(cudaMemcpy(candidate_sectors, host_candidate_sectors,
                          context->candidate_count, cudaMemcpyHostToDevice),
               error, error_capacity, "upload candidate sectors")) {
    cudaFree(terms);
    cudaFree(candidate_sectors);
    return 2;
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaMemset(context->raw, 0, plane * sizeof(int64_t));
  cudaMemset(context->evaluated, 0,
             plane * kPrimesCount * sizeof(uint32_t));
  const uint32_t emit_count = context->candidate_count * kForms;
  emit_witness<<<(emit_count + 255) / 256, 256>>>(
      context->diagrams, context->candidate_diagrams,
      context->candidate_count, terms, term_count, outer_left, outer_right,
      momentum, context->form_axes, context->gamma_row, context->gamma_value,
      context->charge_gamma_row, context->charge_gamma_value, context->raw);
  reduce_raw<<<(plane + 255) / 256, 256>>>(
      context->raw, context->candidate_count, context->modular_raw);
  const int32_t eigenvalues[kSectors] = {55, 183, 163, 135, 99};
  const uint64_t modular_count = plane * kPrimesCount;
  for (uint32_t sector = 0; sector < kSectors; ++sector) {
    cudaMemcpy(context->modular_a, context->modular_raw,
               modular_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    uint32_t* input = context->modular_a;
    uint32_t* output = context->modular_b;
    for (uint32_t other = 0; other < kSectors; ++other) {
      if (other == sector) continue;
      casimir_shift<<<(modular_count + 255) / 256, 256>>>(
          input, output, context->casimir_rows, context->candidate_count,
          eigenvalues[other]);
      uint32_t* swap = input;
      input = output;
      output = swap;
    }
    retain_candidate_sector<<<(modular_count + 255) / 256, 256>>>(
        input, context->evaluated, candidate_sectors,
        context->candidate_count, sector);
  }
  cudaEventRecord(stop);
  const cudaError_t status = cudaEventSynchronize(stop);
  if (status != cudaSuccess || cudaGetLastError() != cudaSuccess ||
      !checked(cudaMemcpy(host_evaluated, context->evaluated,
                          modular_count * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download evaluated witness")) {
    set_error(error, error_capacity, "D21 evaluated-witness CUDA stage failed");
    cudaFree(terms);
    cudaFree(candidate_sectors);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 3;
  }
  if (stage_milliseconds)
    cudaEventElapsedTime(stage_milliseconds, start, stop);
  cudaFree(terms);
  cudaFree(candidate_sectors);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

extern "C" int32_t adynkra_d21_witness_apply(
    void* opaque, uint8_t outer_left, uint8_t outer_right, uint8_t momentum,
    const HTerm* host_terms, uint32_t term_count, uint64_t source_row_base,
    const uint32_t* expected_ranks, float* stage_milliseconds, char* error,
    uint64_t error_capacity) {
  Context* context = static_cast<Context*>(opaque);
  if (!context || !host_terms || term_count == 0 || term_count > 2 ||
      outer_left >= outer_right || outer_right >= kSpin || momentum >= 11 ||
      !expected_ranks) {
    set_error(error, error_capacity, "invalid D21 witness input");
    return 1;
  }
  HTerm* terms = nullptr;
  uint32_t expected[kSectors];
  if (!checked(cudaMalloc(&terms, term_count * sizeof(HTerm)), error,
               error_capacity, "allocate H terms") ||
      !checked(cudaMemcpy(terms, host_terms, term_count * sizeof(HTerm),
                          cudaMemcpyHostToDevice),
               error, error_capacity, "upload H terms") ||
      !checked(cudaMemcpy(expected, expected_ranks, sizeof(expected),
                          cudaMemcpyHostToHost),
               error, error_capacity, "copy expected ranks")) {
    cudaFree(terms);
    return 2;
  }
  const size_t plane = size_t(kTarget) * context->candidate_count;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaMemset(context->raw, 0, plane * sizeof(int64_t));
  const uint32_t emit_count = context->candidate_count * kForms;
  emit_witness<<<(emit_count + 255) / 256, 256>>>(
      context->diagrams, context->candidate_diagrams,
      context->candidate_count, terms, term_count, outer_left, outer_right,
      momentum, context->form_axes, context->gamma_row, context->gamma_value,
      context->charge_gamma_row, context->charge_gamma_value, context->raw);
  reduce_raw<<<(plane + 255) / 256, 256>>>(
      context->raw, context->candidate_count, context->modular_raw);
  const int32_t eigenvalues[kSectors] = {55, 183, 163, 135, 99};
  const uint64_t modular_count = 3 * plane;
  for (uint32_t sector = 0; sector < kSectors; ++sector) {
    cudaMemcpy(context->modular_a, context->modular_raw,
               modular_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    uint32_t* input = context->modular_a;
    uint32_t* output = context->modular_b;
    for (uint32_t other = 0; other < kSectors; ++other) {
      if (other == sector) continue;
      casimir_shift<<<(modular_count + 255) / 256, 256>>>(
          input, output, context->casimir_rows, context->candidate_count,
          eigenvalues[other]);
      uint32_t* swap = input;
      input = output;
      output = swap;
    }
    retained_rref<<<3, 256, context->candidate_count * sizeof(uint32_t)>>>(
        input, context->candidate_count, sector, expected[sector],
        source_row_base, context->basis, context->ranks, context->pivots,
        context->pivot_rows);
  }
  cudaEventRecord(stop);
  const cudaError_t status = cudaEventSynchronize(stop);
  if (status != cudaSuccess || cudaGetLastError() != cudaSuccess) {
    set_error(error, error_capacity, "D21 witness CUDA stage failed");
    cudaFree(terms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 3;
  }
  if (stage_milliseconds)
    cudaEventElapsedTime(stage_milliseconds, start, stop);
  cudaFree(terms);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

extern "C" int32_t adynkra_d21_witness_summary(
    void* opaque, uint32_t* ranks, uint16_t* pivots, uint64_t* pivot_rows,
    char* error, uint64_t error_capacity) {
  Context* context = static_cast<Context*>(opaque);
  if (!context || !ranks || !pivots || !pivot_rows) return 1;
  const size_t states = kSectors * kPrimesCount;
  if (!checked(cudaMemcpy(ranks, context->ranks, states * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download D21 ranks") ||
      !checked(cudaMemcpy(pivots, context->pivots,
                          states * kMaximumRank * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download D21 pivots") ||
      !checked(cudaMemcpy(pivot_rows, context->pivot_rows,
                          states * kMaximumRank * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               error, error_capacity, "download D21 pivot rows")) return 2;
  return 0;
}

extern "C" void adynkra_d21_witness_destroy(void* opaque) {
  destroy(static_cast<Context*>(opaque));
}
