#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kGaugeDegrees = 6;
constexpr uint32_t kSpinors = 32;
constexpr uint32_t kMomentumPairs = 66;
constexpr uint32_t kSectors = 2;
constexpr uint32_t kSeeds = 1;
constexpr uint32_t kBuckets = 32;
constexpr uint32_t kContractionAxes = 11;
constexpr uint32_t kMaxColumns = 32;
constexpr uint32_t kX2OutputCoordinates = 55 * 11;
constexpr uint32_t kX5OutputCoordinates = 462 * 11;
constexpr size_t kDeviceHeadroomBytes = 64ULL * 1024 * 1024;
constexpr uint32_t kRows =
    kGaugeDegrees * kMomentumPairs * kSectors * kSeeds * kBuckets;
constexpr uint32_t kP3Rows = kRows * kContractionAxes;
constexpr uint32_t kP3ScheduleCount = kGaugeDegrees * kSpinors;
constexpr uint32_t kP3SpinorPairCount = kSpinors * kSpinors;
constexpr uint32_t kP3PairOffsetCount =
    kP3ScheduleCount * kP3SpinorPairCount + 1;
constexpr int kRecouplingSemanticKeyBits = 45;
__device__ __constant__ uint64_t kFunctionalSeeds[kSeeds] = {
    0x5d120f0213aa0001ULL,
};

struct GaussianResidue {
  uint32_t real;
  uint32_t imaginary;
};

struct GaugeEntry {
  uint32_t derivative_spinor;
  GaussianResidue coefficient;
};

struct TargetEntry {
  uint32_t vector_weight;
  uint32_t spinor_weight;
  uint32_t coefficient;
};

struct TemplateEntry {
  uint32_t derivative_spinor;
  uint32_t sector;
  uint32_t output_coordinate;
  GaussianResidue coefficient;
};

// Fully multiplied gauge x target x template schedule entry. Runtime work
// only applies the two mask-dependent wedge signs and source coefficient.
struct PlanEntry {
  uint32_t gauge_spinor;
  uint32_t template_spinor;
  uint32_t sector;
  uint32_t output_coordinate;
  GaussianResidue coefficient;
  uint64_t functional_salt;
};

struct P3PlanEntry {
  uint32_t contracted_spinor;
  uint32_t template_spinor;
  uint32_t contraction_axis;
  uint32_t sector;
  uint32_t output_coordinate;
  GaussianResidue coefficient;
  uint64_t functional_salt;
};

struct SourceEntry {
  uint32_t exterior_mask;
  uint32_t coefficient;
  uint32_t metadata;
};

static_assert(sizeof(GaussianResidue) == 8);
static_assert(alignof(GaussianResidue) == 4);
static_assert(sizeof(SourceEntry) == 12);
static_assert(alignof(SourceEntry) == 4);
static_assert(offsetof(SourceEntry, metadata) == 8);
static_assert(sizeof(P3PlanEntry) == 40);
static_assert(alignof(P3PlanEntry) == 8);
static_assert(offsetof(P3PlanEntry, coefficient) == 20);
static_assert(offsetof(P3PlanEntry, functional_salt) == 32);

struct SparseEntry {
  uint64_t key;
  int64_t value;
};

struct SparseLoweringStats {
  uint64_t input_count;
  uint64_t expanded_count;
  uint64_t reduced_count;
  uint64_t output_count;
  uint64_t scratch_high_water_bytes;
  uint64_t immutable_handle_bytes;
  float count_milliseconds;
  float scan_milliseconds;
  float emit_milliseconds;
  float sort_milliseconds;
  float reduce_milliseconds;
  float select_milliseconds;
  float total_milliseconds;
};

struct PersistentLoweringContext;

struct PersistentSparseHandle {
  int device = 0;
  PersistentLoweringContext *owner = nullptr;
  SparseEntry *entries = nullptr;
  uint32_t count = 0;
  uint64_t max_abs_coefficient = 0;
};

struct PersistentLoweringContext {
  int device = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t events[8]{};
  SparseEntry *sources = nullptr;
  uint32_t *counts = nullptr;
  uint32_t *offsets = nullptr;
  uint32_t source_capacity = 0;
  uint64_t *keys[2]{};
  int64_t *values[2]{};
  SparseEntry *reduced_entries = nullptr;
  SparseEntry *selected_entries = nullptr;
  uint8_t *nonzero_flags = nullptr;
  uint32_t expanded_capacity = 0;
  uint32_t *unique_count = nullptr;
  uint32_t *selected_count = nullptr;
  uint32_t *overflow = nullptr;
  unsigned long long *max_abs = nullptr;
  void *cub_temporary = nullptr;
  size_t cub_temporary_capacity = 0;
  uint64_t allocated_bytes = 0;
  uint64_t live_handle_bytes = 0;
  uint64_t live_handle_count = 0;
  uint64_t high_water_bytes = 0;
  uint64_t hard_cap_bytes = UINT64_MAX;
  bool destroy_requested = false;
};

// Signed two's-complement 128-bit value with a sticky overflow bit. Keeping
// the high and low words explicit avoids relying on host/device __int128 ABI.
struct WideValue {
  uint64_t low;
  int64_t high;
  uint32_t overflow;
  uint32_t reserved;
};

struct RecouplingStats {
  uint64_t terms_before_reduce;
  uint64_t keys_after_reduce;
  uint64_t nonzero_terms_after_reduce;
  uint64_t expanded_contributions;
  uint64_t buffer_high_water_bytes;
  float upload_milliseconds;
  float sort_milliseconds;
  float reduce_milliseconds;
  float contract_milliseconds;
  float download_milliseconds;
  float total_milliseconds;
};

struct MultiColumnStats {
  uint64_t unique_count;
  uint32_t active_columns;
  uint32_t reserved;
  uint64_t nonzero_terms[32];
  uint64_t expanded_contributions[32];
  uint64_t resident_bytes;
  uint64_t buffer_high_water_bytes;
  uint64_t device_hard_cap_bytes;
  float upload_milliseconds;
  float contract_milliseconds;
  float finalize_milliseconds;
  float download_milliseconds;
  float total_milliseconds;
};

struct Context {
  int device = 0;
  uint32_t prime = 0;
  uint32_t pow2_64_mod = 0;
  uint32_t gauge_count = 0;
  uint32_t target_count = 0;
  uint32_t template_count = 0;
  uint32_t source_capacity = 0;
  uint32_t *gauge_offsets = nullptr;
  GaugeEntry *gauges = nullptr;
  TargetEntry *targets = nullptr;
  uint32_t *template_offsets = nullptr;
  TemplateEntry *templates = nullptr;
  uint32_t *plan_offsets = nullptr;
  PlanEntry *plan_entries = nullptr;
  uint64_t *pair_salts = nullptr;
  uint32_t plan_entry_count = 0;
  uint32_t max_plan_entries_per_degree_free = 0;
  uint32_t max_plan_entries_per_free = 0;
  uint32_t *p3_plan_offsets = nullptr;
  uint32_t *p3_plan_pair_offsets = nullptr;
  P3PlanEntry *p3_plan_entries = nullptr;
  uint32_t p3_plan_entry_count = 0;
  uint32_t p3_active_columns = 0;
  uint32_t *p3_output_real = nullptr;
  uint32_t *p3_output_imaginary = nullptr;
  bool legacy_contraction = false;
  SourceEntry *sources = nullptr;
  uint32_t *output_real = nullptr;
  uint32_t *output_imaginary = nullptr;
  unsigned long long *output_real_wide = nullptr;
  unsigned long long *output_imaginary_wide = nullptr;
  unsigned long long *expanded = nullptr;
  uint32_t recoupling_capacity = 0;
  uint64_t *recoupling_keys[2] = {nullptr, nullptr};
  WideValue *recoupling_values[2] = {nullptr, nullptr};
  uint32_t *recoupling_unique_count = nullptr;
  unsigned long long *recoupling_nonzero_count = nullptr;
  uint32_t *recoupling_overflow = nullptr;
  void *cub_temporary = nullptr;
  size_t cub_temporary_capacity = 0;
  uint32_t multicol_key_capacity = 0;
  uint32_t multicol_lane_capacity = 0;
  uint64_t *multicol_keys = nullptr;
  WideValue *multicol_values = nullptr;
  unsigned long long *multicol_output_real_wide = nullptr;
  unsigned long long *multicol_output_imaginary_wide = nullptr;
  uint32_t *multicol_output_real = nullptr;
  uint32_t *multicol_output_imaginary = nullptr;
  unsigned long long *multicol_expanded = nullptr;
  uint32_t *multicol_overflow = nullptr;
  uint64_t allocated_bytes = 0;
  uint64_t buffer_high_water_bytes = 0;
  uint64_t recoupling_hard_cap_bytes = UINT64_MAX;
  cudaStream_t stream = nullptr;
  cudaEvent_t started = nullptr;
  cudaEvent_t finished = nullptr;
  cudaEvent_t stage_events[6] = {nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr};
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
  char buffer[512];
  std::snprintf(buffer, sizeof(buffer), "%s: %s", action,
                cudaGetErrorString(status));
  set_error(error, capacity, buffer);
  return false;
}

void destroy(Context *context) {
  if (context == nullptr) {
    return;
  }
  cudaSetDevice(context->device);
  if (context->stream != nullptr) {
    cudaStreamSynchronize(context->stream);
  }
  cudaFree(context->gauge_offsets);
  cudaFree(context->gauges);
  cudaFree(context->targets);
  cudaFree(context->template_offsets);
  cudaFree(context->templates);
  cudaFree(context->plan_offsets);
  cudaFree(context->plan_entries);
  cudaFree(context->p3_plan_offsets);
  cudaFree(context->p3_plan_pair_offsets);
  cudaFree(context->p3_plan_entries);
  cudaFree(context->pair_salts);
  cudaFree(context->sources);
  cudaFree(context->output_real);
  cudaFree(context->output_imaginary);
  cudaFree(context->output_real_wide);
  cudaFree(context->output_imaginary_wide);
  cudaFree(context->p3_output_real);
  cudaFree(context->p3_output_imaginary);
  cudaFree(context->expanded);
  cudaFree(context->recoupling_keys[0]);
  cudaFree(context->recoupling_keys[1]);
  cudaFree(context->recoupling_values[0]);
  cudaFree(context->recoupling_values[1]);
  cudaFree(context->recoupling_unique_count);
  cudaFree(context->recoupling_nonzero_count);
  cudaFree(context->recoupling_overflow);
  cudaFree(context->cub_temporary);
  cudaFree(context->multicol_keys);
  cudaFree(context->multicol_values);
  cudaFree(context->multicol_output_real_wide);
  cudaFree(context->multicol_output_imaginary_wide);
  cudaFree(context->multicol_output_real);
  cudaFree(context->multicol_output_imaginary);
  cudaFree(context->multicol_expanded);
  cudaFree(context->multicol_overflow);
  if (context->started != nullptr) {
    cudaEventDestroy(context->started);
  }
  if (context->finished != nullptr) {
    cudaEventDestroy(context->finished);
  }
  for (cudaEvent_t event : context->stage_events) {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }
  if (context->stream != nullptr) {
    cudaStreamDestroy(context->stream);
  }
  delete context;
}

__device__ __forceinline__ uint32_t add_mod(uint32_t left, uint32_t right,
                                             uint32_t prime) {
  uint32_t sum = left + right;
  return sum >= prime ? sum - prime : sum;
}

__device__ __forceinline__ uint32_t subtract_mod(uint32_t left,
                                                  uint32_t right,
                                                  uint32_t prime) {
  return left >= right ? left - right : prime - (right - left);
}

__device__ __forceinline__ uint32_t negate_mod(uint32_t value,
                                                uint32_t prime) {
  return value == 0 ? 0 : prime - value;
}

__device__ __forceinline__ uint32_t reduce_u64_mod(uint64_t value,
                                                    uint32_t prime) {
  // The production primes are 2^30-c for c <= 105. Two exact folds reduce
  // any u64 value below 2p, so one subtraction finishes without division.
  // Keep the generic path for supported custom primes.
  constexpr uint64_t mask = (1ULL << 30) - 1;
  if (prime < (1U << 30)) {
    const uint32_t c = (1U << 30) - prime;
    if (c <= 128) {
      uint64_t residue = (value & mask) + (value >> 30) * c;
      residue = (residue & mask) + (residue >> 30) * c;
      if (residue >= prime) {
        residue -= prime;
      }
      return static_cast<uint32_t>(residue);
    }
  }
  return static_cast<uint32_t>(value % prime);
}

__device__ __forceinline__ uint32_t multiply_mod(uint32_t left,
                                                  uint32_t right,
                                                  uint32_t prime) {
  return reduce_u64_mod(static_cast<uint64_t>(left) * right, prime);
}

__device__ __forceinline__ GaussianResidue multiply_gaussian(
    GaussianResidue left, GaussianResidue right, uint32_t prime) {
  uint32_t ac = multiply_mod(left.real, right.real, prime);
  uint32_t bd = multiply_mod(left.imaginary, right.imaginary, prime);
  uint32_t ad = multiply_mod(left.real, right.imaginary, prime);
  uint32_t bc = multiply_mod(left.imaginary, right.real, prime);
  return {subtract_mod(ac, bd, prime), add_mod(ad, bc, prime)};
}

__device__ __forceinline__ GaussianResidue scale_gaussian(
    GaussianResidue value, uint32_t scalar, uint32_t prime) {
  return {multiply_mod(value.real, scalar, prime),
          multiply_mod(value.imaginary, scalar, prime)};
}

__device__ __forceinline__ GaussianResidue negate_gaussian(
    GaussianResidue value, uint32_t prime) {
  return {negate_mod(value.real, prime), negate_mod(value.imaginary, prime)};
}

struct WideAdd {
  __host__ __device__ __forceinline__ WideValue operator()(
      const WideValue &left, const WideValue &right) const {
    WideValue output{};
    output.low = left.low + right.low;
    const uint64_t carry = output.low < left.low ? 1ULL : 0ULL;
    const uint64_t left_high = static_cast<uint64_t>(left.high);
    const uint64_t right_high = static_cast<uint64_t>(right.high);
    const uint64_t result_high = left_high + right_high + carry;
    output.high = static_cast<int64_t>(result_high);
    const bool left_negative = left.high < 0;
    const bool right_negative = right.high < 0;
    const bool result_negative = output.high < 0;
    output.overflow = left.overflow | right.overflow |
                      static_cast<uint32_t>(left_negative == right_negative &&
                                            result_negative != left_negative);
    return output;
  }
};

__device__ __forceinline__ bool wide_is_zero(WideValue value) {
  return value.low == 0 && value.high == 0;
}

__device__ __forceinline__ uint32_t unsigned_wide_mod(
    uint64_t low, uint64_t high, uint32_t prime, uint32_t pow2_64_mod) {
  const uint64_t low_mod = low % prime;
  const uint64_t high_mod = high % prime;
  return static_cast<uint32_t>(
      (low_mod + high_mod * pow2_64_mod) % prime);
}

__device__ __forceinline__ uint32_t wide_mod(WideValue value,
                                              uint32_t prime,
                                              uint32_t pow2_64_mod) {
  const bool negative = value.high < 0;
  uint64_t low = value.low;
  uint64_t high = static_cast<uint64_t>(value.high);
  if (negative) {
    low = ~low + 1ULL;
    high = ~high + (low == 0 ? 1ULL : 0ULL);
  }
  const uint32_t magnitude =
      unsigned_wide_mod(low, high, prime, pow2_64_mod);
  return negative && magnitude != 0 ? prime - magnitude : magnitude;
}

__device__ __forceinline__ uint64_t rotate_left(uint64_t value,
                                                 uint32_t amount) {
  amount &= 63;
  return amount == 0 ? value : (value << amount) | (value >> (64 - amount));
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__device__ __forceinline__ uint32_t pair_ordinal(uint32_t left,
                                                  uint32_t right) {
  return left * 11 - (left == 0 ? 0 : (left - 1) * left / 2) + right - left;
}

__device__ __forceinline__ int wedge_sign(uint32_t mask, uint32_t spinor) {
  uint32_t bit = 1U << spinor;
  if ((mask & bit) != 0) {
    return 0;
  }
  uint32_t greater = spinor == 31 ? 0 : __popc(mask >> (spinor + 1));
  return (greater & 1U) == 0 ? 1 : -1;
}

__device__ __forceinline__ int contraction_sign(uint32_t mask,
                                                uint32_t spinor) {
  const uint32_t bit = 1U << spinor;
  if ((mask & bit) == 0) {
    return 0;
  }
  const uint32_t greater =
      spinor == 31 ? 0 : __popc(mask >> (spinor + 1));
  return (greater & 1U) == 0 ? 1 : -1;
}

__device__ __forceinline__ uint32_t nth_set_bit(uint32_t mask,
                                                uint32_t ordinal) {
  while (ordinal != 0) {
    mask &= mask - 1U;
    --ordinal;
  }
  return static_cast<uint32_t>(__ffs(mask) - 1);
}

__device__ __forceinline__ uint64_t functional_base(
    uint32_t degree, uint32_t pair_left, uint32_t pair_right,
    uint32_t output_coordinate, uint32_t mask, uint32_t sector) {
  uint64_t value = rotate_left(static_cast<uint64_t>(degree), 9) ^
                   rotate_left(static_cast<uint64_t>(output_coordinate), 31) ^
                   rotate_left(static_cast<uint64_t>(mask), 43);
  value ^= 0x02d1300000000001ULL;
  value ^= sector == 0 ? 0x1100000000000002ULL : 0x1000200000000005ULL;
  for (uint32_t axis = 0; axis < 11; ++axis) {
    uint32_t exponent = (axis == pair_left ? 1U : 0U) +
                        (axis == pair_right ? 1U : 0U);
    value ^= static_cast<uint64_t>(exponent + 1) *
             rotate_left(0x9e3779b97f4a7c15ULL, axis);
  }
  return splitmix64(value);
}

__device__ __forceinline__ uint32_t row_ordinal(
    uint32_t degree, uint32_t pair, uint32_t sector, uint32_t seed,
    uint32_t bucket) {
  return ((((degree * kMomentumPairs + pair) * kSectors + sector) * kSeeds +
            seed) *
           kBuckets) +
          bucket;
}

__device__ __forceinline__ uint32_t p3_row_ordinal(
    uint32_t degree, uint32_t pair, uint32_t sector, uint32_t axis,
    uint32_t seed, uint32_t bucket) {
  return ((((((degree * kMomentumPairs + pair) * kSectors + sector) *
              kContractionAxes + axis) *
             kSeeds + seed) *
            kBuckets) +
          bucket);
}

__device__ __forceinline__ uint64_t planned_functional_base(
    const PlanEntry &entry, uint32_t functional_mask, uint32_t pair,
    const uint64_t *__restrict__ pair_salts) {
  return splitmix64(entry.functional_salt ^
                    rotate_left(static_cast<uint64_t>(functional_mask), 43) ^
                    pair_salts[pair]);
}

__device__ __forceinline__ void atomic_add_mod(uint32_t *address,
                                                uint32_t value,
                                                uint32_t prime) {
  if (value == 0) {
    return;
  }
  uint32_t observed = *address;
  while (true) {
    uint32_t desired = add_mod(observed, value, prime);
    uint32_t prior = atomicCAS(address, observed, desired);
    if (prior == observed) {
      return;
    }
    observed = prior;
  }
}

__device__ void accumulate_source(
    SourceEntry source,
    const uint32_t *__restrict__ gauge_offsets,
    const GaugeEntry *__restrict__ gauges,
    const TargetEntry *__restrict__ targets, uint32_t target_count,
    const uint32_t *__restrict__ template_offsets,
    const TemplateEntry *__restrict__ templates, uint32_t prime,
    uint32_t *__restrict__ output_real,
    uint32_t *__restrict__ output_imaginary,
    unsigned long long *__restrict__ expanded) {
  uint32_t pair_left = source.metadata & 15U;
  uint32_t pair_right = (source.metadata >> 4) & 15U;
  uint32_t free_spinor = (source.metadata >> 8) & 31U;
  uint32_t pair = pair_ordinal(pair_left, pair_right);
  GaussianResidue source_value{source.coefficient, 0};

  for (uint32_t degree = 0; degree < kGaugeDegrees; ++degree) {
    uint32_t gauge_begin = gauge_offsets[degree * kSpinors + free_spinor];
    uint32_t gauge_end = gauge_offsets[degree * kSpinors + free_spinor + 1];
    for (uint32_t gauge_index = gauge_begin; gauge_index < gauge_end;
         ++gauge_index) {
      GaugeEntry gauge = gauges[gauge_index];
      int first_sign = wedge_sign(source.exterior_mask,
                                  gauge.derivative_spinor);
      if (first_sign == 0) {
        continue;
      }
      GaussianResidue gauged = multiply_gaussian(source_value,
                                                  gauge.coefficient, prime);
      if (first_sign < 0) {
        gauged = negate_gaussian(gauged, prime);
      }
      uint32_t degree13_mask =
          source.exterior_mask | (1U << gauge.derivative_spinor);
      for (uint32_t target_index = 0; target_index < target_count;
           ++target_index) {
        TargetEntry target = targets[target_index];
        GaussianResidue targeted =
            scale_gaussian(gauged, target.coefficient, prime);
        uint32_t raw = target.vector_weight * kSpinors + target.spinor_weight;
        uint32_t template_begin = template_offsets[raw];
        uint32_t template_end = template_offsets[raw + 1];
        for (uint32_t template_index = template_begin;
             template_index < template_end; ++template_index) {
          TemplateEntry item = templates[template_index];
          int second_sign = wedge_sign(degree13_mask,
                                       item.derivative_spinor);
          if (second_sign == 0) {
            continue;
          }
          uint32_t output_mask = degree13_mask | (1U << item.derivative_spinor);
          uint32_t highest = 31U - __clz(output_mask);
          uint32_t functional_mask = output_mask ^ (1U << highest);
          GaussianResidue value =
              multiply_gaussian(targeted, item.coefficient, prime);
          if (second_sign < 0) {
            value = negate_gaussian(value, prime);
          }
          if (value.real == 0 && value.imaginary == 0) {
            continue;
          }
          uint64_t base = functional_base(
              degree, pair_left, pair_right, item.output_coordinate,
              functional_mask, item.sector);
          for (uint32_t seed = 0; seed < kSeeds; ++seed) {
            uint64_t hash = splitmix64(base ^ kFunctionalSeeds[seed]);
            uint32_t bucket = static_cast<uint32_t>(hash) % kBuckets;
            GaussianResidue contribution =
                (hash >> 63) == 0 ? value : negate_gaussian(value, prime);
            uint32_t row = row_ordinal(degree, pair, item.sector, seed, bucket);
            atomic_add_mod(&output_real[row], contribution.real, prime);
            atomic_add_mod(&output_imaginary[row], contribution.imaginary,
                           prime);
          }
          atomicAdd(expanded, 1ULL);
        }
      }
    }
  }
}

__global__ void accumulate_kernel(
    const SourceEntry *__restrict__ sources, uint32_t source_count,
    const uint32_t *__restrict__ gauge_offsets,
    const GaugeEntry *__restrict__ gauges,
    const TargetEntry *__restrict__ targets, uint32_t target_count,
    const uint32_t *__restrict__ template_offsets,
    const TemplateEntry *__restrict__ templates, uint32_t prime,
    uint32_t *__restrict__ output_real,
    uint32_t *__restrict__ output_imaginary,
    unsigned long long *__restrict__ expanded) {
  const uint32_t source_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (source_index < source_count) {
    accumulate_source(sources[source_index], gauge_offsets, gauges, targets,
                      target_count, template_offsets, templates, prime,
                      output_real, output_imaginary, expanded);
  }
}

__global__ void accumulate_p3_kernel(
    const SourceEntry *__restrict__ sources, uint32_t source_count,
    const uint32_t *__restrict__ plan_offsets,
    const P3PlanEntry *__restrict__ plan_entries,
    const uint64_t *__restrict__ pair_salts, uint32_t prime,
    uint32_t *__restrict__ output_real,
    uint32_t *__restrict__ output_imaginary,
    unsigned long long *__restrict__ expanded) {
  const uint32_t source_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (source_index >= source_count) {
    return;
  }
  const SourceEntry source = sources[source_index];
  const uint32_t pair_left = source.metadata & 15U;
  const uint32_t pair_right = (source.metadata >> 4) & 15U;
  const uint32_t free_spinor = (source.metadata >> 8) & 31U;
  const uint32_t pair = pair_ordinal(pair_left, pair_right);
  const GaussianResidue source_value{source.coefficient, 0};
  for (uint32_t degree = 0; degree < kGaugeDegrees; ++degree) {
    const uint32_t schedule = degree * kSpinors + free_spinor;
    for (uint32_t index = plan_offsets[schedule];
         index < plan_offsets[schedule + 1]; ++index) {
      const P3PlanEntry entry = plan_entries[index];
      const int first_sign =
          contraction_sign(source.exterior_mask, entry.contracted_spinor);
      if (first_sign == 0) {
        continue;
      }
      const uint32_t degree11_mask =
          source.exterior_mask ^ (1U << entry.contracted_spinor);
      const int second_sign = wedge_sign(degree11_mask, entry.template_spinor);
      if (second_sign == 0) {
        continue;
      }
      const uint32_t degree12_mask =
          degree11_mask | (1U << entry.template_spinor);
      const uint32_t highest = 31U - __clz(degree12_mask);
      const uint32_t functional_mask = degree12_mask ^ (1U << highest);
      GaussianResidue value =
          multiply_gaussian(source_value, entry.coefficient, prime);
      if ((first_sign < 0) != (second_sign < 0)) {
        value = negate_gaussian(value, prime);
      }
      if (value.real == 0 && value.imaginary == 0) {
        continue;
      }
      const uint64_t base = splitmix64(
          entry.functional_salt ^
          rotate_left(static_cast<uint64_t>(functional_mask), 43) ^
          pair_salts[pair]);
#pragma unroll
      for (uint32_t seed = 0; seed < kSeeds; ++seed) {
        const uint64_t hash = splitmix64(base ^ kFunctionalSeeds[seed]);
        const uint32_t bucket = static_cast<uint32_t>(hash) & (kBuckets - 1);
        const GaussianResidue contribution =
            (hash >> 63) == 0 ? value : negate_gaussian(value, prime);
        const uint32_t row = p3_row_ordinal(
            degree, pair, entry.sector, entry.contraction_axis, seed, bucket);
        atomic_add_mod(&output_real[row], contribution.real, prime);
        atomic_add_mod(&output_imaginary[row], contribution.imaginary, prime);
      }
      atomicAdd(expanded, 1ULL);
    }
  }
}

// One block owns one reduced semantic source key. Contracted/template offsets
// route its four warps directly through the 12 * 21 admissible spinor-pair
// spans, skipping the other 772 of 1,024 pair spans before any plan entry is
// loaded. A whole warp stripes each selected span so the 91-to-208-entry
// canonical spans do not serialize on one thread. Each surviving contribution
// is then fanned across the active output columns while retaining disjoint
// persistent rows. Coefficients arrive already reduced modulo the active prime.
__global__ void accumulate_p3_multicol_kernel(
    const uint64_t *__restrict__ keys,
    const uint32_t *__restrict__ key_major_coefficients,
    uint32_t unique_count, uint32_t active_columns,
    const uint32_t *__restrict__ plan_pair_offsets,
    const P3PlanEntry *__restrict__ plan_entries,
    const uint64_t *__restrict__ pair_salts, uint32_t prime,
    uint32_t *__restrict__ output_real,
    uint32_t *__restrict__ output_imaginary,
    unsigned long long *__restrict__ expanded,
    uint32_t *__restrict__ invalid) {
  constexpr uint32_t kThreads = 128;
  const uint32_t source_index = blockIdx.x;
  if (source_index >= unique_count) return;
  __shared__ uint32_t shared_mask;
  __shared__ uint32_t shared_free_spinor;
  __shared__ uint32_t shared_pair;
  __shared__ uint32_t shared_valid;
  __shared__ uint32_t shared_coefficients[32];
  __shared__ unsigned long long warp_expanded[4];
  constexpr uint32_t kContractedChoices = 12;
  constexpr uint32_t kTemplateChoices = kSpinors - 11;
  constexpr uint32_t kAdmissiblePairs =
      kContractedChoices * kTemplateChoices;
  __shared__ uint16_t shared_plan_pairs[kAdmissiblePairs];
  __shared__ uint64_t shared_source_salts[kAdmissiblePairs];
  __shared__ uint8_t shared_negative[kAdmissiblePairs];
  constexpr uint32_t kRowsPerDegreePair =
      kSectors * kContractionAxes * kSeeds * kBuckets;
  constexpr uint32_t kAggregatedColumns = 3;
  // A bin receives no more than the u32-bounded flat-plan entry count and
  // every residue is below the pinned 30-bit prime, so the unreduced sum is
  // strictly below 2^62 and cannot overflow u64.
  __shared__ unsigned long long
      shared_real[kAggregatedColumns * kRowsPerDegreePair];
  __shared__ unsigned long long
      shared_imaginary[kAggregatedColumns * kRowsPerDegreePair];

  if (threadIdx.x == 0) {
    shared_valid = 0;
    const uint64_t key = keys[source_index];
    const uint32_t metadata = static_cast<uint32_t>(key >> 32);
    const uint32_t pair_left = metadata & 15U;
    const uint32_t pair_right = (metadata >> 4) & 15U;
    const uint32_t free_spinor = (metadata >> 8) & 31U;
    const uint32_t mask = static_cast<uint32_t>(key);
    if ((metadata >> 13) != 0 || pair_left > pair_right || pair_right >= 11 ||
        free_spinor >= 32 || __popc(mask) != 12) {
      atomicExch(invalid, 1U);
    } else {
      shared_mask = mask;
      shared_free_spinor = free_spinor;
      shared_pair = pair_ordinal(pair_left, pair_right);
      shared_valid = 1;
    }
  }
  if (threadIdx.x < 32) {
    const uint32_t column = threadIdx.x;
    shared_coefficients[column] = column < active_columns
        ? key_major_coefficients[
              static_cast<uint64_t>(source_index) * active_columns + column]
        : 0;
  }
  __syncthreads();

  if (threadIdx.x < active_columns &&
      shared_coefficients[threadIdx.x] >= prime) {
    atomicExch(invalid, 2U);
  }
  if (shared_valid != 0) {
    for (uint32_t pair_slot = threadIdx.x; pair_slot < kAdmissiblePairs;
         pair_slot += kThreads) {
      const uint32_t contracted_ordinal = pair_slot / kTemplateChoices;
      const uint32_t template_ordinal = pair_slot % kTemplateChoices;
      const uint32_t contracted_spinor =
          nth_set_bit(shared_mask, contracted_ordinal);
      const uint32_t degree11_mask =
          shared_mask ^ (1U << contracted_spinor);
      const uint32_t template_spinor =
          nth_set_bit(~degree11_mask, template_ordinal);
      const uint32_t degree12_mask =
          degree11_mask | (1U << template_spinor);
      const uint32_t highest = 31U - __clz(degree12_mask);
      const uint32_t functional_mask = degree12_mask ^ (1U << highest);
      shared_plan_pairs[pair_slot] = static_cast<uint16_t>(
          contracted_spinor * kSpinors + template_spinor);
      shared_source_salts[pair_slot] =
          rotate_left(static_cast<uint64_t>(functional_mask), 43) ^
          pair_salts[shared_pair];
      shared_negative[pair_slot] = static_cast<uint8_t>(
          (contraction_sign(shared_mask, contracted_spinor) < 0) !=
          (wedge_sign(degree11_mask, template_spinor) < 0));
    }
  }
  __syncthreads();
  unsigned long long local_expanded = 0;
  if (shared_valid != 0) {
    for (uint32_t degree = 0; degree < kGaugeDegrees; ++degree) {
      if (active_columns <= kAggregatedColumns) {
        const uint32_t shared_count =
            active_columns * kRowsPerDegreePair;
        for (uint32_t index = threadIdx.x; index < shared_count;
             index += kThreads) {
          shared_real[index] = 0;
          shared_imaginary[index] = 0;
        }
        __syncthreads();
      }
      const uint32_t schedule = degree * kSpinors + shared_free_spinor;
      const uint32_t lane = threadIdx.x & 31U;
      const uint32_t warp = threadIdx.x >> 5;
      constexpr uint32_t kWarps = kThreads / 32;
      constexpr uint32_t kPairRounds =
          (kAdmissiblePairs + kWarps - 1) / kWarps;
      for (uint32_t pair_round = 0; pair_round < kPairRounds; ++pair_round) {
        const uint32_t pair_slot = pair_round * kWarps + warp;
        if (pair_slot >= kAdmissiblePairs) continue;
        const uint32_t pair_index = shared_plan_pairs[pair_slot];
        const uint32_t offset_index =
            schedule * kP3SpinorPairCount + pair_index;
        const uint32_t begin = plan_pair_offsets[offset_index];
        const uint32_t end = plan_pair_offsets[offset_index + 1];
        for (uint32_t index = begin + lane; index < end; index += 32) {
          const P3PlanEntry entry = plan_entries[index];
          const uint64_t base = splitmix64(
              entry.functional_salt ^ shared_source_salts[pair_slot]);
#pragma unroll
          for (uint32_t seed = 0; seed < kSeeds; ++seed) {
            const uint64_t hash = splitmix64(base ^ kFunctionalSeeds[seed]);
            const uint32_t bucket =
                static_cast<uint32_t>(hash) & (kBuckets - 1);
            const uint32_t row = p3_row_ordinal(
                degree, shared_pair, entry.sector, entry.contraction_axis,
                seed, bucket);
            const uint32_t local_row =
                (((entry.sector * kContractionAxes + entry.contraction_axis) *
                   kSeeds +
                   seed) *
                  kBuckets) +
                bucket;
            for (uint32_t column = 0; column < active_columns; ++column) {
              const uint32_t coefficient = shared_coefficients[column];
              if (coefficient == 0 || coefficient >= prime) continue;
              GaussianResidue contribution =
                  scale_gaussian(entry.coefficient, coefficient, prime);
              if ((shared_negative[pair_slot] != 0) !=
                  ((hash >> 63) != 0)) {
                contribution = negate_gaussian(contribution, prime);
              }
              if (contribution.real == 0 && contribution.imaginary == 0) {
                continue;
              }
              if (active_columns <= kAggregatedColumns) {
                const uint32_t shared_output =
                    column * kRowsPerDegreePair + local_row;
                atomicAdd(&shared_real[shared_output],
                          static_cast<unsigned long long>(contribution.real));
                atomicAdd(
                    &shared_imaginary[shared_output],
                    static_cast<unsigned long long>(contribution.imaginary));
              } else {
                const uint64_t output =
                    static_cast<uint64_t>(column) * kP3Rows + row;
                atomic_add_mod(&output_real[output], contribution.real, prime);
                atomic_add_mod(&output_imaginary[output],
                               contribution.imaginary, prime);
              }
            }
          }
          ++local_expanded;
        }
      }
      if (active_columns <= kAggregatedColumns) {
        __syncthreads();
        const uint32_t shared_count =
            active_columns * kRowsPerDegreePair;
        for (uint32_t index = threadIdx.x; index < shared_count;
             index += kThreads) {
          const uint32_t column = index / kRowsPerDegreePair;
          const uint32_t local_row = index % kRowsPerDegreePair;
          const uint32_t row =
              (degree * kMomentumPairs + shared_pair) * kRowsPerDegreePair +
              local_row;
          const uint64_t output =
              static_cast<uint64_t>(column) * kP3Rows + row;
          const uint32_t real =
              static_cast<uint32_t>(shared_real[index] % prime);
          const uint32_t imaginary =
              static_cast<uint32_t>(shared_imaginary[index] % prime);
          atomic_add_mod(&output_real[output], real, prime);
          atomic_add_mod(&output_imaginary[output], imaginary, prime);
        }
        __syncthreads();
      }
    }
  }
  for (uint32_t delta = 16; delta != 0; delta >>= 1) {
    local_expanded += __shfl_down_sync(0xffffffffU, local_expanded, delta);
  }
  const uint32_t thread_in_warp = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5;
  if (thread_in_warp == 0) {
    warp_expanded[warp] = local_expanded;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long block_expanded = 0;
#pragma unroll
    for (uint32_t warp_index = 0; warp_index < 4; ++warp_index) {
      block_expanded += warp_expanded[warp_index];
    }
    if (block_expanded != 0 && shared_valid != 0) {
      for (uint32_t column = 0; column < active_columns; ++column) {
        const uint32_t coefficient = shared_coefficients[column];
        if (coefficient != 0 && coefficient < prime) {
          atomicAdd(&expanded[column], block_expanded);
        }
      }
    }
  }
}

// Retained only as an exact canary for flat-plan parity and benchmarking.
__global__ void accumulate_reduced_legacy_kernel(
    const uint64_t *__restrict__ keys, const WideValue *__restrict__ values,
    uint32_t unique_count, const uint32_t *__restrict__ gauge_offsets,
    const GaugeEntry *__restrict__ gauges,
    const TargetEntry *__restrict__ targets, uint32_t target_count,
    const uint32_t *__restrict__ template_offsets,
    const TemplateEntry *__restrict__ templates, uint32_t prime,
    uint32_t pow2_64_mod, uint32_t *__restrict__ output_real,
    uint32_t *__restrict__ output_imaginary,
    unsigned long long *__restrict__ expanded,
    unsigned long long *__restrict__ nonzero_count,
    uint32_t *__restrict__ overflow) {
  const uint32_t source_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (source_index >= unique_count) {
    return;
  }
  const WideValue value = values[source_index];
  if (value.overflow != 0) {
    atomicExch(overflow, 1U);
    return;
  }
  if (wide_is_zero(value)) {
    return;
  }
  const uint64_t key = keys[source_index];
  const uint32_t metadata = static_cast<uint32_t>(key >> 32);
  const uint32_t pair_left = metadata & 15U;
  const uint32_t pair_right = (metadata >> 4) & 15U;
  const uint32_t free_spinor = (metadata >> 8) & 31U;
  const uint32_t mask = static_cast<uint32_t>(key);
  if ((metadata >> 13) != 0 || pair_left > pair_right || pair_right >= 11 ||
      free_spinor >= 32 || __popc(mask) != 12) {
    atomicExch(overflow, 2U);
    return;
  }
  atomicAdd(nonzero_count, 1ULL);
  const uint32_t coefficient = wide_mod(value, prime, pow2_64_mod);
  if (coefficient == 0) {
    return;
  }
  accumulate_source(SourceEntry{mask, coefficient, metadata}, gauge_offsets,
                    gauges, targets, target_count, template_offsets, templates,
                    prime, output_real, output_imaginary, expanded);
}

__global__ void accumulate_reduced_plan_kernel(
    const uint64_t *__restrict__ keys,
    const WideValue *__restrict__ values,
    uint32_t unique_count,
    const uint32_t *__restrict__ plan_offsets,
    const PlanEntry *__restrict__ plan_entries,
    const uint64_t *__restrict__ pair_salts, uint32_t prime,
    uint32_t pow2_64_mod, unsigned long long *__restrict__ output_real,
    unsigned long long *__restrict__ output_imaginary,
    unsigned long long *__restrict__ expanded,
    unsigned long long *__restrict__ nonzero_count,
    uint32_t *__restrict__ overflow) {
  constexpr uint32_t kThreads = 128;
  const uint32_t source_index = blockIdx.x;
  if (source_index >= unique_count) {
    return;
  }
  __shared__ uint32_t shared_mask;
  __shared__ uint32_t shared_free_spinor;
  __shared__ uint32_t shared_pair;
  __shared__ uint32_t shared_coefficient;
  __shared__ uint32_t shared_active;
  using CountReduce = cub::BlockReduce<unsigned long long, kThreads>;
  __shared__ typename CountReduce::TempStorage count_storage;

  if (threadIdx.x == 0) {
    shared_active = 0;
    const WideValue value = values[source_index];
    if (value.overflow != 0) {
      atomicExch(overflow, 1U);
    } else if (!wide_is_zero(value)) {
      const uint64_t key = keys[source_index];
      const uint32_t metadata = static_cast<uint32_t>(key >> 32);
      const uint32_t pair_left = metadata & 15U;
      const uint32_t pair_right = (metadata >> 4) & 15U;
      const uint32_t free_spinor = (metadata >> 8) & 31U;
      const uint32_t mask = static_cast<uint32_t>(key);
      if ((metadata >> 13) != 0 || pair_left > pair_right ||
          pair_right >= 11 || free_spinor >= 32 || __popc(mask) != 12) {
        atomicExch(overflow, 2U);
      } else {
        shared_mask = mask;
        shared_free_spinor = free_spinor;
        shared_pair = pair_ordinal(pair_left, pair_right);
        shared_coefficient = wide_mod(value, prime, pow2_64_mod);
        shared_active = 1;
        atomicAdd(nonzero_count, 1ULL);
      }
    }
  }
  __syncthreads();
  unsigned long long local_expanded = 0;
  if (shared_active != 0 && shared_coefficient != 0) {
    for (uint32_t degree = 0; degree < kGaugeDegrees; ++degree) {
      const uint32_t schedule = degree * kSpinors + shared_free_spinor;
      const uint32_t begin = plan_offsets[schedule];
      const uint32_t end = plan_offsets[schedule + 1];
      for (uint32_t index = begin + threadIdx.x; index < end;
           index += blockDim.x) {
        const PlanEntry entry = plan_entries[index];
        const int first_sign = wedge_sign(shared_mask, entry.gauge_spinor);
        if (first_sign == 0) {
          continue;
        }
        const uint32_t degree13_mask =
            shared_mask | (1U << entry.gauge_spinor);
        const int second_sign =
            wedge_sign(degree13_mask, entry.template_spinor);
        if (second_sign == 0) {
          continue;
        }
        const uint32_t output_mask =
            degree13_mask | (1U << entry.template_spinor);
        const uint32_t highest = 31U - __clz(output_mask);
        const uint32_t functional_mask = output_mask ^ (1U << highest);
        GaussianResidue contribution =
            scale_gaussian(entry.coefficient, shared_coefficient, prime);
        if ((first_sign < 0) != (second_sign < 0)) {
          contribution = negate_gaussian(contribution, prime);
        }
        if (contribution.real == 0 && contribution.imaginary == 0) {
          continue;
        }
        const uint64_t base = planned_functional_base(
            entry, functional_mask, shared_pair, pair_salts);
#pragma unroll
        for (uint32_t seed = 0; seed < kSeeds; ++seed) {
          const uint64_t hash = splitmix64(base ^ kFunctionalSeeds[seed]);
          const uint32_t bucket = static_cast<uint32_t>(hash) & (kBuckets - 1);
          const GaussianResidue value =
              (hash >> 63) == 0 ? contribution
                                : negate_gaussian(contribution, prime);
          const uint32_t row = row_ordinal(degree, shared_pair, entry.sector,
                                           seed, bucket);
          atomicAdd(&output_real[row], static_cast<unsigned long long>(value.real));
          atomicAdd(&output_imaginary[row],
                    static_cast<unsigned long long>(value.imaginary));
        }
        ++local_expanded;
      }
    }
  }
  const unsigned long long block_expanded =
      CountReduce(count_storage).Sum(local_expanded);
  if (threadIdx.x == 0 && block_expanded != 0) {
    atomicAdd(expanded, block_expanded);
  }
}

// Reference direct-to-global implementation retained for parity profiling.
__global__ void accumulate_reduced_plan_multicol_direct_kernel(
    const uint64_t *__restrict__ keys,
    const WideValue *__restrict__ key_major_values, uint32_t unique_count,
    uint32_t active_columns, const uint32_t *__restrict__ plan_offsets,
    const PlanEntry *__restrict__ plan_entries,
    const uint64_t *__restrict__ pair_salts, uint32_t prime,
    uint32_t pow2_64_mod, unsigned long long *__restrict__ output_real,
    unsigned long long *__restrict__ output_imaginary,
    unsigned long long *__restrict__ expanded,
    uint32_t *__restrict__ overflow) {
  constexpr uint32_t kThreads = 128;
  const uint32_t source_index = blockIdx.x;
  if (source_index >= unique_count) return;
  const uint32_t lane_width =
      active_columns <= 1  ? 1
      : active_columns <= 2  ? 2
      : active_columns <= 4  ? 4
      : active_columns <= 8  ? 8
      : active_columns <= 16 ? 16
                             : 32;
  const uint32_t lane = threadIdx.x & (lane_width - 1U);
  const uint32_t plan_lane = threadIdx.x / lane_width;
  const uint32_t plan_lanes = kThreads / lane_width;
  __shared__ uint32_t shared_mask;
  __shared__ uint32_t shared_free_spinor;
  __shared__ uint32_t shared_pair;
  __shared__ uint32_t shared_valid;
  __shared__ uint32_t shared_coefficients[32];
  __shared__ uint32_t shared_active[32];
  __shared__ unsigned long long warp_expanded[4][32];

  if (threadIdx.x == 0) {
    shared_valid = 0;
    const uint64_t key = keys[source_index];
    const uint32_t metadata = static_cast<uint32_t>(key >> 32);
    const uint32_t pair_left = metadata & 15U;
    const uint32_t pair_right = (metadata >> 4) & 15U;
    const uint32_t free_spinor = (metadata >> 8) & 31U;
    const uint32_t mask = static_cast<uint32_t>(key);
    if ((metadata >> 13) != 0 || pair_left > pair_right || pair_right >= 11 ||
        free_spinor >= 32 || __popc(mask) != 12) {
      atomicExch(overflow, 2U);
    } else {
      shared_mask = mask;
      shared_free_spinor = free_spinor;
      shared_pair = pair_ordinal(pair_left, pair_right);
      shared_valid = 1;
    }
  }
  if (threadIdx.x < 32) {
    const uint32_t column = threadIdx.x;
    shared_active[column] = 0;
    shared_coefficients[column] = 0;
    if (column < active_columns) {
      const WideValue value = key_major_values[
          static_cast<uint64_t>(source_index) * active_columns + column];
      if (value.overflow != 0 || value.reserved != 0) {
        atomicExch(overflow, 1U);
      } else if (!wide_is_zero(value)) {
        shared_coefficients[column] = wide_mod(value, prime, pow2_64_mod);
        shared_active[column] = 1;
      }
    }
  }
  __syncthreads();
  const bool lane_active = shared_valid != 0 && lane < active_columns &&
                           shared_active[lane] != 0 &&
                           shared_coefficients[lane] != 0;
  unsigned long long local_expanded = 0;
  if (lane_active) {
    for (uint32_t degree = 0; degree < kGaugeDegrees; ++degree) {
      const uint32_t schedule = degree * kSpinors + shared_free_spinor;
      const uint32_t begin = plan_offsets[schedule];
      const uint32_t end = plan_offsets[schedule + 1];
      for (uint32_t index = begin + plan_lane; index < end;
           index += plan_lanes) {
        const PlanEntry entry = plan_entries[index];
        const int first_sign = wedge_sign(shared_mask, entry.gauge_spinor);
        if (first_sign == 0) continue;
        const uint32_t degree13_mask =
            shared_mask | (1U << entry.gauge_spinor);
        const int second_sign =
            wedge_sign(degree13_mask, entry.template_spinor);
        if (second_sign == 0) continue;
        const uint32_t output_mask =
            degree13_mask | (1U << entry.template_spinor);
        const uint32_t highest = 31U - __clz(output_mask);
        const uint32_t functional_mask = output_mask ^ (1U << highest);
        GaussianResidue contribution = scale_gaussian(
            entry.coefficient, shared_coefficients[lane], prime);
        if ((first_sign < 0) != (second_sign < 0)) {
          contribution = negate_gaussian(contribution, prime);
        }
        if (contribution.real == 0 && contribution.imaginary == 0) continue;
        const uint64_t base = planned_functional_base(
            entry, functional_mask, shared_pair, pair_salts);
#pragma unroll
        for (uint32_t seed = 0; seed < kSeeds; ++seed) {
          const uint64_t hash = splitmix64(base ^ kFunctionalSeeds[seed]);
          const uint32_t bucket =
              static_cast<uint32_t>(hash) & (kBuckets - 1);
          const GaussianResidue value =
              (hash >> 63) == 0 ? contribution
                                : negate_gaussian(contribution, prime);
          const uint32_t row = row_ordinal(degree, shared_pair, entry.sector,
                                           seed, bucket);
          const uint64_t output =
              static_cast<uint64_t>(row) * active_columns + lane;
          atomicAdd(&output_real[output],
                    static_cast<unsigned long long>(value.real));
          atomicAdd(&output_imaginary[output],
                    static_cast<unsigned long long>(value.imaginary));
        }
        ++local_expanded;
      }
    }
  }
  for (uint32_t delta = 16; delta >= lane_width; delta >>= 1) {
    local_expanded += __shfl_down_sync(0xffffffffU, local_expanded, delta);
    if (delta == lane_width) break;
  }
  const uint32_t thread_in_warp = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5;
  if (thread_in_warp < lane_width) {
    warp_expanded[warp][thread_in_warp] = local_expanded;
  }
  __syncthreads();
  if (threadIdx.x < active_columns) {
    unsigned long long block_expanded = 0;
#pragma unroll
    for (uint32_t warp_index = 0; warp_index < 4; ++warp_index) {
      block_expanded += warp_expanded[warp_index][threadIdx.x];
    }
    if (block_expanded != 0) {
      atomicAdd(&expanded[threadIdx.x], block_expanded);
    }
  }
}


// One block owns one (source, degree) pair. The plan response depends only
// on the semantic source key, not on its per-column scalar coefficients. Build
// that 64-bin Gaussian response once, then scale and flush it for every active
// column. This preserves exact finite-field linearity while eliminating both
// per-entry coefficient multiplies and duplicated per-column plan traversal.
__global__ void accumulate_reduced_plan_multicol_shared_kernel(
    const uint64_t *__restrict__ keys,
    const WideValue *__restrict__ key_major_values, uint32_t unique_count,
    uint32_t active_columns, const uint32_t *__restrict__ plan_offsets,
    const PlanEntry *__restrict__ plan_entries,
    const uint64_t *__restrict__ pair_salts, uint32_t prime,
    uint32_t pow2_64_mod, unsigned long long *__restrict__ output_real,
    unsigned long long *__restrict__ output_imaginary,
    unsigned long long *__restrict__ expanded,
    uint32_t *__restrict__ overflow) {
  constexpr uint32_t kThreads = 128;
  constexpr uint32_t kRowsPerDegreePair = kSectors * kSeeds * kBuckets;
  const uint32_t source_index = blockIdx.x;
  const uint32_t degree = blockIdx.y;
  if (source_index >= unique_count || degree >= kGaugeDegrees) return;
  extern __shared__ unsigned long long shared_rows[];
  unsigned long long *shared_real = shared_rows;
  unsigned long long *shared_imaginary =
      shared_rows + kRowsPerDegreePair;
  for (uint32_t index = threadIdx.x; index < kRowsPerDegreePair;
       index += blockDim.x) {
    shared_real[index] = 0;
    shared_imaginary[index] = 0;
  }
  __shared__ uint32_t shared_mask;
  __shared__ uint32_t shared_free_spinor;
  __shared__ uint32_t shared_pair;
  __shared__ uint32_t shared_valid;
  __shared__ uint32_t shared_coefficients[32];
  __shared__ uint32_t shared_active[32];
  __shared__ unsigned long long shared_expanded;
  using CountReduce = cub::BlockReduce<unsigned long long, kThreads>;
  __shared__ typename CountReduce::TempStorage count_storage;

  if (threadIdx.x == 0) {
    shared_valid = 0;
    const uint64_t key = keys[source_index];
    const uint32_t metadata = static_cast<uint32_t>(key >> 32);
    const uint32_t pair_left = metadata & 15U;
    const uint32_t pair_right = (metadata >> 4) & 15U;
    const uint32_t free_spinor = (metadata >> 8) & 31U;
    const uint32_t mask = static_cast<uint32_t>(key);
    if ((metadata >> 13) != 0 || pair_left > pair_right || pair_right >= 11 ||
        free_spinor >= 32 || __popc(mask) != 12) {
      atomicExch(overflow, 2U);
    } else {
      shared_mask = mask;
      shared_free_spinor = free_spinor;
      shared_pair = pair_ordinal(pair_left, pair_right);
      shared_valid = 1;
    }
  }
  if (threadIdx.x < 32) {
    const uint32_t column = threadIdx.x;
    shared_active[column] = 0;
    shared_coefficients[column] = 0;
    if (column < active_columns) {
      const WideValue value = key_major_values[
          static_cast<uint64_t>(source_index) * active_columns + column];
      if (value.overflow != 0 || value.reserved != 0) {
        atomicExch(overflow, 1U);
      } else if (!wide_is_zero(value)) {
        shared_coefficients[column] = wide_mod(value, prime, pow2_64_mod);
        shared_active[column] = 1;
      }
    }
  }
  __syncthreads();
  unsigned long long local_expanded = 0;
  if (shared_valid != 0) {
    const uint32_t schedule = degree * kSpinors + shared_free_spinor;
    const uint32_t begin = plan_offsets[schedule];
    const uint32_t end = plan_offsets[schedule + 1];
    for (uint32_t index = begin + threadIdx.x; index < end;
         index += blockDim.x) {
      const PlanEntry entry = plan_entries[index];
      const int first_sign = wedge_sign(shared_mask, entry.gauge_spinor);
      if (first_sign == 0) continue;
      const uint32_t degree13_mask =
          shared_mask | (1U << entry.gauge_spinor);
      const int second_sign =
          wedge_sign(degree13_mask, entry.template_spinor);
      if (second_sign == 0) continue;
      const uint32_t output_mask =
          degree13_mask | (1U << entry.template_spinor);
      const uint32_t highest = 31U - __clz(output_mask);
      const uint32_t functional_mask = output_mask ^ (1U << highest);
      const uint64_t base = planned_functional_base(
          entry, functional_mask, shared_pair, pair_salts);
#pragma unroll
      for (uint32_t seed = 0; seed < kSeeds; ++seed) {
        const uint64_t hash = splitmix64(base ^ kFunctionalSeeds[seed]);
        GaussianResidue coefficient = entry.coefficient;
        if (((first_sign < 0) != (second_sign < 0)) !=
            ((hash >> 63) != 0)) {
          coefficient = negate_gaussian(coefficient, prime);
        }
        const uint32_t bucket =
            static_cast<uint32_t>(hash) & (kBuckets - 1);
        const uint32_t local_row =
            (entry.sector * kSeeds + seed) * kBuckets + bucket;
        atomicAdd(&shared_real[local_row],
                  static_cast<unsigned long long>(coefficient.real));
        atomicAdd(&shared_imaginary[local_row],
                  static_cast<unsigned long long>(coefficient.imaginary));
      }
      ++local_expanded;
    }
  }
  const unsigned long long block_expanded =
      CountReduce(count_storage).Sum(local_expanded);
  if (threadIdx.x == 0) shared_expanded = block_expanded;
  __syncthreads();
  const uint32_t output_values = kRowsPerDegreePair * active_columns;
  for (uint32_t index = threadIdx.x; index < output_values;
       index += blockDim.x) {
    const uint32_t local_row = index / active_columns;
    const uint32_t column = index % active_columns;
    if (shared_active[column] != 0 && shared_coefficients[column] != 0) {
      const GaussianResidue response{
          reduce_u64_mod(shared_real[local_row], prime),
          reduce_u64_mod(shared_imaginary[local_row], prime)};
      const GaussianResidue value =
          scale_gaussian(response, shared_coefficients[column], prime);
      if (value.real != 0 || value.imaginary != 0) {
        const uint32_t sector = local_row / (kSeeds * kBuckets);
        const uint32_t seed_bucket = local_row % (kSeeds * kBuckets);
        const uint32_t seed = seed_bucket / kBuckets;
        const uint32_t bucket = seed_bucket % kBuckets;
        const uint32_t row =
            row_ordinal(degree, shared_pair, sector, seed, bucket);
        const uint64_t output =
            static_cast<uint64_t>(row) * active_columns + column;
        if (value.real != 0) {
          atomicAdd(&output_real[output],
                    static_cast<unsigned long long>(value.real));
        }
        if (value.imaginary != 0) {
          atomicAdd(&output_imaginary[output],
                    static_cast<unsigned long long>(value.imaginary));
        }
      }
    }
  }
  if (threadIdx.x < active_columns && shared_active[threadIdx.x] != 0 &&
      shared_coefficients[threadIdx.x] != 0 && shared_expanded != 0) {
    atomicAdd(&expanded[threadIdx.x], shared_expanded);
  }
}

__global__ void finalize_wide_rows_kernel(
    const unsigned long long *__restrict__ real_wide,
    const unsigned long long *__restrict__ imaginary_wide, uint32_t prime,
    uint32_t *__restrict__ real, uint32_t *__restrict__ imaginary) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < kRows) {
    real[row] = static_cast<uint32_t>(real_wide[row] % prime);
    imaginary[row] = static_cast<uint32_t>(imaginary_wide[row] % prime);
  }
}

__global__ void finalize_wide_multicol_kernel(
    const unsigned long long *__restrict__ real_wide,
    const unsigned long long *__restrict__ imaginary_wide, uint64_t count,
    uint32_t prime, uint32_t *__restrict__ real,
    uint32_t *__restrict__ imaginary) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    real[index] = static_cast<uint32_t>(real_wide[index] % prime);
    imaginary[index] = static_cast<uint32_t>(imaginary_wide[index] % prime);
  }
}

__device__ __forceinline__ int lowered_spinor(uint32_t index,
                                               uint32_t root) {
  if (root < 4) {
    uint32_t left = 1U << (4 - root);
    uint32_t right = 1U << (3 - root);
    if ((index & left) != 0 || (index & right) == 0) {
      return -1;
    }
    return static_cast<int>(index ^ left ^ right);
  }
  return (index & 1U) == 0 ? static_cast<int>(index | 1U) : -1;
}

__device__ __forceinline__ int64_t replacement_sign(uint32_t mask,
                                                     uint32_t first,
                                                     uint32_t second) {
  uint32_t low = min(first, second);
  uint32_t high = max(first, second);
  uint32_t interval = high == low + 1
                          ? 0
                          : ((1U << high) - 1) ^ ((1U << (low + 1)) - 1);
  return (__popc(mask & interval) & 1U) == 0 ? 1 : -1;
}

__global__ void expand_sparse_lowering_kernel(
    const uint64_t *__restrict__ source_keys,
    const int64_t *__restrict__ source_values, uint32_t source_count,
    uint32_t root, uint64_t *__restrict__ output_keys,
    int64_t *__restrict__ output_values) {
  uint32_t source_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (source_index >= source_count) {
    return;
  }
  constexpr uint32_t stride = 13;
  uint32_t output = source_index * stride;
  for (uint32_t slot = 0; slot < stride; ++slot) {
    output_keys[output + slot] = UINT64_MAX;
    output_values[output + slot] = 0;
  }
  uint64_t key = source_keys[source_index];
  uint32_t free_spinor = static_cast<uint32_t>(key >> 32);
  uint32_t mask = static_cast<uint32_t>(key);
  int64_t coefficient = source_values[source_index];
  uint32_t slot = 0;
  int lowered_free = lowered_spinor(free_spinor, root);
  if (lowered_free >= 0) {
    output_keys[output + slot] =
        (static_cast<uint64_t>(lowered_free) << 32) | mask;
    output_values[output + slot] = coefficient;
    ++slot;
  }
  uint32_t occupied = mask;
  while (occupied != 0) {
    uint32_t upper = __ffs(occupied) - 1;
    occupied &= occupied - 1;
    int lower = lowered_spinor(upper, root);
    if (lower < 0 || (mask & (1U << lower)) != 0) {
      continue;
    }
    uint32_t output_mask = mask ^ (1U << upper) ^ (1U << lower);
    output_keys[output + slot] =
        (static_cast<uint64_t>(free_spinor) << 32) | output_mask;
    output_values[output + slot] =
        coefficient * replacement_sign(mask, upper, static_cast<uint32_t>(lower));
    ++slot;
  }
}

__device__ __forceinline__ uint32_t sparse_lowering_count(SparseEntry source,
                                                           uint32_t root) {
  const uint32_t free_spinor = static_cast<uint32_t>(source.key >> 32);
  const uint32_t mask = static_cast<uint32_t>(source.key);
  uint32_t count = lowered_spinor(free_spinor, root) >= 0 ? 1U : 0U;
  uint32_t occupied = mask;
  while (occupied != 0) {
    const uint32_t upper = __ffs(occupied) - 1;
    occupied &= occupied - 1;
    const int lower = lowered_spinor(upper, root);
    count += lower >= 0 && (mask & (1U << lower)) == 0 ? 1U : 0U;
  }
  return count;
}

__global__ void count_sparse_lowering_kernel(
    const SparseEntry *__restrict__ sources, uint32_t source_count,
    uint32_t root, uint32_t *__restrict__ counts) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < source_count) {
    counts[index] = sparse_lowering_count(sources[index], root);
  }
}

__global__ void finish_sparse_offsets_kernel(
    const uint32_t *__restrict__ counts, uint32_t source_count,
    uint32_t *__restrict__ offsets, uint32_t *__restrict__ overflow) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const uint64_t total = static_cast<uint64_t>(offsets[source_count - 1]) +
                           counts[source_count - 1];
    if (total > UINT32_MAX) {
      *overflow = 1;
    } else {
      offsets[source_count] = static_cast<uint32_t>(total);
    }
  }
}

__global__ void emit_sparse_lowering_kernel(
    const SparseEntry *__restrict__ sources, uint32_t source_count,
    uint32_t root, const uint32_t *__restrict__ offsets,
    uint64_t *__restrict__ output_keys, int64_t *__restrict__ output_values) {
  const uint32_t source_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (source_index >= source_count) {
    return;
  }
  const SparseEntry source = sources[source_index];
  const uint32_t free_spinor = static_cast<uint32_t>(source.key >> 32);
  const uint32_t mask = static_cast<uint32_t>(source.key);
  uint32_t output = offsets[source_index];
  const int lowered_free = lowered_spinor(free_spinor, root);
  if (lowered_free >= 0) {
    output_keys[output] =
        (static_cast<uint64_t>(lowered_free) << 32) | mask;
    output_values[output] = source.value;
    ++output;
  }
  uint32_t occupied = mask;
  while (occupied != 0) {
    const uint32_t upper = __ffs(occupied) - 1;
    occupied &= occupied - 1;
    const int lower = lowered_spinor(upper, root);
    if (lower < 0 || (mask & (1U << lower)) != 0) {
      continue;
    }
    const uint32_t output_mask =
        mask ^ (1U << upper) ^ (1U << static_cast<uint32_t>(lower));
    output_keys[output] =
        (static_cast<uint64_t>(free_spinor) << 32) | output_mask;
    output_values[output] =
        source.value * replacement_sign(mask, upper, lower);
    ++output;
  }
}

__global__ void pack_sparse_nonzero_kernel(
    const uint64_t *__restrict__ keys, const int64_t *__restrict__ values,
    uint32_t count, SparseEntry *__restrict__ entries,
    uint8_t *__restrict__ flags,
    unsigned long long *__restrict__ max_abs) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    entries[index] = SparseEntry{keys[index], values[index]};
    flags[index] = values[index] != 0 ? 1U : 0U;
    if (values[index] != 0) {
      const unsigned long long magnitude =
          values[index] < 0 ? static_cast<unsigned long long>(-values[index])
                            : static_cast<unsigned long long>(values[index]);
      atomicMax(max_abs, magnitude);
    }
  }
}

bool checked_multiply_size(size_t left, size_t right, size_t *output) {
  if (right != 0 && left > SIZE_MAX / right) {
    return false;
  }
  *output = left * right;
  return true;
}

bool checked_product_u64(uint64_t left, uint64_t middle, uint64_t right,
                         uint64_t *output) {
  if (middle != 0 && left > UINT64_MAX / middle) {
    return false;
  }
  const uint64_t partial = left * middle;
  if (right != 0 && partial > UINT64_MAX / right) {
    return false;
  }
  *output = partial * right;
  return true;
}

uint64_t rotate_left_host(uint64_t value, uint32_t amount) {
  amount &= 63;
  return amount == 0 ? value : (value << amount) | (value >> (64 - amount));
}

uint64_t canonical_functional_salt(uint32_t degree,
                                   uint32_t output_coordinate,
                                   uint32_t sector) {
  return rotate_left_host(degree, 9) ^
         rotate_left_host(output_coordinate, 31) ^ 0x02d1300000000001ULL ^
         (sector == 0 ? 0x1100000000000002ULL : 0x1000200000000005ULL);
}

uint64_t canonical_p3_functional_salt(uint32_t degree,
                                      uint32_t output_coordinate,
                                      uint32_t sector, uint32_t axis) {
  return rotate_left_host(degree, 9) ^
         rotate_left_host(output_coordinate, 31) ^
         0x03d1100000000002ULL ^ rotate_left_host(axis, 53) ^
         (sector == 0 ? 0x1100000000000002ULL : 0x1000200000000005ULL);
}

uint64_t canonical_pair_salt(uint32_t left, uint32_t right) {
  uint64_t salt = 0;
  for (uint32_t axis = 0; axis < 11; ++axis) {
    const uint32_t exponent = (axis == left ? 1U : 0U) +
                              (axis == right ? 1U : 0U);
    salt ^= static_cast<uint64_t>(exponent + 1) *
            rotate_left_host(0x9e3779b97f4a7c15ULL, axis);
  }
  return salt;
}

bool ensure_source_capacity(Context *context, uint32_t source_count,
                            char *error, size_t error_capacity) {
  if (source_count <= context->source_capacity) {
    return true;
  }
  size_t new_bytes = 0;
  size_t old_bytes = 0;
  if (!checked_multiply_size(source_count, sizeof(SourceEntry), &new_bytes) ||
      !checked_multiply_size(context->source_capacity, sizeof(SourceEntry),
                             &old_bytes) ||
      old_bytes > context->allocated_bytes ||
      new_bytes > UINT64_MAX - context->allocated_bytes) {
    set_error(error, error_capacity, "CUDA source buffer byte count overflow");
    return false;
  }
  const uint64_t transient_peak = context->allocated_bytes + new_bytes;
  const uint64_t new_resident =
      context->allocated_bytes - old_bytes + new_bytes;
  if (transient_peak > context->recoupling_hard_cap_bytes ||
      new_resident > context->recoupling_hard_cap_bytes) {
    set_error(error, error_capacity,
              "CUDA source buffer exceeds configured device cap");
    return false;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (!check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), error,
                  error_capacity, "query CUDA source buffer memory") ||
      new_bytes > free_bytes || free_bytes - new_bytes < kDeviceHeadroomBytes) {
    if (new_bytes > free_bytes || free_bytes - new_bytes < kDeviceHeadroomBytes) {
      set_error(error, error_capacity,
                "insufficient CUDA memory for source buffer");
    }
    return false;
  }
  SourceEntry *replacement = nullptr;
  if (!check_cuda(cudaMalloc(&replacement, new_bytes), error, error_capacity,
                  "allocate CUDA source buffer")) {
    return false;
  }
  SourceEntry *old = context->sources;
  if (!check_cuda(cudaFree(old), error, error_capacity,
                  "release previous CUDA source buffer")) {
    cudaFree(replacement);
    return false;
  }
  context->sources = replacement;
  context->source_capacity = source_count;
  context->allocated_bytes = new_resident;
  context->buffer_high_water_bytes =
      max(context->buffer_high_water_bytes, transient_peak);
  return true;
}

bool ensure_recoupling_workspace(Context *context, uint32_t count,
                                 char *error, size_t error_capacity) {
  if (count <= context->recoupling_capacity) {
    return true;
  }
  uint32_t capacity = 1;
  while (capacity < count && capacity <= UINT32_MAX / 2) {
    capacity *= 2;
  }
  if (capacity < count) {
    capacity = count;
  }
  size_t key_bytes = 0;
  size_t value_bytes = 0;
  if (!checked_multiply_size(capacity, sizeof(uint64_t), &key_bytes) ||
      !checked_multiply_size(capacity, sizeof(WideValue), &value_bytes)) {
    set_error(error, error_capacity, "recoupling workspace size overflow");
    return false;
  }
  const size_t array_bytes = 2 * key_bytes + 2 * value_bytes;
  size_t sort_bytes = 0;
  size_t reduce_bytes = 0;
  if (!check_cuda(cub::DeviceRadixSort::SortPairs(
                      nullptr, sort_bytes, context->recoupling_keys[0],
                      context->recoupling_keys[1],
                      context->recoupling_values[0],
                      context->recoupling_values[1], capacity, 0,
                      kRecouplingSemanticKeyBits,
                      context->stream),
                  error, error_capacity, "size recoupling radix sort") ||
      !check_cuda(cub::DeviceReduce::ReduceByKey(
                      nullptr, reduce_bytes, context->recoupling_keys[1],
                      context->recoupling_keys[0],
                      context->recoupling_values[1],
                      context->recoupling_values[0],
                      context->recoupling_unique_count, WideAdd(), capacity,
                      context->stream),
                  error, error_capacity, "size recoupling reduce-by-key")) {
    return false;
  }
  const size_t temporary_bytes = max(sort_bytes, reduce_bytes);
  const bool replace_temporary =
      temporary_bytes > context->cub_temporary_capacity;
  const size_t additional_temporary =
      replace_temporary ? temporary_bytes : 0;
  if (array_bytes > SIZE_MAX - additional_temporary) {
    set_error(error, error_capacity, "recoupling peak size overflow");
    return false;
  }
  const size_t growth_peak_bytes = array_bytes + additional_temporary;
  const uint64_t transient_peak_bytes =
      context->allocated_bytes + static_cast<uint64_t>(growth_peak_bytes);
  if (transient_peak_bytes > context->recoupling_hard_cap_bytes) {
    set_error(error, error_capacity,
              "recoupling workspace exceeds configured device hard cap");
    return false;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (!check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), error,
                  error_capacity, "query recoupling device memory") ||
      growth_peak_bytes > free_bytes ||
      free_bytes - growth_peak_bytes < kDeviceHeadroomBytes) {
    if (growth_peak_bytes > free_bytes ||
        free_bytes - growth_peak_bytes < kDeviceHeadroomBytes) {
      set_error(error, error_capacity,
                "insufficient CUDA memory for recoupling buffers");
    }
    return false;
  }

  uint64_t *new_keys[2] = {nullptr, nullptr};
  WideValue *new_values[2] = {nullptr, nullptr};
  if (!check_cuda(cudaMalloc(&new_keys[0], key_bytes), error, error_capacity,
                  "allocate recoupling keys 0") ||
      !check_cuda(cudaMalloc(&new_keys[1], key_bytes), error, error_capacity,
                  "allocate recoupling keys 1") ||
      !check_cuda(cudaMalloc(&new_values[0], value_bytes), error,
                  error_capacity, "allocate recoupling values 0") ||
      !check_cuda(cudaMalloc(&new_values[1], value_bytes), error,
                  error_capacity, "allocate recoupling values 1")) {
    cudaFree(new_keys[0]);
    cudaFree(new_keys[1]);
    cudaFree(new_values[0]);
    cudaFree(new_values[1]);
    return false;
  }

  void *new_temporary = context->cub_temporary;
  if (replace_temporary &&
      !check_cuda(cudaMalloc(&new_temporary, temporary_bytes), error,
                  error_capacity, "allocate reusable CUB workspace")) {
    cudaFree(new_keys[0]);
    cudaFree(new_keys[1]);
    cudaFree(new_values[0]);
    cudaFree(new_values[1]);
    return false;
  }

  const uint64_t old_array_bytes =
      static_cast<uint64_t>(context->recoupling_capacity) *
      (2 * sizeof(uint64_t) + 2 * sizeof(WideValue));
  cudaFree(context->recoupling_keys[0]);
  cudaFree(context->recoupling_keys[1]);
  cudaFree(context->recoupling_values[0]);
  cudaFree(context->recoupling_values[1]);
  context->recoupling_keys[0] = new_keys[0];
  context->recoupling_keys[1] = new_keys[1];
  context->recoupling_values[0] = new_values[0];
  context->recoupling_values[1] = new_values[1];
  context->recoupling_capacity = capacity;
  context->allocated_bytes -= old_array_bytes;
  context->allocated_bytes += array_bytes;
  if (replace_temporary) {
    cudaFree(context->cub_temporary);
    context->allocated_bytes -= context->cub_temporary_capacity;
    context->cub_temporary = new_temporary;
    context->cub_temporary_capacity = temporary_bytes;
    context->allocated_bytes += temporary_bytes;
  }
  context->buffer_high_water_bytes =
      max(context->buffer_high_water_bytes, transient_peak_bytes);
  context->buffer_high_water_bytes =
      max(context->buffer_high_water_bytes, context->allocated_bytes);
  return true;
}

uint32_t growth_capacity(uint32_t required) {
  uint32_t capacity = 1;
  while (capacity < required && capacity <= UINT32_MAX / 2) {
    capacity *= 2;
  }
  return capacity < required ? required : capacity;
}

bool multicol_workspace_bytes(uint32_t key_capacity, uint32_t lane_capacity,
                              size_t *bytes) {
  if (key_capacity == 0 || lane_capacity == 0 || lane_capacity > 32) {
    *bytes = 0;
    return key_capacity == 0 && lane_capacity == 0;
  }
  size_t value_count = 0;
  size_t row_count = 0;
  size_t key_bytes = 0;
  size_t value_bytes = 0;
  size_t wide_row_bytes = 0;
  size_t row_bytes = 0;
  if (!checked_multiply_size(key_capacity, lane_capacity, &value_count) ||
      !checked_multiply_size(kRows, lane_capacity, &row_count) ||
      !checked_multiply_size(key_capacity, sizeof(uint64_t), &key_bytes) ||
      !checked_multiply_size(value_count, sizeof(WideValue), &value_bytes) ||
      !checked_multiply_size(row_count, 2 * sizeof(unsigned long long),
                             &wide_row_bytes) ||
      !checked_multiply_size(row_count, 2 * sizeof(uint32_t), &row_bytes)) {
    return false;
  }
  if (key_bytes > SIZE_MAX - value_bytes ||
      key_bytes + value_bytes > SIZE_MAX - wide_row_bytes ||
      key_bytes + value_bytes + wide_row_bytes > SIZE_MAX - row_bytes ||
      key_bytes + value_bytes + wide_row_bytes + row_bytes >
          SIZE_MAX - 32 * sizeof(unsigned long long) - sizeof(uint32_t)) {
    return false;
  }
  *bytes = key_bytes + value_bytes + wide_row_bytes + row_bytes +
           32 * sizeof(unsigned long long) + sizeof(uint32_t);
  return true;
}

bool ensure_multicol_workspace(Context *context, uint32_t unique_capacity,
                               uint32_t active_columns, char *error,
                               size_t error_capacity) {
  if (unique_capacity == 0 || active_columns == 0 || active_columns > 32) {
    set_error(error, error_capacity,
              "invalid multi-column CUDA reservation dimensions");
    return false;
  }
  if (unique_capacity <= context->multicol_key_capacity &&
      active_columns <= context->multicol_lane_capacity) {
    return true;
  }
  const uint32_t key_capacity =
      max(context->multicol_key_capacity, growth_capacity(unique_capacity));
  const uint32_t lane_capacity =
      max(context->multicol_lane_capacity, growth_capacity(active_columns));
  if (lane_capacity > 32) {
    set_error(error, error_capacity,
              "multi-column CUDA lane reservation exceeds 32");
    return false;
  }
  size_t new_bytes = 0;
  size_t old_bytes = 0;
  if (!multicol_workspace_bytes(key_capacity, lane_capacity, &new_bytes) ||
      !multicol_workspace_bytes(context->multicol_key_capacity,
                               context->multicol_lane_capacity, &old_bytes)) {
    set_error(error, error_capacity,
              "multi-column CUDA workspace size overflow");
    return false;
  }
  if (new_bytes > UINT64_MAX - context->allocated_bytes) {
    set_error(error, error_capacity,
              "multi-column CUDA transient byte count overflow");
    return false;
  }
  const uint64_t transient_peak = context->allocated_bytes + new_bytes;
  if (transient_peak > context->recoupling_hard_cap_bytes) {
    set_error(error, error_capacity,
              "multi-column CUDA workspace exceeds configured device cap");
    return false;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (!check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), error,
                  error_capacity, "query multi-column CUDA memory") ||
      new_bytes > free_bytes || free_bytes - new_bytes < kDeviceHeadroomBytes) {
    if (new_bytes > free_bytes || free_bytes - new_bytes < kDeviceHeadroomBytes) {
      set_error(error, error_capacity,
                "insufficient CUDA memory for multi-column workspace");
    }
    return false;
  }

  const size_t value_count =
      static_cast<size_t>(key_capacity) * lane_capacity;
  const size_t row_count = static_cast<size_t>(kRows) * lane_capacity;
  uint64_t *keys = nullptr;
  WideValue *values = nullptr;
  unsigned long long *real_wide = nullptr;
  unsigned long long *imaginary_wide = nullptr;
  uint32_t *real = nullptr;
  uint32_t *imaginary = nullptr;
  unsigned long long *expanded = nullptr;
  uint32_t *overflow = nullptr;
  auto cleanup = [&]() {
    cudaFree(keys);
    cudaFree(values);
    cudaFree(real_wide);
    cudaFree(imaginary_wide);
    cudaFree(real);
    cudaFree(imaginary);
    cudaFree(expanded);
    cudaFree(overflow);
  };
#define MULTICOL_ALLOC(call, action)                                            \
  do {                                                                          \
    if (!check_cuda((call), error, error_capacity, (action))) {                 \
      cleanup();                                                                \
      return false;                                                             \
    }                                                                           \
  } while (false)
  MULTICOL_ALLOC(cudaMalloc(&keys, key_capacity * sizeof(uint64_t)),
                 "allocate multi-column keys");
  MULTICOL_ALLOC(cudaMalloc(&values, value_count * sizeof(WideValue)),
                 "allocate multi-column values");
  MULTICOL_ALLOC(cudaMalloc(&real_wide,
                            row_count * sizeof(unsigned long long)),
                 "allocate multi-column wide real rows");
  MULTICOL_ALLOC(cudaMalloc(&imaginary_wide,
                            row_count * sizeof(unsigned long long)),
                 "allocate multi-column wide imaginary rows");
  MULTICOL_ALLOC(cudaMalloc(&real, row_count * sizeof(uint32_t)),
                 "allocate multi-column real rows");
  MULTICOL_ALLOC(cudaMalloc(&imaginary, row_count * sizeof(uint32_t)),
                 "allocate multi-column imaginary rows");
  MULTICOL_ALLOC(cudaMalloc(&expanded, 32 * sizeof(unsigned long long)),
                 "allocate multi-column expansion counters");
  MULTICOL_ALLOC(cudaMalloc(&overflow, sizeof(uint32_t)),
                 "allocate multi-column overflow flag");
#undef MULTICOL_ALLOC

  cudaFree(context->multicol_keys);
  cudaFree(context->multicol_values);
  cudaFree(context->multicol_output_real_wide);
  cudaFree(context->multicol_output_imaginary_wide);
  cudaFree(context->multicol_output_real);
  cudaFree(context->multicol_output_imaginary);
  cudaFree(context->multicol_expanded);
  cudaFree(context->multicol_overflow);
  context->multicol_keys = keys;
  context->multicol_values = values;
  context->multicol_output_real_wide = real_wide;
  context->multicol_output_imaginary_wide = imaginary_wide;
  context->multicol_output_real = real;
  context->multicol_output_imaginary = imaginary;
  context->multicol_expanded = expanded;
  context->multicol_overflow = overflow;
  context->multicol_key_capacity = key_capacity;
  context->multicol_lane_capacity = lane_capacity;
  context->allocated_bytes = context->allocated_bytes - old_bytes + new_bytes;
  context->buffer_high_water_bytes =
      max(context->buffer_high_water_bytes, transient_peak);
  context->buffer_high_water_bytes =
      max(context->buffer_high_water_bytes, context->allocated_bytes);
  return true;
}

bool persistent_growth_allowed(PersistentLoweringContext *context,
                               uint64_t additional, char *error,
                               size_t error_capacity) {
  if (context->live_handle_bytes > UINT64_MAX - context->allocated_bytes) {
    set_error(error, error_capacity,
              "persistent sparse resident byte count overflow");
    return false;
  }
  const uint64_t resident =
      context->allocated_bytes + context->live_handle_bytes;
  if (additional > UINT64_MAX - resident) {
    set_error(error, error_capacity,
              "persistent sparse peak byte count overflow");
    return false;
  }
  const uint64_t peak = resident + additional;
  if (peak > context->hard_cap_bytes) {
    set_error(error, error_capacity,
              "persistent sparse workspace exceeds configured hard cap");
    return false;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (!check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), error,
                  error_capacity, "query persistent sparse device memory")) {
    return false;
  }
  if (additional > free_bytes ||
      free_bytes - static_cast<size_t>(additional) < kDeviceHeadroomBytes) {
    set_error(error, error_capacity,
              "insufficient CUDA memory for persistent sparse workspace");
    return false;
  }
  context->high_water_bytes = max(context->high_water_bytes, peak);
  return true;
}

bool ensure_persistent_sources(PersistentLoweringContext *context,
                               uint32_t required, char *error,
                               size_t error_capacity) {
  if (required <= context->source_capacity) {
    return true;
  }
  const uint32_t capacity = growth_capacity(required);
  const uint64_t new_bytes =
      static_cast<uint64_t>(capacity) * 2 * sizeof(uint32_t) +
      sizeof(uint32_t);
  if (!persistent_growth_allowed(context, new_bytes, error, error_capacity)) {
    return false;
  }
  uint32_t *counts = nullptr;
  uint32_t *offsets = nullptr;
  if (!check_cuda(cudaMalloc(&counts,
                             static_cast<size_t>(capacity) * sizeof(uint32_t)),
                  error, error_capacity, "allocate persistent sparse counts") ||
      !check_cuda(cudaMalloc(&offsets,
                             (static_cast<size_t>(capacity) + 1) *
                                 sizeof(uint32_t)),
                  error, error_capacity, "allocate persistent sparse offsets")) {
    cudaFree(counts);
    cudaFree(offsets);
    return false;
  }
  const uint64_t old_bytes =
      static_cast<uint64_t>(context->source_capacity) * 2 * sizeof(uint32_t) +
      (context->source_capacity == 0 ? 0 : sizeof(uint32_t));
  cudaFree(context->counts);
  cudaFree(context->offsets);
  context->counts = counts;
  context->offsets = offsets;
  context->source_capacity = capacity;
  context->allocated_bytes = context->allocated_bytes - old_bytes + new_bytes;
  context->high_water_bytes =
      max(context->high_water_bytes, context->allocated_bytes);
  return true;
}

bool ensure_persistent_expanded(PersistentLoweringContext *context,
                                uint32_t required, char *error,
                                size_t error_capacity) {
  if (required <= context->expanded_capacity) {
    return true;
  }
  const uint32_t capacity = growth_capacity(required);
  const uint64_t bytes = static_cast<uint64_t>(capacity) *
                         (2 * sizeof(uint64_t) + 2 * sizeof(int64_t) +
                          2 * sizeof(SparseEntry) + sizeof(uint8_t));
  if (!persistent_growth_allowed(context, bytes, error, error_capacity)) {
    return false;
  }
  uint64_t *keys[2]{};
  int64_t *values[2]{};
  SparseEntry *reduced = nullptr;
  SparseEntry *selected = nullptr;
  uint8_t *flags = nullptr;
  const size_t key_bytes = static_cast<size_t>(capacity) * sizeof(uint64_t);
  const size_t value_bytes = static_cast<size_t>(capacity) * sizeof(int64_t);
  const size_t entry_bytes = static_cast<size_t>(capacity) * sizeof(SparseEntry);
  if (!check_cuda(cudaMalloc(&keys[0], key_bytes), error, error_capacity,
                  "allocate persistent sparse keys 0") ||
      !check_cuda(cudaMalloc(&keys[1], key_bytes), error, error_capacity,
                  "allocate persistent sparse keys 1") ||
      !check_cuda(cudaMalloc(&values[0], value_bytes), error, error_capacity,
                  "allocate persistent sparse values 0") ||
      !check_cuda(cudaMalloc(&values[1], value_bytes), error, error_capacity,
                  "allocate persistent sparse values 1") ||
      !check_cuda(cudaMalloc(&reduced, entry_bytes), error, error_capacity,
                  "allocate persistent reduced entries") ||
      !check_cuda(cudaMalloc(&selected, entry_bytes), error, error_capacity,
                  "allocate persistent selected entries") ||
      !check_cuda(cudaMalloc(&flags, capacity * sizeof(uint8_t)), error,
                  error_capacity, "allocate persistent nonzero flags")) {
    cudaFree(keys[0]);
    cudaFree(keys[1]);
    cudaFree(values[0]);
    cudaFree(values[1]);
    cudaFree(reduced);
    cudaFree(selected);
    cudaFree(flags);
    return false;
  }
  const uint64_t old_bytes = static_cast<uint64_t>(context->expanded_capacity) *
                             (2 * sizeof(uint64_t) + 2 * sizeof(int64_t) +
                              2 * sizeof(SparseEntry) + sizeof(uint8_t));
  cudaFree(context->keys[0]);
  cudaFree(context->keys[1]);
  cudaFree(context->values[0]);
  cudaFree(context->values[1]);
  cudaFree(context->reduced_entries);
  cudaFree(context->selected_entries);
  cudaFree(context->nonzero_flags);
  context->keys[0] = keys[0];
  context->keys[1] = keys[1];
  context->values[0] = values[0];
  context->values[1] = values[1];
  context->reduced_entries = reduced;
  context->selected_entries = selected;
  context->nonzero_flags = flags;
  context->expanded_capacity = capacity;
  context->allocated_bytes = context->allocated_bytes - old_bytes + bytes;
  context->high_water_bytes =
      max(context->high_water_bytes, context->allocated_bytes);
  return true;
}

bool ensure_persistent_cub(PersistentLoweringContext *context,
                           size_t required, char *error,
                           size_t error_capacity) {
  if (required <= context->cub_temporary_capacity) {
    return true;
  }
  if (!persistent_growth_allowed(context, required, error, error_capacity)) {
    return false;
  }
  void *temporary = nullptr;
  if (!check_cuda(cudaMalloc(&temporary, required), error, error_capacity,
                  "allocate persistent reusable CUB workspace")) {
    return false;
  }
  cudaFree(context->cub_temporary);
  context->allocated_bytes = context->allocated_bytes -
                             context->cub_temporary_capacity + required;
  context->cub_temporary = temporary;
  context->cub_temporary_capacity = required;
  context->high_water_bytes =
      max(context->high_water_bytes, context->allocated_bytes);
  return true;
}

void destroy_persistent_context(PersistentLoweringContext *context) {
  if (context == nullptr) return;
  cudaSetDevice(context->device);
  if (context->stream != nullptr) cudaStreamSynchronize(context->stream);
  cudaFree(context->sources);
  cudaFree(context->counts);
  cudaFree(context->offsets);
  cudaFree(context->keys[0]);
  cudaFree(context->keys[1]);
  cudaFree(context->values[0]);
  cudaFree(context->values[1]);
  cudaFree(context->reduced_entries);
  cudaFree(context->selected_entries);
  cudaFree(context->nonzero_flags);
  cudaFree(context->unique_count);
  cudaFree(context->selected_count);
  cudaFree(context->overflow);
  cudaFree(context->max_abs);
  cudaFree(context->cub_temporary);
  for (cudaEvent_t event : context->events) {
    if (event != nullptr) cudaEventDestroy(event);
  }
  if (context->stream != nullptr) cudaStreamDestroy(context->stream);
  delete context;
}

}  // namespace

extern "C" {

void *adynkra_fx_cuda_create(
    int device, uint32_t prime, const uint32_t *gauge_offsets,
    const GaugeEntry *gauges, uint32_t gauge_count,
    const TargetEntry *targets, uint32_t target_count,
    const uint32_t *template_offsets, const TemplateEntry *templates,
    uint32_t template_count, char *error, size_t error_capacity) {
  if (gauge_offsets == nullptr || gauges == nullptr || targets == nullptr ||
      template_offsets == nullptr || templates == nullptr || prime < 3 ||
      prime > 0x7fffffffU) {
    set_error(error, error_capacity, "invalid CUDA F_X creation input");
    return nullptr;
  }
  if (!check_cuda(cudaSetDevice(device), error, error_capacity,
                  "select CUDA device")) {
    return nullptr;
  }
  Context *context = new Context();
  context->device = device;
  context->prime = prime;
  uint64_t pow2_64_mod = 1;
  for (uint32_t bit = 0; bit < 64; ++bit) {
    pow2_64_mod = (pow2_64_mod * 2ULL) % prime;
  }
  context->pow2_64_mod = static_cast<uint32_t>(pow2_64_mod);
  context->gauge_count = gauge_count;
  context->target_count = target_count;
  context->template_count = template_count;
  if (!check_cuda(cudaStreamCreateWithFlags(&context->stream,
                                             cudaStreamNonBlocking),
                  error, error_capacity, "create CUDA stream") ||
      !check_cuda(cudaEventCreate(&context->started), error, error_capacity,
                  "create CUDA start event") ||
      !check_cuda(cudaEventCreate(&context->finished), error, error_capacity,
                  "create CUDA finish event") ||
      !check_cuda(cudaMalloc(&context->gauge_offsets,
                             (kGaugeDegrees * kSpinors + 1) * sizeof(uint32_t)),
                  error, error_capacity, "allocate gauge offsets") ||
      !check_cuda(cudaMalloc(&context->gauges,
                             gauge_count * sizeof(GaugeEntry)),
                  error, error_capacity, "allocate gauges") ||
      !check_cuda(cudaMalloc(&context->targets,
                             target_count * sizeof(TargetEntry)),
                  error, error_capacity, "allocate targets") ||
      !check_cuda(cudaMalloc(&context->template_offsets,
                             (11 * kSpinors + 1) * sizeof(uint32_t)),
                  error, error_capacity, "allocate template offsets") ||
      !check_cuda(cudaMalloc(&context->templates,
                             template_count * sizeof(TemplateEntry)),
                  error, error_capacity, "allocate templates") ||
      !check_cuda(cudaMalloc(&context->output_real, kRows * sizeof(uint32_t)),
                  error, error_capacity, "allocate real output") ||
      !check_cuda(cudaMalloc(&context->output_imaginary,
                             kRows * sizeof(uint32_t)),
                  error, error_capacity, "allocate imaginary output") ||
      !check_cuda(cudaMalloc(&context->expanded,
                             sizeof(unsigned long long)),
                  error, error_capacity, "allocate contribution counter") ||
      !check_cuda(cudaMalloc(&context->recoupling_unique_count,
                             sizeof(uint32_t)),
                  error, error_capacity, "allocate recoupling unique count") ||
      !check_cuda(cudaMalloc(&context->recoupling_nonzero_count,
                             sizeof(unsigned long long)),
                  error, error_capacity, "allocate recoupling nonzero count") ||
      !check_cuda(cudaMalloc(&context->recoupling_overflow,
                             sizeof(uint32_t)),
                  error, error_capacity, "allocate recoupling overflow flag")) {
    destroy(context);
    return nullptr;
  }
  for (cudaEvent_t &event : context->stage_events) {
    if (!check_cuda(cudaEventCreate(&event), error, error_capacity,
                    "create recoupling stage event")) {
      destroy(context);
      return nullptr;
    }
  }
  context->allocated_bytes =
      (kGaugeDegrees * kSpinors + 1) * sizeof(uint32_t) +
      static_cast<uint64_t>(gauge_count) * sizeof(GaugeEntry) +
      static_cast<uint64_t>(target_count) * sizeof(TargetEntry) +
      (11 * kSpinors + 1) * sizeof(uint32_t) +
      static_cast<uint64_t>(template_count) * sizeof(TemplateEntry) +
      2ULL * kRows * sizeof(uint32_t) + sizeof(unsigned long long) +
      sizeof(uint32_t) + sizeof(unsigned long long) + sizeof(uint32_t);
  context->buffer_high_water_bytes = context->allocated_bytes;
  if (!check_cuda(cudaMemcpyAsync(
                      context->gauge_offsets, gauge_offsets,
                      (kGaugeDegrees * kSpinors + 1) * sizeof(uint32_t),
                      cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload gauge offsets") ||
      !check_cuda(cudaMemcpyAsync(context->gauges, gauges,
                                  gauge_count * sizeof(GaugeEntry),
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload gauges") ||
      !check_cuda(cudaMemcpyAsync(context->targets, targets,
                                  target_count * sizeof(TargetEntry),
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload targets") ||
      !check_cuda(cudaMemcpyAsync(
                      context->template_offsets, template_offsets,
                      (11 * kSpinors + 1) * sizeof(uint32_t),
                      cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload template offsets") ||
      !check_cuda(cudaMemcpyAsync(context->templates, templates,
                                  template_count * sizeof(TemplateEntry),
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload templates") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish CUDA F_X static upload")) {
    destroy(context);
    return nullptr;
  }
  return context;
}

void *adynkra_fx_cuda_create_v2(
    int device, uint32_t prime, const uint32_t *gauge_offsets,
    const GaugeEntry *gauges, uint32_t gauge_count,
    const TargetEntry *targets, uint32_t target_count,
    const uint32_t *template_offsets, const TemplateEntry *templates,
    uint32_t template_count, const uint32_t *plan_offsets,
    const PlanEntry *plan_entries, uint32_t plan_entry_count,
    const uint64_t *pair_salts, char *error, size_t error_capacity) {
  if (plan_offsets == nullptr || plan_entries == nullptr ||
      plan_entry_count == 0 || pair_salts == nullptr || plan_offsets[0] != 0 ||
      plan_offsets[kGaugeDegrees * kSpinors] != plan_entry_count) {
    set_error(error, error_capacity, "invalid CUDA F_X v2 static schedule");
    return nullptr;
  }
  uint32_t max_degree_free = 0;
  uint64_t free_totals[kSpinors]{};
  for (uint32_t schedule = 0; schedule < kGaugeDegrees * kSpinors;
       ++schedule) {
    if (plan_offsets[schedule] > plan_offsets[schedule + 1]) {
      set_error(error, error_capacity, "non-monotone CUDA F_X plan offsets");
      return nullptr;
    }
    const uint32_t count = plan_offsets[schedule + 1] - plan_offsets[schedule];
    max_degree_free = count > max_degree_free ? count : max_degree_free;
    free_totals[schedule % kSpinors] += count;
    const uint32_t degree = schedule / kSpinors;
    for (uint32_t index = plan_offsets[schedule];
         index < plan_offsets[schedule + 1]; ++index) {
      const PlanEntry &entry = plan_entries[index];
      if (entry.functional_salt != canonical_functional_salt(
                                       degree, entry.output_coordinate,
                                       entry.sector)) {
        set_error(error, error_capacity,
                  "noncanonical CUDA F_X functional salt");
        return nullptr;
      }
    }
  }
  uint32_t max_free = 0;
  for (uint32_t free_spinor = 0; free_spinor < kSpinors; ++free_spinor) {
    if (free_totals[free_spinor] > UINT32_MAX) {
      set_error(error, error_capacity, "CUDA F_X plan free-spinor count overflow");
      return nullptr;
    }
    max_free = free_totals[free_spinor] > max_free
                   ? static_cast<uint32_t>(free_totals[free_spinor])
                   : max_free;
  }
  for (uint32_t index = 0; index < plan_entry_count; ++index) {
    const PlanEntry &entry = plan_entries[index];
    if (entry.gauge_spinor >= kSpinors || entry.template_spinor >= kSpinors ||
        entry.gauge_spinor == entry.template_spinor || entry.sector >= kSectors ||
        entry.coefficient.real >= prime || entry.coefficient.imaginary >= prime ||
        (entry.coefficient.real == 0 && entry.coefficient.imaginary == 0)) {
      set_error(error, error_capacity, "invalid CUDA F_X v2 plan entry");
      return nullptr;
    }
  }
  uint32_t pair = 0;
  for (uint32_t left = 0; left < 11; ++left) {
    for (uint32_t right = left; right < 11; ++right, ++pair) {
      if (pair_salts[pair] != canonical_pair_salt(left, right)) {
        set_error(error, error_capacity, "noncanonical CUDA F_X pair salt");
        return nullptr;
      }
    }
  }
  Context *context = static_cast<Context *>(adynkra_fx_cuda_create(
      device, prime, gauge_offsets, gauges, gauge_count, targets, target_count,
      template_offsets, templates, template_count, error, error_capacity));
  if (context == nullptr) {
    return nullptr;
  }
  context->plan_entry_count = plan_entry_count;
  context->max_plan_entries_per_degree_free = max_degree_free;
  context->max_plan_entries_per_free = max_free;
  if (!check_cuda(cudaMalloc(&context->plan_offsets,
                             (kGaugeDegrees * kSpinors + 1) * sizeof(uint32_t)),
                  error, error_capacity, "allocate CUDA F_X plan offsets") ||
      !check_cuda(cudaMalloc(&context->plan_entries,
                             static_cast<size_t>(plan_entry_count) *
                                 sizeof(PlanEntry)),
                  error, error_capacity, "allocate CUDA F_X plan entries") ||
      !check_cuda(cudaMalloc(&context->pair_salts,
                             kMomentumPairs * sizeof(uint64_t)),
                  error, error_capacity, "allocate CUDA F_X pair salts") ||
      !check_cuda(cudaMalloc(&context->output_real_wide,
                             kRows * sizeof(unsigned long long)),
                  error, error_capacity, "allocate wide real output") ||
      !check_cuda(cudaMalloc(&context->output_imaginary_wide,
                             kRows * sizeof(unsigned long long)),
                  error, error_capacity, "allocate wide imaginary output") ||
      !check_cuda(cudaMemcpyAsync(
                      context->plan_offsets, plan_offsets,
                      (kGaugeDegrees * kSpinors + 1) * sizeof(uint32_t),
                      cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload CUDA F_X plan offsets") ||
      !check_cuda(cudaMemcpyAsync(
                      context->plan_entries, plan_entries,
                      static_cast<size_t>(plan_entry_count) * sizeof(PlanEntry),
                      cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload CUDA F_X plan entries") ||
      !check_cuda(cudaMemcpyAsync(context->pair_salts, pair_salts,
                                  kMomentumPairs * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload CUDA F_X pair salts") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish CUDA F_X v2 static upload")) {
    destroy(context);
    return nullptr;
  }
  const uint64_t additional =
      (kGaugeDegrees * kSpinors + 1ULL) * sizeof(uint32_t) +
      static_cast<uint64_t>(plan_entry_count) * sizeof(PlanEntry) +
      kMomentumPairs * sizeof(uint64_t) +
      2ULL * kRows * sizeof(unsigned long long);
  context->allocated_bytes += additional;
  context->buffer_high_water_bytes = context->allocated_bytes;
  return context;
}

int adynkra_fx_cuda_configure_p3(
    void *opaque, const uint32_t *plan_offsets,
    const P3PlanEntry *plan_entries, uint32_t plan_entry_count,
    char *error, size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || plan_offsets == nullptr || plan_entries == nullptr ||
      plan_entry_count == 0 || plan_offsets[0] != 0 ||
      plan_offsets[kGaugeDegrees * kSpinors] != plan_entry_count ||
      context->pair_salts == nullptr || context->p3_plan_offsets != nullptr ||
      context->p3_plan_pair_offsets != nullptr ||
      context->p3_plan_entries != nullptr || context->p3_output_real != nullptr ||
      context->p3_output_imaginary != nullptr) {
    set_error(error, error_capacity, "invalid CUDA p3 static schedule");
    return 1;
  }
  for (uint32_t schedule = 0; schedule < kGaugeDegrees * kSpinors;
       ++schedule) {
    if (plan_offsets[schedule] > plan_offsets[schedule + 1]) {
      set_error(error, error_capacity, "non-monotone CUDA p3 plan offsets");
      return 1;
    }
    const uint32_t degree = schedule / kSpinors;
    for (uint32_t index = plan_offsets[schedule];
         index < plan_offsets[schedule + 1]; ++index) {
      const P3PlanEntry &entry = plan_entries[index];
      if (entry.contracted_spinor >= kSpinors ||
          entry.template_spinor >= kSpinors ||
          entry.contraction_axis >= kContractionAxes ||
          entry.sector >= kSectors ||
          entry.output_coordinate >=
              (entry.sector == 0 ? kX2OutputCoordinates
                                 : kX5OutputCoordinates) ||
          entry.coefficient.real >= context->prime ||
          entry.coefficient.imaginary >= context->prime ||
          (entry.coefficient.real == 0 && entry.coefficient.imaginary == 0) ||
          entry.functional_salt != canonical_p3_functional_salt(
              degree, entry.output_coordinate, entry.sector,
              entry.contraction_axis)) {
        set_error(error, error_capacity, "invalid CUDA p3 plan entry");
        return 1;
      }
    }
  }
  std::vector<uint32_t> host_pair_offsets;
  try {
    host_pair_offsets.resize(kP3PairOffsetCount);
  } catch (...) {
    set_error(error, error_capacity,
              "allocate CUDA p3 contracted/template host offsets");
    return 1;
  }
  for (uint32_t schedule = 0; schedule < kP3ScheduleCount; ++schedule) {
    uint32_t cursor = plan_offsets[schedule];
    const uint32_t schedule_end = plan_offsets[schedule + 1];
    for (uint32_t pair = 0; pair < kP3SpinorPairCount; ++pair) {
      host_pair_offsets[schedule * kP3SpinorPairCount + pair] = cursor;
      while (cursor < schedule_end) {
        const P3PlanEntry &entry = plan_entries[cursor];
        const uint32_t entry_pair =
            entry.contracted_spinor * kSpinors + entry.template_spinor;
        if (entry_pair < pair) {
          set_error(error, error_capacity,
                    "unordered CUDA p3 contracted/template plan entries");
          return 1;
        }
        if (entry_pair != pair) break;
        ++cursor;
      }
    }
    if (cursor != schedule_end) {
      set_error(error, error_capacity,
                "incomplete CUDA p3 contracted/template plan offsets");
      return 1;
    }
  }
  host_pair_offsets[kP3PairOffsetCount - 1] = plan_entry_count;
  size_t entry_bytes = 0;
  if (!checked_multiply_size(plan_entry_count, sizeof(P3PlanEntry),
                             &entry_bytes)) {
    set_error(error, error_capacity, "CUDA p3 plan byte count overflow");
    return 1;
  }
  const size_t offset_bytes =
      (kGaugeDegrees * kSpinors + 1ULL) * sizeof(uint32_t);
  const size_t pair_offset_bytes =
      static_cast<size_t>(kP3PairOffsetCount) * sizeof(uint32_t);
  const size_t output_bytes =
      2ULL * kP3Rows * kMaxColumns * sizeof(uint32_t);
  if (offset_bytes > SIZE_MAX - pair_offset_bytes ||
      offset_bytes + pair_offset_bytes > SIZE_MAX - entry_bytes ||
      offset_bytes + pair_offset_bytes + entry_bytes >
          SIZE_MAX - output_bytes) {
    set_error(error, error_capacity, "CUDA p3 allocation byte count overflow");
    return 1;
  }
  const size_t additional_bytes =
      offset_bytes + pair_offset_bytes + entry_bytes + output_bytes;
  if (additional_bytes > UINT64_MAX - context->allocated_bytes ||
      context->allocated_bytes + additional_bytes >
          context->recoupling_hard_cap_bytes) {
    set_error(error, error_capacity,
              "CUDA p3 static plan exceeds configured device cap");
    return 1;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (!check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), error,
                  error_capacity, "query CUDA p3 device memory") ||
      additional_bytes > free_bytes ||
      free_bytes - additional_bytes < kDeviceHeadroomBytes) {
    if (additional_bytes > free_bytes ||
        free_bytes - additional_bytes < kDeviceHeadroomBytes) {
      set_error(error, error_capacity,
                "insufficient CUDA memory for p3 static plan");
    }
    return 1;
  }
  uint32_t *new_plan_offsets = nullptr;
  uint32_t *new_plan_pair_offsets = nullptr;
  P3PlanEntry *new_plan_entries = nullptr;
  uint32_t *new_output_real = nullptr;
  uint32_t *new_output_imaginary = nullptr;
  const auto rollback = [&]() {
    cudaFree(new_plan_offsets);
    cudaFree(new_plan_pair_offsets);
    cudaFree(new_plan_entries);
    cudaFree(new_output_real);
    cudaFree(new_output_imaginary);
  };
  if (!check_cuda(cudaMalloc(&new_plan_offsets, offset_bytes), error,
                  error_capacity, "allocate CUDA p3 offsets") ||
      !check_cuda(cudaMalloc(&new_plan_pair_offsets, pair_offset_bytes), error,
                  error_capacity,
                  "allocate CUDA p3 contracted/template offsets") ||
      !check_cuda(cudaMalloc(&new_plan_entries, entry_bytes), error,
                  error_capacity, "allocate CUDA p3 entries") ||
      !check_cuda(cudaMalloc(&new_output_real,
                             kP3Rows * kMaxColumns * sizeof(uint32_t)),
                  error, error_capacity, "allocate CUDA p3 real output") ||
      !check_cuda(cudaMalloc(&new_output_imaginary,
                             kP3Rows * kMaxColumns * sizeof(uint32_t)),
                  error, error_capacity, "allocate CUDA p3 imaginary output") ||
      !check_cuda(cudaMemcpyAsync(new_plan_offsets, plan_offsets, offset_bytes,
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload CUDA p3 offsets") ||
      !check_cuda(cudaMemcpyAsync(
                      new_plan_pair_offsets, host_pair_offsets.data(),
                      pair_offset_bytes, cudaMemcpyHostToDevice,
                      context->stream),
                  error, error_capacity,
                  "upload CUDA p3 contracted/template offsets") ||
      !check_cuda(cudaMemcpyAsync(new_plan_entries, plan_entries, entry_bytes,
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload CUDA p3 entries") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish CUDA p3 static upload")) {
    rollback();
    return 1;
  }
  context->p3_plan_offsets = new_plan_offsets;
  context->p3_plan_pair_offsets = new_plan_pair_offsets;
  context->p3_plan_entries = new_plan_entries;
  context->p3_output_real = new_output_real;
  context->p3_output_imaginary = new_output_imaginary;
  context->p3_plan_entry_count = plan_entry_count;
  context->p3_active_columns = 1;
  context->allocated_bytes += additional_bytes;
  context->buffer_high_water_bytes =
      max(context->buffer_high_water_bytes, context->allocated_bytes);
  return 0;
}

int adynkra_fx_cuda_reset_p3_columns(
    void *opaque, const uint32_t *input_real, const uint32_t *input_imaginary,
    uint32_t active_columns, char *error, size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || context->p3_output_real == nullptr ||
      context->p3_output_imaginary == nullptr || active_columns == 0 ||
      active_columns > kMaxColumns ||
      ((input_real == nullptr) != (input_imaginary == nullptr))) {
    set_error(error, error_capacity, "invalid CUDA p3 column reset input");
    return 1;
  }
  const size_t bytes = static_cast<size_t>(active_columns) * kP3Rows *
                       sizeof(uint32_t);
  const bool restore = input_real != nullptr;
  if (!(restore
            ? check_cuda(cudaMemcpyAsync(context->p3_output_real, input_real,
                                         bytes, cudaMemcpyHostToDevice,
                                         context->stream),
                         error, error_capacity,
                         "restore CUDA p3 real columns") &&
                  check_cuda(cudaMemcpyAsync(context->p3_output_imaginary,
                                             input_imaginary, bytes,
                                             cudaMemcpyHostToDevice,
                                             context->stream),
                             error, error_capacity,
                             "restore CUDA p3 imaginary columns")
            : check_cuda(cudaMemsetAsync(context->p3_output_real, 0, bytes,
                                         context->stream),
                         error, error_capacity, "clear CUDA p3 real columns") &&
                  check_cuda(cudaMemsetAsync(context->p3_output_imaginary, 0,
                                             bytes, context->stream),
                             error, error_capacity,
                             "clear CUDA p3 imaginary columns")) ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish CUDA p3 column reset")) {
    return 1;
  }
  context->p3_active_columns = active_columns;
  return 0;
}

int adynkra_fx_cuda_accumulate_p3_column(
    void *opaque, uint32_t column, const SourceEntry *sources,
    uint32_t source_count, uint64_t *expanded_contributions,
    float *kernel_milliseconds, char *error, size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || sources == nullptr || source_count == 0 ||
      column >= context->p3_active_columns || expanded_contributions == nullptr ||
      kernel_milliseconds == nullptr || context->p3_plan_entries == nullptr) {
    set_error(error, error_capacity,
              "invalid CUDA persistent p3 accumulation input");
    return 1;
  }
  if (!ensure_source_capacity(context, source_count, error, error_capacity)) {
    return 1;
  }
  if (!check_cuda(cudaMemcpyAsync(context->sources, sources,
                                  source_count * sizeof(SourceEntry),
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity,
                  "upload CUDA persistent p3 sources") ||
      !check_cuda(cudaMemsetAsync(context->expanded, 0,
                                  sizeof(unsigned long long), context->stream),
                  error, error_capacity,
                  "clear CUDA persistent p3 contribution count") ||
      !check_cuda(cudaEventRecord(context->started, context->stream), error,
                  error_capacity, "record CUDA persistent p3 start")) {
    return 1;
  }
  constexpr uint32_t threads = 128;
  const uint32_t blocks = (source_count + threads - 1) / threads;
  accumulate_p3_kernel<<<blocks, threads, 0, context->stream>>>(
      context->sources, source_count, context->p3_plan_offsets,
      context->p3_plan_entries, context->pair_salts, context->prime,
      context->p3_output_real + static_cast<size_t>(column) * kP3Rows,
      context->p3_output_imaginary + static_cast<size_t>(column) * kP3Rows,
      context->expanded);
  if (!check_cuda(cudaGetLastError(), error, error_capacity,
                  "launch CUDA persistent p3 kernel") ||
      !check_cuda(cudaEventRecord(context->finished, context->stream), error,
                  error_capacity, "record CUDA persistent p3 finish") ||
      !check_cuda(cudaMemcpyAsync(expanded_contributions, context->expanded,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity,
                  "download CUDA persistent p3 count") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish CUDA persistent p3 accumulation") ||
      !check_cuda(cudaEventElapsedTime(kernel_milliseconds, context->started,
                                       context->finished),
                  error, error_capacity,
                  "measure CUDA persistent p3 kernel")) {
    return 1;
  }
  return 0;
}

int adynkra_fx_cuda_accumulate_p3_multicol(
    void *opaque, const uint64_t *keys,
    const uint32_t *key_major_coefficients, uint32_t unique_count,
    uint32_t active_columns, uint64_t *expanded_contributions,
    float *kernel_milliseconds, char *error, size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || keys == nullptr ||
      key_major_coefficients == nullptr || unique_count == 0 ||
      active_columns == 0 || active_columns > 32 ||
      active_columns != context->p3_active_columns ||
      expanded_contributions == nullptr || kernel_milliseconds == nullptr ||
      context->p3_plan_pair_offsets == nullptr ||
      context->p3_plan_entries == nullptr ||
      context->p3_output_real == nullptr ||
      context->p3_output_imaginary == nullptr) {
    set_error(error, error_capacity,
              "invalid CUDA persistent p3 multi-column input");
    return 1;
  }
  if (!ensure_multicol_workspace(context, unique_count, active_columns, error,
                                 error_capacity)) {
    return 1;
  }
  size_t value_count = 0;
  size_t value_bytes = 0;
  if (!checked_multiply_size(unique_count, active_columns, &value_count) ||
      !checked_multiply_size(value_count, sizeof(uint32_t), &value_bytes)) {
    set_error(error, error_capacity,
              "CUDA persistent p3 multi-column input size overflow");
    return 1;
  }
  if (!check_cuda(cudaMemcpyAsync(context->multicol_keys, keys,
                                  unique_count * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity,
                  "upload CUDA persistent p3 multi-column keys") ||
      !check_cuda(cudaMemcpyAsync(
                      reinterpret_cast<uint32_t *>(context->multicol_values),
                      key_major_coefficients, value_bytes,
                      cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity,
                  "upload CUDA persistent p3 multi-column coefficients") ||
      !check_cuda(cudaMemsetAsync(context->multicol_expanded, 0,
                                  active_columns * sizeof(unsigned long long),
                                  context->stream),
                  error, error_capacity,
                  "clear CUDA persistent p3 multi-column counts") ||
      !check_cuda(cudaMemsetAsync(context->multicol_overflow, 0,
                                  sizeof(uint32_t), context->stream),
                  error, error_capacity,
                  "clear CUDA persistent p3 multi-column validity") ||
      !check_cuda(cudaEventRecord(context->started, context->stream), error,
                  error_capacity,
                  "record CUDA persistent p3 multi-column start")) {
    return 1;
  }
  constexpr uint32_t threads = 128;
  accumulate_p3_multicol_kernel<<<unique_count, threads, 0, context->stream>>>(
      context->multicol_keys,
      reinterpret_cast<const uint32_t *>(context->multicol_values),
      unique_count, active_columns, context->p3_plan_pair_offsets,
      context->p3_plan_entries, context->pair_salts, context->prime,
      context->p3_output_real, context->p3_output_imaginary,
      context->multicol_expanded, context->multicol_overflow);
  uint32_t invalid = 0;
  if (!check_cuda(cudaGetLastError(), error, error_capacity,
                  "launch CUDA persistent p3 multi-column kernel") ||
      !check_cuda(cudaEventRecord(context->finished, context->stream), error,
                  error_capacity,
                  "record CUDA persistent p3 multi-column finish") ||
      !check_cuda(cudaMemcpyAsync(
                      expanded_contributions, context->multicol_expanded,
                      active_columns * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity,
                  "download CUDA persistent p3 multi-column counts") ||
      !check_cuda(cudaMemcpyAsync(&invalid, context->multicol_overflow,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                  context->stream),
                  error, error_capacity,
                  "download CUDA persistent p3 multi-column validity") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish CUDA persistent p3 multi-column accumulation") ||
      !check_cuda(cudaEventElapsedTime(kernel_milliseconds, context->started,
                                       context->finished),
                  error, error_capacity,
                  "measure CUDA persistent p3 multi-column kernel")) {
    return 1;
  }
  if (invalid != 0) {
    set_error(error, error_capacity,
              "invalid key or coefficient in CUDA persistent p3 multi-column kernel");
    return 1;
  }
  return 0;
}

int adynkra_fx_cuda_download_p3_columns(
    void *opaque, uint32_t active_columns, uint32_t *output_real,
    uint32_t *output_imaginary, char *error, size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || active_columns == 0 ||
      active_columns != context->p3_active_columns || output_real == nullptr ||
      output_imaginary == nullptr || context->p3_output_real == nullptr ||
      context->p3_output_imaginary == nullptr) {
    set_error(error, error_capacity, "invalid CUDA p3 column download input");
    return 1;
  }
  const size_t bytes = static_cast<size_t>(active_columns) * kP3Rows *
                       sizeof(uint32_t);
  if (!check_cuda(cudaMemcpyAsync(output_real, context->p3_output_real, bytes,
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity, "download CUDA p3 real columns") ||
      !check_cuda(cudaMemcpyAsync(output_imaginary,
                                  context->p3_output_imaginary, bytes,
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity,
                  "download CUDA p3 imaginary columns") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish CUDA p3 column download")) {
    return 1;
  }
  return 0;
}

int adynkra_fx_cuda_accumulate_p3(
    void *opaque, const SourceEntry *sources, uint32_t source_count,
    uint32_t *output_real, uint32_t *output_imaginary,
    uint64_t *expanded_contributions, float *kernel_milliseconds,
    char *error, size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || sources == nullptr || source_count == 0 ||
      output_real == nullptr || output_imaginary == nullptr ||
      expanded_contributions == nullptr || kernel_milliseconds == nullptr ||
      context->p3_plan_entries == nullptr) {
    set_error(error, error_capacity, "invalid CUDA p3 accumulation input");
    return 1;
  }
  if (!ensure_source_capacity(context, source_count, error, error_capacity)) {
    return 1;
  }
  if (!check_cuda(cudaMemcpyAsync(context->sources, sources,
                                  source_count * sizeof(SourceEntry),
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload CUDA p3 sources") ||
      !check_cuda(cudaMemsetAsync(context->p3_output_real, 0,
                                  kP3Rows * sizeof(uint32_t), context->stream),
                  error, error_capacity, "clear CUDA p3 real output") ||
      !check_cuda(cudaMemsetAsync(context->p3_output_imaginary, 0,
                                  kP3Rows * sizeof(uint32_t), context->stream),
                  error, error_capacity, "clear CUDA p3 imaginary output") ||
      !check_cuda(cudaMemsetAsync(context->expanded, 0,
                                  sizeof(unsigned long long), context->stream),
                  error, error_capacity, "clear CUDA p3 contribution count") ||
      !check_cuda(cudaEventRecord(context->started, context->stream), error,
                  error_capacity, "record CUDA p3 start")) {
    return 1;
  }
  constexpr uint32_t threads = 128;
  const uint32_t blocks = (source_count + threads - 1) / threads;
  accumulate_p3_kernel<<<blocks, threads, 0, context->stream>>>(
      context->sources, source_count, context->p3_plan_offsets,
      context->p3_plan_entries, context->pair_salts, context->prime,
      context->p3_output_real, context->p3_output_imaginary,
      context->expanded);
  if (!check_cuda(cudaGetLastError(), error, error_capacity,
                  "launch CUDA p3 kernel") ||
      !check_cuda(cudaEventRecord(context->finished, context->stream), error,
                  error_capacity, "record CUDA p3 finish") ||
      !check_cuda(cudaMemcpyAsync(output_real, context->p3_output_real,
                                  kP3Rows * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity, "download CUDA p3 real output") ||
      !check_cuda(cudaMemcpyAsync(output_imaginary, context->p3_output_imaginary,
                                  kP3Rows * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity, "download CUDA p3 imaginary output") ||
      !check_cuda(cudaMemcpyAsync(expanded_contributions, context->expanded,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity, "download CUDA p3 count") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish CUDA p3 accumulation") ||
      !check_cuda(cudaEventElapsedTime(kernel_milliseconds, context->started,
                                       context->finished),
                  error, error_capacity, "measure CUDA p3 kernel")) {
    return 1;
  }
  return 0;
}

int adynkra_fx_cuda_set_legacy_contraction(void *opaque, int enabled,
                                           char *error,
                                           size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || (enabled != 0 && enabled != 1)) {
    set_error(error, error_capacity,
              "invalid CUDA F_X legacy contraction setting");
    return 1;
  }
  context->legacy_contraction = enabled != 0;
  return 0;
}

int adynkra_fx_cuda_accumulate(
    void *opaque, const SourceEntry *sources, uint32_t source_count,
    uint32_t *output_real, uint32_t *output_imaginary,
    uint64_t *expanded_contributions, float *kernel_milliseconds, char *error,
    size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || sources == nullptr || source_count == 0 ||
      output_real == nullptr || output_imaginary == nullptr ||
      expanded_contributions == nullptr || kernel_milliseconds == nullptr) {
    set_error(error, error_capacity, "invalid CUDA F_X accumulation input");
    return 1;
  }
  if (!ensure_source_capacity(context, source_count, error, error_capacity)) {
    return 1;
  }
  if (!check_cuda(cudaMemcpyAsync(context->sources, sources,
                                  source_count * sizeof(SourceEntry),
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity, "upload source terms") ||
      !check_cuda(cudaMemsetAsync(context->output_real, 0,
                                  kRows * sizeof(uint32_t), context->stream),
                  error, error_capacity, "clear real output") ||
      !check_cuda(cudaMemsetAsync(context->output_imaginary, 0,
                                  kRows * sizeof(uint32_t), context->stream),
                  error, error_capacity, "clear imaginary output") ||
      !check_cuda(cudaMemsetAsync(context->expanded, 0,
                                  sizeof(unsigned long long), context->stream),
                  error, error_capacity, "clear contribution counter") ||
      !check_cuda(cudaEventRecord(context->started, context->stream), error,
                  error_capacity, "record CUDA start event")) {
    return 1;
  }
  constexpr uint32_t threads = 128;
  uint32_t blocks = (source_count + threads - 1) / threads;
  accumulate_kernel<<<blocks, threads, 0, context->stream>>>(
      context->sources, source_count, context->gauge_offsets, context->gauges,
      context->targets, context->target_count, context->template_offsets,
      context->templates, context->prime, context->output_real,
      context->output_imaginary, context->expanded);
  if (!check_cuda(cudaGetLastError(), error, error_capacity,
                  "launch CUDA F_X kernel") ||
      !check_cuda(cudaEventRecord(context->finished, context->stream), error,
                  error_capacity, "record CUDA finish event") ||
      !check_cuda(cudaMemcpyAsync(output_real, context->output_real,
                                  kRows * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity, "download real output") ||
      !check_cuda(cudaMemcpyAsync(output_imaginary, context->output_imaginary,
                                  kRows * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity, "download imaginary output") ||
      !check_cuda(cudaMemcpyAsync(expanded_contributions, context->expanded,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity, "download contribution count") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish CUDA F_X accumulation") ||
      !check_cuda(cudaEventElapsedTime(kernel_milliseconds, context->started,
                                       context->finished),
                  error, error_capacity, "measure CUDA F_X kernel")) {
    return 1;
  }
  return 0;
}

int adynkra_fx_cuda_accumulate_recoupled(
    void *opaque, const uint64_t *keys, const WideValue *values,
    uint32_t source_count, uint32_t *output_real,
    uint32_t *output_imaginary, RecouplingStats *stats, char *error,
    size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || keys == nullptr || values == nullptr ||
      source_count == 0 || output_real == nullptr ||
      output_imaginary == nullptr || stats == nullptr ||
      context->plan_offsets == nullptr || context->plan_entries == nullptr ||
      context->pair_salts == nullptr || context->output_real_wide == nullptr ||
      context->output_imaginary_wide == nullptr) {
    set_error(error, error_capacity,
              "invalid CUDA recoupling accumulation input");
    return 1;
  }
  std::memset(stats, 0, sizeof(*stats));
  uint64_t absolute_sum_low = 0;
  uint64_t absolute_sum_high = 0;
  for (uint32_t index = 0; index < source_count; ++index) {
    const uint32_t metadata = static_cast<uint32_t>(keys[index] >> 32);
    const uint32_t pair_left = metadata & 15U;
    const uint32_t pair_right = (metadata >> 4) & 15U;
    const uint32_t free_spinor = (metadata >> 8) & 31U;
    const uint32_t mask = static_cast<uint32_t>(keys[index]);
    if ((metadata >> 13) != 0 || pair_left > pair_right ||
        pair_right >= 11 || free_spinor >= 32 || __builtin_popcount(mask) != 12 ||
        values[index].overflow != 0 || values[index].reserved != 0) {
      set_error(error, error_capacity,
                "invalid packed CUDA recoupling contribution");
      return 1;
    }
    // CUB may choose a tree reduction order. Bound the global L1 norm by
    // i128::MAX so every possible partial sum is representable and WideAdd is
    // genuinely associative for this invocation. This also rejects i128::MIN,
    // whose absolute value cannot be represented as a signed i128.
    uint64_t magnitude_low = values[index].low;
    uint64_t magnitude_high = static_cast<uint64_t>(values[index].high);
    if (values[index].high < 0) {
      magnitude_low = ~magnitude_low + 1ULL;
      magnitude_high = ~magnitude_high + (magnitude_low == 0 ? 1ULL : 0ULL);
    }
    const uint64_t next_low = absolute_sum_low + magnitude_low;
    const uint64_t carry = next_low < absolute_sum_low ? 1ULL : 0ULL;
    const uint64_t next_high = absolute_sum_high + magnitude_high + carry;
    const bool high_overflow = next_high < absolute_sum_high ||
                               (carry != 0 && next_high == absolute_sum_high);
    if (high_overflow || next_high > static_cast<uint64_t>(INT64_MAX)) {
      set_error(error, error_capacity,
                "recoupling absolute coefficient sum exceeds signed i128");
      return 1;
    }
    absolute_sum_low = next_low;
    absolute_sum_high = next_high;
  }
  if (!check_cuda(cudaSetDevice(context->device), error, error_capacity,
                  "select recoupling CUDA device") ||
      !ensure_recoupling_workspace(context, source_count, error,
                                   error_capacity)) {
    return 1;
  }

  size_t sort_bytes = 0;
  size_t reduce_bytes = 0;
  if (!check_cuda(cub::DeviceRadixSort::SortPairs(
                      nullptr, sort_bytes, context->recoupling_keys[0],
                      context->recoupling_keys[1],
                      context->recoupling_values[0],
                      context->recoupling_values[1], source_count, 0,
                      kRecouplingSemanticKeyBits,
                      context->stream),
                  error, error_capacity, "size active recoupling radix sort") ||
      !check_cuda(cub::DeviceReduce::ReduceByKey(
                      nullptr, reduce_bytes, context->recoupling_keys[1],
                      context->recoupling_keys[0],
                      context->recoupling_values[1],
                      context->recoupling_values[0],
                      context->recoupling_unique_count, WideAdd(), source_count,
                      context->stream),
                  error, error_capacity,
                  "size active recoupling reduce-by-key") ||
      sort_bytes > context->cub_temporary_capacity ||
      reduce_bytes > context->cub_temporary_capacity) {
    if (sort_bytes > context->cub_temporary_capacity ||
        reduce_bytes > context->cub_temporary_capacity) {
      set_error(error, error_capacity,
                "reusable CUB workspace capacity invariant failed");
    }
    return 1;
  }

#define RECOUPLE_CUDA(call, action)                                             \
  do {                                                                          \
    if (!check_cuda((call), error, error_capacity, (action))) {                 \
      return 1;                                                                 \
    }                                                                            \
  } while (false)

  RECOUPLE_CUDA(cudaEventRecord(context->stage_events[0], context->stream),
                  "record recoupling start");
  RECOUPLE_CUDA(cudaMemcpyAsync(
                    context->recoupling_keys[0], keys,
                    source_count * sizeof(uint64_t), cudaMemcpyHostToDevice,
                    context->stream),
                  "upload recoupling keys");
  RECOUPLE_CUDA(cudaMemcpyAsync(
                    context->recoupling_values[0], values,
                    source_count * sizeof(WideValue), cudaMemcpyHostToDevice,
                    context->stream),
                  "upload recoupling values");
  RECOUPLE_CUDA(cudaEventRecord(context->stage_events[1], context->stream),
                  "record recoupling upload finish");
  RECOUPLE_CUDA(cub::DeviceRadixSort::SortPairs(
                    context->cub_temporary, sort_bytes,
                    context->recoupling_keys[0], context->recoupling_keys[1],
                    context->recoupling_values[0],
                    context->recoupling_values[1], source_count, 0,
                    kRecouplingSemanticKeyBits,
                    context->stream),
                  "sort packed recoupling contributions");
  RECOUPLE_CUDA(cudaEventRecord(context->stage_events[2], context->stream),
                  "record recoupling sort finish");
  RECOUPLE_CUDA(cub::DeviceReduce::ReduceByKey(
                    context->cub_temporary, reduce_bytes,
                    context->recoupling_keys[1], context->recoupling_keys[0],
                    context->recoupling_values[1],
                    context->recoupling_values[0],
                    context->recoupling_unique_count, WideAdd(), source_count,
                    context->stream),
                  "reduce packed recoupling contributions");
  RECOUPLE_CUDA(cudaEventRecord(context->stage_events[3], context->stream),
                  "record recoupling reduction finish");
  uint32_t unique_count = 0;
  RECOUPLE_CUDA(cudaMemcpyAsync(&unique_count,
                                  context->recoupling_unique_count,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                  context->stream),
                  "download reduced key count before contraction");
  RECOUPLE_CUDA(cudaStreamSynchronize(context->stream),
                  "finish reduced key count download");
  if (unique_count == 0 || unique_count > source_count) {
    set_error(error, error_capacity,
              "reduced recoupling key count violates its bounds");
    return 1;
  }
  // A source uses exactly one (degree,free) schedule.  Therefore any fixed
  // output component receives at most unique_count*max_schedule contributions,
  // each at most p-1.  The checked product proves every u64 atomic sum exact.
  uint64_t row_sum_bound = 0;
  uint64_t expanded_bound = 0;
  if (!context->legacy_contraction &&
      (!checked_product_u64(unique_count,
                           context->max_plan_entries_per_degree_free,
                           static_cast<uint64_t>(context->prime - 1),
                           &row_sum_bound) ||
      !checked_product_u64(unique_count,
                           context->max_plan_entries_per_free, 1,
                           &expanded_bound))) {
    set_error(error, error_capacity,
              "flat CUDA F_X u64 accumulation bound exceeded");
    return 1;
  }
  if (context->legacy_contraction) {
    RECOUPLE_CUDA(cudaMemsetAsync(context->output_real, 0,
                                    kRows * sizeof(uint32_t), context->stream),
                    "clear legacy recoupled real output");
    RECOUPLE_CUDA(cudaMemsetAsync(context->output_imaginary, 0,
                                    kRows * sizeof(uint32_t), context->stream),
                    "clear legacy recoupled imaginary output");
  } else {
    RECOUPLE_CUDA(cudaMemsetAsync(context->output_real_wide, 0,
                                    kRows * sizeof(unsigned long long),
                                    context->stream),
                    "clear wide recoupled real output");
    RECOUPLE_CUDA(cudaMemsetAsync(context->output_imaginary_wide, 0,
                                    kRows * sizeof(unsigned long long),
                                    context->stream),
                    "clear wide recoupled imaginary output");
  }
  RECOUPLE_CUDA(cudaMemsetAsync(context->expanded, 0,
                                  sizeof(unsigned long long), context->stream),
                  "clear recoupled contribution counter");
  RECOUPLE_CUDA(cudaMemsetAsync(context->recoupling_nonzero_count, 0,
                                  sizeof(unsigned long long), context->stream),
                  "clear reduced nonzero counter");
  RECOUPLE_CUDA(cudaMemsetAsync(context->recoupling_overflow, 0,
                                  sizeof(uint32_t), context->stream),
                  "clear recoupling overflow flag");
  constexpr uint32_t threads = 128;
  if (context->legacy_contraction) {
    const uint32_t blocks = (unique_count + threads - 1) / threads;
    accumulate_reduced_legacy_kernel<<<blocks, threads, 0, context->stream>>>(
        context->recoupling_keys[0], context->recoupling_values[0],
        unique_count, context->gauge_offsets, context->gauges,
        context->targets, context->target_count, context->template_offsets,
        context->templates, context->prime, context->pow2_64_mod,
        context->output_real, context->output_imaginary, context->expanded,
        context->recoupling_nonzero_count, context->recoupling_overflow);
  } else {
    accumulate_reduced_plan_kernel<<<unique_count, threads, 0, context->stream>>>(
        context->recoupling_keys[0], context->recoupling_values[0],
        unique_count, context->plan_offsets, context->plan_entries,
        context->pair_salts, context->prime, context->pow2_64_mod,
        context->output_real_wide, context->output_imaginary_wide,
        context->expanded, context->recoupling_nonzero_count,
        context->recoupling_overflow);
  }
  RECOUPLE_CUDA(cudaGetLastError(),
                  "launch fused recoupling F_X contraction");
  if (!context->legacy_contraction) {
    constexpr uint32_t finalize_threads = 256;
    finalize_wide_rows_kernel<<<
        (kRows + finalize_threads - 1) / finalize_threads, finalize_threads, 0,
        context->stream>>>(context->output_real_wide,
                           context->output_imaginary_wide, context->prime,
                           context->output_real, context->output_imaginary);
    RECOUPLE_CUDA(cudaGetLastError(), "launch wide F_X row finalization");
  }
  RECOUPLE_CUDA(cudaEventRecord(context->stage_events[4], context->stream),
                  "record fused contraction finish");

  uint32_t overflow = 0;
  unsigned long long nonzero_count = 0;
  unsigned long long expanded = 0;
  RECOUPLE_CUDA(cudaMemcpyAsync(output_real, context->output_real,
                                  kRows * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, context->stream),
                  "download fused real output");
  RECOUPLE_CUDA(cudaMemcpyAsync(output_imaginary,
                                  context->output_imaginary,
                                  kRows * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, context->stream),
                  "download fused imaginary output");
  RECOUPLE_CUDA(cudaMemcpyAsync(&nonzero_count,
                                  context->recoupling_nonzero_count,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost, context->stream),
                  "download reduced nonzero count");
  RECOUPLE_CUDA(cudaMemcpyAsync(&expanded, context->expanded,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost, context->stream),
                  "download fused contribution count");
  RECOUPLE_CUDA(cudaMemcpyAsync(&overflow, context->recoupling_overflow,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                  context->stream),
                  "download recoupling overflow flag");
  RECOUPLE_CUDA(cudaEventRecord(context->stage_events[5], context->stream),
                  "record fused download finish");
  RECOUPLE_CUDA(cudaEventSynchronize(context->stage_events[5]),
                  "finish fused recoupling contraction");
  if (overflow != 0) {
    set_error(error, error_capacity,
              overflow == 1 ? "signed i128 recoupling reduction overflow"
                            : "reduced recoupling key violates its domain");
    return 1;
  }

  stats->terms_before_reduce = source_count;
  stats->keys_after_reduce = unique_count;
  stats->nonzero_terms_after_reduce = nonzero_count;
  stats->expanded_contributions = expanded;
  stats->buffer_high_water_bytes = context->buffer_high_water_bytes;
  RECOUPLE_CUDA(cudaEventElapsedTime(
                    &stats->upload_milliseconds, context->stage_events[0],
                    context->stage_events[1]),
                  "measure recoupling upload");
  RECOUPLE_CUDA(cudaEventElapsedTime(
                    &stats->sort_milliseconds, context->stage_events[1],
                    context->stage_events[2]),
                  "measure recoupling sort");
  RECOUPLE_CUDA(cudaEventElapsedTime(
                    &stats->reduce_milliseconds, context->stage_events[2],
                    context->stage_events[3]),
                  "measure recoupling reduction");
  RECOUPLE_CUDA(cudaEventElapsedTime(
                    &stats->contract_milliseconds, context->stage_events[3],
                    context->stage_events[4]),
                  "measure fused F_X contraction");
  RECOUPLE_CUDA(cudaEventElapsedTime(
                    &stats->download_milliseconds, context->stage_events[4],
                    context->stage_events[5]),
                  "measure fused output download");
  RECOUPLE_CUDA(cudaEventElapsedTime(
                    &stats->total_milliseconds, context->stage_events[0],
                    context->stage_events[5]),
                  "measure fused recoupling total");
#undef RECOUPLE_CUDA
  return 0;
}

int adynkra_fx_cuda_set_recoupling_hard_cap(void *opaque,
                                             uint64_t hard_cap_bytes,
                                             char *error,
                                             size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || hard_cap_bytes < context->allocated_bytes) {
    set_error(error, error_capacity,
              "CUDA recoupling hard cap is below fixed device allocation");
    return 1;
  }
  context->recoupling_hard_cap_bytes = hard_cap_bytes;
  return 0;
}

uint64_t adynkra_fx_cuda_resident_bytes(const void *opaque) {
  const Context *context = static_cast<const Context *>(opaque);
  return context == nullptr ? 0 : context->allocated_bytes;
}

uint64_t adynkra_fx_cuda_buffer_high_water_bytes(const void *opaque) {
  const Context *context = static_cast<const Context *>(opaque);
  return context == nullptr ? 0 : context->buffer_high_water_bytes;
}

uint64_t adynkra_fx_cuda_hard_cap_bytes(const void *opaque) {
  const Context *context = static_cast<const Context *>(opaque);
  return context == nullptr ? 0 : context->recoupling_hard_cap_bytes;
}

uint32_t adynkra_fx_cuda_p3_plan_entry_count(const void *opaque) {
  const Context *context = static_cast<const Context *>(opaque);
  return context == nullptr ? 0 : context->p3_plan_entry_count;
}

int adynkra_fx_cuda_reserve_recoupling(void *opaque, uint32_t source_count,
                                       char *error, size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || source_count == 0) {
    set_error(error, error_capacity,
              "invalid CUDA recoupling reservation input");
    return 1;
  }
  return ensure_recoupling_workspace(context, source_count, error,
                                     error_capacity)
             ? 0
             : 1;
}

int adynkra_fx_cuda_reserve_multicol(void *opaque, uint32_t unique_capacity,
                                     uint32_t active_columns, char *error,
                                     size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr ||
      !check_cuda(cudaSetDevice(context->device), error, error_capacity,
                  "select multi-column reservation device")) {
    if (context == nullptr) {
      set_error(error, error_capacity,
                "invalid multi-column CUDA reservation context");
    }
    return 1;
  }
  return ensure_multicol_workspace(context, unique_capacity, active_columns,
                                   error, error_capacity)
             ? 0
             : 1;
}

int adynkra_fx_cuda_accumulate_recoupled_multicol(
    void *opaque, const uint64_t *keys, const WideValue *key_major_values,
    uint32_t unique_count, uint32_t active_columns,
    uint32_t *row_major_output_real, uint32_t *row_major_output_imaginary,
    MultiColumnStats *stats, char *error, size_t error_capacity) {
  Context *context = static_cast<Context *>(opaque);
  if (context == nullptr || keys == nullptr || key_major_values == nullptr ||
      unique_count == 0 || active_columns == 0 || active_columns > 32 ||
      row_major_output_real == nullptr || row_major_output_imaginary == nullptr ||
      stats == nullptr || unique_count > context->multicol_key_capacity ||
      active_columns > context->multicol_lane_capacity ||
      context->multicol_keys == nullptr || context->multicol_values == nullptr ||
      context->multicol_output_real_wide == nullptr ||
      context->multicol_output_imaginary_wide == nullptr ||
      context->multicol_output_real == nullptr ||
      context->multicol_output_imaginary == nullptr ||
      context->multicol_expanded == nullptr ||
      context->multicol_overflow == nullptr) {
    set_error(error, error_capacity,
              "invalid or unreserved multi-column CUDA input");
    return 1;
  }
  std::memset(stats, 0, sizeof(*stats));
  stats->unique_count = unique_count;
  stats->active_columns = active_columns;
  uint64_t row_sum_bound = 0;
  uint64_t expanded_bound = 0;
  if (!checked_product_u64(unique_count,
                           context->max_plan_entries_per_degree_free,
                           static_cast<uint64_t>(context->prime - 1),
                           &row_sum_bound) ||
      !checked_product_u64(unique_count, context->max_plan_entries_per_free, 1,
                           &expanded_bound)) {
    set_error(error, error_capacity,
              "multi-column CUDA exact accumulation bound exceeded");
    return 1;
  }
  for (uint32_t source = 0; source < unique_count; ++source) {
    const uint64_t key = keys[source];
    const uint32_t metadata = static_cast<uint32_t>(key >> 32);
    const uint32_t pair_left = metadata & 15U;
    const uint32_t pair_right = (metadata >> 4) & 15U;
    const uint32_t free_spinor = (metadata >> 8) & 31U;
    const uint32_t mask = static_cast<uint32_t>(key);
    if ((source != 0 && keys[source - 1] >= key) || (metadata >> 13) != 0 ||
        pair_left > pair_right || pair_right >= 11 || free_spinor >= 32 ||
        __builtin_popcount(mask) != 12) {
      set_error(error, error_capacity,
                "multi-column CUDA keys are not canonical sorted degree-12 data");
      return 1;
    }
    bool any_nonzero = false;
    for (uint32_t column = 0; column < active_columns; ++column) {
      const WideValue &value = key_major_values[
          static_cast<uint64_t>(source) * active_columns + column];
      if (value.overflow != 0 || value.reserved != 0) {
        set_error(error, error_capacity,
                  "invalid multi-column CUDA coefficient encoding");
        return 1;
      }
      if (value.low != 0 || value.high != 0) {
        any_nonzero = true;
        ++stats->nonzero_terms[column];
      }
    }
    if (!any_nonzero) {
      set_error(error, error_capacity,
                "multi-column CUDA union contains an all-zero key");
      return 1;
    }
  }
  const size_t value_count =
      static_cast<size_t>(unique_count) * active_columns;
  const size_t output_count = static_cast<size_t>(kRows) * active_columns;
  if (!check_cuda(cudaSetDevice(context->device), error, error_capacity,
                  "select multi-column CUDA device")) {
    return 1;
  }
#define MULTICOL_CUDA(call, action)                                             \
  do {                                                                          \
    if (!check_cuda((call), error, error_capacity, (action))) return 1;          \
  } while (false)
  MULTICOL_CUDA(cudaEventRecord(context->stage_events[0], context->stream),
                "record multi-column start");
  MULTICOL_CUDA(cudaMemcpyAsync(context->multicol_keys, keys,
                                unique_count * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, context->stream),
                "upload multi-column keys");
  MULTICOL_CUDA(cudaMemcpyAsync(context->multicol_values, key_major_values,
                                value_count * sizeof(WideValue),
                                cudaMemcpyHostToDevice, context->stream),
                "upload multi-column values");
  MULTICOL_CUDA(cudaMemsetAsync(context->multicol_output_real_wide, 0,
                                output_count * sizeof(unsigned long long),
                                context->stream),
                "clear multi-column wide real rows");
  MULTICOL_CUDA(cudaMemsetAsync(context->multicol_output_imaginary_wide, 0,
                                output_count * sizeof(unsigned long long),
                                context->stream),
                "clear multi-column wide imaginary rows");
  MULTICOL_CUDA(cudaMemsetAsync(context->multicol_expanded, 0,
                                active_columns * sizeof(unsigned long long),
                                context->stream),
                "clear multi-column expansion counters");
  MULTICOL_CUDA(cudaMemsetAsync(context->multicol_overflow, 0, sizeof(uint32_t),
                                context->stream),
                "clear multi-column overflow flag");
  MULTICOL_CUDA(cudaEventRecord(context->stage_events[1], context->stream),
                "record multi-column upload finish");
  constexpr uint32_t threads = 128;
  constexpr uint32_t rows_per_degree_pair = kSectors * kSeeds * kBuckets;
  const size_t shared_row_bytes = static_cast<size_t>(rows_per_degree_pair) *
                                  2 * sizeof(unsigned long long);
  const dim3 contraction_grid(unique_count, kGaugeDegrees);
  accumulate_reduced_plan_multicol_shared_kernel<<<
      contraction_grid, threads, shared_row_bytes, context->stream>>>(
      context->multicol_keys, context->multicol_values, unique_count,
      active_columns, context->plan_offsets, context->plan_entries,
      context->pair_salts, context->prime, context->pow2_64_mod,
      context->multicol_output_real_wide,
      context->multicol_output_imaginary_wide, context->multicol_expanded,
      context->multicol_overflow);
  MULTICOL_CUDA(cudaGetLastError(),
                "launch multi-column CUDA contraction");
  MULTICOL_CUDA(cudaEventRecord(context->stage_events[2], context->stream),
                "record multi-column contraction finish");
  constexpr uint32_t finalize_threads = 256;
  const uint32_t finalize_blocks = static_cast<uint32_t>(
      (output_count + finalize_threads - 1) / finalize_threads);
  finalize_wide_multicol_kernel<<<finalize_blocks, finalize_threads, 0,
                                  context->stream>>>(
      context->multicol_output_real_wide,
      context->multicol_output_imaginary_wide, output_count, context->prime,
      context->multicol_output_real, context->multicol_output_imaginary);
  MULTICOL_CUDA(cudaGetLastError(),
                "launch multi-column CUDA row finalization");
  MULTICOL_CUDA(cudaEventRecord(context->stage_events[3], context->stream),
                "record multi-column finalization finish");
  MULTICOL_CUDA(cudaMemcpyAsync(row_major_output_real,
                                context->multicol_output_real,
                                output_count * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, context->stream),
                "download multi-column real rows");
  MULTICOL_CUDA(cudaMemcpyAsync(row_major_output_imaginary,
                                context->multicol_output_imaginary,
                                output_count * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, context->stream),
                "download multi-column imaginary rows");
  MULTICOL_CUDA(cudaMemcpyAsync(stats->expanded_contributions,
                                context->multicol_expanded,
                                active_columns * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost, context->stream),
                "download multi-column expansion counters");
  uint32_t overflow = 0;
  MULTICOL_CUDA(cudaMemcpyAsync(&overflow, context->multicol_overflow,
                                sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                context->stream),
                "download multi-column overflow flag");
  MULTICOL_CUDA(cudaEventRecord(context->stage_events[4], context->stream),
                "record multi-column download finish");
  MULTICOL_CUDA(cudaEventSynchronize(context->stage_events[4]),
                "finish multi-column CUDA batch");
  MULTICOL_CUDA(cudaEventElapsedTime(&stats->upload_milliseconds,
                                     context->stage_events[0],
                                     context->stage_events[1]),
                "measure multi-column upload");
  MULTICOL_CUDA(cudaEventElapsedTime(&stats->contract_milliseconds,
                                     context->stage_events[1],
                                     context->stage_events[2]),
                "measure multi-column contraction");
  MULTICOL_CUDA(cudaEventElapsedTime(&stats->finalize_milliseconds,
                                     context->stage_events[2],
                                     context->stage_events[3]),
                "measure multi-column finalization");
  MULTICOL_CUDA(cudaEventElapsedTime(&stats->download_milliseconds,
                                     context->stage_events[3],
                                     context->stage_events[4]),
                "measure multi-column download");
  MULTICOL_CUDA(cudaEventElapsedTime(&stats->total_milliseconds,
                                     context->stage_events[0],
                                     context->stage_events[4]),
                "measure multi-column total");
#undef MULTICOL_CUDA
  if (overflow != 0) {
    set_error(error, error_capacity,
              "multi-column CUDA kernel rejected canonical input");
    return 1;
  }
  stats->resident_bytes = context->allocated_bytes;
  stats->buffer_high_water_bytes = context->buffer_high_water_bytes;
  stats->device_hard_cap_bytes = context->recoupling_hard_cap_bytes;
  return 0;
}

void adynkra_fx_cuda_destroy(void *opaque) {
  destroy(static_cast<Context *>(opaque));
}

int adynkra_fx_cuda_device_name(int device, char *name, size_t capacity,
                                char *error, size_t error_capacity) {
  if (name == nullptr || capacity == 0) {
    set_error(error, error_capacity, "invalid CUDA device-name output");
    return 1;
  }
  cudaDeviceProp properties{};
  if (!check_cuda(cudaGetDeviceProperties(&properties, device), error,
                  error_capacity, "query CUDA device")) {
    return 1;
  }
  std::snprintf(name, capacity, "%s", properties.name);
  return 0;
}

void *adynkra_fx_cuda_sparse_context_create(int device, uint64_t hard_cap_bytes,
                                             char *error,
                                             size_t error_capacity) {
  if (hard_cap_bytes < 4 * sizeof(uint32_t) + sizeof(unsigned long long)) {
    set_error(error, error_capacity,
              "persistent sparse hard cap is below fixed allocation");
    return nullptr;
  }
  if (!check_cuda(cudaSetDevice(device), error, error_capacity,
                  "select persistent sparse CUDA device")) {
    return nullptr;
  }
  PersistentLoweringContext *context = new PersistentLoweringContext();
  context->device = device;
  context->hard_cap_bytes = hard_cap_bytes;
  if (!check_cuda(cudaStreamCreateWithFlags(&context->stream,
                                             cudaStreamNonBlocking),
                  error, error_capacity,
                  "create persistent sparse CUDA stream") ||
      !check_cuda(cudaMalloc(&context->unique_count, sizeof(uint32_t)), error,
                  error_capacity, "allocate persistent unique count") ||
      !check_cuda(cudaMalloc(&context->selected_count, sizeof(uint32_t)), error,
                  error_capacity, "allocate persistent selected count") ||
      !check_cuda(cudaMalloc(&context->overflow, sizeof(uint32_t)), error,
                  error_capacity, "allocate persistent overflow flag") ||
      !check_cuda(cudaMalloc(&context->max_abs,
                             sizeof(unsigned long long)),
                  error, error_capacity,
                  "allocate persistent maximum coefficient")) {
    destroy_persistent_context(context);
    return nullptr;
  }
  for (cudaEvent_t &event : context->events) {
    if (!check_cuda(cudaEventCreate(&event), error, error_capacity,
                    "create persistent sparse timing event")) {
      destroy_persistent_context(context);
      return nullptr;
    }
  }
  context->allocated_bytes =
      3 * sizeof(uint32_t) + sizeof(unsigned long long);
  context->high_water_bytes = context->allocated_bytes;
  return context;
}

void adynkra_fx_cuda_sparse_context_destroy(void *opaque) {
  PersistentLoweringContext *context =
      static_cast<PersistentLoweringContext *>(opaque);
  if (context == nullptr) return;
  if (context->live_handle_count != 0) {
    context->destroy_requested = true;
    return;
  }
  destroy_persistent_context(context);
}

uint64_t adynkra_fx_cuda_sparse_resident_bytes(const void *opaque) {
  const PersistentLoweringContext *context =
      static_cast<const PersistentLoweringContext *>(opaque);
  if (context == nullptr ||
      context->live_handle_bytes > UINT64_MAX - context->allocated_bytes) {
    return UINT64_MAX;
  }
  return context->allocated_bytes + context->live_handle_bytes;
}

void *adynkra_fx_cuda_sparse_handle_upload(
    void *opaque, const SparseEntry *entries, uint32_t count, char *error,
    size_t error_capacity) {
  PersistentLoweringContext *context =
      static_cast<PersistentLoweringContext *>(opaque);
  if (context == nullptr || entries == nullptr || count == 0 ||
      count > UINT32_MAX / 13U) {
    set_error(error, error_capacity,
              "invalid persistent sparse handle upload");
    return nullptr;
  }
  if (!check_cuda(cudaSetDevice(context->device), error, error_capacity,
                  "select persistent sparse upload device")) {
    return nullptr;
  }
  uint64_t max_abs = 0;
  for (uint32_t index = 0; index < count; ++index) {
    const uint32_t free_spinor = static_cast<uint32_t>(entries[index].key >> 32);
    const uint32_t mask = static_cast<uint32_t>(entries[index].key);
    if (free_spinor >= 32 || __builtin_popcount(mask) != 12 ||
        entries[index].value == 0 || entries[index].value == INT64_MIN ||
        (index != 0 && entries[index - 1].key >= entries[index].key)) {
      set_error(error, error_capacity,
                "persistent sparse upload is not canonical degree-12 data");
      return nullptr;
    }
    const uint64_t magnitude =
        entries[index].value < 0
            ? static_cast<uint64_t>(-entries[index].value)
            : static_cast<uint64_t>(entries[index].value);
    max_abs = max(max_abs, magnitude);
  }
  const uint64_t bytes = static_cast<uint64_t>(count) * sizeof(SparseEntry);
  if (!persistent_growth_allowed(context, bytes, error, error_capacity)) {
    return nullptr;
  }
  PersistentSparseHandle *handle = new PersistentSparseHandle();
  handle->device = context->device;
  handle->owner = context;
  handle->count = count;
  handle->max_abs_coefficient = max_abs;
  if (!check_cuda(cudaMalloc(&handle->entries, static_cast<size_t>(bytes)),
                  error, error_capacity,
                  "allocate immutable persistent sparse handle") ||
      !check_cuda(cudaMemcpyAsync(handle->entries, entries,
                                  static_cast<size_t>(bytes),
                                  cudaMemcpyHostToDevice, context->stream),
                  error, error_capacity,
                  "upload immutable persistent sparse handle") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish persistent sparse handle upload")) {
    cudaFree(handle->entries);
    delete handle;
    return nullptr;
  }
  context->live_handle_bytes += bytes;
  ++context->live_handle_count;
  context->high_water_bytes = max(
      context->high_water_bytes,
      context->allocated_bytes + context->live_handle_bytes);
  return handle;
}

void adynkra_fx_cuda_sparse_handle_destroy(void *opaque) {
  PersistentSparseHandle *handle =
      static_cast<PersistentSparseHandle *>(opaque);
  if (handle == nullptr) return;
  cudaSetDevice(handle->device);
  PersistentLoweringContext *owner = handle->owner;
  if (owner != nullptr) {
    const uint64_t bytes =
        static_cast<uint64_t>(handle->count) * sizeof(SparseEntry);
    owner->live_handle_bytes -= bytes;
    --owner->live_handle_count;
  }
  cudaFree(handle->entries);
  delete handle;
  if (owner != nullptr && owner->live_handle_count == 0 &&
      owner->destroy_requested) {
    destroy_persistent_context(owner);
  }
}

uint32_t adynkra_fx_cuda_sparse_handle_count(const void *opaque) {
  const PersistentSparseHandle *handle =
      static_cast<const PersistentSparseHandle *>(opaque);
  return handle == nullptr ? 0 : handle->count;
}

uint64_t adynkra_fx_cuda_sparse_handle_max_abs(const void *opaque) {
  const PersistentSparseHandle *handle =
      static_cast<const PersistentSparseHandle *>(opaque);
  return handle == nullptr ? 0 : handle->max_abs_coefficient;
}

int adynkra_fx_cuda_sparse_handle_download_range(
    void *context_opaque, const void *handle_opaque, uint32_t start,
    SparseEntry *entries, uint32_t capacity, char *error,
    size_t error_capacity) {
  PersistentLoweringContext *context =
      static_cast<PersistentLoweringContext *>(context_opaque);
  const PersistentSparseHandle *handle =
      static_cast<const PersistentSparseHandle *>(handle_opaque);
  if (context == nullptr || handle == nullptr || handle->owner != context ||
      handle->device != context->device || start > handle->count ||
      capacity > handle->count - start ||
      (capacity != 0 && entries == nullptr)) {
    set_error(error, error_capacity,
              "invalid persistent sparse ranged download input");
    return 1;
  }
  if (capacity == 0) return 0;
  if (!check_cuda(cudaSetDevice(context->device), error, error_capacity,
                  "select persistent sparse ranged-download device") ||
      !check_cuda(cudaMemcpyAsync(
                      entries, handle->entries + start,
                      static_cast<size_t>(capacity) * sizeof(SparseEntry),
                      cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity,
                  "download immutable persistent sparse handle range") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish persistent sparse ranged download")) {
    return 1;
  }
  return 0;
}

int adynkra_fx_cuda_sparse_handle_download(
    void *context_opaque, const void *handle_opaque, SparseEntry *entries,
    uint32_t capacity, char *error, size_t error_capacity) {
  PersistentLoweringContext *context =
      static_cast<PersistentLoweringContext *>(context_opaque);
  const PersistentSparseHandle *handle =
      static_cast<const PersistentSparseHandle *>(handle_opaque);
  if (context == nullptr || handle == nullptr || handle->owner != context ||
      handle->device != context->device ||
      capacity < handle->count || (handle->count != 0 && entries == nullptr)) {
    set_error(error, error_capacity,
              "invalid persistent sparse handle download");
    return 1;
  }
  if (handle->count == 0) return 0;
  if (!check_cuda(cudaSetDevice(context->device), error, error_capacity,
                  "select persistent sparse download device") ||
      !check_cuda(cudaMemcpyAsync(
                      entries, handle->entries,
                      static_cast<size_t>(handle->count) * sizeof(SparseEntry),
                      cudaMemcpyDeviceToHost, context->stream),
                  error, error_capacity,
                  "download immutable persistent sparse handle") ||
      !check_cuda(cudaStreamSynchronize(context->stream), error, error_capacity,
                  "finish persistent sparse handle download")) {
    return 1;
  }
  return 0;
}

void *adynkra_fx_cuda_sparse_handle_lower(
    void *context_opaque, const void *handle_opaque, uint32_t root,
    SparseLoweringStats *stats, char *error, size_t error_capacity) {
  PersistentLoweringContext *context =
      static_cast<PersistentLoweringContext *>(context_opaque);
  const PersistentSparseHandle *input =
      static_cast<const PersistentSparseHandle *>(handle_opaque);
  if (context == nullptr || input == nullptr || stats == nullptr || root >= 5 ||
      input->owner != context || input->device != context->device) {
    set_error(error, error_capacity,
              "invalid persistent sparse lowering input");
    return nullptr;
  }
  std::memset(stats, 0, sizeof(*stats));
  stats->input_count = input->count;
  if (input->count == 0) {
    PersistentSparseHandle *empty = new PersistentSparseHandle();
    empty->device = context->device;
    empty->owner = context;
    ++context->live_handle_count;
    stats->scratch_high_water_bytes = context->high_water_bytes;
    return empty;
  }
  if (input->count > UINT32_MAX / 13U) {
    set_error(error, error_capacity,
              "persistent sparse input exceeds u32 expansion bound");
    return nullptr;
  }
  if (input->max_abs_coefficient >
      static_cast<uint64_t>(INT64_MAX) / 13ULL) {
    set_error(error, error_capacity,
              "persistent sparse root coefficient bound exceeded");
    return nullptr;
  }
  if (!check_cuda(cudaSetDevice(context->device), error, error_capacity,
                  "select persistent sparse lowering device") ||
      !ensure_persistent_sources(context, input->count, error,
                                 error_capacity)) {
    return nullptr;
  }
  size_t scan_bytes = 0;
  if (!check_cuda(cub::DeviceScan::ExclusiveSum(
                      nullptr, scan_bytes, context->counts, context->offsets,
                      input->count, context->stream),
                  error, error_capacity, "size persistent sparse scan") ||
      !ensure_persistent_cub(context, scan_bytes, error, error_capacity)) {
    return nullptr;
  }

#define PERSISTENT_CUDA(call, action)                                            \
  do {                                                                           \
    if (!check_cuda((call), error, error_capacity, (action))) {                  \
      return nullptr;                                                            \
    }                                                                            \
  } while (false)

  constexpr uint32_t threads = 256;
  const uint32_t source_blocks = (input->count - 1) / threads + 1;
  PERSISTENT_CUDA(cudaEventRecord(context->events[0], context->stream),
                  "record persistent sparse start");
  count_sparse_lowering_kernel<<<source_blocks, threads, 0, context->stream>>>(
      input->entries, input->count, root, context->counts);
  PERSISTENT_CUDA(cudaGetLastError(), "launch persistent sparse count");
  PERSISTENT_CUDA(cudaEventRecord(context->events[1], context->stream),
                  "record persistent sparse count finish");
  PERSISTENT_CUDA(cub::DeviceScan::ExclusiveSum(
                      context->cub_temporary, scan_bytes, context->counts,
                      context->offsets, input->count, context->stream),
                  "scan persistent sparse counts");
  PERSISTENT_CUDA(cudaMemsetAsync(context->overflow, 0, sizeof(uint32_t),
                                  context->stream),
                  "clear persistent sparse overflow");
  finish_sparse_offsets_kernel<<<1, 1, 0, context->stream>>>(
      context->counts, input->count, context->offsets, context->overflow);
  PERSISTENT_CUDA(cudaGetLastError(), "finish persistent sparse offsets");
  PERSISTENT_CUDA(cudaEventRecord(context->events[2], context->stream),
                  "record persistent sparse scan finish");
  uint32_t expanded_count = 0;
  uint32_t overflow = 0;
  PERSISTENT_CUDA(cudaMemcpyAsync(&expanded_count,
                                  &context->offsets[input->count],
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                  context->stream),
                  "download persistent expanded count");
  PERSISTENT_CUDA(cudaMemcpyAsync(&overflow, context->overflow,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                  context->stream),
                  "download persistent sparse overflow");
  PERSISTENT_CUDA(cudaStreamSynchronize(context->stream),
                  "finish persistent sparse scan");
  if (overflow != 0) {
    set_error(error, error_capacity,
              "persistent sparse expansion count overflow");
    return nullptr;
  }
  stats->expanded_count = expanded_count;
  if (expanded_count == 0) {
    PersistentSparseHandle *empty = new PersistentSparseHandle();
    empty->device = context->device;
    empty->owner = context;
    ++context->live_handle_count;
    stats->scratch_high_water_bytes = context->high_water_bytes;
    if (!check_cuda(cudaEventElapsedTime(&stats->count_milliseconds,
                                         context->events[0],
                                         context->events[1]),
                    error, error_capacity,
                    "measure zero persistent sparse count") ||
        !check_cuda(cudaEventElapsedTime(&stats->scan_milliseconds,
                                         context->events[1],
                                         context->events[2]),
                    error, error_capacity,
                    "measure zero persistent sparse scan") ||
        !check_cuda(cudaEventElapsedTime(&stats->total_milliseconds,
                                         context->events[0],
                                         context->events[2]),
                    error, error_capacity,
                    "measure zero persistent sparse total")) {
      --context->live_handle_count;
      delete empty;
      return nullptr;
    }
    return empty;
  }
  if (!ensure_persistent_expanded(context, expanded_count, error,
                                  error_capacity)) {
    return nullptr;
  }
  size_t sort_bytes = 0;
  size_t reduce_bytes = 0;
  size_t select_bytes = 0;
  if (!check_cuda(cub::DeviceRadixSort::SortPairs(
                      nullptr, sort_bytes, context->keys[0], context->keys[1],
                      context->values[0], context->values[1], expanded_count,
                      0, 37, context->stream),
                  error, error_capacity, "size persistent sparse sort") ||
      !check_cuda(cub::DeviceReduce::ReduceByKey(
                      nullptr, reduce_bytes, context->keys[1], context->keys[0],
                      context->values[1], context->values[0],
                      context->unique_count, cub::Sum(), expanded_count,
                      context->stream),
                  error, error_capacity, "size persistent sparse reduction") ||
      !check_cuda(cub::DeviceSelect::Flagged(
                      nullptr, select_bytes, context->reduced_entries,
                      context->nonzero_flags, context->selected_entries,
                      context->selected_count, expanded_count, context->stream),
                  error, error_capacity, "size persistent sparse selection") ||
      !ensure_persistent_cub(context, max(scan_bytes, max(sort_bytes,
                                                          max(reduce_bytes,
                                                              select_bytes))),
                             error, error_capacity)) {
    return nullptr;
  }
  emit_sparse_lowering_kernel<<<source_blocks, threads, 0, context->stream>>>(
      input->entries, input->count, root, context->offsets, context->keys[0],
      context->values[0]);
  PERSISTENT_CUDA(cudaGetLastError(), "launch persistent sparse emit");
  PERSISTENT_CUDA(cudaEventRecord(context->events[3], context->stream),
                  "record persistent sparse emit finish");
  PERSISTENT_CUDA(cub::DeviceRadixSort::SortPairs(
                      context->cub_temporary, sort_bytes, context->keys[0],
                      context->keys[1], context->values[0], context->values[1],
                      expanded_count, 0, 37, context->stream),
                  "sort persistent sparse contributions");
  PERSISTENT_CUDA(cudaEventRecord(context->events[4], context->stream),
                  "record persistent sparse sort finish");
  PERSISTENT_CUDA(cub::DeviceReduce::ReduceByKey(
                      context->cub_temporary, reduce_bytes, context->keys[1],
                      context->keys[0], context->values[1], context->values[0],
                      context->unique_count, cub::Sum(), expanded_count,
                      context->stream),
                  "reduce persistent sparse contributions");
  PERSISTENT_CUDA(cudaEventRecord(context->events[5], context->stream),
                  "record persistent sparse reduction finish");
  uint32_t reduced_count = 0;
  PERSISTENT_CUDA(cudaMemcpyAsync(&reduced_count, context->unique_count,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                  context->stream),
                  "download persistent reduced count");
  PERSISTENT_CUDA(cudaStreamSynchronize(context->stream),
                  "finish persistent sparse reduction");
  if (reduced_count == 0 || reduced_count > expanded_count) {
    set_error(error, error_capacity,
              "persistent sparse reduced count violates bounds");
    return nullptr;
  }
  stats->reduced_count = reduced_count;
  PERSISTENT_CUDA(cudaMemsetAsync(context->max_abs, 0,
                                  sizeof(unsigned long long), context->stream),
                  "clear persistent maximum coefficient");
  pack_sparse_nonzero_kernel<<<
      (reduced_count - 1) / threads + 1, threads, 0, context->stream>>>(
      context->keys[0], context->values[0], reduced_count,
      context->reduced_entries, context->nonzero_flags, context->max_abs);
  PERSISTENT_CUDA(cudaGetLastError(), "pack persistent nonzero flags");
  PERSISTENT_CUDA(cub::DeviceSelect::Flagged(
                      context->cub_temporary, select_bytes,
                      context->reduced_entries, context->nonzero_flags,
                      context->selected_entries, context->selected_count,
                      reduced_count, context->stream),
                  "select persistent nonzero entries");
  PERSISTENT_CUDA(cudaEventRecord(context->events[6], context->stream),
                  "record persistent sparse selection finish");
  uint32_t output_count = 0;
  unsigned long long max_abs = 0;
  PERSISTENT_CUDA(cudaMemcpyAsync(&output_count, context->selected_count,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                  context->stream),
                  "download persistent output count");
  PERSISTENT_CUDA(cudaMemcpyAsync(&max_abs, context->max_abs,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost, context->stream),
                  "download persistent maximum coefficient");
  PERSISTENT_CUDA(cudaStreamSynchronize(context->stream),
                  "finish persistent sparse selection");
  if (output_count > reduced_count ||
      ((output_count == 0) != (max_abs == 0))) {
    set_error(error, error_capacity,
              "persistent sparse nonzero selection violates bounds");
    return nullptr;
  }
  const uint64_t handle_bytes =
      static_cast<uint64_t>(output_count) * sizeof(SparseEntry);
  if (handle_bytes != 0 &&
      !persistent_growth_allowed(context, handle_bytes, error,
                                 error_capacity)) {
    return nullptr;
  }
  PersistentSparseHandle *output = new PersistentSparseHandle();
  output->device = context->device;
  output->owner = context;
  output->count = output_count;
  output->max_abs_coefficient = max_abs;
  if (handle_bytes != 0) {
    if (!check_cuda(cudaMalloc(&output->entries,
                               static_cast<size_t>(handle_bytes)),
                    error, error_capacity,
                    "allocate exact persistent sparse output handle") ||
        !check_cuda(cudaMemcpyAsync(output->entries, context->selected_entries,
                                    static_cast<size_t>(handle_bytes),
                                    cudaMemcpyDeviceToDevice, context->stream),
                    error, error_capacity,
                    "seal exact persistent sparse output handle")) {
      cudaFree(output->entries);
      delete output;
      return nullptr;
    }
  }
  if (!check_cuda(cudaEventRecord(context->events[7], context->stream), error,
                  error_capacity, "record persistent sparse finish") ||
      !check_cuda(cudaEventSynchronize(context->events[7]), error,
                  error_capacity, "finish exact persistent sparse handle")) {
    cudaFree(output->entries);
    delete output;
    return nullptr;
  }
  stats->output_count = output_count;
  stats->scratch_high_water_bytes = context->high_water_bytes;
  stats->immutable_handle_bytes = handle_bytes;
  if (!check_cuda(cudaEventElapsedTime(&stats->count_milliseconds,
                                       context->events[0], context->events[1]),
                  error, error_capacity, "measure persistent sparse count") ||
      !check_cuda(cudaEventElapsedTime(&stats->scan_milliseconds,
                                       context->events[1], context->events[2]),
                  error, error_capacity, "measure persistent sparse scan") ||
      !check_cuda(cudaEventElapsedTime(&stats->emit_milliseconds,
                                       context->events[2], context->events[3]),
                  error, error_capacity, "measure persistent sparse emit") ||
      !check_cuda(cudaEventElapsedTime(&stats->sort_milliseconds,
                                       context->events[3], context->events[4]),
                  error, error_capacity, "measure persistent sparse sort") ||
      !check_cuda(cudaEventElapsedTime(&stats->reduce_milliseconds,
                                       context->events[4], context->events[5]),
                  error, error_capacity,
                  "measure persistent sparse reduction") ||
      !check_cuda(cudaEventElapsedTime(&stats->select_milliseconds,
                                       context->events[5], context->events[7]),
                  error, error_capacity,
                  "measure persistent sparse selection") ||
      !check_cuda(cudaEventElapsedTime(&stats->total_milliseconds,
                                       context->events[0], context->events[7]),
                  error, error_capacity, "measure persistent sparse total")) {
    cudaFree(output->entries);
    delete output;
    return nullptr;
  }
  context->live_handle_bytes += handle_bytes;
  ++context->live_handle_count;
  context->high_water_bytes = max(
      context->high_water_bytes,
      context->allocated_bytes + context->live_handle_bytes);
#undef PERSISTENT_CUDA
  return output;
}

int adynkra_fx_cuda_lower_sparse(
    int device, const uint64_t *source_keys, const int64_t *source_values,
    uint32_t source_count, uint32_t root, uint64_t *output_keys,
    int64_t *output_values, uint32_t output_capacity, uint32_t *output_count,
    float *kernel_milliseconds, char *error, size_t error_capacity) {
  if (source_keys == nullptr || source_values == nullptr || source_count == 0 ||
      source_count > UINT32_MAX / 13U || root >= 5 || output_keys == nullptr ||
      output_values == nullptr ||
      output_count == nullptr || kernel_milliseconds == nullptr ||
      output_capacity < source_count * 13ULL) {
    set_error(error, error_capacity, "invalid CUDA sparse-lowering input");
    return 1;
  }
  for (uint32_t index = 0; index < source_count; ++index) {
    const uint32_t free_spinor = static_cast<uint32_t>(source_keys[index] >> 32);
    const uint32_t mask = static_cast<uint32_t>(source_keys[index]);
    if (free_spinor >= 32 || __builtin_popcount(mask) != 12 ||
        source_values[index] == 0 || source_values[index] == INT64_MIN ||
        (index != 0 && source_keys[index - 1] >= source_keys[index])) {
      set_error(error, error_capacity,
                "sparse-lowering input is not canonical degree-12 data");
      return 1;
    }
    const uint64_t magnitude = source_values[index] < 0
                                   ? static_cast<uint64_t>(-source_values[index])
                                   : static_cast<uint64_t>(source_values[index]);
    if (magnitude > static_cast<uint64_t>(INT64_MAX) / 13ULL) {
      set_error(error, error_capacity,
                "sparse-lowering coefficient exceeds exact reduction bound");
      return 1;
    }
  }
  if (!check_cuda(cudaSetDevice(device), error, error_capacity,
                  "select sparse-lowering CUDA device")) {
    return 1;
  }
  uint32_t expanded_count = source_count * 13;
  uint64_t *device_source_keys = nullptr;
  int64_t *device_source_values = nullptr;
  uint64_t *expanded_keys = nullptr;
  int64_t *expanded_values = nullptr;
  uint64_t *sorted_keys = nullptr;
  int64_t *sorted_values = nullptr;
  uint64_t *unique_keys = nullptr;
  int64_t *unique_values = nullptr;
  uint32_t *device_unique_count = nullptr;
  void *sort_temporary = nullptr;
  void *reduce_temporary = nullptr;
  cudaEvent_t started = nullptr;
  cudaEvent_t finished = nullptr;
  int result = 1;
  size_t sort_bytes = 0;
  size_t reduce_bytes = 0;

  if (!check_cuda(cub::DeviceRadixSort::SortPairs(
                      nullptr, sort_bytes, expanded_keys, sorted_keys,
                      expanded_values, sorted_values, expanded_count),
                  error, error_capacity, "size sparse radix sort") ||
      !check_cuda(cub::DeviceReduce::ReduceByKey(
                      nullptr, reduce_bytes, sorted_keys, unique_keys,
                      sorted_values, unique_values, device_unique_count,
                      cub::Sum(), expanded_count),
                  error, error_capacity, "size sparse reduce-by-key")) {
    return 1;
  }
  size_t fixed_bytes = 0;
  size_t expanded_array_bytes = 0;
  if (!checked_multiply_size(source_count,
                             sizeof(uint64_t) + sizeof(int64_t),
                             &fixed_bytes) ||
      !checked_multiply_size(expanded_count,
                             3 * (sizeof(uint64_t) + sizeof(int64_t)),
                             &expanded_array_bytes) ||
      fixed_bytes > SIZE_MAX - expanded_array_bytes - sizeof(uint32_t) ||
      fixed_bytes + expanded_array_bytes + sizeof(uint32_t) >
          SIZE_MAX - sort_bytes ||
      fixed_bytes + expanded_array_bytes + sizeof(uint32_t) + sort_bytes >
          SIZE_MAX - reduce_bytes) {
    set_error(error, error_capacity, "sparse-lowering memory size overflow");
    return 1;
  }
  const size_t requested_bytes = fixed_bytes + expanded_array_bytes +
                                 sizeof(uint32_t) + sort_bytes + reduce_bytes;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (!check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), error,
                  error_capacity, "query sparse-lowering device memory") ||
      requested_bytes > free_bytes ||
      free_bytes - requested_bytes < kDeviceHeadroomBytes) {
    if (requested_bytes > free_bytes ||
        free_bytes - requested_bytes < kDeviceHeadroomBytes) {
      set_error(error, error_capacity,
                "insufficient CUDA memory for sparse lowering");
    }
    return 1;
  }

#define LOWER_CUDA(call, action)                                                \
  do {                                                                          \
    if (!check_cuda((call), error, error_capacity, (action))) {                 \
      goto cleanup;                                                              \
    }                                                                            \
  } while (false)

  LOWER_CUDA(cudaMalloc(&device_source_keys, source_count * sizeof(uint64_t)),
             "allocate sparse source keys");
  LOWER_CUDA(cudaMalloc(&device_source_values, source_count * sizeof(int64_t)),
             "allocate sparse source values");
  LOWER_CUDA(cudaMalloc(&expanded_keys, expanded_count * sizeof(uint64_t)),
             "allocate expanded sparse keys");
  LOWER_CUDA(cudaMalloc(&expanded_values, expanded_count * sizeof(int64_t)),
             "allocate expanded sparse values");
  LOWER_CUDA(cudaMalloc(&sorted_keys, expanded_count * sizeof(uint64_t)),
             "allocate sorted sparse keys");
  LOWER_CUDA(cudaMalloc(&sorted_values, expanded_count * sizeof(int64_t)),
             "allocate sorted sparse values");
  LOWER_CUDA(cudaMalloc(&unique_keys, expanded_count * sizeof(uint64_t)),
             "allocate reduced sparse keys");
  LOWER_CUDA(cudaMalloc(&unique_values, expanded_count * sizeof(int64_t)),
             "allocate reduced sparse values");
  LOWER_CUDA(cudaMalloc(&device_unique_count, sizeof(uint32_t)),
             "allocate sparse unique count");
  LOWER_CUDA(cudaEventCreate(&started), "create sparse start event");
  LOWER_CUDA(cudaEventCreate(&finished), "create sparse finish event");
  LOWER_CUDA(cudaMemcpy(device_source_keys, source_keys,
                        source_count * sizeof(uint64_t), cudaMemcpyHostToDevice),
             "upload sparse source keys");
  LOWER_CUDA(cudaMemcpy(device_source_values, source_values,
                        source_count * sizeof(int64_t), cudaMemcpyHostToDevice),
             "upload sparse source values");
  LOWER_CUDA(cudaMalloc(&sort_temporary, sort_bytes),
             "allocate sparse radix-sort workspace");
  LOWER_CUDA(cudaMalloc(&reduce_temporary, reduce_bytes),
             "allocate sparse reduction workspace");
  LOWER_CUDA(cudaEventRecord(started), "record sparse start event");
  {
    constexpr uint32_t threads = 256;
    uint32_t blocks = (source_count + threads - 1) / threads;
    expand_sparse_lowering_kernel<<<blocks, threads>>>(
        device_source_keys, device_source_values, source_count, root,
        expanded_keys, expanded_values);
  }
  LOWER_CUDA(cudaGetLastError(), "launch sparse-lowering expansion");
  LOWER_CUDA(cub::DeviceRadixSort::SortPairs(
                 sort_temporary, sort_bytes, expanded_keys, sorted_keys,
                 expanded_values, sorted_values, expanded_count),
             "sort sparse-lowering contributions");
  LOWER_CUDA(cub::DeviceReduce::ReduceByKey(
                 reduce_temporary, reduce_bytes, sorted_keys, unique_keys,
                 sorted_values, unique_values, device_unique_count, cub::Sum(),
                 expanded_count),
             "reduce sparse-lowering contributions");
  LOWER_CUDA(cudaEventRecord(finished), "record sparse finish event");
  LOWER_CUDA(cudaMemcpy(output_count, device_unique_count, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost),
             "download sparse unique count");
  LOWER_CUDA(cudaEventSynchronize(finished), "finish sparse lowering");
  if (*output_count > output_capacity) {
    set_error(error, error_capacity, "CUDA sparse-lowering output overflow");
    goto cleanup;
  }
  LOWER_CUDA(cudaMemcpy(output_keys, unique_keys,
                        *output_count * sizeof(uint64_t), cudaMemcpyDeviceToHost),
             "download sparse output keys");
  LOWER_CUDA(cudaMemcpy(output_values, unique_values,
                        *output_count * sizeof(int64_t), cudaMemcpyDeviceToHost),
             "download sparse output values");
  LOWER_CUDA(cudaEventElapsedTime(kernel_milliseconds, started, finished),
             "measure sparse lowering");
  result = 0;

cleanup:
  cudaFree(device_source_keys);
  cudaFree(device_source_values);
  cudaFree(expanded_keys);
  cudaFree(expanded_values);
  cudaFree(sorted_keys);
  cudaFree(sorted_values);
  cudaFree(unique_keys);
  cudaFree(unique_values);
  cudaFree(device_unique_count);
  cudaFree(sort_temporary);
  cudaFree(reduce_temporary);
  if (started != nullptr) cudaEventDestroy(started);
  if (finished != nullptr) cudaEventDestroy(finished);
#undef LOWER_CUDA
  return result;
}

}  // extern "C"
