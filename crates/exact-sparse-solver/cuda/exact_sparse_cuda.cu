#include <cuda_runtime.h>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>

namespace {

constexpr uint32_t PRIME = 2147483647u;
constexpr uint64_t PRIME_U64 = 2147483647ull;
constexpr uint32_t SIGN_BIT = 0x80000000u;
constexpr uint32_t INDEX_MASK = 0x7fffffffu;
constexpr uint32_t BLOCK_WIDTH = 32;
constexpr uint32_t THREADS = 256;
constexpr uint32_t REDUCTION_PARTS = 256;
constexpr uint32_t CG_ACTIVE = 0;
constexpr uint32_t CG_CONVERGED = 1;
constexpr uint32_t CG_BROKEN = 2;
constexpr int INVALID_ARGUMENT = -1;
constexpr int HOST_ALLOCATION_FAILED = -2;
constexpr int INPUT_NOT_UPLOADED = -3;
constexpr int CG_NOT_INITIALIZED = -4;

struct Operator {
    int device = 0;
    uint32_t rows = 0;
    uint32_t columns = 0;
    uint32_t nonzeros = 0;
    uint32_t active_block = 0;
    bool input_uploaded = false;
    cudaStream_t stream = nullptr;
    uint32_t* csr_offsets = nullptr;
    uint32_t* csr_entries = nullptr;
    uint32_t* transpose_offsets = nullptr;
    uint32_t* transpose_entries = nullptr;
    uint32_t* diagonal = nullptr;
    uint32_t* column_blocks[2] = {nullptr, nullptr};
    uint32_t* row_block = nullptr;
    bool cg_allocated = false;
    bool cg_initialized = false;
    uint64_t cg_rounds = 0;
    uint32_t* cg_border = nullptr;
    uint32_t* cg_x = nullptr;
    uint32_t* cg_r = nullptr;
    uint32_t* cg_p = nullptr;
    uint32_t* cg_q = nullptr;
    uint32_t* cg_rr = nullptr;
    uint32_t* cg_sigma = nullptr;
    uint32_t* cg_alpha = nullptr;
    uint32_t* cg_beta = nullptr;
    uint32_t* cg_status = nullptr;
    uint32_t* cg_lane_steps = nullptr;
    uint64_t* cg_transcript = nullptr;
    uint32_t* cg_partials = nullptr;
};

void release_cg_allocations(Operator* operator_handle) {
    cudaFree(operator_handle->cg_partials);
    cudaFree(operator_handle->cg_transcript);
    cudaFree(operator_handle->cg_lane_steps);
    cudaFree(operator_handle->cg_status);
    cudaFree(operator_handle->cg_beta);
    cudaFree(operator_handle->cg_alpha);
    cudaFree(operator_handle->cg_sigma);
    cudaFree(operator_handle->cg_rr);
    cudaFree(operator_handle->cg_q);
    cudaFree(operator_handle->cg_p);
    cudaFree(operator_handle->cg_r);
    cudaFree(operator_handle->cg_x);
    cudaFree(operator_handle->cg_border);
    operator_handle->cg_partials = nullptr;
    operator_handle->cg_transcript = nullptr;
    operator_handle->cg_lane_steps = nullptr;
    operator_handle->cg_status = nullptr;
    operator_handle->cg_beta = nullptr;
    operator_handle->cg_alpha = nullptr;
    operator_handle->cg_sigma = nullptr;
    operator_handle->cg_rr = nullptr;
    operator_handle->cg_q = nullptr;
    operator_handle->cg_p = nullptr;
    operator_handle->cg_r = nullptr;
    operator_handle->cg_x = nullptr;
    operator_handle->cg_border = nullptr;
    operator_handle->cg_allocated = false;
    operator_handle->cg_initialized = false;
}

void set_error(char* output, size_t capacity, const char* format, ...) {
    if (output == nullptr || capacity == 0) {
        return;
    }
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(output, capacity, format, arguments);
    va_end(arguments);
    output[capacity - 1] = '\0';
}

int cuda_failure(cudaError_t error, const char* operation, char* message, size_t capacity) {
    set_error(message, capacity, "%s: %s", operation, cudaGetErrorString(error));
    return static_cast<int>(error) + 1;
}

int invalid(const char* detail, char* message, size_t capacity) {
    set_error(message, capacity, "%s", detail);
    return INVALID_ARGUMENT;
}

bool checked_block_entries(uint32_t coordinates, size_t* entries) {
    constexpr size_t width = BLOCK_WIDTH;
    if (static_cast<size_t>(coordinates) > SIZE_MAX / width) {
        return false;
    }
    *entries = static_cast<size_t>(coordinates) * width;
    return true;
}

bool validate_offsets(const uint32_t* offsets, uint32_t coordinates, uint32_t nonzeros) {
    if (offsets == nullptr || offsets[0] != 0 || offsets[coordinates] != nonzeros) {
        return false;
    }
    for (uint32_t coordinate = 0; coordinate < coordinates; ++coordinate) {
        if (offsets[coordinate] > offsets[coordinate + 1]) {
            return false;
        }
    }
    return true;
}

bool validate_entries(const uint32_t* entries, uint32_t nonzeros, uint32_t bound) {
    if (nonzeros != 0 && entries == nullptr) {
        return false;
    }
    for (uint32_t entry = 0; entry < nonzeros; ++entry) {
        if ((entries[entry] & INDEX_MASK) >= bound) {
            return false;
        }
    }
    return true;
}

template <typename T>
cudaError_t allocate(T** pointer, size_t count) {
    const size_t allocation_count = count == 0 ? 1 : count;
    return cudaMalloc(reinterpret_cast<void**>(pointer), allocation_count * sizeof(T));
}

template <typename T>
cudaError_t upload(T* destination, const T* source, size_t count, cudaStream_t stream) {
    if (count == 0) {
        return cudaSuccess;
    }
    return cudaMemcpyAsync(
        destination, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
}

void release(Operator* operator_handle) {
    if (operator_handle == nullptr) {
        return;
    }
    cudaSetDevice(operator_handle->device);
    if (operator_handle->stream != nullptr) {
        cudaStreamSynchronize(operator_handle->stream);
    }
    release_cg_allocations(operator_handle);
    cudaFree(operator_handle->row_block);
    cudaFree(operator_handle->column_blocks[1]);
    cudaFree(operator_handle->column_blocks[0]);
    cudaFree(operator_handle->diagonal);
    cudaFree(operator_handle->transpose_entries);
    cudaFree(operator_handle->transpose_offsets);
    cudaFree(operator_handle->csr_entries);
    cudaFree(operator_handle->csr_offsets);
    if (operator_handle->stream != nullptr) {
        cudaStreamDestroy(operator_handle->stream);
    }
    delete operator_handle;
}

__host__ __device__ __forceinline__ uint32_t reduce_mersenne(uint64_t value) {
    value = (value & PRIME_U64) + (value >> 31);
    value = (value & PRIME_U64) + (value >> 31);
    return static_cast<uint32_t>(value >= PRIME_U64 ? value - PRIME_U64 : value);
}

__host__ __device__ __forceinline__ uint32_t multiply_mersenne(uint32_t left, uint32_t right) {
    return reduce_mersenne(static_cast<uint64_t>(left) * static_cast<uint64_t>(right));
}

__device__ __forceinline__ uint32_t add_field(uint32_t left, uint32_t right) {
    const uint64_t sum = static_cast<uint64_t>(left) + right;
    return static_cast<uint32_t>(sum >= PRIME_U64 ? sum - PRIME_U64 : sum);
}

__device__ __forceinline__ uint32_t subtract_field(uint32_t left, uint32_t right) {
    return left >= right ? left - right
                         : static_cast<uint32_t>(static_cast<uint64_t>(left) + PRIME_U64 - right);
}

__device__ uint32_t inverse_field(uint32_t value) {
    uint32_t result = 1;
    uint32_t base = value;
    uint32_t exponent = PRIME - 2;
    while (exponent != 0) {
        if ((exponent & 1u) != 0) {
            result = multiply_mersenne(result, base);
        }
        base = multiply_mersenne(base, base);
        exponent >>= 1;
    }
    return result;
}

__device__ __forceinline__ uint64_t transcript_mix(uint64_t digest, uint64_t value) {
    digest ^= value;
    return digest * 0x100000001b3ull;
}

__global__ void csr_diagonal_block32(
    uint32_t rows,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ entries,
    const uint32_t* __restrict__ diagonal,
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output) {
    const uint32_t warps_per_block = blockDim.x / BLOCK_WIDTH;
    const uint32_t row = blockIdx.x * warps_per_block + threadIdx.x / BLOCK_WIDTH;
    const uint32_t lane = threadIdx.x & (BLOCK_WIDTH - 1);
    if (row >= rows) {
        return;
    }

    uint64_t accumulator = 0;
    for (uint32_t entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
        const uint32_t packed = entries[entry];
        const uint32_t value = input[static_cast<size_t>(packed & INDEX_MASK) * BLOCK_WIDTH + lane];
        accumulator += (packed & SIGN_BIT) != 0 ? static_cast<uint64_t>(PRIME - value)
                                                : static_cast<uint64_t>(value);
    }
    const uint32_t reduced = reduce_mersenne(accumulator);
    output[static_cast<size_t>(row) * BLOCK_WIDTH + lane] =
        multiply_mersenne(reduced, diagonal[row]);
}

__global__ void transpose_block32(
    uint32_t columns,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ entries,
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output) {
    const uint32_t warps_per_block = blockDim.x / BLOCK_WIDTH;
    const uint32_t column = blockIdx.x * warps_per_block + threadIdx.x / BLOCK_WIDTH;
    const uint32_t lane = threadIdx.x & (BLOCK_WIDTH - 1);
    if (column >= columns) {
        return;
    }

    uint64_t accumulator = 0;
    for (uint32_t entry = offsets[column]; entry < offsets[column + 1]; ++entry) {
        const uint32_t packed = entries[entry];
        const uint32_t value = input[static_cast<size_t>(packed & INDEX_MASK) * BLOCK_WIDTH + lane];
        accumulator += (packed & SIGN_BIT) != 0 ? static_cast<uint64_t>(PRIME - value)
                                                : static_cast<uint64_t>(value);
    }
    output[static_cast<size_t>(column) * BLOCK_WIDTH + lane] =
        reduce_mersenne(accumulator);
}

// Each warp visits whole coordinate-major rows, so its 32 lane loads are
// coalesced. Warp 0 then reduces the fixed per-warp partial order lane by lane.
__global__ void cg_direction_partials(
    uint32_t coordinates,
    const uint32_t* __restrict__ status,
    const uint32_t* __restrict__ border,
    const uint32_t* __restrict__ p,
    const uint32_t* __restrict__ bp,
    uint32_t* __restrict__ partials) {
    const uint32_t lane = threadIdx.x & (BLOCK_WIDTH - 1);
    const uint32_t warp = threadIdx.x / BLOCK_WIDTH;
    const uint32_t warps = blockDim.x / BLOCK_WIDTH;
    const uint32_t part = blockIdx.x;
    const uint32_t partial_index = lane * REDUCTION_PARTS + part;
    const bool active = status[lane] == CG_ACTIVE;

    uint64_t border_p = 0;
    uint64_t p_bp = 0;
    if (active) {
        for (uint32_t coordinate = part * warps + warp;
             coordinate < coordinates;
             coordinate += REDUCTION_PARTS * warps) {
            const size_t index = static_cast<size_t>(coordinate) * BLOCK_WIDTH + lane;
            border_p += multiply_mersenne(border[index], p[index]);
            p_bp += multiply_mersenne(p[index], bp[index]);
        }
    }
    __shared__ uint64_t shared_border_p[THREADS];
    __shared__ uint64_t shared_p_bp[THREADS];
    shared_border_p[threadIdx.x] = border_p;
    shared_p_bp[threadIdx.x] = p_bp;
    __syncthreads();
    if (warp == 0) {
        uint64_t lane_border_p = 0;
        uint64_t lane_p_bp = 0;
        for (uint32_t source_warp = 0; source_warp < warps; ++source_warp) {
            const uint32_t source = source_warp * BLOCK_WIDTH + lane;
            lane_border_p += shared_border_p[source];
            lane_p_bp += shared_p_bp[source];
        }
        partials[partial_index] = active ? reduce_mersenne(lane_border_p) : 0;
        partials[BLOCK_WIDTH * REDUCTION_PARTS + partial_index] =
            active ? reduce_mersenne(lane_p_bp) : 0;
    }
}

__global__ void cg_finish_direction(
    uint32_t* __restrict__ status,
    const uint32_t* __restrict__ rr,
    uint32_t* __restrict__ sigma,
    uint32_t* __restrict__ alpha,
    uint64_t* __restrict__ transcript,
    const uint32_t* __restrict__ partials) {
    const uint32_t lane = threadIdx.x;
    if (lane >= BLOCK_WIDTH || status[lane] != CG_ACTIVE) {
        return;
    }
    uint32_t border_p = 0;
    uint32_t p_bp = 0;
    for (uint32_t part = 0; part < REDUCTION_PARTS; ++part) {
        const uint32_t index = lane * REDUCTION_PARTS + part;
        border_p = add_field(border_p, partials[index]);
        p_bp = add_field(
            p_bp, partials[BLOCK_WIDTH * REDUCTION_PARTS + index]);
    }
    sigma[lane] = border_p;
    const uint32_t p_cp = add_field(p_bp, multiply_mersenne(border_p, border_p));
    uint64_t digest = transcript[lane];
    digest = transcript_mix(digest, 0x444952454354494full);
    digest = transcript_mix(digest, rr[lane]);
    digest = transcript_mix(digest, border_p);
    digest = transcript_mix(digest, p_cp);
    if (rr[lane] == 0 || p_cp == 0) {
        status[lane] = CG_BROKEN;
        alpha[lane] = 0;
        transcript[lane] = transcript_mix(digest, CG_BROKEN);
        return;
    }
    alpha[lane] = multiply_mersenne(rr[lane], inverse_field(p_cp));
    transcript[lane] = transcript_mix(digest, alpha[lane]);
}

__global__ void cg_update_xr(
    size_t entries,
    const uint32_t* __restrict__ status,
    const uint32_t* __restrict__ sigma,
    const uint32_t* __restrict__ alpha,
    const uint32_t* __restrict__ border,
    uint32_t* __restrict__ x,
    uint32_t* __restrict__ r,
    const uint32_t* __restrict__ p,
    const uint32_t* __restrict__ bp) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= entries) {
        return;
    }
    const uint32_t lane = index & (BLOCK_WIDTH - 1);
    if (status[lane] != CG_ACTIVE) {
        return;
    }
    const uint32_t cp = add_field(
        bp[index], multiply_mersenne(border[index], sigma[lane]));
    x[index] = add_field(x[index], multiply_mersenne(alpha[lane], p[index]));
    r[index] = subtract_field(r[index], multiply_mersenne(alpha[lane], cp));
}

__global__ void cg_residual_partials(
    uint32_t coordinates,
    const uint32_t* __restrict__ status,
    const uint32_t* __restrict__ r,
    uint32_t* __restrict__ partials) {
    const uint32_t lane = threadIdx.x & (BLOCK_WIDTH - 1);
    const uint32_t warp = threadIdx.x / BLOCK_WIDTH;
    const uint32_t warps = blockDim.x / BLOCK_WIDTH;
    const uint32_t part = blockIdx.x;
    const uint32_t partial_index = lane * REDUCTION_PARTS + part;
    const bool active = status[lane] == CG_ACTIVE;

    uint64_t squared_norm = 0;
    uint32_t any_nonzero = 0;
    if (active) {
        for (uint32_t coordinate = part * warps + warp;
             coordinate < coordinates;
             coordinate += REDUCTION_PARTS * warps) {
            const uint32_t value = r[static_cast<size_t>(coordinate) * BLOCK_WIDTH + lane];
            squared_norm += multiply_mersenne(value, value);
            any_nonzero |= value;
        }
    }
    __shared__ uint64_t shared_squared_norm[THREADS];
    __shared__ uint32_t shared_nonzero[THREADS];
    shared_squared_norm[threadIdx.x] = squared_norm;
    shared_nonzero[threadIdx.x] = any_nonzero;
    __syncthreads();
    if (warp == 0) {
        uint64_t lane_squared_norm = 0;
        uint32_t lane_nonzero = 0;
        for (uint32_t source_warp = 0; source_warp < warps; ++source_warp) {
            const uint32_t source = source_warp * BLOCK_WIDTH + lane;
            lane_squared_norm += shared_squared_norm[source];
            lane_nonzero |= shared_nonzero[source];
        }
        partials[partial_index] = active ? reduce_mersenne(lane_squared_norm) : 0;
        partials[BLOCK_WIDTH * REDUCTION_PARTS + partial_index] =
            active ? lane_nonzero : 0;
    }
}

__global__ void cg_finish_residual(
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ rr,
    uint32_t* __restrict__ beta,
    uint32_t* __restrict__ lane_steps,
    uint64_t* __restrict__ transcript,
    const uint32_t* __restrict__ partials) {
    const uint32_t lane = threadIdx.x;
    if (lane >= BLOCK_WIDTH || status[lane] != CG_ACTIVE) {
        return;
    }
    uint32_t next_rr = 0;
    uint32_t any_nonzero = 0;
    for (uint32_t part = 0; part < REDUCTION_PARTS; ++part) {
        const uint32_t index = lane * REDUCTION_PARTS + part;
        next_rr = add_field(next_rr, partials[index]);
        any_nonzero |= partials[BLOCK_WIDTH * REDUCTION_PARTS + index];
    }
    ++lane_steps[lane];
    uint64_t digest = transcript[lane];
    digest = transcript_mix(digest, 0x524553494455414cull);
    digest = transcript_mix(digest, next_rr);
    digest = transcript_mix(digest, any_nonzero);
    digest = transcript_mix(digest, lane_steps[lane]);
    if (any_nonzero == 0) {
        status[lane] = CG_CONVERGED;
        rr[lane] = 0;
        beta[lane] = 0;
        transcript[lane] = transcript_mix(digest, CG_CONVERGED);
        return;
    }
    if (next_rr == 0) {
        status[lane] = CG_BROKEN;
        rr[lane] = 0;
        beta[lane] = 0;
        transcript[lane] = transcript_mix(digest, CG_BROKEN);
        return;
    }
    beta[lane] = multiply_mersenne(next_rr, inverse_field(rr[lane]));
    rr[lane] = next_rr;
    transcript[lane] = transcript_mix(digest, beta[lane]);
}

__global__ void cg_update_p(
    size_t entries,
    const uint32_t* __restrict__ status,
    const uint32_t* __restrict__ beta,
    const uint32_t* __restrict__ r,
    uint32_t* __restrict__ p) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= entries) {
        return;
    }
    const uint32_t lane = index & (BLOCK_WIDTH - 1);
    if (status[lane] == CG_ACTIVE) {
        p[index] = add_field(r[index], multiply_mersenne(beta[lane], p[index]));
    }
}

int set_device(Operator* operator_handle, char* message, size_t capacity) {
    const cudaError_t error = cudaSetDevice(operator_handle->device);
    return error == cudaSuccess ? 0 : cuda_failure(error, "cudaSetDevice", message, capacity);
}

int launch_operator_once(
    Operator* operator_handle,
    const uint32_t* input,
    uint32_t* output,
    char* message,
    size_t capacity) {
    const uint32_t warps_per_block = THREADS / BLOCK_WIDTH;
    const uint32_t csr_blocks =
        (operator_handle->rows + warps_per_block - 1) / warps_per_block;
    const uint32_t transpose_blocks =
        (operator_handle->columns + warps_per_block - 1) / warps_per_block;
    if (operator_handle->rows != 0) {
        csr_diagonal_block32<<<csr_blocks, THREADS, 0, operator_handle->stream>>>(
            operator_handle->rows,
            operator_handle->csr_offsets,
            operator_handle->csr_entries,
            operator_handle->diagonal,
            input,
            operator_handle->row_block);
        const cudaError_t error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            return cuda_failure(error, "launch csr_diagonal_block32", message, capacity);
        }
    }
    if (operator_handle->columns != 0) {
        transpose_block32<<<transpose_blocks, THREADS, 0, operator_handle->stream>>>(
            operator_handle->columns,
            operator_handle->transpose_offsets,
            operator_handle->transpose_entries,
            operator_handle->row_block,
            output);
        const cudaError_t error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            return cuda_failure(error, "launch transpose_block32", message, capacity);
        }
    }
    return 0;
}

int launch_steps(Operator* operator_handle, uint32_t steps, char* message, size_t capacity) {
    if (!operator_handle->input_uploaded) {
        set_error(message, capacity, "no input block has been uploaded");
        return INPUT_NOT_UPLOADED;
    }
    for (uint32_t step = 0; step < steps; ++step) {
        const uint32_t next = operator_handle->active_block ^ 1u;
        const int result = launch_operator_once(
            operator_handle,
            operator_handle->column_blocks[operator_handle->active_block],
            operator_handle->column_blocks[next],
            message,
            capacity);
        if (result != 0) {
            return result;
        }
        operator_handle->active_block = next;
    }
    const cudaError_t error = cudaStreamSynchronize(operator_handle->stream);
    return error == cudaSuccess ? 0
                                : cuda_failure(error, "synchronize exact sparse kernels", message, capacity);
}

int allocate_cg(Operator* operator_handle, char* message, size_t capacity) {
    if (operator_handle->cg_allocated) {
        return 0;
    }
    size_t entries = 0;
    if (!checked_block_entries(operator_handle->columns, &entries)) {
        return invalid("CG block32 buffer size overflow", message, capacity);
    }
    cudaError_t error = cudaSuccess;
#define CUDA_CG_ALLOC(operation, expression)                                                   \
    do {                                                                                        \
        error = (expression);                                                                   \
        if (error != cudaSuccess) {                                                             \
            const int code = cuda_failure(error, operation, message, capacity);                 \
            release_cg_allocations(operator_handle);                                            \
            return code;                                                                        \
        }                                                                                       \
    } while (false)
    CUDA_CG_ALLOC("allocate CG border", allocate(&operator_handle->cg_border, entries));
    CUDA_CG_ALLOC("allocate CG x", allocate(&operator_handle->cg_x, entries));
    CUDA_CG_ALLOC("allocate CG r", allocate(&operator_handle->cg_r, entries));
    CUDA_CG_ALLOC("allocate CG p", allocate(&operator_handle->cg_p, entries));
    CUDA_CG_ALLOC("allocate CG q", allocate(&operator_handle->cg_q, entries));
    CUDA_CG_ALLOC("allocate CG rr", allocate(&operator_handle->cg_rr, BLOCK_WIDTH));
    CUDA_CG_ALLOC("allocate CG sigma", allocate(&operator_handle->cg_sigma, BLOCK_WIDTH));
    CUDA_CG_ALLOC("allocate CG alpha", allocate(&operator_handle->cg_alpha, BLOCK_WIDTH));
    CUDA_CG_ALLOC("allocate CG beta", allocate(&operator_handle->cg_beta, BLOCK_WIDTH));
    CUDA_CG_ALLOC("allocate CG status", allocate(&operator_handle->cg_status, BLOCK_WIDTH));
    CUDA_CG_ALLOC("allocate CG lane steps", allocate(&operator_handle->cg_lane_steps, BLOCK_WIDTH));
    CUDA_CG_ALLOC("allocate CG transcript", allocate(&operator_handle->cg_transcript, BLOCK_WIDTH));
    CUDA_CG_ALLOC(
        "allocate CG reduction partials",
        allocate(&operator_handle->cg_partials, 2 * BLOCK_WIDTH * REDUCTION_PARTS));
#undef CUDA_CG_ALLOC
    operator_handle->cg_allocated = true;
    return 0;
}

int launch_cg_round(Operator* operator_handle, char* message, size_t capacity) {
    int result = launch_operator_once(
        operator_handle,
        operator_handle->cg_p,
        operator_handle->cg_q,
        message,
        capacity);
    if (result != 0) {
        return result;
    }
    const dim3 reduction_grid(REDUCTION_PARTS);
    cg_direction_partials<<<reduction_grid, THREADS, 0, operator_handle->stream>>>(
        operator_handle->columns,
        operator_handle->cg_status,
        operator_handle->cg_border,
        operator_handle->cg_p,
        operator_handle->cg_q,
        operator_handle->cg_partials);
    cg_finish_direction<<<1, BLOCK_WIDTH, 0, operator_handle->stream>>>(
        operator_handle->cg_status,
        operator_handle->cg_rr,
        operator_handle->cg_sigma,
        operator_handle->cg_alpha,
        operator_handle->cg_transcript,
        operator_handle->cg_partials);
    size_t entries = 0;
    checked_block_entries(operator_handle->columns, &entries);
    const uint32_t vector_blocks = static_cast<uint32_t>((entries + THREADS - 1) / THREADS);
    if (entries != 0) {
        cg_update_xr<<<vector_blocks, THREADS, 0, operator_handle->stream>>>(
            entries,
            operator_handle->cg_status,
            operator_handle->cg_sigma,
            operator_handle->cg_alpha,
            operator_handle->cg_border,
            operator_handle->cg_x,
            operator_handle->cg_r,
            operator_handle->cg_p,
            operator_handle->cg_q);
    }
    cg_residual_partials<<<reduction_grid, THREADS, 0, operator_handle->stream>>>(
        operator_handle->columns,
        operator_handle->cg_status,
        operator_handle->cg_r,
        operator_handle->cg_partials);
    cg_finish_residual<<<1, BLOCK_WIDTH, 0, operator_handle->stream>>>(
        operator_handle->cg_status,
        operator_handle->cg_rr,
        operator_handle->cg_beta,
        operator_handle->cg_lane_steps,
        operator_handle->cg_transcript,
        operator_handle->cg_partials);
    if (entries != 0) {
        cg_update_p<<<vector_blocks, THREADS, 0, operator_handle->stream>>>(
            entries,
            operator_handle->cg_status,
            operator_handle->cg_beta,
            operator_handle->cg_r,
            operator_handle->cg_p);
    }
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess
               ? 0
               : cuda_failure(error, "launch bordered CG kernels", message, capacity);
}

}  // namespace

extern "C" {

uint32_t adynkra_exact_cuda_abi_version() {
    return 2;
}

int adynkra_exact_cuda_create(
    int device,
    uint32_t rows,
    uint32_t columns,
    uint32_t nonzeros,
    const uint32_t* csr_offsets,
    const uint32_t* csr_entries,
    const uint32_t* transpose_offsets,
    const uint32_t* transpose_entries,
    const uint32_t* diagonal,
    Operator** output,
    char* message,
    size_t capacity) {
    if (output == nullptr) {
        return invalid("output operator pointer is null", message, capacity);
    }
    *output = nullptr;
    if (!validate_offsets(csr_offsets, rows, nonzeros)) {
        return invalid("invalid CSR offsets", message, capacity);
    }
    if (!validate_offsets(transpose_offsets, columns, nonzeros)) {
        return invalid("invalid transpose offsets", message, capacity);
    }
    if (!validate_entries(csr_entries, nonzeros, columns)) {
        return invalid("CSR entry index is outside the column dimension", message, capacity);
    }
    if (!validate_entries(transpose_entries, nonzeros, rows)) {
        return invalid("transpose entry index is outside the row dimension", message, capacity);
    }
    if (rows != 0 && diagonal == nullptr) {
        return invalid("diagonal pointer is null", message, capacity);
    }
    for (uint32_t row = 0; row < rows; ++row) {
        if (diagonal[row] == 0 || diagonal[row] >= PRIME) {
            return invalid("diagonal entry is not a nonzero canonical field element", message, capacity);
        }
    }

    size_t row_block_entries = 0;
    size_t column_block_entries = 0;
    if (!checked_block_entries(rows, &row_block_entries) ||
        !checked_block_entries(columns, &column_block_entries)) {
        return invalid("block32 buffer size overflow", message, capacity);
    }

    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) {
        return cuda_failure(error, "cudaSetDevice", message, capacity);
    }
    Operator* operator_handle = new (std::nothrow) Operator;
    if (operator_handle == nullptr) {
        set_error(message, capacity, "host allocation failed");
        return HOST_ALLOCATION_FAILED;
    }
    operator_handle->device = device;
    operator_handle->rows = rows;
    operator_handle->columns = columns;
    operator_handle->nonzeros = nonzeros;

#define CUDA_CREATE(operation, expression)                                                     \
    do {                                                                                        \
        error = (expression);                                                                   \
        if (error != cudaSuccess) {                                                             \
            const int code = cuda_failure(error, operation, message, capacity);                 \
            release(operator_handle);                                                           \
            return code;                                                                        \
        }                                                                                       \
    } while (false)

    CUDA_CREATE("cudaStreamCreateWithFlags",
                cudaStreamCreateWithFlags(&operator_handle->stream, cudaStreamNonBlocking));
    CUDA_CREATE("allocate CSR offsets", allocate(&operator_handle->csr_offsets, size_t(rows) + 1));
    CUDA_CREATE("allocate CSR entries", allocate(&operator_handle->csr_entries, nonzeros));
    CUDA_CREATE("allocate transpose offsets",
                allocate(&operator_handle->transpose_offsets, size_t(columns) + 1));
    CUDA_CREATE("allocate transpose entries", allocate(&operator_handle->transpose_entries, nonzeros));
    CUDA_CREATE("allocate diagonal", allocate(&operator_handle->diagonal, rows));
    CUDA_CREATE("allocate column block 0", allocate(&operator_handle->column_blocks[0], column_block_entries));
    CUDA_CREATE("allocate column block 1", allocate(&operator_handle->column_blocks[1], column_block_entries));
    CUDA_CREATE("allocate row block", allocate(&operator_handle->row_block, row_block_entries));
    CUDA_CREATE("upload CSR offsets",
                upload(operator_handle->csr_offsets, csr_offsets, size_t(rows) + 1, operator_handle->stream));
    CUDA_CREATE("upload CSR entries",
                upload(operator_handle->csr_entries, csr_entries, nonzeros, operator_handle->stream));
    CUDA_CREATE("upload transpose offsets",
                upload(operator_handle->transpose_offsets,
                       transpose_offsets,
                       size_t(columns) + 1,
                       operator_handle->stream));
    CUDA_CREATE("upload transpose entries",
                upload(operator_handle->transpose_entries,
                       transpose_entries,
                       nonzeros,
                       operator_handle->stream));
    CUDA_CREATE("upload diagonal",
                upload(operator_handle->diagonal, diagonal, rows, operator_handle->stream));
    CUDA_CREATE("synchronize matrix upload", cudaStreamSynchronize(operator_handle->stream));
#undef CUDA_CREATE

    *output = operator_handle;
    if (message != nullptr && capacity != 0) {
        message[0] = '\0';
    }
    return 0;
}

void adynkra_exact_cuda_destroy(Operator* operator_handle) {
    release(operator_handle);
}

int adynkra_exact_cuda_upload(
    Operator* operator_handle,
    const uint32_t* input,
    size_t input_entries,
    char* message,
    size_t capacity) {
    if (operator_handle == nullptr) {
        return invalid("operator pointer is null", message, capacity);
    }
    size_t expected = 0;
    checked_block_entries(operator_handle->columns, &expected);
    if (input_entries != expected || (input_entries != 0 && input == nullptr)) {
        return invalid("input block length does not match columns times 32", message, capacity);
    }
    int result = set_device(operator_handle, message, capacity);
    if (result != 0) {
        return result;
    }
    cudaError_t error = upload(
        operator_handle->column_blocks[0], input, input_entries, operator_handle->stream);
    if (error != cudaSuccess) {
        return cuda_failure(error, "upload input block", message, capacity);
    }
    error = cudaStreamSynchronize(operator_handle->stream);
    if (error != cudaSuccess) {
        return cuda_failure(error, "synchronize input upload", message, capacity);
    }
    operator_handle->active_block = 0;
    operator_handle->input_uploaded = true;
    return 0;
}

int adynkra_exact_cuda_apply_steps(
    Operator* operator_handle,
    uint32_t steps,
    char* message,
    size_t capacity) {
    if (operator_handle == nullptr) {
        return invalid("operator pointer is null", message, capacity);
    }
    int result = set_device(operator_handle, message, capacity);
    return result == 0 ? launch_steps(operator_handle, steps, message, capacity) : result;
}

int adynkra_exact_cuda_download(
    Operator* operator_handle,
    uint32_t* output,
    size_t output_entries,
    char* message,
    size_t capacity) {
    if (operator_handle == nullptr) {
        return invalid("operator pointer is null", message, capacity);
    }
    if (!operator_handle->input_uploaded) {
        set_error(message, capacity, "no input block has been uploaded");
        return INPUT_NOT_UPLOADED;
    }
    size_t expected = 0;
    checked_block_entries(operator_handle->columns, &expected);
    if (output_entries != expected || (output_entries != 0 && output == nullptr)) {
        return invalid("output block length does not match columns times 32", message, capacity);
    }
    int result = set_device(operator_handle, message, capacity);
    if (result != 0) {
        return result;
    }
    cudaError_t error = cudaSuccess;
    if (output_entries != 0) {
        error = cudaMemcpyAsync(output,
                                operator_handle->column_blocks[operator_handle->active_block],
                                output_entries * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost,
                                operator_handle->stream);
    }
    if (error != cudaSuccess) {
        return cuda_failure(error, "download output block", message, capacity);
    }
    error = cudaStreamSynchronize(operator_handle->stream);
    return error == cudaSuccess
               ? 0
               : cuda_failure(error, "synchronize output download", message, capacity);
}

int adynkra_exact_cuda_cg_initialize(
    Operator* operator_handle,
    const uint32_t* border,
    size_t block_entries,
    const uint32_t* initial_rr,
    char* message,
    size_t capacity) {
    if (operator_handle == nullptr || initial_rr == nullptr) {
        return invalid("CG initialize received a null pointer", message, capacity);
    }
    size_t expected = 0;
    checked_block_entries(operator_handle->columns, &expected);
    if (block_entries != expected || (block_entries != 0 && border == nullptr)) {
        return invalid("CG border length does not match columns times 32", message, capacity);
    }
    uint64_t norm_accumulators[BLOCK_WIDTH] = {};
    for (size_t index = 0; index < block_entries; ++index) {
        if (border[index] >= PRIME) {
            return invalid("CG border contains a noncanonical field element", message, capacity);
        }
        const uint32_t value = border[index];
        norm_accumulators[index & (BLOCK_WIDTH - 1)] += multiply_mersenne(value, value);
    }
    uint32_t status[BLOCK_WIDTH];
    uint64_t transcript[BLOCK_WIDTH];
    for (uint32_t lane = 0; lane < BLOCK_WIDTH; ++lane) {
        if (initial_rr[lane] >= PRIME) {
            return invalid("CG initial squared norm is noncanonical", message, capacity);
        }
        if (initial_rr[lane] != reduce_mersenne(norm_accumulators[lane])) {
            return invalid("CG initial squared norm does not equal u^T u", message, capacity);
        }
        status[lane] = initial_rr[lane] == 0 ? CG_BROKEN : CG_ACTIVE;
        transcript[lane] = 0xcbf29ce484222325ull ^ lane;
    }
    operator_handle->cg_initialized = false;
    int result = set_device(operator_handle, message, capacity);
    if (result != 0) {
        return result;
    }
    result = allocate_cg(operator_handle, message, capacity);
    if (result != 0) {
        return result;
    }
    cudaError_t error = cudaSuccess;
#define CUDA_CG_INIT(operation, expression)                                                    \
    do {                                                                                        \
        error = (expression);                                                                   \
        if (error != cudaSuccess) {                                                             \
            return cuda_failure(error, operation, message, capacity);                           \
        }                                                                                       \
    } while (false)
    CUDA_CG_INIT("upload CG border",
                 upload(operator_handle->cg_border, border, block_entries, operator_handle->stream));
    CUDA_CG_INIT("zero CG solution",
                 cudaMemsetAsync(operator_handle->cg_x, 0, block_entries * sizeof(uint32_t), operator_handle->stream));
    CUDA_CG_INIT("initialize CG residual",
                 cudaMemcpyAsync(operator_handle->cg_r,
                                 operator_handle->cg_border,
                                 block_entries * sizeof(uint32_t),
                                 cudaMemcpyDeviceToDevice,
                                 operator_handle->stream));
    CUDA_CG_INIT("initialize CG direction",
                 cudaMemcpyAsync(operator_handle->cg_p,
                                 operator_handle->cg_border,
                                 block_entries * sizeof(uint32_t),
                                 cudaMemcpyDeviceToDevice,
                                 operator_handle->stream));
    CUDA_CG_INIT("upload CG squared norms",
                 upload(operator_handle->cg_rr, initial_rr, BLOCK_WIDTH, operator_handle->stream));
    CUDA_CG_INIT("upload CG statuses",
                 upload(operator_handle->cg_status, status, BLOCK_WIDTH, operator_handle->stream));
    CUDA_CG_INIT("upload CG transcript",
                 upload(operator_handle->cg_transcript,
                        transcript,
                        BLOCK_WIDTH,
                        operator_handle->stream));
    CUDA_CG_INIT("zero CG lane steps",
                 cudaMemsetAsync(operator_handle->cg_lane_steps,
                                 0,
                                 BLOCK_WIDTH * sizeof(uint32_t),
                                 operator_handle->stream));
    CUDA_CG_INIT("synchronize CG initialization", cudaStreamSynchronize(operator_handle->stream));
#undef CUDA_CG_INIT
    operator_handle->cg_rounds = 0;
    operator_handle->cg_initialized = true;
    return 0;
}

int adynkra_exact_cuda_cg_run(
    Operator* operator_handle,
    uint32_t rounds,
    uint64_t* total_rounds,
    uint32_t* status,
    uint32_t* lane_steps,
    char* message,
    size_t capacity) {
    if (operator_handle == nullptr || total_rounds == nullptr || status == nullptr ||
        lane_steps == nullptr) {
        return invalid("CG run received a null pointer", message, capacity);
    }
    if (!operator_handle->cg_initialized) {
        set_error(message, capacity, "bordered CG is not initialized");
        return CG_NOT_INITIALIZED;
    }
    if (operator_handle->cg_rounds > operator_handle->columns ||
        rounds > operator_handle->columns - operator_handle->cg_rounds) {
        return invalid("CG run would exceed the matrix column dimension", message, capacity);
    }
    int result = set_device(operator_handle, message, capacity);
    if (result != 0) {
        return result;
    }
    for (uint32_t round = 0; round < rounds; ++round) {
        result = launch_cg_round(operator_handle, message, capacity);
        if (result != 0) {
            operator_handle->cg_initialized = false;
            return result;
        }
    }
    cudaError_t error = cudaStreamSynchronize(operator_handle->stream);
    if (error != cudaSuccess) {
        operator_handle->cg_initialized = false;
        return cuda_failure(error, "synchronize bordered CG", message, capacity);
    }
    operator_handle->cg_rounds += rounds;
    error = cudaMemcpy(status,
                       operator_handle->cg_status,
                       BLOCK_WIDTH * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) {
        error = cudaMemcpy(lane_steps,
                           operator_handle->cg_lane_steps,
                           BLOCK_WIDTH * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) {
        operator_handle->cg_initialized = false;
        return cuda_failure(error, "download bordered CG progress", message, capacity);
    }
    *total_rounds = operator_handle->cg_rounds;
    return 0;
}

int adynkra_exact_cuda_cg_download_state(
    Operator* operator_handle,
    uint32_t* x,
    uint32_t* r,
    uint32_t* p,
    size_t block_entries,
    uint32_t* rr,
    uint32_t* status,
    uint32_t* lane_steps,
    uint64_t* transcript,
    uint64_t* total_rounds,
    char* message,
    size_t capacity) {
    if (operator_handle == nullptr || rr == nullptr || status == nullptr ||
        lane_steps == nullptr || transcript == nullptr || total_rounds == nullptr) {
        return invalid("CG state download received a null pointer", message, capacity);
    }
    if (!operator_handle->cg_initialized) {
        set_error(message, capacity, "bordered CG is not initialized");
        return CG_NOT_INITIALIZED;
    }
    size_t expected = 0;
    checked_block_entries(operator_handle->columns, &expected);
    if (block_entries != expected ||
        (block_entries != 0 && (x == nullptr || r == nullptr || p == nullptr))) {
        return invalid("CG state block length does not match columns times 32", message, capacity);
    }
    int result = set_device(operator_handle, message, capacity);
    if (result != 0) {
        return result;
    }
    cudaError_t error = cudaMemcpy(
        x, operator_handle->cg_x, block_entries * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            r, operator_handle->cg_r, block_entries * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            p, operator_handle->cg_p, block_entries * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(
            rr, operator_handle->cg_rr, BLOCK_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(status,
                           operator_handle->cg_status,
                           BLOCK_WIDTH * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(lane_steps,
                           operator_handle->cg_lane_steps,
                           BLOCK_WIDTH * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(transcript,
                           operator_handle->cg_transcript,
                           BLOCK_WIDTH * sizeof(uint64_t),
                           cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) {
        return cuda_failure(error, "download bordered CG state", message, capacity);
    }
    *total_rounds = operator_handle->cg_rounds;
    return 0;
}

int adynkra_exact_cuda_cg_upload_state(
    Operator* operator_handle,
    const uint32_t* border,
    const uint32_t* x,
    const uint32_t* r,
    const uint32_t* p,
    size_t block_entries,
    const uint32_t* rr,
    const uint32_t* status,
    const uint32_t* lane_steps,
    const uint64_t* transcript,
    uint64_t total_rounds,
    char* message,
    size_t capacity) {
    if (operator_handle == nullptr || rr == nullptr || status == nullptr ||
        lane_steps == nullptr || transcript == nullptr) {
        return invalid("CG state upload received a null pointer", message, capacity);
    }
    size_t expected = 0;
    checked_block_entries(operator_handle->columns, &expected);
    if (block_entries != expected ||
        (block_entries != 0 &&
         (border == nullptr || x == nullptr || r == nullptr || p == nullptr))) {
        return invalid("CG state block length does not match columns times 32", message, capacity);
    }
    for (size_t index = 0; index < block_entries; ++index) {
        if (border[index] >= PRIME || x[index] >= PRIME || r[index] >= PRIME || p[index] >= PRIME) {
            return invalid("CG state contains a noncanonical field element", message, capacity);
        }
    }
    for (uint32_t lane = 0; lane < BLOCK_WIDTH; ++lane) {
        if (rr[lane] >= PRIME || status[lane] > CG_BROKEN ||
            lane_steps[lane] > total_rounds || lane_steps[lane] > operator_handle->columns) {
            return invalid("CG scalar state is invalid", message, capacity);
        }
        if ((status[lane] == CG_ACTIVE && rr[lane] == 0) ||
            (status[lane] == CG_CONVERGED && (rr[lane] != 0 || lane_steps[lane] == 0))) {
            return invalid("CG status is inconsistent with its residual scalar", message, capacity);
        }
    }
    if (total_rounds > operator_handle->columns) {
        return invalid("CG total rounds exceed the matrix column dimension", message, capacity);
    }
    if (total_rounds == operator_handle->columns) {
        for (uint32_t lane = 0; lane < BLOCK_WIDTH; ++lane) {
            if (status[lane] == CG_ACTIVE) {
                return invalid("CG lane remains active at the matrix dimension", message, capacity);
            }
        }
    }
    operator_handle->cg_initialized = false;
    int result = set_device(operator_handle, message, capacity);
    if (result != 0) {
        return result;
    }
    result = allocate_cg(operator_handle, message, capacity);
    if (result != 0) {
        return result;
    }
    cudaError_t error = cudaSuccess;
#define CUDA_CG_RESTORE(operation, expression)                                                 \
    do {                                                                                        \
        error = (expression);                                                                   \
        if (error != cudaSuccess) {                                                             \
            return cuda_failure(error, operation, message, capacity);                           \
        }                                                                                       \
    } while (false)
    CUDA_CG_RESTORE("restore CG border",
                    upload(operator_handle->cg_border, border, block_entries, operator_handle->stream));
    CUDA_CG_RESTORE("restore CG x",
                    upload(operator_handle->cg_x, x, block_entries, operator_handle->stream));
    CUDA_CG_RESTORE("restore CG r",
                    upload(operator_handle->cg_r, r, block_entries, operator_handle->stream));
    CUDA_CG_RESTORE("restore CG p",
                    upload(operator_handle->cg_p, p, block_entries, operator_handle->stream));
    CUDA_CG_RESTORE("restore CG rr",
                    upload(operator_handle->cg_rr, rr, BLOCK_WIDTH, operator_handle->stream));
    CUDA_CG_RESTORE("restore CG statuses",
                    upload(operator_handle->cg_status, status, BLOCK_WIDTH, operator_handle->stream));
    CUDA_CG_RESTORE("restore CG lane steps",
                    upload(operator_handle->cg_lane_steps, lane_steps, BLOCK_WIDTH, operator_handle->stream));
    CUDA_CG_RESTORE("restore CG transcript",
                    upload(operator_handle->cg_transcript,
                           transcript,
                           BLOCK_WIDTH,
                           operator_handle->stream));
    CUDA_CG_RESTORE("synchronize CG restore", cudaStreamSynchronize(operator_handle->stream));
#undef CUDA_CG_RESTORE
    operator_handle->cg_rounds = total_rounds;
    operator_handle->cg_initialized = true;
    return 0;
}

int adynkra_exact_cuda_device_name(
    Operator* operator_handle,
    char* output,
    size_t capacity) {
    if (operator_handle == nullptr || output == nullptr || capacity == 0) {
        return INVALID_ARGUMENT;
    }
    cudaDeviceProp properties{};
    const cudaError_t error = cudaGetDeviceProperties(&properties, operator_handle->device);
    if (error != cudaSuccess) {
        return cuda_failure(error, "cudaGetDeviceProperties", output, capacity);
    }
    set_error(output, capacity, "%s", properties.name);
    return 0;
}

}  // extern "C"
