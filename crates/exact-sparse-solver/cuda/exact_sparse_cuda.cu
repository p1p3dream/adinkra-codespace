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
constexpr int INVALID_ARGUMENT = -1;
constexpr int HOST_ALLOCATION_FAILED = -2;
constexpr int INPUT_NOT_UPLOADED = -3;

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
};

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

__device__ __forceinline__ uint32_t reduce_mersenne(uint64_t value) {
    value = (value & PRIME_U64) + (value >> 31);
    value = (value & PRIME_U64) + (value >> 31);
    return static_cast<uint32_t>(value >= PRIME_U64 ? value - PRIME_U64 : value);
}

__device__ __forceinline__ uint32_t multiply_mersenne(uint32_t left, uint32_t right) {
    return reduce_mersenne(static_cast<uint64_t>(left) * static_cast<uint64_t>(right));
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

int set_device(Operator* operator_handle, char* message, size_t capacity) {
    const cudaError_t error = cudaSetDevice(operator_handle->device);
    return error == cudaSuccess ? 0 : cuda_failure(error, "cudaSetDevice", message, capacity);
}

int launch_steps(Operator* operator_handle, uint32_t steps, char* message, size_t capacity) {
    if (!operator_handle->input_uploaded) {
        set_error(message, capacity, "no input block has been uploaded");
        return INPUT_NOT_UPLOADED;
    }
    const uint32_t warps_per_block = THREADS / BLOCK_WIDTH;
    const uint32_t csr_blocks =
        (operator_handle->rows + warps_per_block - 1) / warps_per_block;
    const uint32_t transpose_blocks =
        (operator_handle->columns + warps_per_block - 1) / warps_per_block;

    for (uint32_t step = 0; step < steps; ++step) {
        const uint32_t next = operator_handle->active_block ^ 1u;
        if (operator_handle->rows != 0) {
            csr_diagonal_block32<<<csr_blocks, THREADS, 0, operator_handle->stream>>>(
                operator_handle->rows,
                operator_handle->csr_offsets,
                operator_handle->csr_entries,
                operator_handle->diagonal,
                operator_handle->column_blocks[operator_handle->active_block],
                operator_handle->row_block);
            cudaError_t error = cudaPeekAtLastError();
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
                operator_handle->column_blocks[next]);
            cudaError_t error = cudaPeekAtLastError();
            if (error != cudaSuccess) {
                return cuda_failure(error, "launch transpose_block32", message, capacity);
            }
        }
        operator_handle->active_block = next;
    }
    const cudaError_t error = cudaStreamSynchronize(operator_handle->stream);
    return error == cudaSuccess ? 0
                                : cuda_failure(error, "synchronize exact sparse kernels", message, capacity);
}

}  // namespace

extern "C" {

uint32_t adynkra_exact_cuda_abi_version() {
    return 1;
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
