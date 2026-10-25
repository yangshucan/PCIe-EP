// S0: publication and slot-reuse test using only CPU mapped memory.
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -Xcompiler -pthread mapped_litmus.cu -o mapped_litmus
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using Atomic = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>;
constexpr unsigned kThreads = 256;
constexpr std::uint64_t kNoIndex = ~std::uint64_t{0};

// Each handoff field has exactly one GPU writer. No atomic RMW is performed on
// mapped memory. Separate cache lines also avoid false sharing between flags.
struct alignas(128) Control {
    alignas(128) std::uint64_t ready;
    alignas(128) std::uint64_t ack;
    alignas(128) std::uint64_t producer_abort;
    alignas(128) std::uint64_t consumer_abort;
};
static_assert(offsetof(Control, ready) % 128 == 0);
static_assert(offsetof(Control, ack) % 128 == 0);
static_assert(offsetof(Control, producer_abort) % 128 == 0);
static_assert(offsetof(Control, consumer_abort) % 128 == 0);

enum Error : std::uint64_t {
    kSuccess = 0,
    kReadyTimeout = 1,
    kAckTimeout = 2,
    kPeerAborted = 3,
    kPayloadMismatch = 4,
};

struct alignas(128) Result {
    std::uint64_t error;
    std::uint64_t completed;
    std::uint64_t sequence;
    std::uint64_t index;
    std::uint64_t observed;
    std::uint64_t expected;
};

__device__ __forceinline__ std::uint64_t pattern(std::uint64_t sequence,
                                                 std::uint64_t index) {
    // Both halves of every 64-bit word vary with both sequence and word index.
    std::uint64_t x = sequence * 0x9e3779b97f4a7c15ULL;
    x ^= (index + 1) * 0xd6e8feb86659fd93ULL;
    x ^= x >> 32;
    x *= 0xa24baed4963ee407ULL;
    return x ^ (x >> 29);
}

__device__ __forceinline__ void delay_cycles(std::uint64_t cycles) {
    const auto start = clock64();
    while (clock64() - start < cycles) {
        __nanosleep(32);
    }
}

// Only a CTA's leader calls this function. The caller broadcasts the outcome
// with __syncthreads before any thread accesses the payload or exits.
__device__ bool wait_sequence(std::uint64_t* flag, std::uint64_t expected,
                             std::uint64_t* peer_abort, std::uint64_t timeout_cycles,
                             Result* result, Error timeout_error,
                             std::uint64_t sequence) {
    const auto start = clock64();
    while (true) {
        if (Atomic(*flag).load(cuda::memory_order_acquire) == expected) {
            return true;
        }
        if (Atomic(*peer_abort).load(cuda::memory_order_acquire) != 0) {
            result->error = kPeerAborted;
            result->sequence = sequence;
            return false;
        }
        if (clock64() - start >= timeout_cycles) {
            result->error = timeout_error;
            result->sequence = sequence;
            result->observed = Atomic(*flag).load(cuda::memory_order_relaxed);
            result->expected = expected;
            return false;
        }
        __nanosleep(64);
    }
}

__global__ void producer(volatile std::uint64_t* payload, Control* control,
                         Result* result, std::uint64_t words,
                         std::uint64_t iterations, std::uint64_t timeout_cycles,
                         std::uint64_t writer_delay_cycles) {
    __shared__ bool proceed;
    for (std::uint64_t sequence = 1; sequence <= iterations; ++sequence) {
        if (threadIdx.x == 0) {
            proceed = wait_sequence(&control->ack, sequence - 1,
                                    &control->consumer_abort, timeout_cycles,
                                    result, kAckTimeout, sequence);
            if (!proceed) {
                Atomic(control->producer_abort).store(1, cuda::memory_order_release);
            }
        }
        __syncthreads();
        if (!proceed) return;

        if (threadIdx.x == kThreads - 1 && writer_delay_cycles != 0) {
            delay_cycles(writer_delay_cycles);
        }
        for (std::uint64_t i = threadIdx.x; i < words; i += blockDim.x) {
            payload[i] = pattern(sequence, i);
        }
        // A leader-only fence does not publish the other 255 writers. Every
        // writer fences its own stores, then the CTA joins before READY.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            Atomic(control->ready).store(sequence, cuda::memory_order_release);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (!wait_sequence(&control->ack, iterations, &control->consumer_abort,
                           timeout_cycles, result, kAckTimeout, iterations)) {
            Atomic(control->producer_abort).store(1, cuda::memory_order_release);
        } else {
            result->completed = iterations;
        }
    }
}

__global__ void consumer(volatile std::uint64_t* payload, Control* control,
                         Result* result, std::uint64_t words,
                         std::uint64_t iterations, std::uint64_t timeout_cycles,
                         std::uint64_t reader_delay_cycles) {
    __shared__ bool proceed;
    __shared__ unsigned long long first_bad;
    for (std::uint64_t sequence = 1; sequence <= iterations; ++sequence) {
        if (threadIdx.x == 0) {
            first_bad = kNoIndex;
            proceed = wait_sequence(&control->ready, sequence,
                                    &control->producer_abort, timeout_cycles,
                                    result, kReadyTimeout, sequence);
            if (!proceed) {
                Atomic(control->consumer_abort).store(1, cuda::memory_order_release);
            }
        }
        __syncthreads();
        if (!proceed) return;

        // Delay a reader before its loads, so an early ACK would expose reuse
        // while the last reader still needs the previous sequence's payload.
        if (threadIdx.x == kThreads - 1 && reader_delay_cycles != 0) {
            delay_cycles(reader_delay_cycles);
        }
        for (std::uint64_t i = threadIdx.x; i < words; i += blockDim.x) {
            if (payload[i] != pattern(sequence, i)) {
                // This RMW targets CTA shared memory, never mapped host memory.
                atomicMin(&first_bad, static_cast<unsigned long long>(i));
            }
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            if (first_bad != kNoIndex) {
                result->error = kPayloadMismatch;
                result->sequence = sequence;
                result->index = first_bad;
                result->observed = payload[first_bad];
                result->expected = pattern(sequence, first_bad);
                Atomic(control->consumer_abort).store(1, cuda::memory_order_release);
                proceed = false;
            } else {
                // All readers have finished before the producer may overwrite.
                Atomic(control->ack).store(sequence, cuda::memory_order_release);
                result->completed = sequence;
            }
        }
        __syncthreads();
        if (!proceed) return;
    }
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

std::string quoted(const std::string& input) {
    std::ostringstream output;
    output << '"';
    for (unsigned char c : input) {
        if (c == '"' || c == '\\') output << '\\' << c;
        else if (c == '\n') output << "\\n";
        else if (c < 32) output << '?';
        else output << c;
    }
    output << '"';
    return output.str();
}

struct Options {
    std::uint64_t iterations = 1000000;
    std::uint64_t bytes = 7168 * 2;
    std::uint64_t timeout_ms = 10000;
    std::uint64_t reader_delay = 0;
    std::uint64_t writer_delay = 0;
    std::array<int, 2> devices{0, 1};
};

std::uint64_t number(const std::string& value) {
    if (value.empty() || value.front() == '-') throw std::runtime_error("invalid unsigned number");
    std::size_t consumed = 0;
    const auto result = std::stoull(value, &consumed);
    if (consumed != value.size()) throw std::runtime_error("invalid unsigned number");
    return result;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "mapped_litmus [--iterations N] [--bytes N] [--producer GPU] "
                         "[--consumer GPU] [--timeout-ms N] [--delay CYCLES] "
                         "[--writer-delay CYCLES]\n";
            std::exit(0);
        }
        if (++i == argc) throw std::runtime_error("missing value for " + key);
        const auto value = number(argv[i]);
        if (key == "--iterations") options.iterations = value;
        else if (key == "--bytes") options.bytes = value;
        else if (key == "--timeout-ms") options.timeout_ms = value;
        else if (key == "--delay") options.reader_delay = value;
        else if (key == "--writer-delay") options.writer_delay = value;
        else if (key == "--producer" || key == "--consumer") {
            if (value > std::numeric_limits<int>::max()) throw std::runtime_error("GPU index overflow");
            options.devices[key == "--producer" ? 0 : 1] = static_cast<int>(value);
        } else throw std::runtime_error("unknown argument: " + key);
    }
    if (!options.iterations || options.iterations == std::numeric_limits<std::uint64_t>::max())
        throw std::runtime_error("iterations must be in [1, UINT64_MAX-1]");
    if (!options.bytes || options.bytes % sizeof(std::uint64_t) != 0)
        throw std::runtime_error("bytes must be a positive multiple of 8");
    if (!options.timeout_ms || options.timeout_ms > 3600000)
        throw std::runtime_error("timeout-ms must be in [1, 3600000]");
    if (options.devices[0] == options.devices[1])
        throw std::runtime_error("producer and consumer must be different GPUs");
    return options;
}

// This gate coordinates launches only. CPU threads do not touch any handoff or
// payload while the kernels are running.
struct LaunchGate {
    std::mutex mutex;
    std::condition_variable cv;
    int arrivals = 0;
    bool failed = false;
    bool arrive() {
        std::unique_lock<std::mutex> lock(mutex);
        ++arrivals;
        cv.notify_all();
        cv.wait(lock, [&] { return arrivals == 2 || failed; });
        return !failed;
    }
    void fail() {
        std::lock_guard<std::mutex> lock(mutex);
        failed = true;
        cv.notify_all();
    }
};

struct DeviceInfo {
    cudaDeviceProp properties{};
    int native_host_atomics = 0;
    int clock_rate_khz = 0;
    float milliseconds = 0;
    std::string error;
};

}  // namespace

int main(int argc, char** argv) {
    void* payload_host = nullptr;
    Control* control_host = nullptr;
    Result* results_host = nullptr;
    try {
        const auto options = parse(argc, argv);
        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        std::array<DeviceInfo, 2> info;
        for (int role = 0; role < 2; ++role) {
            if (options.devices[role] < 0 || options.devices[role] >= count)
                throw std::runtime_error("GPU index is unavailable");
            check(cudaGetDeviceProperties(&info[role].properties, options.devices[role]),
                  "cudaGetDeviceProperties");
            check(cudaDeviceGetAttribute(&info[role].native_host_atomics,
                                         cudaDevAttrHostNativeAtomicSupported,
                                         options.devices[role]), "hostNativeAtomicSupported");
            check(cudaDeviceGetAttribute(&info[role].clock_rate_khz, cudaDevAttrClockRate,
                                         options.devices[role]), "clockRate");
            if (!info[role].properties.canMapHostMemory)
                throw std::runtime_error("selected GPU cannot map host memory");
        }
        check(cudaSetDevice(options.devices[0]), "cudaSetDevice");
        constexpr auto allocation_flags = cudaHostAllocMapped | cudaHostAllocPortable;
        check(cudaHostAlloc(&payload_host, options.bytes, allocation_flags), "allocate payload");
        check(cudaHostAlloc(reinterpret_cast<void**>(&control_host), sizeof(Control), allocation_flags),
              "allocate control");
        check(cudaHostAlloc(reinterpret_cast<void**>(&results_host), sizeof(Result) * 2, allocation_flags),
              "allocate results");
        *control_host = {};
        results_host[0] = {};
        results_host[1] = {};

        LaunchGate gate;
        const auto wall_start = std::chrono::steady_clock::now();
        auto run = [&](int role) {
            cudaStream_t stream = nullptr;
            cudaEvent_t begin = nullptr, end = nullptr;
            try {
                check(cudaSetDevice(options.devices[role]), "worker cudaSetDevice");
                check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
                check(cudaEventCreate(&begin), "create begin event");
                check(cudaEventCreate(&end), "create end event");
                std::uint64_t* payload_device = nullptr;
                Control* control_device = nullptr;
                Result* results_device = nullptr;
                check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&payload_device), payload_host, 0),
                      "map payload");
                check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&control_device), control_host, 0),
                      "map control");
                check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&results_device), results_host, 0),
                      "map results");
                const std::uint64_t timeout_cycles = options.timeout_ms *
                    static_cast<std::uint64_t>(info[role].clock_rate_khz);
                if (gate.arrive()) {
                    check(cudaEventRecord(begin, stream), "record begin");
                    if (role == 0) {
                        producer<<<1, kThreads, 0, stream>>>(payload_device, control_device,
                            results_device, options.bytes / 8, options.iterations,
                            timeout_cycles, options.writer_delay);
                    } else {
                        consumer<<<1, kThreads, 0, stream>>>(payload_device, control_device,
                            results_device + 1, options.bytes / 8, options.iterations,
                            timeout_cycles, options.reader_delay);
                    }
                    check(cudaGetLastError(), "launch kernel");
                    check(cudaEventRecord(end, stream), "record end");
                    // Test-only final synchronization; there is no CPU handoff
                    // in the per-sequence communication loop.
                    check(cudaEventSynchronize(end), "wait final test result");
                    check(cudaEventElapsedTime(&info[role].milliseconds, begin, end), "elapsed time");
                }
            } catch (const std::exception& error) {
                info[role].error = error.what();
                gate.fail();
            }
            if (stream) cudaStreamSynchronize(stream);
            if (begin) cudaEventDestroy(begin);
            if (end) cudaEventDestroy(end);
            if (stream) cudaStreamDestroy(stream);
        };
        std::thread producer_thread(run, 0);
        std::thread consumer_thread(run, 1);
        producer_thread.join();
        consumer_thread.join();
        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();
        bool passed = true;
        for (int role = 0; role < 2; ++role) {
            passed &= info[role].error.empty() && results_host[role].error == kSuccess &&
                      results_host[role].completed == options.iterations;
        }
        int runtime_version = 0, driver_version = 0;
        check(cudaRuntimeGetVersion(&runtime_version), "runtime version");
        check(cudaDriverGetVersion(&driver_version), "driver version");
        std::cout << std::setprecision(9)
                  << "{\"test\":\"mapped_publication_litmus\",\"passed\":"
                  << (passed ? "true" : "false")
                  << ",\"iterations\":" << options.iterations
                  << ",\"payload_bytes\":" << options.bytes
                  << ",\"reader_delay_cycles\":" << options.reader_delay
                  << ",\"writer_delay_cycles\":" << options.writer_delay
                  << ",\"host_elapsed_seconds\":" << wall_seconds
                  << ",\"cuda_runtime\":" << runtime_version
                  << ",\"cuda_driver\":" << driver_version
                  << ",\"transport\":\"host_mapped_only\",\"host_atomic_operations\":\"load_store_only\""
                  << ",\"devices\":[";
        for (int role = 0; role < 2; ++role) {
            const auto& result = results_host[role];
            if (role) std::cout << ',';
            std::cout << "{\"role\":" << quoted(role == 0 ? "producer" : "consumer")
                      << ",\"ordinal\":" << options.devices[role]
                      << ",\"name\":" << quoted(info[role].properties.name)
                      << ",\"can_map_host_memory\":" << info[role].properties.canMapHostMemory
                      << ",\"host_native_atomic_supported\":" << info[role].native_host_atomics
                      << ",\"kernel_elapsed_ms\":" << info[role].milliseconds
                      << ",\"completed\":" << result.completed
                      << ",\"error_code\":" << result.error
                      << ",\"error_sequence\":" << result.sequence
                      << ",\"error_word\":" << result.index
                      << ",\"observed\":" << result.observed
                      << ",\"expected\":" << result.expected
                      << ",\"runtime_error\":" << quoted(info[role].error) << '}';
        }
        std::cout << "]}\n";
        check(cudaFreeHost(results_host), "free results"); results_host = nullptr;
        check(cudaFreeHost(control_host), "free control"); control_host = nullptr;
        check(cudaFreeHost(payload_host), "free payload"); payload_host = nullptr;
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "{\"test\":\"mapped_publication_litmus\",\"passed\":false,\"error\":"
                  << quoted(error.what()) << "}\n";
        if (results_host) cudaFreeHost(results_host);
        if (control_host) cudaFreeHost(control_host);
        if (payload_host) cudaFreeHost(payload_host);
        return 1;
    }
}
