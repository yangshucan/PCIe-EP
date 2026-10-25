#pragma once

#include "advanced_shared_memory.hpp"

#include <pthread.h>
#include <sched.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

namespace pcie_ep::host_staged {

static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free,
    "CPU/GPU mapped handshakes require lock-free naturally aligned 64-bit atomic load/store");

inline std::uint64_t host_acquire(std::uint64_t* address) {
    return std::atomic_ref<std::uint64_t>(*address).load(std::memory_order_acquire);
}
inline void host_release(std::uint64_t* address, std::uint64_t value) {
    std::atomic_ref<std::uint64_t>(*address).store(value, std::memory_order_release);
}
inline std::uint64_t* host_signal(char* segment, size_t offset, size_t index = 0) {
    auto* value = &reinterpret_cast<Signal*>(segment + offset)[index].value;
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(value) % std::atomic_ref<std::uint64_t>::required_alignment == 0,
        "unaligned mapped atomic signal");
    return value;
}
inline std::uint64_t relay_sequence(std::uint64_t epoch, int ordinal) {
    return (epoch << 32) | std::uint64_t(ordinal + 1);
}
inline std::uint64_t host_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline std::vector<int> numa_cpus(int node) {
    std::ifstream input("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string text;
    input >> text;
    std::vector<int> result;
    std::stringstream parts(text);
    std::string part;
    while (std::getline(parts, part, ',')) {
        const auto separator = part.find('-');
        const int first = std::stoi(part.substr(0, separator));
        const int last = separator == std::string::npos ? first : std::stoi(part.substr(separator + 1));
        for (int cpu = first; cpu <= last; ++cpu) result.push_back(cpu);
    }
    return result;
}

// One worker owns a disjoint subset of channels for one source/target domain.
// Its progress step never waits on a single full ring: other channels continue.
class NumaRelay {
    struct Flow {
        int channel;
        int domain;
        std::uint64_t epoch = 0;
        std::uint64_t started_ns = 0;
        std::uint64_t delayed_until_ns = 0;
        int expected = 0;
        int copied = 0;
        bool drained = false;
        std::array<int, kNumMaxRanks> sent{};
        std::vector<std::uint64_t> replica_sequences;
        std::vector<std::uint64_t> replica_masks;
        std::atomic<std::uint64_t> diagnostic_epoch{0};
        std::atomic<std::uint64_t> diagnostic_copies{0};
        std::atomic<std::uint64_t> diagnostic_bytes{0};
        std::atomic<std::uint64_t> copies_total{0};
        std::atomic<std::uint64_t> bytes_total{0};
        std::atomic<std::uint64_t> waits_total{0};
        std::atomic<std::uint64_t> copy_time_ns{0};
        std::atomic<int> diagnostic_expected{0};
        std::atomic<int> failure{0};
        Flow(int c, int n, int slots): channel(c), domain(n),
            replica_sequences(slots, 0), replica_masks(slots, 0) {}
    };
    struct Worker {
        int domain;
        int index;
        int cpu = -1;
        std::string initialization_error;
        std::string runtime_error;
        std::mutex initialization_mutex;
        mutable std::mutex diagnostic_mutex;
        std::condition_variable initialization_cv;
        bool initialized = false;
        std::vector<std::unique_ptr<Flow>> flows;
        std::thread thread;
    };

    AdvancedConfig config_;
    AdvancedSharedMemoryLayout shared_;
    int rank_;
    char* own_;
    std::vector<char*> segments_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<bool> stopping_{false};
    std::atomic<std::uint64_t> delay_ns_{0};

public:
    NumaRelay(const AdvancedConfig& config, const AdvancedSharedMemoryLayout& shared,
              int rank, const std::vector<MappedHostSegment>& segments):
        config_(config), shared_(shared), rank_(rank),
        own_(static_cast<char*>(segments[rank].host_address)), delay_ns_(config.relay_delay_ns) {
        for (const auto& segment : segments) segments_.push_back(static_cast<char*>(segment.host_address));
        try {
            for (int domain = 0; domain < shared.num_numa_nodes; ++domain) {
                for (int index = 0; index < config.relay_workers_per_numa; ++index) {
                    auto worker = std::make_unique<Worker>();
                    worker->domain = domain;
                    worker->index = index;
                    for (int channel = index; channel < config.channels; channel += config.relay_workers_per_numa)
                        worker->flows.push_back(std::make_unique<Flow>(channel, domain, shared.replica_slots));
                    auto* pointer = worker.get();
                    workers_.push_back(std::move(worker));
                    pointer->thread = std::thread([this, pointer] { run(*pointer); });
                    std::unique_lock<std::mutex> lock(pointer->initialization_mutex);
                    pointer->initialization_cv.wait(lock, [&] { return pointer->initialized; });
                    TORCH_CHECK(pointer->initialization_error.empty(), pointer->initialization_error);
                }
            }
        } catch (...) {
            stop_and_join();
            throw;
        }
    }
    ~NumaRelay() { stop_and_join(); }

    void stop_and_join() {
        stopping_.store(true, std::memory_order_release);
        for (auto& worker : workers_) if (worker->thread.joinable()) worker->thread.join();
    }
    void set_delay(std::uint64_t delay_ns) { delay_ns_.store(delay_ns, std::memory_order_relaxed); }

    py::dict diagnostics() const {
        py::dict result;
        py::list workers;
        std::uint64_t copies = 0, bytes = 0, copies_total = 0, bytes_total = 0;
        const auto ready_epoch = host_acquire(host_signal(own_, shared_.base.counts_ready_offset));
        for (const auto& worker : workers_) {
            py::dict item;
            item["numa_domain"] = worker->domain;
            item["numa_node"] = config_.physical_numa_nodes[worker->domain];
            item["cpu"] = worker->cpu;
            {
                std::lock_guard<std::mutex> lock(worker->diagnostic_mutex);
                item["runtime_error"] = worker->runtime_error;
            }
            py::list flows;
            for (const auto& flow : worker->flows) {
                py::dict record;
                const auto epoch = flow->diagnostic_epoch.load(std::memory_order_acquire);
                const auto count = flow->diagnostic_copies.load(std::memory_order_relaxed);
                const auto payload_bytes = flow->diagnostic_bytes.load(std::memory_order_relaxed);
                record["channel"] = flow->channel;
                record["epoch"] = epoch;
                record["expected_tokens"] = flow->diagnostic_expected.load(std::memory_order_relaxed);
                record["copied_tokens"] = count;
                record["copied_payload_bytes"] = payload_bytes;
                record["copy_time_ns_total"] = flow->copy_time_ns.load(std::memory_order_relaxed);
                record["credit_wait_probes_total"] = flow->waits_total.load(std::memory_order_relaxed);
                record["drained_epoch"] = host_acquire(host_signal(own_, shared_.relay_drained_offset,
                    flow->channel * shared_.num_numa_nodes + flow->domain));
                record["error"] = flow->failure.load(std::memory_order_relaxed);
                flows.append(record);
                if (epoch == ready_epoch) { copies += count; bytes += payload_bytes; }
                copies_total += flow->copies_total.load(std::memory_order_relaxed);
                bytes_total += flow->bytes_total.load(std::memory_order_relaxed);
            }
            item["flows"] = flows;
            workers.append(item);
        }
        result["epoch"] = ready_epoch;
        result["copies"] = copies;
        result["payload_bytes"] = bytes;
        result["copies_total"] = copies_total;
        result["payload_bytes_total"] = bytes_total;
        result["host_read_write_bytes"] = bytes * 2;
        result["workers"] = workers;
        result["metadata_and_scales_excluded_from_payload_bytes"] = true;
        return result;
    }

private:
    void pin_worker(Worker& worker) {
        const int physical_node = config_.physical_numa_nodes[worker.domain];
        const auto node_cpus = numa_cpus(physical_node);
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed) != 0)
            throw std::runtime_error("cannot query relay thread CPU affinity");
        std::vector<int> candidates;
        for (int cpu : node_cpus) {
            const bool selected = config_.relay_cpu_affinity.empty() ||
                std::find(config_.relay_cpu_affinity.begin(), config_.relay_cpu_affinity.end(), cpu) != config_.relay_cpu_affinity.end();
            if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed) && selected) candidates.push_back(cpu);
        }
        if (candidates.empty()) throw std::runtime_error("no allowed CPU on requested relay NUMA node");
        worker.cpu = candidates[(rank_ * config_.relay_workers_per_numa + worker.index) % candidates.size()];
        cpu_set_t target;
        CPU_ZERO(&target);
        CPU_SET(worker.cpu, &target);
        const int status = pthread_setaffinity_np(pthread_self(), sizeof(target), &target);
        if (status) throw std::runtime_error("cannot pin relay worker: " + std::to_string(status));
    }

    void fail(Flow& flow, int code) {
        flow.failure.store(code, std::memory_order_relaxed);
        host_release(host_signal(own_, shared_.relay_failed_offset,
            flow.channel * shared_.num_numa_nodes + flow.domain), code);
    }

    bool peer_aborted() const {
        for (auto* segment : segments_) {
            for (int channel = 0; channel < shared_.channels; ++channel)
                if (host_acquire(host_signal(segment, shared_.channel_abort_offset, channel))) return true;
            if (host_acquire(host_signal(segment, shared_.base.abort_offset))) return true;
        }
        return false;
    }

    bool replica_released(const Flow& flow, int slot) const {
        const auto sequence = flow.replica_sequences[slot];
        const auto targets = flow.replica_masks[slot];
        const size_t index = (size_t(flow.channel) * shared_.num_numa_nodes + flow.domain) * shared_.replica_slots + slot;
        for (int rank = 0; rank < shared_.base.num_ranks; ++rank) {
            if ((targets & (1ull << rank)) && host_acquire(host_signal(own_, shared_.replica_ack_offset,
                    index * shared_.base.num_ranks + rank)) != sequence) return false;
        }
        return true;
    }

    bool progress(Flow& flow) {
        const auto ready_epoch = host_acquire(host_signal(own_, shared_.base.counts_ready_offset));
        // Construction precedes the first GPU publication. Epoch zero has no
        // active work or timeout origin; do not diagnose it as a stalled flow.
        if (!ready_epoch) {
            return false;
        }
        if (ready_epoch != flow.epoch) {
            if (flow.epoch && !flow.drained) { fail(flow, 101); return false; }
            flow.epoch = ready_epoch;
            flow.started_ns = host_time_ns();
            flow.expected = reinterpret_cast<int*>(own_ + shared_.endpoint_counts_offset)[
                flow.channel * shared_.num_endpoints + shared_.base.num_ranks + flow.domain];
            flow.copied = 0;
            flow.drained = false;
            flow.delayed_until_ns = 0;
            flow.sent.fill(0);
            std::fill(flow.replica_sequences.begin(), flow.replica_sequences.end(), 0);
            std::fill(flow.replica_masks.begin(), flow.replica_masks.end(), 0);
            flow.diagnostic_expected.store(flow.expected, std::memory_order_relaxed);
            flow.diagnostic_copies.store(0, std::memory_order_relaxed);
            flow.diagnostic_bytes.store(0, std::memory_order_relaxed);
            flow.diagnostic_epoch.store(flow.epoch, std::memory_order_release);
        }
        if (flow.drained || flow.failure.load(std::memory_order_relaxed)) return false;
        if (host_time_ns() - flow.started_ns > std::uint64_t(config_.timeout_ms) * 1000000) {
            fail(flow, 102); return false;
        }
        if (flow.copied == flow.expected) {
            for (int slot = 0; slot < shared_.replica_slots; ++slot)
                if (!replica_released(flow, slot)) return false;
            host_release(host_signal(own_, shared_.relay_drained_offset,
                flow.channel * shared_.num_numa_nodes + flow.domain), flow.epoch);
            flow.drained = true;
            return true;
        }
        const int endpoint = shared_.base.num_ranks + flow.domain;
        auto* descriptor = reinterpret_cast<Descriptor*>(own_ + shared_.source_descriptors_offset) +
            (flow.channel * shared_.num_endpoints + endpoint) * shared_.base.num_descriptor_slots +
            flow.copied % shared_.base.num_descriptor_slots;
        if (host_acquire(&descriptor->ready_sequence) != relay_sequence(flow.epoch, flow.copied)) return false;
        const int source_slot = descriptor->payload_slot;
        const auto sequence = descriptor->payload_sequence;
        if (source_slot < 0 || source_slot >= shared_.base.num_payload_slots) { fail(flow, 103); return false; }
        const size_t source_index = size_t(flow.channel) * shared_.base.num_payload_slots + source_slot;
        TokenMetadata metadata = reinterpret_cast<TokenMetadata*>(own_ + shared_.source_metadata_offset)[source_index];
        if (sequence != relay_sequence(flow.epoch, metadata.token_index)) { fail(flow, 104); return false; }
        const int replica_slot = flow.copied % shared_.replica_slots;
        if (!replica_released(flow, replica_slot)) { flow.waits_total.fetch_add(1, std::memory_order_relaxed); return false; }
        std::uint64_t targets = 0;
        for (int rank = 0; rank < shared_.base.num_ranks; ++rank) {
            if (rank == rank_ || config_.rank_numa_domains[rank] != flow.domain || !(metadata.destination_mask & (1ull << rank))) continue;
            targets |= 1ull << rank;
            if (flow.sent[rank] >= shared_.base.num_descriptor_slots) {
                auto* output = reinterpret_cast<Descriptor*>(own_ + shared_.replica_descriptors_offset) +
                    ((flow.channel * shared_.num_numa_nodes + flow.domain) * shared_.base.num_ranks + rank) *
                    shared_.base.num_descriptor_slots + flow.sent[rank] % shared_.base.num_descriptor_slots;
                if (host_acquire(&output->consumed_sequence) != relay_sequence(flow.epoch, flow.sent[rank] - shared_.base.num_descriptor_slots)) {
                    flow.waits_total.fetch_add(1, std::memory_order_relaxed); return false;
                }
            }
        }
        if (!targets) { fail(flow, 105); return false; }
        const auto delay = delay_ns_.load(std::memory_order_relaxed);
        if (delay && !flow.delayed_until_ns) flow.delayed_until_ns = host_time_ns() + delay;
        if (flow.delayed_until_ns && host_time_ns() < flow.delayed_until_ns) return false;
        flow.delayed_until_ns = 0;
        const size_t replica_index = (size_t(flow.channel) * shared_.num_numa_nodes + flow.domain) * shared_.replica_slots + replica_slot;
        const size_t payload_bytes = size_t(shared_.base.num_hidden) * (shared_.payload_dtype ? 1 : 2);
        const auto begin = host_time_ns();
        std::memcpy(own_ + shared_.replica_payload_offset + replica_index * shared_.payload_stride_bytes,
                    own_ + shared_.source_payload_offset + source_index * shared_.payload_stride_bytes, payload_bytes);
        std::memcpy(reinterpret_cast<TokenMetadata*>(own_ + shared_.replica_metadata_offset) + replica_index, &metadata, sizeof(metadata));
        reinterpret_cast<float*>(own_ + shared_.replica_scale_offset)[replica_index] =
            reinterpret_cast<float*>(own_ + shared_.source_scale_offset)[source_index];
        flow.copy_time_ns.fetch_add(host_time_ns() - begin, std::memory_order_relaxed);
        for (int rank = 0; rank < shared_.base.num_ranks; ++rank) {
            if (!(targets & (1ull << rank))) continue;
            auto* output = reinterpret_cast<Descriptor*>(own_ + shared_.replica_descriptors_offset) +
                ((flow.channel * shared_.num_numa_nodes + flow.domain) * shared_.base.num_ranks + rank) *
                shared_.base.num_descriptor_slots + flow.sent[rank] % shared_.base.num_descriptor_slots;
            output->payload_sequence = sequence;
            output->payload_slot = replica_slot;
            output->token_index = metadata.token_index;
            output->rank_ordinal = metadata.rank_ordinals[rank];
            host_release(&output->ready_sequence, relay_sequence(flow.epoch, flow.sent[rank]));
            ++flow.sent[rank];
        }
        flow.replica_sequences[replica_slot] = sequence;
        flow.replica_masks[replica_slot] = targets;
        // CPU no longer accesses source payload/metadata after this COPY ACK.
        host_release(host_signal(own_, shared_.source_ack_offset,
            source_index * shared_.num_endpoints + endpoint), sequence);
        host_release(&descriptor->consumed_sequence, relay_sequence(flow.epoch, flow.copied));
        ++flow.copied;
        flow.diagnostic_copies.store(flow.copied, std::memory_order_relaxed);
        flow.diagnostic_bytes.store(std::uint64_t(flow.copied) * payload_bytes, std::memory_order_relaxed);
        flow.copies_total.fetch_add(1, std::memory_order_relaxed);
        flow.bytes_total.fetch_add(payload_bytes, std::memory_order_relaxed);
        return true;
    }

    void run(Worker& worker) {
        try { pin_worker(worker); }
        catch (const std::exception& error) { worker.initialization_error = error.what(); }
        {
            std::lock_guard<std::mutex> lock(worker.initialization_mutex);
            worker.initialized = true;
        }
        worker.initialization_cv.notify_one();
        if (!worker.initialization_error.empty()) return;
        try {
            while (!stopping_.load(std::memory_order_acquire)) {
                if (peer_aborted()) return;
                bool advanced = false;
                for (auto& flow : worker.flows) advanced |= progress(*flow);
                if (!advanced) std::this_thread::yield();
            }
        } catch (const std::exception& error) {
            {
                std::lock_guard<std::mutex> lock(worker.diagnostic_mutex);
                worker.runtime_error = error.what();
            }
            for (auto& flow : worker.flows) fail(*flow, 106);
        }
    }
};

// An explicit diagnostic, never called by dispatch. Both query methods return
// the resident page's NUMA node, rather than its allocation-policy preference.
// Some container profiles permit get_mempolicy but reject move_pages query mode.
inline py::dict query_numa_pages(void* address, size_t length) {
    const size_t pages = (length + kNumSharedMemoryPageBytes - 1) / kNumSharedMemoryPageBytes;
    const size_t count = std::min<size_t>(pages, 64);
    std::vector<void*> addresses(count);
    std::vector<int> status(count, -1);
    for (size_t i = 0; i < count; ++i) {
        const size_t page = count < 2 ? 0 : i * (pages - 1) / (count - 1);
        addresses[i] = static_cast<char*>(address) + page * kNumSharedMemoryPageBytes;
    }

    const long move_pages_result = count ? syscall(
        SYS_move_pages, 0, count, addresses.data(), nullptr, status.data(), 0) : 0;
    const int move_pages_error = move_pages_result < 0 ? errno : 0;
    long result = move_pages_result;
    int error = move_pages_error;
    const char* method = "move_pages";
    if (move_pages_result < 0) {
        method = "get_mempolicy";
        result = 0;
        error = 0;
        for (size_t i = 0; i < count; ++i) {
            int node = -1;
            const long queried = syscall(SYS_get_mempolicy, &node, nullptr, 0ul,
                addresses[i], MPOL_F_NODE | MPOL_F_ADDR);
            if (queried == 0) {
                status[i] = node;
            } else {
                const int query_error = errno;
                status[i] = -query_error;
                result = -1;
                if (!error) {
                    error = query_error;
                }
            }
        }
    }

    std::map<int, int> nodes;
    for (int node : status) {
        if (node >= 0) {
            nodes[node]++;
        }
    }
    py::dict record;
    record["query_return"] = result;
    record["query_errno"] = error;
    record["query_method"] = method;
    record["move_pages_return"] = move_pages_result;
    record["move_pages_errno"] = move_pages_error;
    record["total_pages"] = pages;
    record["sampled_pages"] = count;
    record["node_page_counts"] = nodes;
    record["sample_status"] = status;
    return record;
}

}  // namespace pcie_ep::host_staged
