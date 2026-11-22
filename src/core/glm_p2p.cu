// src/core/glm_p2p.cu - probe and run the opt-in decode peer hop (include/strata/core/glm_p2p.hpp).
//
// The decision is per ordered device pair and lives for the process: the driver either grants peer
// access or it does not. A later copy that fails to enqueue latches that pair back to the host path.
#include "strata/core/glm_p2p.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core::glmp2p {

namespace {

struct DevRestore {
    int cur = 0;
    bool ok = false;
    DevRestore() { ok = cudaGetDevice(&cur) == cudaSuccess; }
    ~DevRestore() {
        if (ok) cudaSetDevice(cur);
    }
};

struct Pair {
    int state = 0;   // 0 unknown, 1 peer, -1 host
    std::string why;
    float* slot[kPeerSlots] = {};
    size_t slot_bytes = 0;
    double probe_ms = 0;
    bool said_decode = false;
    bool said_window = false;
    bool issue_valid[kPeerSlots] = {};
    bool issue_peer[kPeerSlots] = {};
};

std::mutex g_mu;
Pair g_pair[kMaxDev][kMaxDev];
std::atomic<uint64_t> g_peer_hops{0};
std::atomic<uint64_t> g_host_hops{0};
std::atomic<uint64_t> g_peer_bytes{0};
std::atomic<uint64_t> g_host_bytes{0};

bool dev_ok(int d) { return d >= 0 && d < kMaxDev; }

void free_slots(Pair& p) {
    for (int i = 0; i < kPeerSlots; ++i) {
        if (p.slot[i] != nullptr) cudaFree(p.slot[i]);
        p.slot[i] = nullptr;
        p.issue_valid[i] = false;
    }
    p.slot_bytes = 0;
}

// Caller holds g_mu. Devices are restored. On failure the pair is host and slots are freed.
bool probe_unlocked(Pair& p, int src, int dst, size_t bytes) {
    p.why.clear();
    p.state = -1;
    p.probe_ms = 0;
    if (!dev_ok(src) || !dev_ok(dst) || src == dst) {
        p.why = "device id out of range";
        return false;
    }
    if (bytes == 0 || bytes > (size_t{4} << 20)) {
        p.why = HopReasons::kBytes;
        return false;
    }
    DevRestore restore;
    (void) cudaGetLastError();
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, dst, src) != cudaSuccess || can == 0) {
        (void) cudaGetLastError();
        p.why = HopReasons::kNoPeer;
        return false;
    }
    cudaSetDevice(dst);
    cudaError_t pe = cudaDeviceEnablePeerAccess(src, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) {
        p.why = std::string(HopReasons::kEnable) + ": " + cudaGetErrorString(pe);
        (void) cudaGetLastError();
        return false;
    }
    // The copy runs on the destination stream. The reverse enable is best-effort: some pairs only
    // report one direction, and a failure here must not hide a forward path that already enabled.
    cudaSetDevice(src);
    pe = cudaDeviceEnablePeerAccess(dst, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) (void) cudaGetLastError();

    cudaSetDevice(src);
    for (int i = 0; i < kPeerSlots; ++i) {
        if (cudaMalloc(&p.slot[i], bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            free_slots(p);
            p.why = HopReasons::kAlloc;
            return false;
        }
    }
    p.slot_bytes = bytes;
    cudaSetDevice(dst);
    void* probe_dst = nullptr;
    if (cudaMalloc(&probe_dst, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        cudaSetDevice(src);
        free_slots(p);
        p.why = HopReasons::kAlloc;
        return false;
    }
    std::vector<unsigned char> host(bytes), back(bytes);
    for (size_t i = 0; i < bytes; ++i) host[i] = (unsigned char) (i * 131u + 17u);
    bool matched = true;
    std::string fail;
    double ms = 0;
    for (int s = 0; s < kPeerSlots && matched; ++s) {
        cudaSetDevice(src);
        if (cudaMemcpy(p.slot[s], host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            fail = std::string(HopReasons::kProbe) + ": " + cudaGetErrorString(cudaGetLastError());
            matched = false;
            break;
        }
        cudaSetDevice(dst);
        const auto t0 = std::chrono::steady_clock::now();
        const cudaError_t ce = cudaMemcpyPeer(probe_dst, dst, p.slot[s], src, bytes);
        const auto t1 = std::chrono::steady_clock::now();
        if (ce != cudaSuccess) {
            fail = std::string(HopReasons::kProbe) + ": " + cudaGetErrorString(ce);
            (void) cudaGetLastError();
            matched = false;
            break;
        }
        ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (cudaMemcpy(back.data(), probe_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            fail = std::string(HopReasons::kProbe) + ": " + cudaGetErrorString(cudaGetLastError());
            matched = false;
            break;
        }
        if (std::memcmp(host.data(), back.data(), bytes) != 0) {
            fail = std::string(HopReasons::kProbe) + ": byte mismatch";
            matched = false;
        }
    }
    cudaSetDevice(dst);
    cudaFree(probe_dst);
    if (!matched) {
        cudaSetDevice(src);
        free_slots(p);
        p.why = fail.empty() ? HopReasons::kProbe : fail;
        p.state = -1;
        return false;
    }
    p.state = 1;
    p.probe_ms = ms / kPeerSlots;
    p.why = HopReasons::kPeer;
    return true;
}

Pair* pair_unlocked(int src, int dst) {
    if (!dev_ok(src) || !dev_ok(dst) || src == dst) return nullptr;
    return &g_pair[src][dst];
}

void ensure_unlocked(Pair& p, int src, int dst, size_t bytes) {
    if (p.state != 0) return;
    if (!env_on()) {
        p.state = -1;
        p.why = HopReasons::kOff;
        return;
    }
    probe_unlocked(p, src, dst, bytes == 0 ? (size_t{64} << 10) : bytes);
}

HopDecision decision_of(const Pair& p) {
    if (p.state > 0) return {HopPath::Peer, HopReasons::kPeer};
    return {HopPath::Host, p.why.empty() ? HopReasons::kOff : p.why.c_str()};
}

void count(bool peer, size_t bytes) {
    if (peer) {
        g_peer_hops.fetch_add(1, std::memory_order_relaxed);
        g_peer_bytes.fetch_add(bytes, std::memory_order_relaxed);
    } else {
        g_host_hops.fetch_add(1, std::memory_order_relaxed);
        g_host_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

// hop_out already counted a peer enqueue that hop_in could not run. The bytes went through the host.
void recount_as_host(size_t bytes) {
    if (g_peer_hops.load(std::memory_order_relaxed) > 0) g_peer_hops.fetch_sub(1, std::memory_order_relaxed);
    const uint64_t pb = g_peer_bytes.load(std::memory_order_relaxed);
    if (pb >= bytes) g_peer_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    g_host_hops.fetch_add(1, std::memory_order_relaxed);
    g_host_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

bool record_ev(cudaEvent_t ev, cudaStream_t stream, std::string& err) {
    if (ev == nullptr) {
        err = "glm p2p: hop event was not created";
        return false;
    }
    if (cudaEventRecord(ev, stream) != cudaSuccess) {
        err = std::string("glm p2p: event record failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

}  // namespace

bool env_on() {
    static const bool on = env_value_on(std::getenv("MAYA_GLM_P2P"));
    return on;
}

std::string prepare_split(const int* devs, int n, size_t hop_bytes) {
    if (!env_on()) return kHostSplitLog;
    if (devs == nullptr || n < 2) return std::string("MAYA_GLM_P2P=1, no split boundary");
    std::string out;
    std::lock_guard<std::mutex> lk(g_mu);
    for (int i = 0; i + 1 < n; ++i) {
        const int src = devs[i], dst = devs[i + 1];
        if (!out.empty()) out += ", ";
        char head[64];
        std::snprintf(head, sizeof(head), "CUDA%d->CUDA%d ", src, dst);
        out += head;
        Pair* p = pair_unlocked(src, dst);
        if (p == nullptr) {
            out += "host (device id out of range)";
            continue;
        }
        if (p->state == 0) probe_unlocked(*p, src, dst, hop_bytes);
        else if (p->state > 0 && hop_bytes > p->slot_bytes) {
            // A larger residual than the slots already probed: do not realloc under a live copy.
            p->state = -1;
            p->why = HopReasons::kBytes;
        }
        if (p->state > 0) {
            char tail[96];
            std::snprintf(tail, sizeof(tail), "peer (%zu B probe matched, %.3f ms)", p->slot_bytes, p->probe_ms);
            out += tail;
        } else {
            out += "host (";
            out += p->why.empty() ? HopReasons::kProbe : p->why;
            out += ")";
        }
    }
    out += "; MAYA_GLM_P2P=1";
    std::fprintf(stderr, "glm p2p: %s\n", out.c_str());
    std::fflush(stderr);
    return out;
}

bool pair_ready(int dst_dev, int src_dev) {
    if (!env_on() || !dev_ok(src_dev) || !dev_ok(dst_dev) || src_dev == dst_dev) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    Pair& p = g_pair[src_dev][dst_dev];
    ensure_unlocked(p, src_dev, dst_dev, size_t{64} << 10);
    return p.state > 0;
}

void latch_host(int src_dev, int dst_dev, const char* why) {
    if (!dev_ok(src_dev) || !dev_ok(dst_dev)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    Pair& p = g_pair[src_dev][dst_dev];
    if (p.state < 0) return;
    p.state = -1;
    p.why = why != nullptr ? why : HopReasons::kProbe;
    std::fprintf(stderr, "glm p2p: CUDA%d -> CUDA%d fell back to the host hop (%s)\n", src_dev, dst_dev,
                 p.why.c_str());
    std::fflush(stderr);
}

void note(int src_dev, int dst_dev, bool peer, size_t bytes, int slot, const char* where, bool announce) {
    if (!env_on()) return;
    count(peer, bytes);
    if (!announce || !dev_ok(src_dev) || !dev_ok(dst_dev)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    Pair* p = pair_unlocked(src_dev, dst_dev);
    if (p == nullptr) return;
    const bool window = where != nullptr && std::strcmp(where, "window") == 0;
    bool& said = window ? p->said_window : p->said_decode;
    if (said) return;
    said = true;
    std::fprintf(stderr, "glm p2p: first runtime hop CUDA%d -> CUDA%d queued %s (%s, %zu bytes, slot %d)\n", src_dev,
                 dst_dev, peer ? "peer" : "host", where != nullptr ? where : "decode", bytes, slot);
    std::fflush(stderr);
}

bool hop_out(int src_dev, int dst_dev, const float* src, size_t bytes, float* host, cudaStream_t src_stream,
             cudaEvent_t ev, int slot, std::string& err) {
    if (bytes == 0) return record_ev(ev, src_stream, err);
    bool peer = false;
    float* slotp = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        Pair* p = pair_unlocked(src_dev, dst_dev);
        if (p != nullptr) {
            ensure_unlocked(*p, src_dev, dst_dev, bytes);
            peer = select_hop(decision_of(*p), slot, kPeerSlots, bytes, p->slot_bytes).peer;
            if (peer) slotp = p->slot[slot];
        }
    }
    DevRestore restore;
    cudaSetDevice(src_dev);
    if (peer && slotp == nullptr) peer = false;
    if (peer) {
        if (cudaMemcpyAsync(slotp, src, bytes, cudaMemcpyDeviceToDevice, src_stream) != cudaSuccess) {
            const std::string ce = cudaGetErrorString(cudaGetLastError());
            latch_host(src_dev, dst_dev, ("peer snapshot failed: " + ce).c_str());
            peer = false;
        }
    }
    if (!peer) {
        if (host == nullptr) {
            err = "glm p2p: no host hop buffer";
            return false;
        }
        cudaSetDevice(src_dev);
        if (cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost, src_stream) != cudaSuccess) {
            err = std::string("glm p2p: host hop out failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    if (!record_ev(ev, src_stream, err)) return false;
    if (slot >= 0 && slot < kPeerSlots && dev_ok(src_dev) && dev_ok(dst_dev)) {
        std::lock_guard<std::mutex> lk(g_mu);
        Pair& p = g_pair[src_dev][dst_dev];
        p.issue_valid[slot] = true;
        p.issue_peer[slot] = peer;
    }
    note(src_dev, dst_dev, peer, bytes, slot, "decode", true);
    return true;
}

bool hop_in(int src_dev, int dst_dev, float* dst, size_t bytes, float* host, cudaStream_t dst_stream,
            cudaStream_t src_stream, cudaEvent_t ev, int slot, std::string& err, cudaEvent_t mark_after_wait) {
    if (bytes == 0) return true;
    bool peer = false;
    bool issued = false;
    float* slotp = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        Pair* p = pair_unlocked(src_dev, dst_dev);
        if (slot >= 0 && slot < kPeerSlots && p != nullptr && p->issue_valid[slot]) {
            peer = p->issue_peer[slot];
            issued = true;
            p->issue_valid[slot] = false;
            slotp = p->slot[slot];
        } else if (slot < 0 || slot >= kPeerSlots) {
            peer = false;
            issued = true;   // both sides treat an out-of-range slot as the host buffer
        }
    }
    if (!issued) {
        err = "glm p2p: hop in without a matching hop out";
        return false;
    }
    if (peer && slotp == nullptr) {
        err = "glm p2p: peer slot missing";
        return false;
    }
    DevRestore restore;
    cudaSetDevice(dst_dev);
    if (cudaStreamWaitEvent(dst_stream, ev, 0) != cudaSuccess) {
        err = std::string("glm p2p: hop wait failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    if (mark_after_wait != nullptr && !record_ev(mark_after_wait, dst_stream, err)) return false;
    bool fell_back = false;
    if (peer) {
        if (cudaMemcpyPeerAsync(dst, dst_dev, slotp, src_dev, bytes, dst_stream) == cudaSuccess) return true;
        const std::string ce = cudaGetErrorString(cudaGetLastError());
        latch_host(src_dev, dst_dev, ("peer copy failed: " + ce).c_str());
        // The snapshot is in the source slot. Move it through the host buffer on the source stream
        // and wait for that second record. The wait already queued above completes first.
        if (host == nullptr || src_stream == nullptr) {
            err = "glm p2p: peer copy failed and the host fallback has no buffer";
            return false;
        }
        cudaSetDevice(src_dev);
        if (cudaMemcpyAsync(host, slotp, bytes, cudaMemcpyDeviceToHost, src_stream) != cudaSuccess) {
            err = std::string("glm p2p: host fallback out failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (!record_ev(ev, src_stream, err)) return false;
        cudaSetDevice(dst_dev);
        if (cudaStreamWaitEvent(dst_stream, ev, 0) != cudaSuccess) {
            err = std::string("glm p2p: host fallback wait failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        fell_back = true;
    }
    if (host == nullptr) {
        err = "glm p2p: no host hop buffer";
        return false;
    }
    cudaSetDevice(dst_dev);
    if (cudaMemcpyAsync(dst, host, bytes, cudaMemcpyHostToDevice, dst_stream) != cudaSuccess) {
        err = std::string("glm p2p: host hop in failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    if (fell_back) recount_as_host(bytes);
    return true;
}

bool hop_sync(int src_dev, int dst_dev, float* dst, const float* src, size_t bytes, float* host, std::string& err) {
    if (bytes == 0) return true;
    bool peer = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        Pair* p = pair_unlocked(src_dev, dst_dev);
        if (p != nullptr) {
            ensure_unlocked(*p, src_dev, dst_dev, bytes);
            peer = p->state > 0 && bytes <= p->slot_bytes;
        }
    }
    DevRestore restore;
    if (peer) {
        cudaSetDevice(dst_dev);
        if (cudaMemcpyPeer(dst, dst_dev, src, src_dev, bytes) == cudaSuccess) {
            note(src_dev, dst_dev, true, bytes, 0, "step", true);
            return true;
        }
        latch_host(src_dev, dst_dev, (std::string("sync peer copy failed: ") + cudaGetErrorString(cudaGetLastError())).c_str());
    }
    if (host == nullptr) {
        err = "glm p2p: no host hop buffer";
        return false;
    }
    cudaSetDevice(src_dev);
    if (cudaMemcpy(host, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = std::string("glm p2p: sync host out failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    cudaSetDevice(dst_dev);
    if (cudaMemcpy(dst, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = std::string("glm p2p: sync host in failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    note(src_dev, dst_dev, false, bytes, 0, "step", true);
    return true;
}

std::string stat_line() {
    if (!env_on()) return {};
    const uint64_t ph = g_peer_hops.load(std::memory_order_relaxed);
    const uint64_t hh = g_host_hops.load(std::memory_order_relaxed);
    const uint64_t pb = g_peer_bytes.load(std::memory_order_relaxed);
    const uint64_t hb = g_host_bytes.load(std::memory_order_relaxed);
    std::string out = "glm p2p: runtime hops peer=" + std::to_string(ph) + " host=" + std::to_string(hh) +
                      " (peer " + std::to_string(pb) + " B, host " + std::to_string(hb) + " B)";
    std::lock_guard<std::mutex> lk(g_mu);
    bool any = false;
    for (int s = 0; s < kMaxDev; ++s) {
        for (int d = 0; d < kMaxDev; ++d) {
            const Pair& p = g_pair[s][d];
            if (p.state == 0) continue;
            if (!any) {
                out += "; ";
                any = true;
            } else {
                out += ", ";
            }
            out += "CUDA" + std::to_string(s) + "->CUDA" + std::to_string(d) + (p.state > 0 ? " peer" : " host");
            if (p.state < 0 && !p.why.empty()) {
                out += " (";
                out += p.why;
                out += ")";
            }
        }
    }
    return out;
}

}  // namespace strata::core::glmp2p
