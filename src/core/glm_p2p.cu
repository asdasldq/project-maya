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
    int home = -1;   // destination device: bounce[] lives there
    std::string why;
    float* slot[kWindowSlots] = {};   // destination bounce, one residual each
    size_t slot_bytes = 0;
    unsigned char* got[kWindowSlots] = {};   // pinned consumer copy, only with MAYA_GLM_P2P_CHECK
    double probe_ms = 0;
    bool said_decode = false;
    bool said_window = false;
    bool issue_valid[kWindowSlots] = {};
    bool issue_peer[kWindowSlots] = {};
};

struct CheckCookie {
    const unsigned char* ref = nullptr;
    const unsigned char* got = nullptr;
    size_t bytes = 0;
    int src = 0, dst = 0, slot = 0;
};

std::mutex g_mu;
Pair g_pair[kMaxDev][kMaxDev];
CheckCookie g_cookie[kMaxDev][kMaxDev][kWindowSlots];
std::atomic<uint64_t> g_peer_hops{0};
std::atomic<uint64_t> g_host_hops{0};
std::atomic<uint64_t> g_peer_bytes{0};
std::atomic<uint64_t> g_host_bytes{0};
std::atomic<uint64_t> g_check_match{0};
std::atomic<uint64_t> g_check_mismatch{0};
std::atomic<uint64_t> g_queued{0};
std::atomic<uint64_t> g_overwritten{0};
std::atomic<uint64_t> g_consumed{0};
std::atomic<uint64_t> g_launched{0};
std::atomic<uint64_t> g_skipped{0};
std::atomic<int> g_check_logged{0};

bool dev_ok(int d) { return d >= 0 && d < kMaxDev; }

bool check_on() {
    static const bool on = env_value_on(std::getenv("MAYA_GLM_P2P_CHECK"));
    return on;
}

void free_slots(Pair& p) {
    if (p.home >= 0) cudaSetDevice(p.home);
    for (int i = 0; i < kWindowSlots; ++i) {
        if (p.slot[i] != nullptr) cudaFree(p.slot[i]);
        if (p.got[i] != nullptr) cudaFreeHost(p.got[i]);
        p.slot[i] = nullptr;
        p.got[i] = nullptr;
        p.issue_valid[i] = false;
    }
    p.slot_bytes = 0;
}

void check_cb(void* raw) {
    std::atomic_thread_fence(std::memory_order_acquire);
    const auto* c = static_cast<const CheckCookie*>(raw);
    if (c == nullptr || c->ref == nullptr || c->got == nullptr || c->bytes == 0) return;
    const size_t at = first_mismatch(c->ref, c->got, c->bytes);
    if (at == c->bytes) {
        g_check_match.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_check_mismatch.fetch_add(1, std::memory_order_relaxed);
    if (g_check_logged.exchange(1, std::memory_order_relaxed) == 0) {
        std::fprintf(stderr,
                     "glm p2p: CHECK mismatch CUDA%d -> CUDA%d slot %d byte %zu (producer 0x%02x, consumer 0x%02x, %zu bytes)\n",
                     c->src, c->dst, c->slot, at, c->ref[at], c->got[at], c->bytes);
        std::fflush(stderr);
    }
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
    // The runtime copy is a producer-stream peer write into destination memory. Enable the source
    // to address the destination. The reverse enable stays best-effort.
    cudaSetDevice(src);
    pe = cudaDeviceEnablePeerAccess(dst, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) {
        p.why = std::string(HopReasons::kEnable) + ": " + cudaGetErrorString(pe);
        (void) cudaGetLastError();
        return false;
    }

    p.home = dst;
    cudaSetDevice(dst);
    for (int i = 0; i < kPeerSlots; ++i) {
        if (cudaMalloc(&p.slot[i], bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            free_slots(p);
            p.why = HopReasons::kAlloc;
            return false;
        }
        if (check_on() && cudaHostAlloc(&p.got[i], bytes, cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "glm p2p: CUDA%d -> CUDA%d check buffer did not allocate; comparison off for this pair\n",
                         src, dst);
            std::fflush(stderr);
            if (p.got[i] != nullptr) cudaFreeHost(p.got[i]);
            p.got[i] = nullptr;
        }
    }
    p.slot_bytes = bytes;
    cudaSetDevice(src);
    void* src_tmp = nullptr;
    if (cudaMalloc(&src_tmp, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        free_slots(p);
        p.why = HopReasons::kAlloc;
        return false;
    }
    std::vector<unsigned char> host(bytes), back(bytes);
    for (size_t i = 0; i < bytes; ++i) host[i] = (unsigned char) (i * 131u + 17u);
    bool matched = true;
    std::string fail;
    double ms = 0;
    // Same direction as the runtime hop: source memory, producer pushes into the destination bounce.
    if (cudaMemcpy(src_tmp, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fail = std::string(HopReasons::kProbe) + ": " + cudaGetErrorString(cudaGetLastError());
        matched = false;
    }
    for (int s = 0; s < kPeerSlots && matched; ++s) {
        cudaSetDevice(src);
        const auto t0 = std::chrono::steady_clock::now();
        const cudaError_t ce = cudaMemcpyPeer(p.slot[s], dst, src_tmp, src, bytes);
        const auto t1 = std::chrono::steady_clock::now();
        if (ce != cudaSuccess) {
            fail = std::string(HopReasons::kProbe) + ": " + cudaGetErrorString(ce);
            (void) cudaGetLastError();
            matched = false;
            break;
        }
        ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        cudaSetDevice(dst);
        if (cudaMemcpy(back.data(), p.slot[s], bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            fail = std::string(HopReasons::kProbe) + ": " + cudaGetErrorString(cudaGetLastError());
            matched = false;
            break;
        }
        if (first_mismatch(host.data(), back.data(), bytes) != bytes) {
            fail = std::string(HopReasons::kProbe) + ": byte mismatch";
            matched = false;
        }
    }
    cudaSetDevice(src);
    cudaFree(src_tmp);
    if (!matched) {
        free_slots(p);
        p.why = fail.empty() ? HopReasons::kProbe : fail;
        p.state = -1;
        return false;
    }
    p.state = 1;
    p.probe_ms = ms / kPeerSlots;
    // Slots past the probe (the pipe window's bank * 16 + token) are allocated on first use.
    p.why = check_on() ? "peer src-stream" : HopReasons::kPeer;
    if (check_on()) p.why += " check";
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

// Caller holds g_mu. Slots [0, kPeerSlots) come from the probe. A pipe-window slot past those
// is allocated here, on the destination, the first time it is used. A failed alloc does not
// latch the pair, so decode slots 0/1 stay on the peer path.
bool ensure_window_slot(Pair& p, int slot) {
    if (slot < 0 || slot >= kWindowSlots) return false;
    if (slot < kPeerSlots) return p.slot[slot] != nullptr;
    if (p.slot[slot] != nullptr) return true;
    if (p.state != 1 || p.home < 0 || p.slot_bytes == 0) return false;
    DevRestore restore;
    cudaSetDevice(p.home);
    if (cudaMalloc(&p.slot[slot], p.slot_bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        p.slot[slot] = nullptr;
        return false;
    }
    if (check_on() && p.got[slot] == nullptr) {
        if (cudaHostAlloc(&p.got[slot], p.slot_bytes, cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            p.got[slot] = nullptr;
        }
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
            char tail[128];
            std::snprintf(tail, sizeof(tail), "peer src-stream (%zu B probe matched, %.3f ms%s)", p->slot_bytes,
                          p->probe_ms, check_on() ? ", check" : "");
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
             cudaEvent_t ev, int slot, std::string& err, int slot_limit, const char* where) {
    if (bytes == 0) return record_ev(ev, src_stream, err);
    bool peer = false;
    float* slotp = nullptr;
    const int lim = slot_limit < 1 ? kPeerSlots : (slot_limit > kWindowSlots ? kWindowSlots : slot_limit);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        Pair* p = pair_unlocked(src_dev, dst_dev);
        if (p != nullptr) {
            ensure_unlocked(*p, src_dev, dst_dev, bytes);
            // A decode or sequence-id limit of 2 must not allocate a pipe-window slot.
            const bool need_extra = slot >= kPeerSlots && slot < lim;
            const bool slot_ok = !need_extra || ensure_window_slot(*p, slot);
            peer = slot_ok && select_hop(decision_of(*p), slot, lim, bytes, p->slot_bytes).peer;
            if (peer && slot >= 0 && slot < kWindowSlots) slotp = p->slot[slot];
        }
    }
    DevRestore restore;
    cudaSetDevice(src_dev);
    if (peer && slotp == nullptr) peer = false;
    if (peer) {
        // Push from the stream that just wrote the residual. The event below means the bounce holds
        // those bytes, which is what the destination waits for.
        if (cudaMemcpyPeerAsync(slotp, dst_dev, src, src_dev, bytes, src_stream) != cudaSuccess) {
            const std::string ce = cudaGetErrorString(cudaGetLastError());
            latch_host(src_dev, dst_dev, ("peer write failed: " + ce).c_str());
            peer = false;
        } else if (check_on() && host != nullptr) {
            if (cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost, src_stream) != cudaSuccess) {
                (void) cudaGetLastError();
                std::fprintf(stderr, "glm p2p: CUDA%d -> CUDA%d check snapshot failed; comparison skipped\n", src_dev,
                             dst_dev);
                std::fflush(stderr);
            }
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
    if (slot >= 0 && slot < lim && dev_ok(src_dev) && dev_ok(dst_dev)) {
        std::lock_guard<std::mutex> lk(g_mu);
        Pair& p = g_pair[src_dev][dst_dev];
        // A rejected draft writes this slot, then the redo writes it again before the tail reads.
        // The first snapshot is never installed. Count it here, not as a missed consumer check.
        if (peer && p.issue_valid[slot] && p.issue_peer[slot])
            g_overwritten.fetch_add(1, std::memory_order_relaxed);
        p.issue_valid[slot] = true;
        p.issue_peer[slot] = peer;
    }
    if (peer) g_queued.fetch_add(1, std::memory_order_relaxed);
    note(src_dev, dst_dev, peer, bytes, slot, where != nullptr ? where : "decode", true);
    return true;
}

bool hop_in(int src_dev, int dst_dev, float* dst, size_t bytes, float* host, cudaStream_t dst_stream,
            cudaStream_t src_stream, cudaEvent_t ev, int slot, std::string& err, cudaEvent_t mark_after_wait,
            int slot_limit) {
    if (bytes == 0) return true;
    bool peer = false;
    bool issued = false;
    float* slotp = nullptr;
    const int lim = slot_limit < 1 ? kPeerSlots : (slot_limit > kWindowSlots ? kWindowSlots : slot_limit);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        Pair* p = pair_unlocked(src_dev, dst_dev);
        if (slot >= 0 && slot < lim && p != nullptr && p->issue_valid[slot]) {
            peer = p->issue_peer[slot];
            issued = true;
            p->issue_valid[slot] = false;
            slotp = p->slot[slot];
        } else if (slot < 0 || slot >= lim) {
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
    if (peer) {
        unsigned char* got = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            Pair* p = pair_unlocked(src_dev, dst_dev);
            if (p != nullptr && slot >= 0 && slot < lim) got = p->got[slot];
        }
        // Local copy. The bounce already holds the residual: the event waited for the producer push.
        if (cudaMemcpyAsync(dst, slotp, bytes, cudaMemcpyDeviceToDevice, dst_stream) != cudaSuccess) {
            const std::string ce = cudaGetErrorString(cudaGetLastError());
            latch_host(src_dev, dst_dev, ("bounce install failed: " + ce).c_str());
            err = "glm p2p: the destination bounce did not install";
            return false;
        }
        bool launched = false;
        if (check_on() && got != nullptr && host != nullptr && dev_ok(src_dev) && dev_ok(dst_dev) && slot >= 0 &&
            slot < lim) {
            if (cudaMemcpyAsync(got, dst, bytes, cudaMemcpyDeviceToHost, dst_stream) != cudaSuccess) {
                (void) cudaGetLastError();
            } else {
                CheckCookie& c = g_cookie[src_dev][dst_dev][slot];
                c.ref = reinterpret_cast<const unsigned char*>(host);
                c.got = got;
                c.bytes = bytes;
                c.src = src_dev;
                c.dst = dst_dev;
                c.slot = slot;
                std::atomic_thread_fence(std::memory_order_release);
                if (cudaLaunchHostFunc(dst_stream, check_cb, &c) == cudaSuccess) launched = true;
                else (void) cudaGetLastError();
            }
        }
        g_consumed.fetch_add(1, std::memory_order_relaxed);
        if (launched) g_launched.fetch_add(1, std::memory_order_relaxed);
        else g_skipped.fetch_add(1, std::memory_order_relaxed);
        (void) src_stream;
        return true;
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
    (void) src_stream;
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
    // Once per request print. Host callbacks already queued on the consumer streams finish here,
    // so pending is not "the compare has not run yet".
    if (check_on()) {
        int n = 0;
        if (cudaGetDeviceCount(&n) == cudaSuccess) {
            DevRestore restore;
            for (int d = 0; d < n; ++d) {
                cudaSetDevice(d);
                cudaDeviceSynchronize();
            }
        }
    }
    const uint64_t ph = g_peer_hops.load(std::memory_order_relaxed);
    const uint64_t hh = g_host_hops.load(std::memory_order_relaxed);
    const uint64_t pb = g_peer_bytes.load(std::memory_order_relaxed);
    const uint64_t hb = g_host_bytes.load(std::memory_order_relaxed);
    std::string out = "glm p2p: runtime hops peer=" + std::to_string(ph) + " host=" + std::to_string(hh) +
                      " (peer " + std::to_string(pb) + " B, host " + std::to_string(hb) + " B)";
    if (check_on()) {
        HopLedger led;
        led.queued = g_queued.load(std::memory_order_relaxed);
        led.overwritten = g_overwritten.load(std::memory_order_relaxed);
        led.consumed = g_consumed.load(std::memory_order_relaxed);
        led.launched = g_launched.load(std::memory_order_relaxed);
        led.match = g_check_match.load(std::memory_order_relaxed);
        led.mismatch = g_check_mismatch.load(std::memory_order_relaxed);
        led.skipped = g_skipped.load(std::memory_order_relaxed);
        const HopLedgerView v = view_ledger(led);
        out += " queued=" + std::to_string(led.queued) + " consumed=" + std::to_string(led.consumed) +
               " overwritten=" + std::to_string(led.overwritten) + " outstanding=" + std::to_string(v.outstanding) +
               " check match=" + std::to_string(led.match) + " mismatch=" + std::to_string(led.mismatch) +
               " pending=" + std::to_string(v.pending) + " skipped=" + std::to_string(led.skipped);
        if (!v.arithmetic_ok) out += " ledger=bad";
        else if (v.consumed_checked) out += " consumed_checked=1";
        else out += " consumed_checked=0";
    }
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
