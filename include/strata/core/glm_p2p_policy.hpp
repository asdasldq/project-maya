// include/strata/core/glm_p2p_policy.hpp - host-side choice for a GLM split's decode residual hop.
//
// No CUDA here: the runtime in src/core/glm_p2p.cu applies this, and tests/glm_p2p_policy_test.cpp locks it.
// MAYA_GLM_P2P stays off unless the variable is a non-zero integer (the same atoi test the window path used).
// A peer hop is chosen only when the knob is on AND the driver probe succeeded AND this hop fits a slot.
#pragma once

#include <cstddef>
#include <cstdlib>

namespace strata::core::glmp2p {

inline constexpr int kPeerSlots = 2;
inline constexpr int kMaxDev = 16;
// The split log when the knob is off. Byte-for-byte the line Maya printed before this path existed.
inline constexpr const char* kHostSplitLog = "one 64 KB host hop per boundary per token";

enum class HopPath { Host, Peer };

struct HopDecision {
    HopPath path;
    const char* reason;
};

struct HopReasons {
    static constexpr const char* kOff = "MAYA_GLM_P2P off";
    static constexpr const char* kNoPeer = "cudaDeviceCanAccessPeer=0";
    static constexpr const char* kEnable = "cudaDeviceEnablePeerAccess failed";
    static constexpr const char* kAlloc = "peer slot alloc failed";
    static constexpr const char* kProbe = "peer copy probe failed";
    static constexpr const char* kPeer = "peer";
    static constexpr const char* kSlot = "no peer slot for this hop";
    static constexpr const char* kBytes = "hop larger than the peer slot";
};

// True only for the existing knob: unset, empty, and "0" stay off. "on" stays off (atoi).
inline bool env_value_on(const char* v) { return v != nullptr && std::atoi(v) != 0; }

// env_on is first so a closed knob cannot become a peer hop, whatever the probe flags say.
inline HopDecision decide(bool env_on, bool can_access, bool enable_ok, bool slots_ok, bool probe_ok) {
    if (!env_on) return {HopPath::Host, HopReasons::kOff};
    if (!can_access) return {HopPath::Host, HopReasons::kNoPeer};
    if (!enable_ok) return {HopPath::Host, HopReasons::kEnable};
    if (!slots_ok) return {HopPath::Host, HopReasons::kAlloc};
    if (!probe_ok) return {HopPath::Host, HopReasons::kProbe};
    return {HopPath::Peer, HopReasons::kPeer};
}

struct HopSelect {
    bool peer;
    const char* reason;
};

// slot is the caller's buffer index (spec double-buffer 0/1, or a sequence id). Past the probed
// slots, or a hop bigger than the slot, the bytes stay on the host copy for that hop.
inline HopSelect select_hop(const HopDecision& d, int slot, int nslots, size_t bytes, size_t slot_bytes) {
    if (d.path != HopPath::Peer) return {false, d.reason};
    if (slot < 0 || slot >= nslots) return {false, HopReasons::kSlot};
    if (bytes == 0 || slot_bytes == 0 || bytes > slot_bytes) return {false, HopReasons::kBytes};
    return {true, HopReasons::kPeer};
}

// Byte offset of the first difference, or n when the ranges are equal. The runtime check compares
// the consumer's residual to the producer snapshot with this.
inline size_t first_mismatch(const void* a, const void* b, size_t n) {
    const auto* x = static_cast<const unsigned char*>(a);
    const auto* y = static_cast<const unsigned char*>(b);
    for (size_t i = 0; i < n; ++i)
        if (x[i] != y[i]) return i;
    return n;
}

}  // namespace strata::core::glmp2p
