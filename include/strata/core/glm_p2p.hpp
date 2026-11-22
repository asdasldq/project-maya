// include/strata/core/glm_p2p.hpp - opt-in CUDA peer handoff for a GLM layer-split decode hop.
//
// MAYA_GLM_P2P=1 (off when unset). The token residual between two parts is snapshotted on the source
// stream into a small device slot, then the destination stream peer-copies that slot after the source
// event. Direct state-to-state is not used: the next token on the source can be queued before the
// destination reads. The host pinned buffer remains the fallback when the pair was not probed peer,
// the slot does not fit, or the peer copy fails to enqueue. Prefill hops are unchanged.
#pragma once

#include "strata/core/glm_p2p_policy.hpp"

#include <cuda_runtime.h>

#include <string>

namespace strata::core::glmp2p {

bool env_on();

// Probe each consecutive pair. hop_bytes is one residual (Flash: hc * n_embd * 4). The returned
// text is the parenthetical of the "glm split:" line. Does not enable peer access when the knob is off.
std::string prepare_split(const int* devs, int n, size_t hop_bytes);

// Knob on and this ordered pair's probe matched. Window code asks this instead of enabling on its own.
bool pair_ready(int dst_dev, int src_dev);

// After a peer enqueue fails: later hops of this ordered pair use the host copy. Slots are kept
// (a copy already queued may still read them).
void latch_host(int src_dev, int dst_dev, const char* why);

// Queue the snapshot on src_stream and record ev. slot 0/1 are the spec double-buffer; any other
// index uses the host buffer (parallel sequence ids past the two slots).
bool hop_out(int src_dev, int dst_dev, const float* src, size_t bytes, float* host, cudaStream_t src_stream,
             cudaEvent_t ev, int slot, std::string& err);

// Wait ev on dst_stream, then fill dst. mark_after_wait, when set, is recorded on dst_stream after
// the wait and before the copy (MAYA_GLM_SPEC_PROF). src_stream is used only if the peer copy fails
// and this hop falls back to the host buffer.
bool hop_in(int src_dev, int dst_dev, float* dst, size_t bytes, float* host, cudaStream_t dst_stream,
            cudaStream_t src_stream, cudaEvent_t ev, int slot, std::string& err,
            cudaEvent_t mark_after_wait = nullptr);

// Reference step(): both sides idle. Peer memcpy of the residual, or the host vector if that fails.
bool hop_sync(int src_dev, int dst_dev, float* dst, const float* src, size_t bytes, float* host, std::string& err);

// Count one residual crossing. announce prints the first runtime line for this pair and class
// ("decode" or "window"). where "mtp-rows" is counted and not announced (those copies stay on the host).
void note(int src_dev, int dst_dev, bool peer, size_t bytes, int slot, const char* where, bool announce);

// Empty when the knob is off, so the default run prints nothing new.
std::string stat_line();

}  // namespace strata::core::glmp2p
