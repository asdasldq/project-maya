// include/strata/core/glm_mtp_subset.hpp - when a NextN draft argmax stays on the draft vocabulary.
//
// No CUDA. fast_mtp applies this. decode_spec passes caller_sub false and updates mtp_out on every
// emitted token. On 3+ GPUs the subset is then used only while that share stays under 0.02.
// caller_sub true forces the subset whenever mtp_dv > 0, which a full-head verify cannot accept
// once the answer has left the draft vocabulary.
#pragma once

namespace strata::core {

inline bool mtp_draft_subset(bool caller_sub, bool full_head, bool sample_draft, int n_parts, double mtp_out,
                             int mtp_dv) {
    return (caller_sub || (!full_head && !sample_draft && n_parts > 2 && mtp_out < 0.02)) && mtp_dv > 0;
}

}  // namespace strata::core
