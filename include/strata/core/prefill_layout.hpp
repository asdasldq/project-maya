#pragma once
// What one measured prompt-path chunk does before any expert is lent and before any token is computed.
//
// A chunk that fits runs. A chunk longer than the shrink floor (256) that does not fit is measured again
// shorter. A chunk already under that floor has no smaller batched layout; the same tokens go through the
// token path. A miss exactly at the floor is the long prompt's memory limit and stays an error. This does
// not classify a CUDA failure, a lend failure, or a prompt that already ran.

namespace strata::core {

enum class PrefillLayoutStep { Run, Shrink, TokenPath, Error };

inline PrefillLayoutStep prefill_layout_step(bool fits, int trun, int floor) {
    if (fits) return PrefillLayoutStep::Run;
    if (trun > floor) return PrefillLayoutStep::Shrink;
    if (trun < floor) return PrefillLayoutStep::TokenPath;
    return PrefillLayoutStep::Error;
}

}  // namespace strata::core
