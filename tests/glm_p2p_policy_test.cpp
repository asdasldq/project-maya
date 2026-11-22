// tests/glm_p2p_policy_test.cpp - host-only rules for MAYA_GLM_P2P. No GPU.
#include "strata/core/glm_p2p_policy.hpp"

#include <cstdio>
#include <cstring>

namespace gp = strata::core::glmp2p;

static int g_fail = 0;

#define CHECK(c)                                                                                 \
    do {                                                                                         \
        if (!(c)) {                                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                    \
            ++g_fail;                                                                            \
        }                                                                                        \
    } while (0)

static void test_env() {
    CHECK(!gp::env_value_on(nullptr));
    CHECK(!gp::env_value_on(""));
    CHECK(!gp::env_value_on("0"));
    CHECK(!gp::env_value_on("00"));
    CHECK(!gp::env_value_on("on"));     // atoi, same as the old window test
    CHECK(!gp::env_value_on("false"));
    CHECK(gp::env_value_on("1"));
    CHECK(gp::env_value_on("2"));
    CHECK(gp::env_value_on(" 1"));
}

static void test_default_is_host() {
    // Every probe flag true still cannot select peer while the knob is off.
    const gp::HopDecision d = gp::decide(false, true, true, true, true);
    CHECK(d.path == gp::HopPath::Host);
    CHECK(std::strcmp(d.reason, gp::HopReasons::kOff) == 0);
    const gp::HopSelect s = gp::select_hop(d, 0, gp::kPeerSlots, 65536, 65536);
    CHECK(!s.peer);
    CHECK(std::strcmp(s.reason, gp::HopReasons::kOff) == 0);
    CHECK(std::strcmp(gp::kHostSplitLog, "one 64 KB host hop per boundary per token") == 0);
}

static void test_probe_failures_stay_host() {
    CHECK(gp::decide(true, false, true, true, true).path == gp::HopPath::Host);
    CHECK(std::strcmp(gp::decide(true, false, true, true, true).reason, gp::HopReasons::kNoPeer) == 0);
    CHECK(std::strcmp(gp::decide(true, true, false, true, true).reason, gp::HopReasons::kEnable) == 0);
    CHECK(std::strcmp(gp::decide(true, true, true, false, true).reason, gp::HopReasons::kAlloc) == 0);
    CHECK(std::strcmp(gp::decide(true, true, true, true, false).reason, gp::HopReasons::kProbe) == 0);
    const gp::HopDecision peer = gp::decide(true, true, true, true, true);
    CHECK(peer.path == gp::HopPath::Peer);
    CHECK(std::strcmp(peer.reason, gp::HopReasons::kPeer) == 0);
}

static void test_slot_and_size() {
    const gp::HopDecision peer = gp::decide(true, true, true, true, true);
    CHECK(gp::select_hop(peer, 0, gp::kPeerSlots, 65536, 65536).peer);
    CHECK(gp::select_hop(peer, 1, gp::kPeerSlots, 65536, 65536).peer);
    CHECK(gp::select_hop(peer, 1, gp::kPeerSlots, 4096, 65536).peer);   // smaller residual still fits
    const gp::HopSelect over = gp::select_hop(peer, 2, gp::kPeerSlots, 65536, 65536);
    CHECK(!over.peer);
    CHECK(std::strcmp(over.reason, gp::HopReasons::kSlot) == 0);
    const gp::HopSelect big = gp::select_hop(peer, 0, gp::kPeerSlots, 65537, 65536);
    CHECK(!big.peer);
    CHECK(std::strcmp(big.reason, gp::HopReasons::kBytes) == 0);
    CHECK(!gp::select_hop(peer, 0, gp::kPeerSlots, 0, 65536).peer);
    CHECK(!gp::select_hop(peer, -1, gp::kPeerSlots, 65536, 65536).peer);
}

int main() {
    test_env();
    test_default_is_host();
    test_probe_failures_stay_host();
    test_slot_and_size();
    if (g_fail != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("glm_p2p_policy_test ok\n");
    return 0;
}
