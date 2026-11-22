# GLM decode peer hop (`MAYA_GLM_P2P`)

Opt-in device-to-device handoff of the residual between GPUs during GLM-5.3-Flash decode. The host copy stays the default.

This was written against Maya **v1.1.74** (`861e211f0c9c5894ea9c8e6e033fe0b70bd5a94d`). It was not run on a V100. A 4x V100-SXM2 machine has to build it and check the logs below.

## What was wrong

`MAYA_GLM_P2P=1` only changed the speculative window (`src/core/glm_window.cu`). Token decode did not read it.

- `Glm5Model::fast_token` (plain decode, and `MAYA_GLM_NO_SPEC=1`) copied each boundary with `cudaMemcpyAsync` device-to-host, then host-to-device.
- `spec_head` / `spec_tail` (the default 4-GPU path: head group of two parts, tail group of two) did the same through pinned `spec_hop_h_` / `hop_h`.
- The load line `one 64 KB host hop per boundary per token` is printed for every split. It does not mean the copy that ran.

`MAYA_GLM_DUAL_PREFILL` is a two-GPU prompt path. It does not move decode residuals.

On the 4x V100-SXM2 split the residual is `hc * n_embd * 4` bytes (Flash: 4 * 4096 * 4 = 64 KiB). Three boundaries per token: layers 0-12, 13-23, 24-34, 35-44.

## What `1` does

At the end of the split load, each consecutive pair is probed:

1. `cudaDeviceCanAccessPeer` from the destination to the source.
2. `cudaDeviceEnablePeerAccess` both ways (the reverse enable is best-effort).
3. Two device slots of one residual (128 KiB on the source). A pattern is peer-copied into a temporary buffer and compared. The temporary buffer is freed.

If that matches, a decode hop snapshots the residual into the slot on the source stream, records the same event as before, and the destination stream peer-copies the slot after it waits on that event. The next token on the source is not allowed to overwrite the live residual before the destination has read it, so the copy is not state-to-state. Slots are the same indexes the speculative path already double-buffers (`p & 1` and `spec_n_ & 1`).

If the probe fails, the slot does not fit (sequence id 2 or higher on the parallel path; only two slots), or `cudaMemcpyPeerAsync` fails to enqueue, that hop uses the existing pinned host buffer. A failed enqueue latches the pair to the host path for the rest of the process and says so on stderr. Slots already queued are not freed.

Unset, empty, and `0` do not call the peer APIs. The split log stays exactly:

```
glm split: layers ... (one 64 KB host hop per boundary per token)
```

`on` is off: the knob is `atoi`, the same test as before. Use `1`.

Prefill hops are unchanged. Multi-conversation MTP window rows (`MAYA_GLM_MULTI_MTP`) stay on the host copy and are counted as host hops so they are not invisible.

## Build on the V100 host

CUDA 12.8 and g++ 13. Use a **new** build directory so the running `build/maya` is not replaced.

```sh
cmake -S . -B build-p2p -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DSTRATA_GGML_DIR="$HOME/builds/env/llama.cpp-3cf0325"
cmake --build build-p2p --target maya -j "$(nproc)"
```

Point `CMAKE_CUDA_COMPILER` and `CUDAToolkit_ROOT` at the toolkit whose `nvcc` is 12.8 (the driver build in use is CUDA 12.8.93). `STRATA_GGML_DIR` is the llama.cpp checkout Maya already uses (`3cf03257f219afbe7334045ff7c6a06ac68c627d`). If `ninja` is not installed, drop `-G Ninja`.

Host-only check, no GPU:

```sh
c++ -std=c++20 -I include tests/glm_p2p_policy_test.cpp -o /tmp/glm_p2p_policy_test && /tmp/glm_p2p_policy_test
```

## Run

Keep the current server up until this binary is ready. Start it separately, with the same weights, context, and `MAYA_GLM_P2P=1` already in the environment. Do not point it at the live binary's port until you mean to switch.

Startup, four cards, probe matched:

```
glm p2p: CUDA0->CUDA1 peer (65536 B probe matched, ... ms), CUDA1->CUDA2 peer ..., CUDA2->CUDA3 peer ...; MAYA_GLM_P2P=1
glm split: layers [0, 13) on CUDA0, ... (CUDA0->CUDA1 peer ...)
```

A boundary that did not probe peer is named `host` and the reason (`cudaDeviceCanAccessPeer=0`, `cudaDeviceEnablePeerAccess failed`, `peer slot alloc failed`, `peer copy probe failed: ...`).

During the first token of a boundary:

```
glm p2p: first runtime hop CUDA0 -> CUDA1 queued peer (decode, 65536 bytes, slot 0)
```

`window` in that line is the speculative window, not the token path. `step` is the reference runner (`MAYA_GLM_SLOW=1`), not the server's fast path.

After the reply, next to `glm stat:`:

```
glm p2p: runtime hops peer=N host=0 (peer ... B, host 0 B); CUDA0->CUDA1 peer, ...
```

For one 4-GPU sequence, `N` is 3 per token that crossed the split, and higher while speculative decode drafts and redoes. `host=0` with every boundary `peer` means the copies that ran were the peer path. `host` greater than 0 means at least one copy used the host buffer; the latch line says which pair.

## Quality and speed

The peer copy is the same residual bytes. KV cache, the sampler, and the speculative accept/reject rule are not changed. A temp-0 replay of the same prompt should match the host-path tokens. Cached prompt tokens should still be reported.

The 64 KiB NVLink copy is a small part of a token. A microbench of that copy (about 0.02 ms peer versus about 0.08 ms through the host) is not the generation speed. Judge tokens/s from `glm stat`, not from the probe time.

## Rollback

Stop the new process. Start the previous binary, or start this binary with `MAYA_GLM_P2P` unset or `0`. Peer access enabled in a process stays enabled until that process exits; rollback is a new process. The split log returns to `one 64 KB host hop per boundary per token`.

If a generation faults after the probe (illegal address, a hang), treat that as unsupported on this process: restart with the knob off. No weight, driver, IOMMU, or kernel change is required.

## Not tested here

No V100 and no CUDA runtime were available where this patch was built. The policy test checks the host default and the fallback rules only. Application, compile of `maya`, and generation checks belong on the server.
