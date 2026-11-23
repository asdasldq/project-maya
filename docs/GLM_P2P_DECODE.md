# GLM decode peer hop (`MAYA_GLM_P2P`)

Opt-in device-to-device handoff of the residual between GPUs during GLM-5.3-Flash decode. The host copy stays the default.

## What a short prompt actually runs

On a split, `prefill()` returns without doing work when the prompt is shorter than 32 tokens (`MAYA_GLM_PREFILL_MIN`, default 32). An 18-token prompt is forwarded one token at a time by `fast_token`, and every one of those tokens crosses the peer hop when `MAYA_GLM_P2P=1`. The first completion token is sampled from that prompt forward. A wrong residual on any of those hops changes the opening of the reply. Speculative decode (`spec_head` / `spec_tail`) uses the same hop for later tokens. This was not measured here; the 18-versus-32 cut is what the source does.

## Ordering

The first revision peer-read the source buffer from the destination stream after a cross-device event. A synchronous probe can still match on that machine: the probe waits until the copy finishes. The token path does not. The producer stream now peer-writes the residual straight into a 2-slot bounce on the destination (128 KiB on each GPU that receives a hop), records the hop event after that write, and the destination copies the bounce into its residual with an ordinary same-device copy. The event means the bytes have landed, which is the host hop's contract. The window path (`MAYA_GLM_SPEC_WIN`) is separate and is not what an 18-token prompt uses.

A text difference between `MAYA_GLM_P2P=1` and `0` does not by itself prove a bad copy, and a matching opening does not by itself prove the bytes match. `MAYA_GLM_P2P_CHECK=1` compares them.

## Byte check

Set both:

```sh
MAYA_GLM_P2P=1
MAYA_GLM_P2P_CHECK=1
```

On each peer hop the producer stream also copies the residual into the existing pinned hop buffer before the event. After the consumer has installed the bounce, and before its layers, that stream copies its residual back out and compares it to the producer snapshot. The residual the layers read is not modified. The check adds one 64 KiB device-to-host copy each way and a host callback; it is for a correctness run, not a speed run. Host RAM for the consumer side is two 64 KiB buffers per boundary. No extra device memory beyond the bounce.

After the reply the stat line has, once the consumer callbacks have been waited for:

```text
glm p2p: runtime hops peer=N host=0 ... queued=N consumed=C overwritten=R outstanding=U check match=C mismatch=0 pending=0 skipped=0 consumed_checked=1
```

`queued` is every peer snapshot. A rejected draft writes the head-to-tail slot and the redo writes that same slot before the tail reads it: the first snapshot is `overwritten`, not a consumer that skipped the check. The request can also stop with one speculative handoff still `outstanding`. `consumed` is a residual the next GPU actually installed. `consumed_checked=1` with `pending=0` and `mismatch=0` means every installed residual matched its producer snapshot. `match` is not supposed to equal `queued`. A mismatch line names the boundary, slot, and first byte offset.

On the rev2 log this arithmetic is 223 queued, 200 consumed, gap 23. That is 22 rejected drafts (40 drafted, 18 accepted) plus one speculative handoff left queued when the reply stopped, and the follow-up (8 of 8 drafts accepted) adds one more outstanding. The check callbacks that did run reported no byte mismatch. The text still differed from the host run. The same logs show different expert traffic on that first reply (peer: vram hit 86.74%, ram fetches 0.33/tok, promotions 0.36/tok, cpu lane 56.36/tok; host: 87.95%, 0.67, 0.67, 48.08). The hop check does not compare those two runs to each other. It only compares each consumer to its own producer. A different expert set can change the producer residual before the copy.

Next distinction, only after this accounting build prints `consumed_checked=1` and `mismatch=0`: stop the server, then run the same temp-0 prompt twice with `MAYA_GLM_CPU_LANE=0`, once with `MAYA_GLM_P2P=0` and once with `MAYA_GLM_P2P=1` (check on). That turns the CPU expert lane off so timing cannot move experts between CPU and GPU. If the texts still differ, the divergence is not that lane. If they match, the earlier difference followed the lane's timing. Do not treat either result as a speed measurement.

Do not start this through `maya.py` or `maya.sh`. Those rebuild `build/maya` from the tree and can replace the restored engine. Stop the running server first so two processes do not map the same weights. Point the existing `server.py` wrapper at the new binary only, and check that file's sha256 before the request. Leave the restored official binary in place.

## Window drafts versus one-draft decode

`MAYA_GLM_SPEC_WIN=1` with `MAYA_GLM_MTP_FIXED=4` or `2` was not faster than `decode_spec` on the short prompts. K=2 cut the hops and left the reported CPU-lane ms/tok essentially unchanged, so that counter is not evidence that CPU experts are the wall-clock limit. `glm stat`'s cpu lane ms/tok adds every GPU part's own CPU compute and divides by output tokens. Parts overlap, so the sum is not the time a GPU waited. A window hop logged at slot 16 is the second pipeline bank's first token (`kPipeMax` is 16), one 64 KiB residual, not sixteen copies.

`MAYA_GLM_PIPE_PROF=1` (off by default; leave it off for a speed run) prints, after the reply:

- `glm cpu prof`: the sum and each part, plus the largest part. Compare the max-part ms/tok with the summed stat line.
- On the window path only, `glm pipe prof`: this thread's ms/round in draft, pick, commit, and cache, plus rows queued and kept, and whether the draft chain waited once (`chain rounds`) or once per draft (`step rounds`).

Use one warmed request of about 128–256 tokens after a throwaway of the same prompt, temp 0, thinking off, for `decode_spec` and for the window. Do not compare a cold start with a warm one. No speed claim follows from this instrumentation alone.

Leave the running service on the previous binary until this one is started beside it. Roll back by exiting the process; `MAYA_GLM_P2P` unset or `0` does not enable peer access and does not run the check.

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

If that matches, the producer stream peer-writes the live residual into a 2-slot bounce allocated on the destination (128 KiB there), then records the hop event. The destination waits on that event and copies the bounce into its residual locally. The same slot indexes as the speculative host buffers are used (`p & 1`, `spec_n_ & 1`, and slot 0 for the single-buffer tail hop). The producer's next kernels are behind the peer write on its own stream, so they do not overwrite the residual early.

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

```text
glm p2p: CUDA0->CUDA1 peer src-stream (65536 B probe matched, ... ms), ...; MAYA_GLM_P2P=1
glm split: layers [0, 13) on CUDA0, ... (CUDA0->CUDA1 peer src-stream ...)
```

A boundary that did not probe peer is named `host` and the reason (`cudaDeviceCanAccessPeer=0`, `cudaDeviceEnablePeerAccess failed`, `peer slot alloc failed`, `peer copy probe failed: ...`).

During the first token of a boundary:

```
glm p2p: first runtime hop CUDA0 -> CUDA1 queued peer (decode, 65536 bytes, slot 0)
```

`window` in that line is the speculative window, not the token path. `step` is the reference runner (`MAYA_GLM_SLOW=1`), not the server's fast path.

After the reply, next to `glm stat:`:

```text
glm p2p: runtime hops peer=N host=0 (peer ... B, host 0 B) check match=N mismatch=0; CUDA0->CUDA1 peer, ...
```

`check match=` is present only with `MAYA_GLM_P2P_CHECK=1`. Without that variable the line has no check counts, and a matched load probe still does not say the consumer residual matched.

For one 4-GPU sequence, `N` is 3 per token that crossed the split, and higher while speculative decode drafts and redoes. `host=0` with every boundary `peer` means the copies that ran were the peer path. `host` greater than 0 means at least one copy used the host buffer; the latch line says which pair.

## Quality and speed

The peer copy is the same residual bytes. KV cache, the sampler, and the speculative accept/reject rule are not changed. A temp-0 replay of the same prompt should match the host-path tokens. Cached prompt tokens should still be reported.

The 64 KiB NVLink copy is a small part of a token. A microbench of that copy (about 0.02 ms peer versus about 0.08 ms through the host) is not the generation speed. Judge tokens/s from `glm stat`, not from the probe time.

## Rollback

Stop the new process. Start the previous binary, or start this binary with `MAYA_GLM_P2P` unset or `0`. Peer access enabled in a process stays enabled until that process exits; rollback is a new process. The split log returns to `one 64 KB host hop per boundary per token`.

If a generation faults after the probe (illegal address, a hang), treat that as unsupported on this process: restart with the knob off. No weight, driver, IOMMU, or kernel change is required.

## Not tested here

No V100 and no CUDA runtime were available where this patch was written. The policy test checks the host default, the fallback rules, and the byte-compare helper only. It does not show that a consumer residual matches. That is `MAYA_GLM_P2P_CHECK=1` on the server. A previous run's token text is not a substitute for `mismatch=0`.
