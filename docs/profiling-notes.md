# Profiling vLLM: measured facts

Everything here was measured on 2026-08-08 against qwen3-32b fp8 on one
H100 SXM (RunPod secure cloud, vLLM 0.26.0). The `kvlm profile`
command carries the same facts as structured recipes; this file holds
the background that does not fit in a table cell.

## CUDA graphs: what is captured and what is not

vLLM (cudagraph_mode FULL_AND_PIECEWISE) captures the static-shape
decode compute — GEMMs, norms, quant, elementwise — as FULL graphs for
uniform decode batches and PIECEWISE graphs otherwise, at batch sizes
1..64. Its splitting_ops list is the roster of what never enters a
graph: unified attention (variable sequence lengths), KV-cache updates,
mamba and linear-attention (KDA) ops, the sparse-attention indexer.
Sampling and arbitrary-length prefill also run eagerly; decode batches
above max_cudagraph_capture_size (64) fall back to eager.

Profiler consequence, measured directly: an nsys trace WITHOUT
`--cuda-graph-trace=node` records cudaGraphLaunch as opaque blocks. An
A/B on identical load: 59,669 kernel records covering 5% of GPU time
(graph mode) vs 1,174,714 records covering 98% (node mode). The
graph-mode trace made prefill look dominant (nvjet GEMM 50%) purely
because decode was invisible; node mode showed the truth (decode fp8
GEMM 71.5%, FlashAttention 11.6%). Never read a vLLM trace taken
without node-level graph tracing. Cost of node mode: about 4x
cudaGraphLaunch API overhead and a 20x larger trace — use bounded
windows (`nsys start` / `nsys stop`).

## What is impossible on RunPod pods

- py-spy attach: ptrace_scope=1 on a read-only sysfs (error 13 copying
  Py_Version). Not fixable from inside the pod.
- perf: perf_event_paranoid=4; perf_event_open refused for every mode.
- ncu: ERR_NVGPUCTRPERM; GPU performance counters are disabled at the
  driver level (NVreg_RestrictProfilingToAdminUsers) and only the host
  owner can lift it.

What works: the Prometheus /metrics endpoint, and nsys in launch mode
(`nsys launch --session-new=...`), because CUPTI injection at process
start needs no ptrace. Trace flags belong to nsys launch, not nsys
start — start rejects them.

## vLLM 0.26 contract changes hit during the run

- VLLM_TORCH_PROFILER_DIR is no longer a known env var and POST
  /start_profile returns 404 even with VLLM_SERVER_DEV_MODE=1. The
  working replacement, encoded in the launch recipe since: the
  --profiler-config serve flag (it parses even though serve --help
  omits it; trust the ProfilerConfig line the server logs at startup).
- Metric renames: time_per_output_token_seconds became
  request_time_per_output_token_seconds; gpu_cache_usage_perc became
  kv_cache_usage_perc. request_success_total is one series per
  finished_reason and must be summed.
- --kv-cache-memory: the startup log prints the exact value to fully
  use the card (44.78 GiB on the measured H100 vs 37.4 default).

## Reference numbers (qwen3-32b fp8, 1xH100 SXM)

- single-stream decode 63 tok/s idle; per-user 64.5 / 49 / 31.6 tok/s
  at concurrency 1 / 8 / 32; aggregate 63 / 279 / 769 tok/s (unique
  random prompts)
- shared-prefix traffic added ~47% throughput (98.4% prefix-cache hits)
- prefill ~8,650 tok/s single stream, chunked
- KV pool 37.42 GiB = 306,528 tokens = 131,072 bytes/token exactly;
  "Maximum concurrency for 32,768 tokens per request: 9.35x"
- at KV saturation (96.2%, 18 running of pool/ctx = 18.4) vLLM queues
  rather than preempts; TTFT degrades to 31 s mean while decode holds
- measured non-KV overhead: 1.26 GiB activation + 0.22 non-torch +
  0.18 CUDA graphs
- SM utilization 100%, peak power 702 W against the 700 W cap under
  every load
