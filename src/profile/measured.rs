// measured: the baseline reference numbers as structured data. The
// dashboard and any future estimator read these; nothing re-types
// them as prose. Provenance for every value is the Run's Provenance
// field (measured 2026-08-08, qwen3-32b fp8 on 1xH100 SXM, RunPod
// secure cloud, vLLM 0.26.0).
#![allow(non_snake_case)]

use goish::string;
use goish::{float64, int, slice};

// ConcurrencyPoint is one row of the bench sweep (unique random
// prompts, 1K in / 256 out).
#[derive(Clone, Default)]
pub struct ConcurrencyPoint {
    pub Concurrency: int,
    pub PerUserTokS: float64,
    pub AggregateTokS: float64,
    pub TTFTMeanMs: float64,
    pub TTFTP99Ms: float64,
    pub TPOTMeanMs: float64,
}

// KernelShare is one row of the node-mode nsys kernel summary (the
// honest one; graph-mode hides 95% of GPU work).
#[derive(Clone, Default)]
pub struct KernelShare {
    pub Name: string,
    pub Pct: float64,
}

// Fact is one headline number with its unit.
#[derive(Clone, Default)]
pub struct Fact {
    pub Label: string,
    pub Value: string,
}

// Metric is one measured number as data: the value and unit stay
// numeric so displays are computed, never re-typed as prose. Max > 0
// gives the metric a scale (the dashboard draws a gauge; value over
// Max renders as an overshoot). Future collection runs append Metric
// entries and every consumer picks them up unchanged.
#[derive(Clone, Default)]
pub struct Metric {
    pub Key: string,
    pub Label: string,
    pub Value: float64,
    pub Unit: string,
    pub Max: float64,
    pub Note: string,
}

// Run bundles the whole measured reference set.
#[derive(Clone, Default)]
pub struct Run {
    pub Provenance: string,
    pub Curve: slice<ConcurrencyPoint>,
    pub Kernels: slice<KernelShare>,
    pub Metrics: slice<Metric>,
    pub Saturation: slice<Metric>,
    pub GraphMetrics: slice<Metric>,
    pub CudaGraphs: slice<Fact>,
    pub NeverCaptured: slice<string>,
}

fn fact(label: &'static str, value: &'static str) -> Fact {
    Fact {
        Label: string(label),
        Value: string(value),
    }
}

// Reference returns the measured run data.
pub fn Reference() -> Run {
    Run {
        Provenance: string("measured 2026-08-08, qwen3-32b fp8 on 1xH100 SXM, RunPod secure cloud, vLLM 0.26.0"),
        Curve: slice!([]ConcurrencyPoint{
            ConcurrencyPoint{ Concurrency: 1, PerUserTokS: 64.5, AggregateTokS: 63.0, TTFTMeanMs: 123.5, TTFTP99Ms: 134.2, TPOTMeanMs: 15.5, ..Default::default() },
            ConcurrencyPoint{ Concurrency: 8, PerUserTokS: 49.1, AggregateTokS: 279.0, TTFTMeanMs: 2142.3, TTFTP99Ms: 3564.3, TPOTMeanMs: 20.4, ..Default::default() },
            ConcurrencyPoint{ Concurrency: 32, PerUserTokS: 31.6, AggregateTokS: 769.0, TTFTMeanMs: 2581.4, TTFTP99Ms: 8664.6, TPOTMeanMs: 31.6, ..Default::default() },
        }),
        Kernels: slice!([]KernelShare{
            KernelShare{ Name: string("cutlass fp8 blockwise GEMM (decode)"), Pct: 71.5, ..Default::default() },
            KernelShare{ Name: string("FlashAttention forward (sm90)"), Pct: 11.6, ..Default::default() },
            KernelShare{ Name: string("fused rms-norm + fp8 quant"), Pct: 4.5, ..Default::default() },
            KernelShare{ Name: string("nvjet prefill GEMM"), Pct: 2.7, ..Default::default() },
            KernelShare{ Name: string("sampling (flashinfer top-k/top-p)"), Pct: 2.3, ..Default::default() },
            KernelShare{ Name: string("other"), Pct: 7.4, ..Default::default() },
        }),
        Metrics: slice!([]Metric{
            Metric{ Key: string("sm_util"), Label: string("SM utilization under load"), Value: 100.0, Unit: string("%"), Max: 100.0, ..Default::default() },
            Metric{ Key: string("power"), Label: string("peak power under load"), Value: 702.0, Unit: string("W"), Max: 700.0, Note: string("700 W cap"), ..Default::default() },
            Metric{ Key: string("kv_pool"), Label: string("KV pool"), Value: 40.18, Unit: string("GB"), Max: 80.0, Note: string("37.42 GiB = 306,528 tokens of 80 GB VRAM"), ..Default::default() },
            Metric{ Key: string("kv_per_token"), Label: string("KV cache per token"), Value: 131072.0, Unit: string("bytes"), Note: string("exact"), ..Default::default() },
            Metric{ Key: string("max_concurrency_32k"), Label: string("max concurrency at 32K"), Value: 9.35, Unit: string("x"), Note: string("vLLM's own line"), ..Default::default() },
            Metric{ Key: string("prefill_rate"), Label: string("prefill rate"), Value: 8650.0, Unit: string("tok/s"), Note: string("single stream"), ..Default::default() },
            Metric{ Key: string("decode_idle"), Label: string("single-stream decode"), Value: 63.0, Unit: string("tok/s"), Note: string("idle server"), ..Default::default() },
            Metric{ Key: string("prefix_bonus"), Label: string("shared-prefix bonus"), Value: 47.0, Unit: string("%"), Note: string("throughput gain at 98.4% hit rate"), ..Default::default() },
            Metric{ Key: string("nonkv_overhead"), Label: string("non-KV overhead"), Value: 1.66, Unit: string("GiB"), Note: string("activation + graphs"), ..Default::default() },
        }),
        Saturation: slice!([]Metric{
            Metric{ Key: string("demand"), Label: string("demand"), Value: 32.0, Unit: string("requests"), Note: string("16K tokens each against a 306K-token pool"), ..Default::default() },
            Metric{ Key: string("admitted"), Label: string("admitted"), Value: 18.0, Unit: string("running"), Max: 32.0, Note: string("pool / ctx = 18.4, exact"), ..Default::default() },
            Metric{ Key: string("queued"), Label: string("queued"), Value: 14.0, Unit: string("waiting"), Max: 32.0, Note: string("zero preemptions, vLLM queues instead"), ..Default::default() },
            Metric{ Key: string("kv_usage"), Label: string("KV usage"), Value: 96.2, Unit: string("%"), Max: 100.0, ..Default::default() },
            Metric{ Key: string("ttft_saturated"), Label: string("TTFT under saturation"), Value: 31.6, Unit: string("s"), Note: string("mean; 60 s p99, pure queueing"), ..Default::default() },
            Metric{ Key: string("decode_saturated"), Label: string("per-user decode"), Value: 8.0, Unit: string("tok/s"), Note: string("at 123 ms TPOT"), ..Default::default() },
        }),
        GraphMetrics: slice!([]Metric{
            Metric{ Key: string("graphs_full"), Label: string("FULL graphs captured"), Value: 7.0, Unit: string(""), Note: string("uniform decode batches"), ..Default::default() },
            Metric{ Key: string("graphs_piecewise"), Label: string("PIECEWISE graphs captured"), Value: 11.0, Unit: string(""), Note: string("per-layer, attention runs eager between"), ..Default::default() },
            Metric{ Key: string("graph_memory"), Label: string("graph memory"), Value: 0.18, Unit: string("GiB"), ..Default::default() },
            Metric{ Key: string("default_trace_vis"), Label: string("default nsys trace sees"), Value: 5.0, Unit: string("% of GPU time"), Max: 100.0, Note: string("59,669 kernel records; graphs opaque"), ..Default::default() },
            Metric{ Key: string("node_trace_vis"), Label: string("node-mode trace sees"), Value: 98.0, Unit: string("% of GPU time"), Max: 100.0, Note: string("1,174,714 records"), ..Default::default() },
            Metric{ Key: string("node_trace_cost"), Label: string("node-mode launch overhead"), Value: 4.0, Unit: string("x"), Note: string("cudaGraphLaunch 193 us to 767 us avg, 20x trace size"), ..Default::default() },
        }),
        CudaGraphs: slice!([]Fact{
            fact("capture mode", "FULL_AND_PIECEWISE"),
            fact("capture batch sizes", "1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64; larger decode runs eager"),
        }),
        NeverCaptured: slice!([]string{
            string("unified attention (variable sequence lengths)"),
            string("KV-cache updates"),
            string("mamba, linear-attention, and KDA ops"),
            string("sparse-attention indexer"),
            string("sampling (flashinfer top-k/top-p)"),
            string("arbitrary-length prefill (chunked, piecewise)"),
        }),
        ..Default::default()
    }
}
