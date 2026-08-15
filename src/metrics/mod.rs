// metrics package: vLLM Prometheus /metrics collector and formatter.
//
// vLLM exposes a Prometheus-compatible /metrics endpoint (default :8000).
// This module fetches, parses, and categorizes the metrics for display.
//
// Metric categories (all from vLLM's exporter):
//   Latency:    TTFT, TPOT, e2e, prefill, decode
//   Throughput: request rate, token rate, prompt tokens
//   Queue:      running, waiting, swapped
//   Cache:      GPU/CPU KV cache utilization
//   Speculative: acceptance rate, draft acceptance, efficiency
#![allow(non_snake_case)]

use goish::errors::error;
use goish::fmt;
use goish::io;
use goish::net::http;
use goish::strconv;
use goish::strings;
use goish::string;
use goish::slice;
use goish::gomap::map;
use goish::{append, float64, int, nil, range};

// Metric is one parsed Prometheus sample.
#[derive(Clone, Default)]
pub struct Metric {
    pub Name: string,
    pub Value: float64,
    pub Labels: map<string, string>,
}

// Snapshot is a categorized collection of vLLM metrics.
#[derive(Clone, Default)]
pub struct Snapshot {
    // Latency (seconds).
    pub TTFTSum: float64,
    pub TTFTCount: float64,
    pub TPOTSum: float64,
    pub TPOTCount: float64,
    pub E2ESum: float64,
    pub E2ECount: float64,
    pub PrefillSum: float64,
    pub PrefillCount: float64,
    pub DecodeSum: float64,
    pub DecodeCount: float64,
    pub QueueSum: float64,
    pub QueueCount: float64,

    // Scheduler pressure (cumulative counter): each preemption is a
    // request evicted for KV space, vLLM's own first tuning signal.
    pub Preemptions: float64,

    // Throughput (cumulative counters).
    pub RequestSuccess: float64,
    pub GenerationTokens: float64,
    pub PromptTokens: float64,

    // Queue (gauges).
    pub Running: float64,
    pub Waiting: float64,
    pub Swapped: float64,

    // Cache (gauges, 0.0-1.0).
    pub GPUCacheUsage: float64,
    pub CPUCacheUsage: float64,

    // Prefix cache (cumulative block counters, vLLM 0.26).
    pub PrefixCacheQueries: float64,
    pub PrefixCacheHits: float64,

    // Speculative decoding (gauges).
    pub SpecAcceptance: float64,
    pub SpecDraftAcceptance: float64,
    pub SpecEfficiency: float64,

    // Raw metrics for extensibility.
    pub Raw: slice<Metric>,
}

// Parse parses Prometheus text exposition format into a Snapshot.
// Each line is either:
//   # HELP <name> <text>
//   # TYPE <name> <type>
//   <name>[{labels}] <value> [timestamp]
//   (blank or comment)
pub fn Parse(text: string) -> Snapshot {
    let mut s = Snapshot::default();
    let lines = strings::Split(text, "\n");
    for (_, line) in range!(lines) {
        let trimmed = strings::TrimSpace(line.clone());
        if trimmed == "" || strings::HasPrefix(trimmed.clone(), "#") {
            continue;
        }
        let (name, value) = parseLine(trimmed);
        if name == "" {
            continue;
        }
        let m = Metric {
            Name: name.clone(),
            Value: value,
            ..Default::default()
        };
        s.Raw = append!(s.Raw.clone(), m);

        // Categorize by metric name. Where vLLM 0.26 renamed a metric
        // the old name stays as a fallback (verified against real
        // 0.26 output, profile-output/metrics-after.txt).
        match name.as_ref() {
            "vllm:time_to_first_token_seconds_sum" => s.TTFTSum = value,
            "vllm:time_to_first_token_seconds_count" => s.TTFTCount = value,
            // 0.26 name, then the pre-0.26 name
            "vllm:request_time_per_output_token_seconds_sum" => s.TPOTSum = value,
            "vllm:request_time_per_output_token_seconds_count" => s.TPOTCount = value,
            "vllm:time_per_output_token_seconds_sum" => s.TPOTSum = value,
            "vllm:time_per_output_token_seconds_count" => s.TPOTCount = value,
            "vllm:e2e_request_latency_seconds_sum" => s.E2ESum = value,
            "vllm:e2e_request_latency_seconds_count" => s.E2ECount = value,
            "vllm:request_prefill_time_seconds_sum" => s.PrefillSum = value,
            "vllm:request_prefill_time_seconds_count" => s.PrefillCount = value,
            "vllm:request_decode_time_seconds_sum" => s.DecodeSum = value,
            "vllm:request_decode_time_seconds_count" => s.DecodeCount = value,
            "vllm:request_queue_time_seconds_sum" => s.QueueSum = value,
            "vllm:request_queue_time_seconds_count" => s.QueueCount = value,
            // one series per engine; sum across engines
            "vllm:num_preemptions_total" => s.Preemptions += value,
            // one series per finished_reason; the total is their sum,
            // an assignment would keep only the last label seen
            "vllm:request_success_total" => s.RequestSuccess += value,
            "vllm:generation_tokens_total" => s.GenerationTokens = value,
            "vllm:prompt_tokens_total" => s.PromptTokens = value,
            "vllm:num_requests_running" => s.Running = value,
            "vllm:num_requests_waiting" => s.Waiting = value,
            "vllm:num_requests_swapped" => s.Swapped = value,
            // 0.26 name, then the pre-0.26 name
            "vllm:kv_cache_usage_perc" => s.GPUCacheUsage = value,
            "vllm:gpu_cache_usage_perc" => s.GPUCacheUsage = value,
            "vllm:cpu_cache_usage_perc" => s.CPUCacheUsage = value,
            "vllm:prefix_cache_queries_total" => s.PrefixCacheQueries = value,
            "vllm:prefix_cache_hits_total" => s.PrefixCacheHits = value,
            "vllm:spec_decode_acceptance_rate" => s.SpecAcceptance = value,
            "vllm:spec_decode_draft_acceptance_rate" => s.SpecDraftAcceptance = value,
            "vllm:spec_decode_efficiency" => s.SpecEfficiency = value,
            _ => {}
        }
    }
    s
}

// Fetch pulls and parses a live /metrics endpoint. addr is host:port
// (scheme optional; plain http when absent).
pub fn Fetch<S: Into<string>>(addr: S) -> (Snapshot, error) {
    let addr = addr.into();
    let mut url = addr.clone();
    if !strings::HasPrefix(url.clone(), "http://") && !strings::HasPrefix(url.clone(), "https://") {
        url = ("http://") + (url);
    }
    url = (url) + ("/metrics");
    let (resp, err) = http::Get(url.clone());
    if err != nil {
        return (Snapshot::default(), err);
    }
    let mut resp = resp;
    if resp.StatusCode != 200 {
        return (
            Snapshot::default(),
            fmt::Errorf!("%s returned status %d", url, resp.StatusCode),
        );
    }
    // Go shape: resp.Body is a reader; drain it fully.
    let (data, err) = io::ReadAll(&mut resp.Body);
    if err != nil {
        return (Snapshot::default(), err);
    }
    (Parse(string(data)), nil.into())
}

// Derived is what two snapshots plus an elapsed wall time yield: the
// means and rates over the window. Has* flags mark which sections had
// any samples; gauges (Running, Waiting, KVUsage) come from the after
// snapshot.
#[derive(Clone, Default)]
pub struct Derived {
    pub ElapsedS: float64,
    pub Requests: float64,
    pub TTFTMeanMs: float64,
    pub HasTTFT: bool,
    pub TPOTMeanMs: float64,
    pub HasTPOT: bool,
    pub E2EMeanS: float64,
    pub HasE2E: bool,
    // the latency split: where a finished request's wall time went;
    // queue = capacity, prefill = compute bound, decode = memory bound
    pub QueueMeanS: float64,
    pub HasQueue: bool,
    pub PrefillMeanS: float64,
    pub HasPrefillT: bool,
    pub DecodeMeanS: float64,
    pub HasDecodeT: bool,
    // preemptions in the window; any nonzero value is KV pressure
    pub Preemptions: float64,
    pub GenTokS: float64,
    pub PromptTokS: float64,
    pub PrefixHitRate: float64,
    pub HasPrefix: bool,
    pub Running: float64,
    pub Waiting: float64,
    pub KVUsage: float64,
}

// Derive computes window statistics from two snapshots of the same
// server taken elapsedS seconds apart.
pub fn Derive(before: &Snapshot, after: &Snapshot, elapsedS: float64) -> Derived {
    let mut d = Derived::default();
    d.ElapsedS = elapsedS;
    d.Requests = after.RequestSuccess - before.RequestSuccess;

    let ttftC = after.TTFTCount - before.TTFTCount;
    if ttftC > 0.0 {
        d.TTFTMeanMs = (after.TTFTSum - before.TTFTSum) / ttftC * 1000.0;
        d.HasTTFT = true;
    }
    let tpotC = after.TPOTCount - before.TPOTCount;
    if tpotC > 0.0 {
        d.TPOTMeanMs = (after.TPOTSum - before.TPOTSum) / tpotC * 1000.0;
        d.HasTPOT = true;
    }
    let e2eC = after.E2ECount - before.E2ECount;
    if e2eC > 0.0 {
        d.E2EMeanS = (after.E2ESum - before.E2ESum) / e2eC;
        d.HasE2E = true;
    }
    let queueC = after.QueueCount - before.QueueCount;
    if queueC > 0.0 {
        d.QueueMeanS = (after.QueueSum - before.QueueSum) / queueC;
        d.HasQueue = true;
    }
    let prefC = after.PrefillCount - before.PrefillCount;
    if prefC > 0.0 {
        d.PrefillMeanS = (after.PrefillSum - before.PrefillSum) / prefC;
        d.HasPrefillT = true;
    }
    let decC = after.DecodeCount - before.DecodeCount;
    if decC > 0.0 {
        d.DecodeMeanS = (after.DecodeSum - before.DecodeSum) / decC;
        d.HasDecodeT = true;
    }
    d.Preemptions = after.Preemptions - before.Preemptions;
    if elapsedS > 0.0 {
        d.GenTokS = (after.GenerationTokens - before.GenerationTokens) / elapsedS;
        d.PromptTokS = (after.PromptTokens - before.PromptTokens) / elapsedS;
    }
    let pq = after.PrefixCacheQueries - before.PrefixCacheQueries;
    if pq > 0.0 {
        d.PrefixHitRate = (after.PrefixCacheHits - before.PrefixCacheHits) / pq;
        d.HasPrefix = true;
    }
    d.Running = after.Running;
    d.Waiting = after.Waiting;
    d.KVUsage = after.GPUCacheUsage;
    d
}

// round0 / round1 / round2 — goish fmt has no precision verbs, so
// rounding happens before printing (same convention as model::fmtGB).
fn round0(v: float64) -> int {
    goish::math::Round(v) as int
}
fn round1(v: float64) -> float64 {
    goish::math::Round(v * 10.0) / 10.0
}
fn round2(v: float64) -> float64 {
    goish::math::Round(v * 100.0) / 100.0
}

// PrintDerived renders one window's statistics.
pub fn PrintDerived(d: &Derived) {
    fmt::Printf!("Window:        %d s, %d requests completed\n", round0(d.ElapsedS), round0(d.Requests));
    if d.HasTTFT {
        fmt::Printf!("TTFT mean:     %d ms\n", round0(d.TTFTMeanMs));
    } else {
        fmt::Printf!("TTFT mean:     no data\n");
    }
    if d.HasTPOT {
        fmt::Printf!("TPOT mean:     %v ms (%v tok/s per stream)\n", round1(d.TPOTMeanMs), round1(1000.0 / d.TPOTMeanMs));
    } else {
        fmt::Printf!("TPOT mean:     no data\n");
    }
    if d.HasE2E {
        fmt::Printf!("e2e mean:      %v s\n", round2(d.E2EMeanS));
    }
    if d.HasQueue && d.HasPrefillT && d.HasDecodeT {
        fmt::Printf!(
            "latency split: queue %v s, prefill %v s, decode %v s\n",
            round2(d.QueueMeanS),
            round2(d.PrefillMeanS),
            round2(d.DecodeMeanS)
        );
    }
    if d.Preemptions > 0.0 {
        fmt::Printf!(
            "preemptions:   %d in window (KV pressure: raise --gpu-memory-utilization, lower --max-num-seqs, or shard with -tp)\n",
            round0(d.Preemptions)
        );
    }
    fmt::Printf!("output rate:   %d tok/s\n", round0(d.GenTokS));
    fmt::Printf!("prompt rate:   %d tok/s\n", round0(d.PromptTokS));
    if d.HasPrefix {
        fmt::Printf!("prefix cache:  %v%% hit rate\n", round1(d.PrefixHitRate * 100.0));
    }
    fmt::Printf!("now:           running %d, waiting %d, KV cache %v%%\n", round0(d.Running), round0(d.Waiting), round1(d.KVUsage * 100.0));
}

// parseLine extracts the metric name and value from a Prometheus sample line.
// Format: <name>[{labels}] <value> [timestamp]
// Simplified: just split on whitespace and take first two fields.
fn parseLine(line: string) -> (string, float64) {
    let parts = strings::Fields(line);
    if parts.Len() < 2 {
        return (string(""), 0.0);
    }
    let fullName = parts[0usize].clone();
    let valueStr = parts[1usize].clone();
    let value = parseFloat(valueStr);
    
    // Strip labels: "metric_name{labels}" -> "metric_name"
    // Find position of '{' and take substring before it
    let braceIdx = strings::Index(fullName.clone(), "{");
    let name = if braceIdx >= 0 {
        // Use strings::Split to get the part before '{'
        let splitParts = strings::Split(fullName, "{");
        if splitParts.Len() > 0 {
            splitParts[0usize].clone()
        } else {
            string("")
        }
    } else {
        fullName
    };
    
    (name, value)
}

// parseFloat parses a float64 from a string. Returns 0 on error.
fn parseFloat(s: string) -> float64 {
    // goish: use strconv.ParseFloat when available; for now, simple cases.
    let (v, _) = strconv::ParseFloat(s, 64);
    v
}

// Print displays the snapshot in a human-readable format.
pub fn Print(s: &Snapshot) {
    fmt::Printf!("=== vLLM Metrics ===\n\n");

    // Latency section.
    fmt::Printf!("Latency:\n");
    printAvg(string("  TTFT"), s.TTFTSum, s.TTFTCount);
    printAvg(string("  TPOT"), s.TPOTSum, s.TPOTCount);
    printAvg(string("  E2E "), s.E2ESum, s.E2ECount);
    printAvg(string("  Prefill"), s.PrefillSum, s.PrefillCount);
    printAvg(string("  Decode "), s.DecodeSum, s.DecodeCount);

    // Throughput section.
    fmt::Printf!("\nThroughput (cumulative):\n");
    fmt::Printf!("  Requests:      %d\n", round0(s.RequestSuccess));
    fmt::Printf!("  Gen tokens:    %d\n", round0(s.GenerationTokens));
    fmt::Printf!("  Prompt tokens: %d\n", round0(s.PromptTokens));

    // Queue section.
    fmt::Printf!("\nQueue:\n");
    fmt::Printf!("  Running:  %d\n", round0(s.Running));
    fmt::Printf!("  Waiting:  %d\n", round0(s.Waiting));
    fmt::Printf!("  Swapped:  %d\n", round0(s.Swapped));

    // Cache section.
    fmt::Printf!("\nKV Cache:\n");
    fmt::Printf!("  GPU: %v%%\n", round1(s.GPUCacheUsage * 100.0));
    fmt::Printf!("  CPU: %v%%\n", round1(s.CPUCacheUsage * 100.0));

    // Speculative decoding (only if non-zero).
    if s.SpecAcceptance > 0.0 || s.SpecDraftAcceptance > 0.0 {
        fmt::Printf!("\nSpeculative Decoding:\n");
        if s.SpecAcceptance > 0.0 {
            fmt::Printf!("  Acceptance rate:      %v\n", round2(s.SpecAcceptance));
        }
        if s.SpecDraftAcceptance > 0.0 {
            fmt::Printf!("  Draft acceptance:     %v\n", round2(s.SpecDraftAcceptance));
        }
        if s.SpecEfficiency > 0.0 {
            fmt::Printf!("  Efficiency:           %v\n", round2(s.SpecEfficiency));
        }
    }
}

// printAvg prints "  name: avg ms (count samples)" for a sum/count pair.
fn printAvg(label: string, sum: float64, count: float64) {
    if count <= 0.0 {
        fmt::Printf!("  %s: no data\n", label);
        return;
    }
    let avgMs = round1((sum / count) * 1000.0);
    fmt::Printf!("  %s: %v ms (%d samples)\n", label, avgMs, round0(count));
}

// KnownMetrics lists all vLLM metrics we categorize, for documentation.
pub fn KnownMetrics() -> slice<string> {
    slice!([]string{
        // Latency
        string("vllm:time_to_first_token_seconds (TTFT)"),
        string("vllm:time_per_output_token_seconds (TPOT)"),
        string("vllm:e2e_request_latency_seconds"),
        string("vllm:request_prefill_time_seconds"),
        string("vllm:request_decode_time_seconds"),
        // Throughput
        string("vllm:request_success_total"),
        string("vllm:generation_tokens_total"),
        string("vllm:prompt_tokens_total"),
        // Queue
        string("vllm:num_requests_running"),
        string("vllm:num_requests_waiting"),
        string("vllm:num_requests_swapped"),
        // Cache
        string("vllm:gpu_cache_usage_perc"),
        string("vllm:cpu_cache_usage_perc"),
        // Speculative
        string("vllm:spec_decode_acceptance_rate"),
        string("vllm:spec_decode_draft_acceptance_rate"),
        string("vllm:spec_decode_efficiency"),
    })
}
