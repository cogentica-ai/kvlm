// metrics: vLLM's Prometheus /metrics endpoint. The zero-overhead
// always-on path; works anywhere the port is reachable.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::profile::{cmdOf, strsOf, Cmd, Register, Support, Tool};
use crate::profile::{EnvBareMetal, EnvK8s, EnvRunPod, StatusUntested, StatusWorks};

#[goish::init]
fn init() {
    Register(Tool {
        Name: string("metrics"),
        Summary: string("vLLM Prometheus counters over HTTP"),
        Support: slice!([]Support{
            Support {
                Env: string(EnvRunPod),
                Status: string(StatusWorks),
                Reason: string("plain HTTP on the serving port; expose 8000 or tunnel it"),
                Verified: string("measured 2026-08-08, qwen3-32b fp8 on 1xH100 SXM, RunPod"),
                ..Default::default()
            },
            Support {
                Env: string(EnvK8s),
                Status: string(StatusUntested),
                Reason: string("reachable via service or kubectl port-forward"),
                ..Default::default()
            },
            Support {
                Env: string(EnvBareMetal),
                Status: string(StatusUntested),
                Reason: string(""),
                ..Default::default()
            },
        }),
        Window: slice!([]Cmd{
            cmdOf(&["curl", "-s", "http://{addr}/metrics"], "raw snapshot; kvlm profile metrics parses and summarizes instead"),
        }),
        Analyze: slice!([]Cmd{
            cmdOf(
                &["kvlm", "profile", "metrics", "--addr", "{addr}", "--interval", "30"],
                "two samples 30 s apart give TTFT/TPOT means, token rates, and prefix-cache hit rate",
            ),
        }),
        Notes: strsOf(&[
            "vLLM 0.26 renamed time_per_output_token_seconds to request_time_per_output_token_seconds and gpu_cache_usage_perc to kv_cache_usage_perc; kvlm parses both generations",
            "request_success_total is one series per finished_reason; sum them or the count reads wrong",
            "histograms expose sums and counts, so kvlm derives means; percentiles need the buckets, not parsed yet",
        ]),
        ..Default::default()
    });
}
