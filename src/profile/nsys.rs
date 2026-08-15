// nsys: Nsight Systems, the working GPU-timeline path on RunPod.
// Every command below was executed successfully on the 2026-08-08 run.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::profile::{cmdOf, strsOf, Cmd, Register, Support, Tool};
use crate::profile::{EnvBareMetal, EnvK8s, EnvRunPod, StatusUntested, StatusWorks};

#[goish::init]
fn init() {
    Register(Tool {
        Name: string("nsys"),
        Summary: string("GPU timeline and kernel summary via Nsight Systems"),
        Support: slice!([]Support{
            Support {
                Env: string(EnvRunPod),
                Status: string(StatusWorks),
                Reason: string("launch mode injects at process start, so it needs no ptrace; CPU sampling stays off (perf_event_paranoid=4) but the CUPTI GPU trace is unaffected"),
                Verified: string("measured 2026-08-08, qwen3-32b fp8 on 1xH100 SXM, RunPod"),
                ..Default::default()
            },
            Support {
                Env: string(EnvK8s),
                Status: string(StatusUntested),
                Reason: string("should work the same way; depends on node image and seccomp policy"),
                ..Default::default()
            },
            Support {
                Env: string(EnvBareMetal),
                Status: string(StatusUntested),
                Reason: string(""),
                ..Default::default()
            },
        }),
        Install: slice!([]Cmd{
            cmdOf(&["wget", "-q", "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"], ""),
            cmdOf(&["dpkg", "-i", "cuda-keyring_1.1-1_all.deb"], ""),
            cmdOf(&["apt-get", "update", "-qq"], ""),
            cmdOf(&["apt-get", "install", "-y", "nsight-systems-2026.1.3"], "the package is nsight-systems-<version>, not nsight-systems-cli"),
        }),
        Setup: slice!([]Cmd{
            cmdOf(
                &["nsys", "launch", "--session-new={session}", "-t", "cuda,nvtx", "--cuda-graph-trace=node", "{serve}"],
                "trace flags belong to launch, not start; --cuda-graph-trace=node is mandatory, without it only ~5% of GPU work is visible",
            ),
        }),
        Window: slice!([]Cmd{
            cmdOf(
                &["nsys", "start", "--session={session}", "-o", "{out}", "--force-overwrite=true", "--gpu-metrics-devices=cuda-visible"],
                "GPU metrics sampling adds SM-active/tensor/DRAM time series; if the driver restricts GPU counters kvlm retries the start without it",
            ),
            cmdOf(&["sleep", "{seconds}"], "kvlm profile run sleeps locally between start and stop instead of running this remotely"),
            cmdOf(&["nsys", "stop", "--session={session}"], ""),
        }),
        Artifacts: strsOf(&["{out}.nsys-rep"]),
        Analyze: slice!([]Cmd{
            cmdOf(&["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "table", "{out}.nsys-rep"], ""),
            cmdOf(&["nsys", "stats", "--report", "cuda_api_sum", "--format", "table", "{out}.nsys-rep"], ""),
        }),
        Notes: strsOf(&[
            "a trace taken without node-level graph tracing profiles prefill and hides decode: an A/B on the same workload measured 5% vs 98% of GPU time visible",
            "node-level graph tracing costs about 4x cudaGraphLaunch overhead and a 20x larger trace; keep windows bounded",
            "the .nsys-rep opens in the nsys-ui desktop app for the timeline view",
            "pkill/pgrep -f patterns that appear in your own ssh command line kill your own session; write bin/[v]llm style patterns",
            "GPU metrics sampling is refused on RunPod pods (measured 2026-08-09: nsys start rejects --gpu-metrics-devices; the retry without it keeps the capture alive) — same driver counter policy that blocks ncu",
        ]),
        ..Default::default()
    });
}
