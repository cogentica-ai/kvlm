// torchprof: vLLM's built-in torch profiler, the correlated GPU plus
// python timeline that needs no nsys install and no ptrace. The
// working contract, measured on a real pod: enable at launch with
// --profiler-config (the flag parses on vLLM 0.26.0 even though serve
// --help does not list it), then drive /start_profile and
// /stop_profile. The pre-0.26 env-var contract is dead; postmortem in
// the notes.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::profile::{cmdOf, strsOf, Cmd, Register, Support, Tool};
use crate::profile::{EnvBareMetal, EnvK8s, EnvRunPod, StatusUntested, StatusWorks};

#[goish::init]
fn init() {
    Register(Tool {
        Name: string("torchprof"),
        Summary: string("vLLM built-in torch profiler via --profiler-config"),
        Support: slice!([]Support{
            Support {
                Env: string(EnvRunPod),
                Status: string(StatusWorks),
                Reason: string("in-process, so ptrace_scope and perf_event_paranoid do not apply; a 12 s window under probe load returned an async_llm frontend trace and a 55 MB rank0 worker trace with 3.17M events"),
                Verified: string("measured 2026-08-08, qwen3-32b fp8 on 1xH100 SXM, RunPod, vLLM 0.26.0"),
                ..Default::default()
            },
            Support {
                Env: string(EnvK8s),
                Status: string(StatusUntested),
                Reason: string("in-process, should work anywhere the server runs"),
                ..Default::default()
            },
            Support {
                Env: string(EnvBareMetal),
                Status: string(StatusUntested),
                Reason: string(""),
                ..Default::default()
            },
        }),
        Setup: slice!([]Cmd{
            cmdOf(
                &["vllm", "serve", "<model>", "--profiler-config", "{\"torch_profiler_dir\": \"/tmp/kvlm-profile/torch\"}"],
                "launch flag from current vLLM docs; shape/memory/stack/flops knobs exist under the same config",
            ),
        }),
        Window: slice!([]Cmd{
            cmdOf(&["curl", "-s", "-X", "POST", "http://{addr}/start_profile"], ""),
            cmdOf(&["sleep", "{seconds}"], "keep windows short; torch profiler overhead is well above nsys"),
            cmdOf(&["curl", "-s", "-X", "POST", "http://{addr}/stop_profile"], "the trace file appears after stop returns; large windows take a while to serialize"),
        }),
        Artifacts: strsOf(&["/tmp/kvlm-profile/torch/*.pt.trace.json.gz"]),
        Analyze: slice!([]Cmd{
            cmdOf(&["echo", "open the trace at https://ui.perfetto.dev"], ""),
        }),
        Notes: strsOf(&[
            "two traces come back: <host>_<pid>.async_llm...pt.trace.json.gz (frontend) and rank0...pt.trace.json.gz (the GPU worker; this is the one with kernels)",
            "measured cost: a 12 s window under load produced 66 MB of gzipped trace (3.17M events); keep windows short",
            "--profiler-config is accepted by vllm serve on 0.26.0 but absent from serve --help; trust the ProfilerConfig line the server logs at startup",
            "old contract postmortem, measured 2026-08-08: VLLM_TORCH_PROFILER_DIR + VLLM_SERVER_DEV_MODE=1 gave 404 on both endpoints, no trace written",
            "tensor parallel caveat, measured 2026-08-08 on 2xH100 tp=2: only the frontend async_llm trace is written, the TP workers write no rank traces, and stop_profile raises a Kineto RuntimeError in the server log; single-GPU works fully",
        ]),
        ..Default::default()
    });
}
