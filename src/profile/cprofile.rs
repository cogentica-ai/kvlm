// cprofile: Python's stdlib profiler, run in-process. It exists in
// this registry because it dodges the exact restriction that blocks
// py-spy and perf on RunPod: nothing attaches from outside, the
// profiled interpreter measures itself. The trade: it profiles one
// process you launch, so it fits offline vLLM scripts and single
// components, not the running multi-process API server (that is
// torchprof's job).
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::profile::{cmdOf, strsOf, Cmd, Register, Support, Tool};
use crate::profile::{EnvBareMetal, EnvK8s, EnvRunPod, StatusUntested};

#[goish::init]
fn init() {
    Register(Tool {
        Name: string("cprofile"),
        Summary: string("Python stdlib profiler, in-process (no ptrace needed)"),
        Support: slice!([]Support{
            Support {
                Env: string(EnvRunPod),
                Status: string(StatusUntested),
                Reason: string("should work where py-spy cannot: it is in-process, so ptrace_scope and perf_event_paranoid do not apply; not yet measured here"),
                ..Default::default()
            },
            Support {
                Env: string(EnvK8s),
                Status: string(StatusUntested),
                Reason: string(""),
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
                &["python3", "-m", "cProfile", "-o", "{out}.prof", "your_offline_script.py"],
                "wrap the script under test; for vLLM use an offline LLM() script, not the API server",
            ),
        }),
        Artifacts: strsOf(&["{out}.prof"]),
        Analyze: slice!([]Cmd{
            cmdOf(
                &["python3", "-c", "import pstats; pstats.Stats('{out}.prof').sort_stats('cumulative').print_stats(30)"],
                "or pip install snakeviz && snakeviz {out}.prof for the browser view",
            ),
        }),
        Notes: strsOf(&[
            "CPU-side python time only; pair with nsys for the GPU truth",
            "from the vLLM profiling docs; the docs use it for scheduler and input-processing hot spots",
        ]),
        ..Default::default()
    });
}
