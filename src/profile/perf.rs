// perf: CPU profiling with kernel visibility. Fully disabled on
// RunPod; recorded here so nobody re-walks that dead end.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::profile::{cmdOf, strsOf, Cmd, Register, Support, Tool};
use crate::profile::{EnvBareMetal, EnvK8s, EnvRunPod, StatusBlocked, StatusUntested};

#[goish::init]
fn init() {
    Register(Tool {
        Name: string("perf"),
        Summary: string("CPU profile with kernel stacks via perf"),
        Support: slice!([]Support{
            Support {
                Env: string(EnvRunPod),
                Status: string(StatusBlocked),
                Reason: string("perf_event_paranoid=4: perf_event_open is refused for every mode, not fixable from inside the pod"),
                Verified: string("measured 2026-08-08, qwen3-32b fp8 on 1xH100 SXM, RunPod"),
                ..Default::default()
            },
            Support {
                Env: string(EnvK8s),
                Status: string(StatusUntested),
                Reason: string("needs perf_event_paranoid <= 2 on the node and a perf binary matching its kernel"),
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
            cmdOf(&["apt-get", "install", "-y", "linux-tools-generic"], "the binary must roughly match the host kernel"),
        }),
        Window: slice!([]Cmd{
            cmdOf(
                &["perf", "record", "-F", "99", "-g", "--pid", "{pid}", "-o", "{out}.data", "--", "sleep", "{seconds}"],
                "--pid attaches to the engine process; without it perf profiles the sleep and captures nothing",
            ),
        }),
        Artifacts: strsOf(&["{out}.data"]),
        Analyze: slice!([]Cmd{
            cmdOf(
                &["perf", "script", "-i", "{out}.data"],
                "pipe through stackcollapse-perf.pl, then kvlm profile flamegraph -i folded.txt -o flame.svg",
            ),
        }),
        ..Default::default()
    });
}
