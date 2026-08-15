// ncu: Nsight Compute kernel-level analysis. Impossible on RunPod at
// the driver level; recorded here so nobody re-walks that dead end.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::profile::{cmdOf, strsOf, Cmd, Register, Support, Tool};
use crate::profile::{EnvBareMetal, EnvK8s, EnvRunPod, StatusBlocked, StatusUntested};

#[goish::init]
fn init() {
    Register(Tool {
        Name: string("ncu"),
        Summary: string("per-kernel occupancy and memory analysis via Nsight Compute"),
        Support: slice!([]Support{
            Support {
                Env: string(EnvRunPod),
                Status: string(StatusBlocked),
                Reason: string("ERR_NVGPUCTRPERM: GPU performance counters are disabled at the driver level (NVreg_RestrictProfilingToAdminUsers); only the host owner can lift it"),
                Verified: string("measured 2026-08-08, qwen3-32b fp8 on 1xH100 SXM, RunPod"),
                ..Default::default()
            },
            Support {
                Env: string(EnvK8s),
                Status: string(StatusUntested),
                Reason: string("the node driver must allow counter access"),
                ..Default::default()
            },
            Support {
                Env: string(EnvBareMetal),
                Status: string(StatusUntested),
                Reason: string("works where you own the driver settings"),
                ..Default::default()
            },
        }),
        Install: slice!([]Cmd{
            cmdOf(&["apt-get", "install", "-y", "nsight-compute-2026.2.1"], "needs the cuda-keyring apt repo, same as nsys"),
        }),
        Window: slice!([]Cmd{
            cmdOf(
                &["ncu", "--set", "full", "--launch-count", "10", "-o", "{out}", "python3", "your_workload.py"],
                "ncu must launch its target; it cannot attach to a running server",
            ),
        }),
        Artifacts: strsOf(&["{out}.ncu-rep"]),
        Analyze: slice!([]Cmd{
            cmdOf(&["ncu", "--import", "{out}.ncu-rep"], "or open in the ncu-ui desktop app"),
        }),
        Notes: strsOf(&[
            "use nsys first to find WHICH kernel matters, then ncu on that kernel where counters are permitted",
        ]),
        ..Default::default()
    });
}
