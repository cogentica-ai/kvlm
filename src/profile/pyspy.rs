// pyspy: Python-level flamegraphs. Attach mode is impossible on
// RunPod; recorded here so nobody re-walks that dead end.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::profile::{cmdOf, strsOf, Cmd, Register, Support, Tool};
use crate::profile::{EnvBareMetal, EnvK8s, EnvRunPod, StatusBlocked, StatusUntested};

#[goish::init]
fn init() {
    Register(Tool {
        Name: string("pyspy"),
        Summary: string("Python-level CPU flamegraph via py-spy"),
        Support: slice!([]Support{
            Support {
                Env: string(EnvRunPod),
                Status: string(StatusBlocked),
                Reason: string("ptrace_scope=1 on a read-only sysfs; attaching needs process_vm_readv rights the pod does not have and cannot grant (error 13 copying Py_Version)"),
                Verified: string("measured 2026-08-08, qwen3-32b fp8 on 1xH100 SXM, RunPod"),
                ..Default::default()
            },
            Support {
                Env: string(EnvK8s),
                Status: string(StatusUntested),
                Reason: string("needs SYS_PTRACE capability or ptrace_scope=0 on the node"),
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
            cmdOf(&["pip", "install", "py-spy"], ""),
        }),
        Window: slice!([]Cmd{
            cmdOf(
                &["py-spy", "record", "-o", "{out}.svg", "-d", "{seconds}", "-r", "100", "--pid", "{pid}"],
                "attach mode, blocked on RunPod; launch mode (py-spy record -- <cmd>) traces a child and may work, untested by us",
            ),
        }),
        Artifacts: strsOf(&["{out}.svg"]),
        Notes: strsOf(&[
            "the output is already an SVG flamegraph, no post-processing needed",
        ]),
        ..Default::default()
    });
}
