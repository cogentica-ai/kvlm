// profile::launch regression tests: the vLLM launch composition kvlm
// up runs on a fresh pod. Pins the serve line (exec form, profiler
// flag, tensor parallelism sizing) and the setup script contract
// (pinned install, registry install steps, node-mode launch).
#![no_std]
#![no_main]
#![allow(non_snake_case)]

extern crate alloc;

use goish::fmt;
use goish::strings;
use goish::syscall;
use goish::testing;
use goish::string;
use goish::{append, int32, make, range, slice};

use kvlm::model;
use kvlm::profile::launch;

fn spec(withTP: bool) -> model::ServeSpec {
    let mut flags: goish::slice<model::Flag> = make!([]model::Flag, 0);
    flags = append!(flags.clone(), model::Flag { Name: string("--max-model-len"), Value: string("32768"), ..Default::default() });
    flags = append!(flags.clone(), model::Flag { Name: string("--kv-cache-dtype"), Value: string("fp8"), ..Default::default() });
    if withTP {
        flags = append!(flags.clone(), model::Flag { Name: string("--tensor-parallel-size"), Value: string("4"), ..Default::default() });
    }
    model::ServeSpec {
        Model: string("Qwen/Qwen3-32B-FP8"),
        Flags: flags,
        ..Default::default()
    }
}

fn test_serve_line(t: &mut testing::T) {
    let s = spec(false);
    let line = launch::ServeLine(&s, 1, true);
    if !strings::HasPrefix(line.clone(), string("exec vllm serve Qwen/Qwen3-32B-FP8")) {
        t.Fatal(fmt::Sprintf!("serve line start drifted: %q", line));
    }
    if !strings::Contains(line.clone(), "--max-model-len 32768") {
        t.Fatal(string("catalog flag lost"));
    }
    // the torch profiler flag rides along, single-quoted json
    if !strings::Contains(line.clone(), "--profiler-config '{\"profiler\": \"torch\"") {
        t.Fatal(fmt::Sprintf!("profiler flag drifted: %q", line));
    }
    if strings::Contains(line.clone(), "--tensor-parallel-size") {
        t.Fatal(string("tp must not appear for a single GPU"));
    }
}

fn test_serve_line_tensor_parallel(t: &mut testing::T) {
    // gpu count sizes tp when the recipe does not set it
    let line = launch::ServeLine(&spec(false), 2, true);
    if !strings::Contains(line.clone(), "--tensor-parallel-size 2") {
        t.Fatal(fmt::Sprintf!("tp not sized to the pod: %q", line));
    }
    // a recipe that pins tp wins over the gpu count
    let line = launch::ServeLine(&spec(true), 2, true);
    if !strings::Contains(line.clone(), "--tensor-parallel-size 4") || strings::Contains(line.clone(), "--tensor-parallel-size 2") {
        t.Fatal(fmt::Sprintf!("recipe tp must win: %q", line));
    }
}

fn test_setup_script(t: &mut testing::T) {
    let script = launch::Script(string("exec vllm serve m"), string("0.26.0"), true);
    for want in [
        "pip install -q vllm==0.26.0",
        "cuda-keyring_1.1-1_all.deb",
        "nsight-systems-2026.1.3",
        "mkdir -p /tmp/kvlm-profile/torch",
        "cat > /workspace/serve-kvlm.sh",
        "exec vllm serve m",
        "echo SETUP_DONE",
        "--cuda-graph-trace=node",
        "--session-new=kvlm",
        "/workspace/serve-kvlm.sh > /workspace/vllm.log 2>&1 &",
        "echo LAUNCHED",
    ]
    .iter()
    {
        if !strings::Contains(script.clone(), string(*want)) {
            t.Fatal(fmt::Sprintf!("setup script missing %q", string(*want)));
        }
    }
    // the launch must come after the serve script exists
    let launchIdx = strings::Index(script.clone(), string("--session-new=kvlm"));
    let writeIdx = strings::Index(script.clone(), string("cat > /workspace/serve-kvlm.sh"));
    if launchIdx < writeIdx {
        t.Fatal(string("launch precedes the serve script write"));
    }
}

fn test_production_mode(t: &mut testing::T) {
    // production: same recipe, no profiler flag, no nsys anywhere
    let line = launch::ServeLine(&spec(false), 1, false);
    if strings::Contains(line.clone(), "--profiler-config") {
        t.Fatal(string("production serve line must not expose profiling endpoints"));
    }
    let script = launch::Script(line, string("0.26.0"), false);
    for absent in ["nsys", "nsight-systems", "--cuda-graph-trace"].iter() {
        if strings::Contains(script.clone(), string(*absent)) {
            t.Fatal(fmt::Sprintf!("production script must not contain %q", string(*absent)));
        }
    }
    if !strings::Contains(script.clone(), "/workspace/serve-kvlm.sh > /workspace/vllm.log 2>&1 &") {
        t.Fatal(string("production launch line drifted"));
    }
    if !strings::Contains(script.clone(), "pip install -q vllm==0.26.0") {
        t.Fatal(string("production must still pin vLLM"));
    }
}

fn test_parse_gpu_ref(t: &mut testing::T) {
    let cases: &[(&'static str, i64, &'static str, bool)] = &[
        ("2xH100", 2, "H100", true),
        ("1xL40S", 1, "L40S", true),
        ("8xB200", 8, "B200", true),
        ("H100", 0, "", false),
        ("", 0, "", false),
    ];
    for (input, wantN, wantT, wantOk) in cases.iter() {
        let (n, ty, ok) = model::ParseGPURef(*input);
        if ok != *wantOk || (n as i64) != *wantN || ty != *wantT {
            t.Fatal(fmt::Sprintf!("%q: got %d %q %v", string(*input), n, ty, ok));
        }
    }
}

fn test_serve_argv(t: &mut testing::T) {
    // the production container command: catalog argv plus sized tp
    let argv = launch::ServeArgv(&spec(false), 2);
    let mut joined = string("");
    for (_, a) in range!(argv.clone()) {
        joined = (joined) + (a.clone()) + (" ");
    }
    if !strings::HasPrefix(joined.clone(), string("vllm serve Qwen/Qwen3-32B-FP8 ")) {
        t.Fatal(fmt::Sprintf!("argv start drifted: %q", joined));
    }
    if !strings::Contains(joined.clone(), "--tensor-parallel-size 2") {
        t.Fatal(string("tp not sized"));
    }
    if strings::Contains(joined.clone(), "--profiler-config") {
        t.Fatal(string("production argv must not carry the profiler flag"));
    }
}

// WithFlag overrides in place and appends when absent; the restart
// script kills, rewrites, and relaunches under a fresh session.
fn test_with_flag(t: &mut testing::T) {
    let s = spec(false);
    let over = launch::WithFlag(&s, string("--max-model-len"), string("16384"));
    let (v, ok) = over.FlagValue("--max-model-len");
    if !ok || v != "16384" {
        t.Fatal(fmt::Sprintf!("override did not land: %q", v));
    }
    let added = launch::WithFlag(&s, string("--max-num-seqs"), string("64"));
    let (v, ok) = added.FlagValue("--max-num-seqs");
    if !ok || v != "64" {
        t.Fatal(fmt::Sprintf!("append did not land: %q", v));
    }
    // the original spec must stay untouched
    let (_, had) = s.FlagValue("--max-num-seqs");
    if had {
        t.Fatal(string("WithFlag mutated its input"));
    }
}

fn test_restart_script(t: &mut testing::T) {
    let line = launch::ServeLine(&spec(false), 1, true);
    let script = launch::RestartScript(line.clone());
    for want in [
        "pkill -f '[v]llm serve'",
        "nsys shutdown --session=kvlm",
        "cat > /workspace/serve-kvlm.sh <<'KVLMEOF'",
        "--cuda-graph-trace=node",
        "> /workspace/vllm.log 2>&1 &",
        "echo RESTARTED",
    ]
    .iter()
    {
        if !strings::Contains(script.clone(), *want) {
            t.Fatal(fmt::Sprintf!("restart script lost %q", string(*want)));
        }
    }
    if !strings::Contains(script.clone(), line.clone()) {
        t.Fatal(string("restart script must embed the serve line verbatim"));
    }
}

#[goish::main]
fn main() {
    let tests: &[(&str, testing::TestFn)] = &[
        ("TestServeLine", test_serve_line),
        ("TestServeLineTensorParallel", test_serve_line_tensor_parallel),
        ("TestSetupScript", test_setup_script),
        ("TestProductionMode", test_production_mode),
        ("TestParseGPURef", test_parse_gpu_ref),
        ("TestServeArgv", test_serve_argv),
        ("TestWithFlag", test_with_flag),
        ("TestRestartScript", test_restart_script),
    ];
    let code = testing::Main(tests);
    syscall::Exit(int32(code));
}
