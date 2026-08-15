// profile package regression tests: the registry invariants and the
// measured command contracts. The nsys window contract test exists so
// nobody reintroduces the trace-flags-on-start trap (nsys start
// rejects -t; flags belong to launch — measured 2026-08-08).
#![no_std]
#![no_main]
#![allow(non_snake_case)]

extern crate alloc;

use goish::fmt;
use goish::strings;
use goish::syscall;
use goish::testing;
use goish::string;
use goish::{int32, nil, range, slice};

use kvlm::profile;
use kvlm::profile::{EnvRunPod, StatusBlocked, StatusUntested, StatusWorks};

fn test_registry_has_all_tools(t: &mut testing::T) {
    for name in ["nsys", "metrics", "pyspy", "perf", "ncu", "torchprof", "cprofile"].iter() {
        let (_, ok) = profile::Find(*name);
        if !ok {
            t.Fatal(fmt::Sprintf!("tool %q not registered", string(*name)));
        }
    }
    let (_, ok) = profile::Find("nope");
    if ok {
        t.Fatal(string("Find(nope) should fail"));
    }
}

fn test_every_tool_has_valid_support(t: &mut testing::T) {
    for tool in profile::ToolsSorted().iter() {
        if tool.Support.Len() == 0 {
            t.Fatal(fmt::Sprintf!("%s: no Support entries", tool.Name.clone()));
        }
        for (_, s) in range!(tool.Support.clone()) {
            let env: &str = s.Env.as_ref();
            if env != "runpod" && env != "k8s" && env != "baremetal" {
                t.Fatal(fmt::Sprintf!("%s: bad env %q", tool.Name.clone(), s.Env.clone()));
            }
            let st: &str = s.Status.as_ref();
            if st != StatusWorks && st != StatusBlocked && st != StatusUntested {
                t.Fatal(fmt::Sprintf!("%s: bad status %q", tool.Name.clone(), s.Status.clone()));
            }
            if st == StatusBlocked && s.Reason == "" {
                t.Fatal(fmt::Sprintf!("%s/%s: blocked without a reason", tool.Name.clone(), s.Env.clone()));
            }
            if s.Verified != "" && !strings::HasPrefix(s.Verified.clone(), "measured ") {
                t.Fatal(fmt::Sprintf!("%s/%s: provenance %q must start with 'measured '", tool.Name.clone(), s.Env.clone(), s.Verified.clone()));
            }
            if s.Verified != "" && !strings::Contains(s.Verified.clone(), "2026-08-08") {
                t.Fatal(fmt::Sprintf!("%s/%s: provenance lacks the measurement date", tool.Name.clone(), s.Env.clone()));
            }
        }
    }
}

fn test_every_argv_non_empty(t: &mut testing::T) {
    for tool in profile::ToolsSorted().iter() {
        let phases = slice!([]slice<profile::Cmd>{
            tool.Install.clone(),
            tool.Setup.clone(),
            tool.Window.clone(),
            tool.Analyze.clone(),
        });
        for (_, phase) in range!(phases.clone()) {
            for (_, c) in range!(phase.clone()) {
                if c.Argv.Len() == 0 {
                    t.Fatal(fmt::Sprintf!("%s: empty argv", tool.Name.clone()));
                }
                for (_, a) in range!(c.Argv.clone()) {
                    if *a == "" {
                        t.Fatal(fmt::Sprintf!("%s: empty argv element", tool.Name.clone()));
                    }
                }
            }
        }
    }
}

fn test_nsys_window_contract(t: &mut testing::T) {
    let (nsys, _) = profile::Find("nsys");
    // setup must launch with node-level graph tracing and cuda,nvtx
    let mut setup = string("");
    for (_, c) in range!(nsys.Setup.clone()) {
        setup = (setup) + (profile::RenderCmd(&c)) + (" ");
    }
    if !strings::Contains(setup.clone(), "--cuda-graph-trace=node") {
        t.Fatal(string("nsys setup lost --cuda-graph-trace=node; without it only ~5% of GPU work is visible"));
    }
    if !strings::Contains(setup.clone(), "cuda,nvtx") {
        t.Fatal(string("nsys setup lost -t cuda,nvtx"));
    }
    // window: start exactly, then a sleep, then stop exactly; trace
    // flags must NOT appear on start (nsys start rejects them), but
    // GPU metrics sampling belongs here (kvlm strips it on retry when
    // the target's driver refuses GPU counters)
    let first = profile::RenderCmd(&nsys.Window[0usize]);
    if first != "nsys start --session={session} -o {out} --force-overwrite=true --gpu-metrics-devices=cuda-visible" {
        t.Fatal(fmt::Sprintf!("window start drifted: %q", first));
    }
    let last = profile::RenderCmd(&nsys.Window[(nsys.Window.Len() - 1) as usize]);
    if last != "nsys stop --session={session}" {
        t.Fatal(fmt::Sprintf!("window stop drifted: %q", last));
    }
}

fn test_runpod_matrix(t: &mut testing::T) {
    // cprofile is deliberately absent: its RunPod status is untested
    // and untested rows carry no measured provenance
    let expect = [
        ("nsys", StatusWorks),
        ("metrics", StatusWorks),
        ("torchprof", StatusWorks),
        ("pyspy", StatusBlocked),
        ("perf", StatusBlocked),
        ("ncu", StatusBlocked),
    ];
    for (name, want) in expect.iter() {
        let (tool, _) = profile::Find(*name);
        let (s, ok) = profile::SupportFor(&tool, EnvRunPod);
        if !ok {
            t.Fatal(fmt::Sprintf!("%s: no runpod entry", string(*name)));
        }
        if s.Status != *want {
            t.Fatal(fmt::Sprintf!(
                "%s on runpod: got %q, want %q",
                string(*name),
                s.Status.clone(),
                string(*want)
            ));
        }
        if s.Verified == "" {
            t.Fatal(fmt::Sprintf!("%s: runpod status is a measured fact, provenance required", string(*name)));
        }
    }
}

fn test_expand(t: &mut testing::T) {
    let cmds = slice!([]profile::Cmd{
        profile::Cmd{
            Argv: slice!([]string{"nsys", "start", "--session={session}", "-o", "{out}"}),
            ..Default::default()
        },
    });
    let vars = slice!([]profile::Var{
        profile::Var{ Key: string("session"), Value: string("kvlm"), ..Default::default() },
    });
    let out = profile::Expand(cmds, vars);
    let got = profile::RenderCmd(&out[0usize]);
    // {session} substituted; unknown {out} passes through verbatim
    if got != "nsys start --session=kvlm -o {out}" {
        t.Fatal(fmt::Sprintf!("expand: got %q", got));
    }
}

fn test_render_quoting(t: &mut testing::T) {
    let c = profile::Cmd {
        Argv: slice!([]string{"sh", "-c", "echo hello world"}),
        ..Default::default()
    };
    let got = profile::RenderCmd(&c);
    if got != "sh -c 'echo hello world'" {
        t.Fatal(fmt::Sprintf!("quoting: got %q", got));
    }
}

fn test_torchprof_window_contract(t: &mut testing::T) {
    // endpoint-driven capture: start, bounded wait, stop; the addr is
    // a placeholder so profile run points it at the real server
    let (tp, ok) = profile::Find("torchprof");
    if !ok {
        t.Fatal(string("torchprof not registered"));
    }
    if tp.Window.Len() != 3 {
        t.Fatal(fmt::Sprintf!("torchprof window: %d cmds, want 3", tp.Window.Len()));
    }
    let first = profile::RenderCmd(&tp.Window[0usize]);
    if first != "curl -s -X POST http://{addr}/start_profile" {
        t.Fatal(fmt::Sprintf!("start drifted: %q", first));
    }
    let last = profile::RenderCmd(&tp.Window[2usize]);
    if last != "curl -s -X POST http://{addr}/stop_profile" {
        t.Fatal(fmt::Sprintf!("stop drifted: %q", last));
    }
    let mid: &str = tp.Window[1usize].Argv[0usize].as_ref();
    if mid != "sleep" {
        t.Fatal(fmt::Sprintf!("middle must be the window sleep, got %q", tp.Window[1usize].Argv[0usize].clone()));
    }
}

fn test_probe_script_contract(t: &mut testing::T) {
    let s = profile::probe::ProbeScript();
    // parameters arrive as positional args, never interpolated
    for want in ["ADDR=\"$1\"", "MODEL=\"$2\"", "CONC=\"$3\"", "SECS=\"$4\"", "MAXTOK=\"$5\""].iter() {
        if !strings::Contains(s.clone(), string(*want)) {
            t.Fatal(fmt::Sprintf!("probe script missing %q", string(*want)));
        }
    }
    // unique prompt per request so prefix caching cannot collapse the work
    if !strings::Contains(s.clone(), string("$(date +%s%N)")) {
        t.Fatal(string("probe prompt is not unique per request"));
    }
    // the exact sh-quoted JSON body (validated with sh -n and a live
    // dry run against a dead port)
    if !strings::Contains(
        s.clone(),
        string("-d \"{\\\"model\\\":\\\"$MODEL\\\",\\\"prompt\\\":\\\"kvlm probe $$ $i $(date +%s%N)\\\",\\\"max_tokens\\\":$MAXTOK,\\\"temperature\\\":0.8}\""),
    ) {
        t.Fatal(string("probe curl body quoting drifted from the validated form"));
    }
    // must wait for stragglers so the stop command runs after the load
    if !strings::HasSuffix(strings::TrimSpace(s.clone()), string("wait")) {
        t.Fatal(string("probe script must end with wait"));
    }
    // travels base64-encoded; single quotes would still be a smell in
    // anything ever pasted into an ssh command line
    if strings::Contains(s.clone(), string("'")) {
        t.Fatal(string("probe script contains single quotes"));
    }
}

fn test_next_run_name(t: &mut testing::T) {
    let cases: &[(&[&'static str], &'static str)] = &[
        (&[], "run1"),
        (&["run2", "run3", "run4"], "run5"),
        (&["run9", "run10"], "run11"),
        (&["run4", "imported-thing", "SUMMARY.md"], "run5"),
    ];
    for (existing, want) in cases.iter() {
        let mut e: goish::slice<string> = goish::make!([]string, 0);
        for n in existing.iter() {
            e = goish::append!(e.clone(), string(*n));
        }
        let got = profile::NextRunName(e);
        if got != *want {
            t.Fatal(fmt::Sprintf!("got %q want %q", got, string(*want)));
        }
    }
}

#[goish::main]
fn main() {
    let tests: &[(&str, testing::TestFn)] = &[
        ("TestRegistryHasAllTools", test_registry_has_all_tools),
        ("TestEveryToolHasValidSupport", test_every_tool_has_valid_support),
        ("TestEveryArgvNonEmpty", test_every_argv_non_empty),
        ("TestNsysWindowContract", test_nsys_window_contract),
        ("TestRunPodMatrix", test_runpod_matrix),
        ("TestExpand", test_expand),
        ("TestRenderQuoting", test_render_quoting),
        ("TestTorchprofWindowContract", test_torchprof_window_contract),
        ("TestProbeScriptContract", test_probe_script_contract),
        ("TestNextRunName", test_next_run_name),
    ];
    let code = testing::Main(tests);
    syscall::Exit(int32(code));
}
