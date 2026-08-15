// profile package: the profiling-tool catalog kvlm runs against a
// serving pod. One file per tool (nsys.rs, metrics.rs, ...), each
// self-registering its Tool in init() — same registry idiom as the
// driver and model packages.
//
// The Support matrix, command contracts, and blockers are measured on
// real hardware; each Support entry carries its own provenance in
// Verified.
//
// One source of truth: Cmd argv arrays carry {placeholders}; `profile
// show` and `profile run` expand the SAME data, so what show prints is
// exactly what run executes.
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub mod archive;
pub mod cprofile;
pub mod graph;
pub mod launch;
pub mod measured;
pub mod probe;
pub mod propose;
pub mod vllmcfg;
pub mod vllmflags;
pub mod metrics;
pub mod ncu;
pub mod nsys;
pub mod perf;
pub mod pyspy;
pub mod torchprof;

goish::import! {
    crate::profile::nsys as __init_prof_nsys,
    crate::profile::metrics as __init_prof_metrics,
    crate::profile::pyspy as __init_prof_pyspy,
    crate::profile::perf as __init_prof_perf,
    crate::profile::ncu as __init_prof_ncu,
    crate::profile::torchprof as __init_prof_torchprof,
    crate::profile::cprofile as __init_prof_cprofile,
}

use goish::fmt;
use goish::lazy::Lazy;
use goish::strconv;
use goish::strings;
use goish::sync;
use goish::string;
use goish::{append, make, nil, range, slice};

// Status values for Support.Status.
pub const StatusWorks: &str = "works";
pub const StatusBlocked: &str = "blocked";
pub const StatusUntested: &str = "untested";

// Environment keys for Support.Env.
pub const EnvRunPod: &str = "runpod";
pub const EnvK8s: &str = "k8s";
pub const EnvBareMetal: &str = "baremetal";

// Support is one tool x environment cell of the capability matrix.
// Reason is required when Status is blocked (why, with the exact error
// where we have one) and optional context otherwise. Verified carries
// measurement provenance ("measured 2026-08-08, ..."), "" = untested.
#[derive(Clone, Default)]
pub struct Support {
    pub Env: string,
    pub Status: string,
    pub Reason: string,
    pub Verified: string,
}

// Cmd is one command as data. Argv entries may contain the
// placeholders {session}, {out}, {addr}, {seconds}, {pid}, {serve};
// Expand substitutes them. Note is a one-line caveat.
#[derive(Clone, Default)]
pub struct Cmd {
    pub Argv: slice<string>,
    pub Note: string,
}

// Tool is one profiling tool: where it works, how to install it, how
// to put the server under it, the bounded capture window, what
// artifacts come back, and how to analyze them locally.
#[derive(Clone, Default)]
pub struct Tool {
    pub Name: string,
    pub Summary: string,
    pub Support: slice<Support>,
    pub Install: slice<Cmd>,   // remote, once
    pub Setup: slice<Cmd>,     // launch the server under the profiler
    pub Window: slice<Cmd>,    // bounded capture window, remote
    pub Artifacts: slice<string>, // remote paths `profile run` fetches
    pub Analyze: slice<Cmd>,   // local, after fetch
    pub Notes: slice<string>,
}

// Var is one placeholder binding for Expand.
#[derive(Clone, Default)]
pub struct Var {
    pub Key: string,
    pub Value: string,
}

static tools: Lazy<sync::Mutex<alloc::vec::Vec<Tool>>> =
    Lazy::new(|| sync::Mutex::new(alloc::vec::Vec::new()));

pub fn Register(t: Tool) {
    tools.Lock().push(t);
}

// ToolsSorted returns the catalog sorted by name (init_array order is
// link order, not alphabetical).
pub fn ToolsSorted() -> alloc::vec::Vec<Tool> {
    let mut list = tools.Lock().clone();
    list.sort_by(|a, b| {
        let an: &str = a.Name.as_ref();
        let bn: &str = b.Name.as_ref();
        an.cmp(bn)
    });
    list
}

// Find looks a tool up by name.
pub fn Find<S: Into<string>>(name: S) -> (Tool, bool) {
    let name = name.into();
    for t in ToolsSorted().iter() {
        if t.Name == name {
            return (t.clone(), true);
        }
    }
    (Default::default(), false)
}

// Names returns the sorted tool names as a comma-separated list.
pub fn Names() -> string {
    let mut names: slice<string> = make!([]string, 0);
    for t in ToolsSorted().iter() {
        names = append!(names.clone(), t.Name.clone());
    }
    strings::Join(names, ", ")
}

// SupportFor returns the tool's Support entry for one environment.
pub fn SupportFor(t: &Tool, env: &str) -> (Support, bool) {
    for (_, s) in range!(t.Support.clone()) {
        if s.Env == env {
            return (s.clone(), true);
        }
    }
    (Default::default(), false)
}

// Expand substitutes {key} placeholders in every argv element. Unknown
// placeholders pass through verbatim so show output stays honest about
// what still needs filling in.
pub fn Expand(cmds: slice<Cmd>, vars: slice<Var>) -> slice<Cmd> {
    let mut out: slice<Cmd> = make!([]Cmd, 0);
    for (_, c) in range!(cmds.clone()) {
        let mut argv: slice<string> = make!([]string, 0);
        for (_, a) in range!(c.Argv.clone()) {
            let mut e = a.clone();
            for (_, v) in range!(vars.clone()) {
                let pat = ("{") + (v.Key.clone()) + ("}");
                e = strings::ReplaceAll(e, pat, v.Value.clone());
            }
            argv = append!(argv.clone(), e);
        }
        out = append!(
            out.clone(),
            Cmd {
                Argv: argv,
                Note: c.Note.clone(),
            }
        );
    }
    out
}

// RenderCmd returns the shell form of one Cmd, quoting arguments that
// need it (same convention as model::ServeSpec::Render).
pub fn RenderCmd(c: &Cmd) -> string {
    let mut parts: slice<string> = make!([]string, 0);
    for (_, a) in range!(c.Argv.clone()) {
        let mut e = a.clone();
        if strings::Contains(e.clone(), " ") || strings::Contains(e.clone(), "\"") {
            e = ("'") + (e) + ("'");
        }
        parts = append!(parts.clone(), e);
    }
    strings::Join(parts, " ")
}

// cmdOf / strsOf — literal-noise reducers for the tool data files
// (same role as the model package's flag()).
pub(crate) fn cmdOf(argv: &[&'static str], note: &'static str) -> Cmd {
    let mut a: slice<string> = make!([]string, 0);
    for s in argv.iter() {
        a = append!(a.clone(), string(*s));
    }
    Cmd {
        Argv: a,
        Note: string(note),
    }
}

pub(crate) fn strsOf(items: &[&'static str]) -> slice<string> {
    let mut out: slice<string> = make!([]string, 0);
    for s in items.iter() {
        out = append!(out.clone(), string(*s));
    }
    out
}

// NextRunName picks the next auto-numbered revision name from the
// directory names that already exist: run<max+1>, starting at run1.
// Non-run names are ignored, so imported or hand-named revisions do
// not disturb the sequence.
pub fn NextRunName(existing: slice<string>) -> string {
    let mut max: goish::int = 0;
    for (_, name) in range!(existing.clone()) {
        if !strings::HasPrefix(name.clone(), string("run")) {
            continue;
        }
        let digits = strings::TrimPrefix(name.clone(), string("run"));
        if digits == "" {
            continue;
        }
        let (n, err) = strconv::Atoi(digits);
        if err != nil && n == 0 {
            continue;
        }
        if n > max {
            max = n;
        }
    }
    fmt::Sprintf!("run%d", max + 1)
}

// LatestRunName returns the highest-numbered runN among existing
// directory names ("" when none exist yet).
pub fn LatestRunName(existing: slice<string>) -> string {
    let mut max: goish::int = 0;
    for (_, name) in range!(existing.clone()) {
        if !strings::HasPrefix(name.clone(), string("run")) {
            continue;
        }
        let digits = strings::TrimPrefix(name.clone(), string("run"));
        if digits == "" {
            continue;
        }
        let (n, err) = strconv::Atoi(digits);
        if err != nil && n == 0 {
            continue;
        }
        if n > max {
            max = n;
        }
    }
    if max == 0 {
        return string("");
    }
    fmt::Sprintf!("run%d", max)
}
