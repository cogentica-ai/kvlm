// cmd/profile.go equivalent: the profile command group. Everything
// here renders or executes the structured recipes in the profile
// package — one source of truth: `profile show` prints exactly what
// `profile run` executes. No prose lives in this file.
#![allow(non_snake_case)]
// lowercase type names are Go unexported-type names, kept verbatim
#![allow(non_camel_case_types)]

use goish::encoding::base64;
use goish::encoding::json;
use goish::fmt;
use goish::gomap::map;
use goish::io::Reader as _;
use goish::os;
use goish::os::exec;
use goish::path;
use goish::strconv;
use goish::strings;
use goish::text::tabwriter;
use goish::time;
use goish::string;
use goish::goslice::slice as goslice;
use goish::errors::error;
use goish::math;
use goish::{append, bytes, float64, go, int, make, nil, range, slice, types};

use goish::lazy::Lazy;
use goish::net::http;
use goish::sync;

use spf13_cobra as cobra;

use crate::cmd::rootCmd;
use crate::driver;
use crate::flamegraph;
use crate::metrics;
use crate::model;
use crate::profile;
use crate::profile::{EnvBareMetal, EnvK8s, EnvRunPod};
use crate::state;

// profileCmd represents the profile command group.
fn profileCmd() -> cobra::Command {
    cobra::Command {
        Use: string("profile"),
        Short: string("Profile vLLM serving performance"),
        Long: string(
            "Profile a vLLM serving pod. Recipes are data: kvlm profile show\n\
             prints exactly what kvlm profile run executes.\n\
             \n\
             Start with kvlm profile ls for the capability matrix.",
        ),
        ..Default::default()
    }
}

// lsCmd represents `kvlm profile ls`: the tool x environment matrix.
pub(crate) fn lsCmd() -> cobra::Command {
    cobra::Command {
        Use: string("ls"),
        Aliases: slice!([]string{"list"}),
        Short: string("List profiling tools and where they work"),
        Run: Some(alloc::sync::Arc::new(
            |_cmd: &mut cobra::Command, _args: goslice<string>| {
                let mut tw = tabwriter::NewWriter(os::Stdout(), 10, 1, 3, b' ', 0);
                fmt::Fprintf!(tw, "TOOL\tSUMMARY\tRUNPOD\tK8S\tBARE METAL\n");
                for t in profile::ToolsSorted().iter() {
                    fmt::Fprintf!(
                        tw,
                        "%s\t%s\t%s\t%s\t%s\n",
                        t.Name.clone(),
                        t.Summary.clone(),
                        supportCell(t, EnvRunPod),
                        supportCell(t, EnvK8s),
                        supportCell(t, EnvBareMetal)
                    );
                }
                let _ = tw.Flush();
                fmt::Printf!("\nReasons and recipes: kvlm profile show <tool>\n");
            },
        )),
        ..Default::default()
    }
}

// supportCell renders one matrix cell: the status, with "(verified)"
// when the entry carries measurement provenance.
fn supportCell(t: &profile::Tool, env: &str) -> string {
    let (s, ok) = profile::SupportFor(t, env);
    if !ok {
        return string("-");
    }
    let mut out = s.Status.clone();
    if s.Verified != "" {
        out = (out) + (" (verified)");
    }
    out
}

// defaultVars are the placeholder bindings `profile show` renders
// with; `profile run` binds the same keys from its flags.
fn defaultVars() -> slice<profile::Var> {
    slice!([]profile::Var{
        profile::Var{ Key: string("session"), Value: string("kvlm"), ..Default::default() },
        profile::Var{ Key: string("out"), Value: string("/tmp/kvlm-profile/win"), ..Default::default() },
        profile::Var{ Key: string("addr"), Value: string("localhost:8000"), ..Default::default() },
        profile::Var{ Key: string("seconds"), Value: string("30"), ..Default::default() },
        profile::Var{ Key: string("serve"), Value: string("./serve.sh"), ..Default::default() },
    })
}

// expandOne applies placeholder bindings to a single string.
fn expandOne(s: string, vars: &slice<profile::Var>) -> string {
    let mut e = s;
    for (_, v) in range!(vars.clone()) {
        let pat = ("{") + (v.Key.clone()) + ("}");
        e = strings::ReplaceAll(e, pat, v.Value.clone());
    }
    e
}

// printPhase renders one phase of a recipe: notes as comments, then
// the shell form of each command.
fn printPhase(title: &'static str, cmds: slice<profile::Cmd>) {
    if cmds.Len() == 0 {
        return;
    }
    fmt::Printf!("\n%s\n", string(title));
    for (_, c) in range!(cmds.clone()) {
        if c.Note != "" {
            fmt::Printf!("  # %s\n", c.Note.clone());
        }
        fmt::Printf!("  %s\n", profile::RenderCmd(&c));
    }
}

// showCmd represents `kvlm profile show <tool>`: the full recipe.
pub(crate) fn showCmd() -> cobra::Command {
    cobra::Command {
        Use: string("show <tool>"),
        Short: string("Show a tool's support status and full recipe"),
        Args: Some(cobra::ExactArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |_cmd: &mut cobra::Command, args: goslice<string>| -> error {
                let (t, ok) = profile::Find(args[0usize].clone());
                if !ok {
                    return fmt::Errorf!(
                        "unknown tool %q (have: %s)",
                        args[0usize].clone(),
                        profile::Names()
                    );
                }
                fmt::Printf!("%s: %s\n", t.Name.clone(), t.Summary.clone());

                fmt::Printf!("\nSupport:\n");
                let mut tw = tabwriter::NewWriter(os::Stdout(), 10, 1, 3, b' ', 0);
                fmt::Fprintf!(tw, "ENV\tSTATUS\tNOTES\n");
                for (_, s) in range!(t.Support.clone()) {
                    let mut notes = s.Reason.clone();
                    if s.Verified != "" {
                        if notes != "" {
                            notes = (notes) + ("; ");
                        }
                        notes = (notes) + (s.Verified.clone());
                    }
                    fmt::Fprintf!(tw, "%s\t%s\t%s\n", s.Env.clone(), s.Status.clone(), notes);
                }
                let _ = tw.Flush();

                let vars = defaultVars();
                printPhase("Install (remote, once):", profile::Expand(t.Install.clone(), vars.clone()));
                printPhase("Launch the server under the profiler:", profile::Expand(t.Setup.clone(), vars.clone()));
                printPhase("Capture window:", profile::Expand(t.Window.clone(), vars.clone()));
                if t.Artifacts.Len() > 0 {
                    fmt::Printf!("\nArtifacts (fetched by kvlm profile run):\n");
                    for (_, a) in range!(t.Artifacts.clone()) {
                        fmt::Printf!("  %s\n", expandOne(a.clone(), &vars));
                    }
                }
                printPhase("Analyze locally:", profile::Expand(t.Analyze.clone(), vars.clone()));

                if t.Notes.Len() > 0 {
                    fmt::Printf!("\nNotes:\n");
                    for (_, n) in range!(t.Notes.clone()) {
                        fmt::Printf!("  - %s\n", n.clone());
                    }
                }
                nil.into()
            },
        )),
        ..Default::default()
    }
}

// metricsCmd represents `kvlm profile metrics`: fetch a live /metrics
// endpoint; with --interval, two samples and the derived window stats.
pub(crate) fn metricsCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("metrics"),
        Short: string("Fetch and summarize vLLM Prometheus metrics"),
        Long: string(
            "Fetch a running vLLM server's /metrics endpoint and print a\n\
             categorized summary. With --interval, take two samples that many\n\
             seconds apart and print the window statistics instead: TTFT and\n\
             TPOT means, token rates, and prefix cache hit rate.",
        ),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, _args: goslice<string>| -> error {
                let (addr, _) = cmd.Flags().GetString("addr");
                let (interval, _) = cmd.Flags().GetInt("interval");

                let (snap, err) = metrics::Fetch(addr.clone());
                if err != nil {
                    return fmt::Errorf!("fetch %s: %v", addr, err);
                }
                if interval <= 0 {
                    metrics::Print(&snap);
                    return nil.into();
                }
                fmt::Printf!("sampling again in %d s...\n\n", interval);
                let t0 = time::Now();
                time::Sleep(time::Seconds(interval));
                let (snap2, err) = metrics::Fetch(addr.clone());
                if err != nil {
                    return fmt::Errorf!("fetch %s: %v", addr, err);
                }
                // rates use measured wall time, not the flag, so a
                // slow fetch cannot inflate them
                let elapsed = time::Since(t0).Seconds();
                let d = metrics::Derive(&snap, &snap2, elapsed);
                metrics::PrintDerived(&d);
                nil.into()
            },
        )),
        ..Default::default()
    };
    c.Flags().StringP(
        string("addr"),
        string("a"),
        string("localhost:8000"),
        string("vLLM server address (host:port)"),
    );
    c.Flags().IntP(
        string("interval"),
        string("i"),
        0,
        string("seconds between two samples; 0 prints one snapshot"),
    );
    c
}

// flamegraphCmd represents `kvlm profile flamegraph`: folded stacks in,
// SVG out, using the flamegraph package.
fn flamegraphCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("flamegraph"),
        Short: string("Render an SVG flamegraph from folded stacks"),
        Long: string(
            "Render folded stack samples (one 'funcA;funcB;funcC 42' line per\n\
             stack, the stackcollapse format) into an SVG flamegraph.",
        ),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, _args: goslice<string>| -> error {
                let (input, _) = cmd.Flags().GetString("input");
                let (output, _) = cmd.Flags().GetString("output");
                let (title, _) = cmd.Flags().GetString("title");
                if input == "" {
                    return fmt::Errorf!("--input is required (folded stacks, one 'funcA;funcB 42' line per stack)");
                }
                let (data, err) = os::ReadFile(input.clone());
                if err != nil {
                    return fmt::Errorf!("read %s: %v", input, err);
                }
                let (fg, err) = flamegraph::ParseFolded(string(data));
                if err != nil {
                    return err;
                }
                let mut fg = fg;
                if title != "" {
                    fg.Title = title;
                }
                let (vertical, _) = cmd.Flags().GetBool("vertical");
                fg.Vertical = vertical;
                let (fh, _) = cmd.Flags().GetInt("frame-height");
                if fh > 0 {
                    fg.Height = fh;
                }
                if fg.Total <= 0 {
                    return fmt::Errorf!(
                        "no samples parsed from %s (expected folded stacks: 'funcA;funcB 42' per line)",
                        input
                    );
                }
                let svg = flamegraph::ToSVG(&fg);
                let err = os::WriteFile(output.clone(), svg.clone(), 0o644);
                if err != nil {
                    return fmt::Errorf!("write %s: %v", output, err);
                }
                fmt::Printf!("wrote %s (%d samples)\n", output, fg.Total);
                nil.into()
            },
        )),
        ..Default::default()
    };
    c.Flags().StringP(
        string("input"),
        string("i"),
        string(""),
        string("folded stack file (required)"),
    );
    c.Flags().StringP(
        string("output"),
        string("o"),
        string("flame.svg"),
        string("output SVG path"),
    );
    c.Flags().StringP(
        string("title"),
        string("t"),
        string(""),
        string("graph title (default: Flame Graph)"),
    );
    c.Flags().BoolP(
        string("vertical"),
        string("V"),
        false,
        string("vertical partition layout (depth as columns); best for shallow stacks with long names"),
    );
    let _ = c.Flags().Int_flag(
        string("frame-height"),
        0,
        string("row height in pixels (default 16, like flamegraph.pl --height)"),
    );
    c
}

// queryRows runs one sqlite query through the sqlite3 CLI and returns
// the non-empty output lines, tab-separated so kernel names (which
// contain commas, angle brackets, and pipes) survive as the last field.
fn queryRows(db: string, sql: string) -> (goslice<string>, error) {
    let mut args: goslice<string> = make!([]string, 0);
    args = append!(args.clone(), string("-separator"));
    args = append!(args.clone(), string("\t"));
    args = append!(args.clone(), db.clone());
    args = append!(args.clone(), sql);
    let (out, errOut, err) = runCaptureRetry(string("sqlite3"), args);
    if err != nil {
        return (
            make!([]string, 0),
            fmt::Errorf!("sqlite3 %s: %v: %s", db, err, strings::TrimSpace(errOut)),
        );
    }
    let mut rows: goslice<string> = make!([]string, 0);
    for (_, line) in range!(strings::Split(out, "\n")) {
        if strings::TrimSpace(line.clone()) != "" {
            rows = append!(rows.clone(), line);
        }
    }
    (rows, nil.into())
}

// findNsys discovers the local nsys binary beyond PATH: the KVLM_NSYS
// override first, then the usual Nsight Systems install roots (system
// and user-local), newest version directory first. Returns "" when
// nothing is found. Called from the resolvedTools init so it runs
// before the first fork (Getenv breaks after forking, see below).
fn findNsys() -> string {
    let over = os::Getenv("KVLM_NSYS");
    if over != "" {
        let (_, err) = os::Stat(over.clone());
        if err == nil {
            return over;
        }
    }
    let (p, err) = exec::LookPath("nsys");
    if err == nil {
        return p;
    }
    let mut roots: slice<string> = make!([]string, 0);
    roots = append!(roots.clone(), string("/opt/nvidia/nsight-systems"));
    let home = os::Getenv("HOME");
    if home != "" {
        roots = append!(roots.clone(), (home) + ("/.local/opt/nsight-systems"));
    }
    for (_, root) in range!(roots.clone()) {
        let (entries, err) = os::ReadDir(root.clone());
        if err != nil {
            continue;
        }
        // version dirs sort lexically well enough (2026.1.3 style);
        // walk from the newest
        let mut vs: alloc::vec::Vec<string> = alloc::vec::Vec::new();
        for e in entries.iter() {
            vs.push(e.Name());
        }
        vs.sort_by(|a, b| {
            let an: &str = a.as_ref();
            let bn: &str = b.as_ref();
            an.cmp(bn)
        });
        vs.reverse();
        for v in vs.iter() {
            let cand = fmt::Sprintf!("%s/%s/target-linux-x64/nsys", root.clone(), v.clone());
            let (_, err) = os::Stat(cand.clone());
            if err == nil {
                return cand;
            }
        }
    }
    string("")
}

// resolvedTools caches absolute paths for every external tool this
// file spawns, resolved eagerly on first use, BEFORE the process ever
// forks. Root cause of the long-standing intermittent "exit status 127
// with empty stderr": after the first fork, the goish runtime's
// Getenv("PATH") starts returning empty, so exec::Command's LookPath
// silently falls back to the bare name and the child's execve fails
// with ENOENT (strace-verified). Resolving pre-fork and always
// spawning absolute paths sidesteps the runtime bug entirely.
static resolvedTools: Lazy<sync::Mutex<slice<string>>> = Lazy::new(|| {
    let mut cache: slice<string> = make!([]string, 0);
    for name in ["ssh", "scp", "sqlite3", "echo", "sh", "curl", "python3"].iter() {
        let (p, err) = exec::LookPath(string(*name));
        if err == nil {
            cache = append!(cache.clone(), (string(*name)) + ("\t") + (p));
        }
    }
    // nsys gets richer discovery: PATH, KVLM_NSYS, install roots
    let np = findNsys();
    if np != "" {
        cache = append!(cache.clone(), ("nsys\t") + (np));
    }
    sync::Mutex::new(cache)
});

// toolAvailable reports whether a tool resolved in the pre-fork
// cache. This is the ONLY correct presence check in this file: a live
// LookPath lies after the first fork (the runtime's PATH goes empty),
// reporting an installed tool as missing.
pub(crate) fn toolAvailable(name: &'static str) -> bool {
    strings::Contains(toolPath(string(name)), "/")
}

// nsysLocalPath returns the discovered local nsys, or "" when none is
// available (toolPath returns the bare name in that case).
fn nsysLocalPath() -> string {
    let p = toolPath(string("nsys"));
    if strings::Contains(p.clone(), "/") {
        return p;
    }
    string("")
}

// toolPath returns the pre-fork absolute path for a tool, or the name
// unchanged when it was not resolvable (the caller's LookPath check
// then reports the real "not installed" error).
fn toolPath(name: string) -> string {
    if strings::Contains(name.clone(), "/") {
        return name;
    }
    let cache = resolvedTools.Lock();
    for (_, entry) in range!(cache.clone()) {
        let parts = strings::SplitN(entry.clone(), "\t", 2);
        if parts.len() == 2 && parts[0usize] == name {
            return parts[1usize].clone();
        }
    }
    name
}

// runCaptureRetry wraps runCapture with the absolute-path resolution
// above plus one retry for genuinely transient spawn failures.
fn runCaptureRetry(name: string, args: goslice<string>) -> (string, string, error) {
    let bin = toolPath(name);
    let (out, errOut, err) = runCapture(bin.clone(), args.clone());
    if err != nil && strings::TrimSpace(errOut.clone()) == "" {
        return runCapture(bin, args);
    }
    (out, errOut, err)
}

// writeFolded aggregates the whole capture into two-level folded
// stacks ("in cuda graph;<kernel>[;grid_<G>] <ns>") and renders the
// flamegraph SVG beside them. The grid becomes a third level only
// when one kernel ran with several grids.
fn writeFolded(db: string, outDir: string) -> error {
    let (rows, err) = queryRows(
        db.clone(),
        string("SELECT CASE WHEN k.graphId IS NULL OR k.graphId = 0 THEN 'eager' ELSE 'in cuda graph' END, k.gridX||'x'||k.gridY||'x'||k.gridZ, SUM(k.end-k.start), s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName GROUP BY 1, 2, s.value"),
    );
    if err != nil {
        return err;
    }
    // aggregate under simplified names: scope, name, grid, ns
    let mut scopes: goslice<string> = make!([]string, 0);
    let mut names: goslice<string> = make!([]string, 0);
    let mut grids: goslice<string> = make!([]string, 0);
    let mut sums: goslice<int> = make!([]int, 0);
    for (_, row) in range!(rows.clone()) {
        let parts = strings::SplitN(row.clone(), "\t", 4);
        if parts.len() < 4 {
            continue;
        }
        let scope = parts[0usize].clone();
        let grid = parts[1usize].clone();
        let (ns, _) = strconv::Atoi(parts[2usize].clone());
        let name = profile::graph::SimplifyKernelName(parts[3usize].clone());
        let mut found = false;
        let mut i: int = 0;
        while i < names.len() as int {
            if scopes[i as usize] == scope && names[i as usize] == name && grids[i as usize] == grid {
                sums[i as usize] += ns;
                found = true;
                break;
            }
            i += 1;
        }
        if !found {
            scopes = append!(scopes.clone(), scope);
            names = append!(names.clone(), name);
            grids = append!(grids.clone(), grid);
            sums = append!(sums.clone(), ns);
        }
    }
    let mut b = strings::Builder::new();
    let mut i: int = 0;
    while i < names.len() as int {
        // grid level only when this scope+name ran with several grids
        let mut multi = false;
        let mut j: int = 0;
        while j < names.len() as int {
            if j != i && scopes[j as usize] == scopes[i as usize] && names[j as usize] == names[i as usize] {
                multi = true;
                break;
            }
            j += 1;
        }
        if multi {
            let _ = b.WriteString(fmt::Sprintf!(
                "%s;%s;grid_%s %d\n",
                scopes[i as usize].clone(),
                names[i as usize].clone(),
                grids[i as usize].clone(),
                sums[i as usize]
            ));
        } else {
            let _ = b.WriteString(fmt::Sprintf!(
                "%s;%s %d\n",
                scopes[i as usize].clone(),
                names[i as usize].clone(),
                sums[i as usize]
            ));
        }
        i += 1;
    }
    let folded = string(b.String());
    let err = os::WriteFile((outDir.clone()) + ("/gpu-folded.txt"), folded.clone(), 0o644);
    if err != nil {
        return err;
    }
    let (fg, err) = flamegraph::ParseFolded(folded);
    if err != nil {
        return err;
    }
    let mut fg = fg;
    fg.Title = string("GPU time");
    if fg.Total <= 0 {
        return fmt::Errorf!("no samples in folded output");
    }
    let err = os::WriteFile((outDir.clone()) + ("/gpu-time.svg"), flamegraph::ToSVG(&fg), 0o644);
    if err != nil {
        return err;
    }
    fmt::Printf!("wrote %s/gpu-folded.txt and gpu-time.svg\n", outDir);
    nil.into()
}

// graphNodes reads the per-node aggregates of one executable graph:
// every kernel record with this graphId, grouped by graph node,
// averaged across all replays, in first-replay execution order.
fn graphNodes(db: string, graphID: int) -> (goslice<profile::graph::Node>, error) {
    let sql = fmt::Sprintf!(
        "SELECT k.gridX||'x'||k.gridY||'x'||k.gridZ, k.gridX*k.gridY*k.gridZ, CAST(AVG(k.end-k.start) AS INTEGER), MAX(k.end-k.start), s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName WHERE k.graphId = %d GROUP BY k.graphNodeId ORDER BY MIN(k.start)",
        graphID
    );
    let (rows, err) = queryRows(db.clone(), sql);
    if err != nil {
        return (make!([]profile::graph::Node, 0), err);
    }
    let mut nodes: goslice<profile::graph::Node> = make!([]profile::graph::Node, 0);
    for (i, row) in range!(rows) {
        let parts = strings::SplitN(row.clone(), "\t", 5);
        if parts.len() < 5 {
            continue;
        }
        let (blocks, _) = strconv::Atoi(parts[1usize].clone());
        let (avgNs, _) = strconv::Atoi(parts[2usize].clone());
        let (maxNs, _) = strconv::Atoi(parts[3usize].clone());
        let full = parts[4usize].clone();
        let name = profile::graph::SimplifyKernelName(full.clone());
        nodes = append!(
            nodes.clone(),
            profile::graph::Node {
                Pos: i + 1,
                Name: name.clone(),
                FullName: full,
                Category: profile::graph::ClassifyKernel(name),
                Grid: parts[0usize].clone(),
                Blocks: blocks,
                AvgUs: (avgNs as float64) / 1000.0,
                MaxUs: (maxNs as float64) / 1000.0,
                ..Default::default()
            }
        );
    }
    (nodes, nil.into())
}

fn jnum2(v: float64) -> string {
    fmt::Sprintf!("%v", math::Round(v * 100.0) / 100.0)
}

// writeMetricArray appends the elements of one Metric group to a JSON
// array being built (brackets belong to the caller).
fn writeMetricArray(b: &mut strings::Builder, ms: slice<profile::measured::Metric>) {
    for (i, m) in range!(ms.clone()) {
        if i > 0 {
            let _ = b.WriteString(",");
        }
        let _ = b.WriteString(fmt::Sprintf!(
            "{\"key\":\"%s\",\"label\":\"%s\",\"value\":%v,\"unit\":\"%s\",\"max\":%v,\"note\":\"%s\"}",
            jsonEsc(m.Key.clone()),
            jsonEsc(m.Label.clone()),
            m.Value,
            jsonEsc(m.Unit.clone()),
            m.Max,
            jsonEsc(m.Note.clone())
        ));
    }
}

// writeGraphNode appends one chain-node object to the JSON being built.
fn writeGraphNode(b: &mut strings::Builder, n: &profile::graph::Node) {
    let _ = b.WriteString(fmt::Sprintf!(
        "{\"pos\":%d,\"name\":\"%s\",\"category\":\"%s\",\"grid\":\"%s\",\"blocks\":%d,\"avgUs\":%s,\"maxUs\":%s,\"pctOfLayer\":%s,\"ratio\":%s,\"verdict\":\"%s\"}",
        n.Pos,
        jsonEsc(n.Name.clone()),
        jsonEsc(n.Category.clone()),
        n.Grid,
        n.Blocks,
        jnum(n.AvgUs),
        jnum(n.MaxUs),
        jnum(n.PctOfLayer),
        jnum2(n.Ratio),
        jsonEsc(n.Verdict.clone())
    ));
}

// writeCorrelation appends the measurement-to-action rows to the JSON
// being built (brackets belong to the caller).
// leverKey maps a lever's flag token to the engine config key whose
// current value annotates it; "" when the lever is not a single flag.
fn leverKey(param: string) -> string {
    let tok = strings::SplitN(param.clone(), " ", 2)[0usize].clone();
    if tok == "-tp" {
        return string("tensor_parallel_size");
    }
    if !strings::HasPrefix(tok.clone(), string("--")) {
        return string("");
    }
    strings::ReplaceAll(strings::TrimPrefix(tok, string("--")), "-", "_")
}

fn writeCorrelation(
    b: &mut strings::Builder,
    corr: slice<profile::graph::Correlation>,
    effective: &goslice<profile::vllmcfg::KV>,
) -> string {
    let mut rec = strings::Builder::new();
    for (i, c) in range!(corr.clone()) {
        if i > 0 {
            let _ = b.WriteString(", ");
        }
        let _ = b.WriteString(fmt::Sprintf!(
            "{\"class\":\"%s\",\"pctOfLayer\":%s,\"nodes\":%d,\"levers\":[",
            jsonEsc(c.Class.clone()),
            jnum(c.PctOfLayer),
            c.NodeCount
        ));
        let _ = rec.WriteString(fmt::Sprintf!(
            "  %s: %s%% of the layer across %d nodes\n",
            c.Class.clone(),
            jnum(c.PctOfLayer),
            c.NodeCount
        ));
        for (j, l) in range!(c.Levers.clone()) {
            if j > 0 {
                let _ = b.WriteString(",");
            }
            let key = leverKey(l.Param.clone());
            let mut current = string("");
            if key != "" {
                for (_, kv) in range!(effective.clone()) {
                    if kv.Key == key {
                        current = kv.Value.clone();
                        break;
                    }
                }
            }
            // a boolean lever whose flag is already on is applied
            let applied = strings::HasPrefix(key.clone(), string("enable_")) && current == "True";
            let _ = b.WriteString(fmt::Sprintf!(
                "{\"param\":\"%s\",\"effect\":\"%s\",\"current\":\"%s\",\"applied\":%v}",
                jsonEsc(l.Param.clone()),
                jsonEsc(l.Effect.clone()),
                jsonEsc(current.clone()),
                applied
            ));
            let mut now = string("");
            if current != "" {
                now = fmt::Sprintf!(" (now %s)", current.clone());
            }
            if applied {
                now = string(" (already set)");
            }
            let _ = rec.WriteString(fmt::Sprintf!(
                "    %s%s: %s\n",
                l.Param.clone(),
                now,
                l.Effect.clone()
            ));
        }
        let _ = b.WriteString("],\"kernels\":[");
        for (j, p) in range!(c.KernelPaths.clone()) {
            if j > 0 {
                let _ = b.WriteString(",");
            }
            let _ = b.WriteString(fmt::Sprintf!("\"%s\"", jsonEsc(p.clone())));
        }
        let _ = b.WriteString("]}");
    }
    string(rec.String())
}

// graphCmd represents `kvlm profile graph`: reconstruct the executable
// CUDA graph from a node-mode nsys capture and write the structure plus
// bottleneck verdicts as a JSON artifact the dashboard renders.
fn graphCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("graph"),
        Short: string("Reconstruct a captured CUDA graph and analyze bottlenecks"),
        Long: string(
            "Reconstruct the executable CUDA graph from a node-mode nsys\n\
             capture (.sqlite export, or .nsys-rep if nsys is on PATH): the\n\
             dominant graph's node chain, the repeating layer block found by\n\
             periodicity, and per-node bottleneck verdicts from trace-only\n\
             heuristics: grid blocks vs SMs (latency bound), time flat vs a\n\
             second captured batch size (memory bound), and the GPU-busy\n\
             fraction of the window (host bound).\n\
             \n\
             The capture must have been recorded with\n\
             --cuda-graph-trace=node; the default graph trace hides the\n\
             nodes inside every graph.",
        ),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, _args: goslice<string>| -> error {
                let (input, _) = cmd.Flags().GetString("input");
                let (output, _) = cmd.Flags().GetString("output");
                let (gpuName, _) = cmd.Flags().GetString("gpu");
                let (note, _) = cmd.Flags().GetString("note");
                generateGraph(input, output, gpuName, note)
            },
        )),
        ..Default::default()
    };
    c.Flags().StringP(
        string("input"),
        string("i"),
        string(""),
        string("node-mode capture: .sqlite export, or .nsys-rep if nsys is on PATH (required)"),
    );
    c.Flags().StringP(
        string("output"),
        string("o"),
        string("graph-structure.json"),
        string("output JSON path"),
    );
    let _ = c.Flags().String_flag(
        string("gpu"),
        string(""),
        string("GPU name from the catalog (H100, B200, ...) to enable the blocks-vs-SMs latency heuristic"),
    );
    let _ = c.Flags().String_flag(
        string("note"),
        string(""),
        string("provenance note stored in the JSON (default: derived from the input path)"),
    );
    c
}

// generateGraph is the body of `profile graph`, callable from the
// full-collection run as well.
pub(crate) fn generateGraph(input: string, output: string, gpuName: string, note: string) -> error {
    {
        {
                if input == "" {
                    return fmt::Errorf!("--input is required (.sqlite export or .nsys-rep)");
                }
                if !toolAvailable("sqlite3") {
                    return fmt::Errorf!("sqlite3 not found on PATH; install sqlite3 to read the capture");
                }

                let mut db = input.clone();
                if strings::HasSuffix(input.clone(), ".nsys-rep") {
                    if nsysLocalPath() == "" {
                        return fmt::Errorf!(
                            "%s is a .nsys-rep and no nsys was found (PATH, KVLM_NSYS, /opt/nvidia/nsight-systems, ~/.local/opt/nsight-systems); export it first: nsys export --type sqlite %s",
                            input, input
                        );
                    }
                    db = strings::TrimSuffix(input.clone(), ".nsys-rep") + (".sqlite");
                    let mut args: goslice<string> = make!([]string, 0);
                    for a in ["export", "--type", "sqlite", "--force-overwrite=true", "--output"].iter() {
                        args = append!(args.clone(), string(*a));
                    }
                    args = append!(args.clone(), db.clone());
                    args = append!(args.clone(), input.clone());
                    fmt::Printf!("exporting %s to sqlite (one-time, can take a minute)\n", input);
                    let (_, errOut, err) = runCaptureRetry(string("nsys"), args);
                    if err != nil {
                        return fmt::Errorf!("nsys export: %v: %s", err, strings::TrimSpace(errOut));
                    }
                }

                let mut sms: int = 0;
                let mut gpuSpec: model::GPUSpec = Default::default();
                if gpuName != "" {
                    let (spec, ok) = model::LookupGPU(gpuName.clone());
                    if !ok {
                        return fmt::Errorf!("unknown GPU %q; known: H100, H200, B200, A100, L40S, L4, T4, A10G, MI300X, RTX 4090, RTX Pro 6000", gpuName);
                    }
                    gpuSpec = spec.clone();
                    sms = spec.SMs;
                }

                // every executable graph in the window
                let (rows, err) = queryRows(
                    db.clone(),
                    string("SELECT graphId, COUNT(DISTINCT graphNodeId), COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId IS NOT NULL AND graphId > 0 GROUP BY graphId"),
                );
                if err != nil {
                    return err;
                }
                let mut graphs: goslice<profile::graph::GraphInfo> = make!([]profile::graph::GraphInfo, 0);
                for (_, row) in range!(rows) {
                    let parts = strings::SplitN(row.clone(), "\t", 4);
                    if parts.len() < 4 {
                        continue;
                    }
                    let (id, _) = strconv::Atoi(parts[0usize].clone());
                    let (nodeCount, _) = strconv::Atoi(parts[1usize].clone());
                    let (execs, _) = strconv::Atoi(parts[2usize].clone());
                    let (totalNs, _) = strconv::Atoi(parts[3usize].clone());
                    if nodeCount == 0 {
                        continue;
                    }
                    graphs = append!(
                        graphs.clone(),
                        profile::graph::GraphInfo {
                            ID: id,
                            Nodes: nodeCount,
                            Replays: execs / nodeCount,
                            TotalMs: (totalNs as float64) / 1e6,
                            ..Default::default()
                        }
                    );
                }
                if graphs.len() == 0 {
                    return fmt::Errorf!(
                        "no CUDA graph nodes in %s; was the capture recorded with --cuda-graph-trace=node?",
                        db
                    );
                }
                // sort by total GPU time, heaviest first
                let mut i: int = 1;
                while i < graphs.len() as int {
                    let mut j = i;
                    while j > 0 && graphs[(j - 1) as usize].TotalMs < graphs[j as usize].TotalMs {
                        let tmp = graphs[(j - 1) as usize].clone();
                        graphs[(j - 1) as usize] = graphs[j as usize].clone();
                        graphs[j as usize] = tmp;
                        j -= 1;
                    }
                    i += 1;
                }
                let primary = graphs[0usize].clone();
                // comparison graph: same node roster at another batch size
                let mut cmpID: int = 0;
                for (_, g) in range!(graphs.clone()) {
                    if g.ID != primary.ID && g.Nodes == primary.Nodes {
                        cmpID = g.ID;
                        break;
                    }
                }

                let (mut nodes, err) = graphNodes(db.clone(), primary.ID);
                if err != nil {
                    return err;
                }
                let mut cmpNodes: goslice<profile::graph::Node> = make!([]profile::graph::Node, 0);
                if cmpID != 0 {
                    let (cn, err) = graphNodes(db.clone(), cmpID);
                    if err == nil && cn.len() == nodes.len() {
                        cmpNodes = cn;
                    } else {
                        cmpID = 0;
                    }
                }
                let mut replayUs: float64 = 0.0;
                for (_, n) in range!(nodes.clone()) {
                    replayUs += n.AvgUs;
                }

                // GPU busy fraction of the whole capture window. Kernel
                // durations sum across every device in the trace, so on
                // a multi-GPU capture (tp>1) the fraction is averaged
                // per device; without that a 2-GPU window reads >100%.
                let (busyRows, err) = queryRows(
                    db.clone(),
                    string("SELECT MAX(end)-MIN(start), SUM(end-start), COUNT(DISTINCT deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL"),
                );
                if err != nil {
                    return err;
                }
                let mut wallS: float64 = 0.0;
                let mut busyS: float64 = 0.0;
                let mut devices: int = 1;
                if busyRows.len() > 0 {
                    let parts = strings::SplitN(busyRows[0usize].clone(), "\t", 3);
                    if parts.len() == 3 {
                        let (w, _) = strconv::Atoi(parts[0usize].clone());
                        let (bz, _) = strconv::Atoi(parts[1usize].clone());
                        let (dv, _) = strconv::Atoi(parts[2usize].clone());
                        wallS = (w as float64) / 1e9;
                        busyS = (bz as float64) / 1e9;
                        if dv > 1 {
                            devices = dv;
                        }
                    }
                }
                let mut busyPct: float64 = 0.0;
                if wallS > 0.0 {
                    busyPct = 100.0 * busyS / (devices as float64) / wallS;
                }

                // GPU-busy time series per device, ~240 bins across the
                // window: the dashboard's timeline panel. Bins with no
                // kernel time are filled as zeros; a kernel is counted
                // in the bin its start falls into, so a bin can nudge
                // past 100 and is clamped.
                let bins: int = 240;
                let mut tlJSON = string("");
                if wallS > 0.0 {
                    let binNs: int = (((wallS * 1e9) as int) / bins) + 1;
                    let sql = fmt::Sprintf!(
                        "SELECT k.deviceId, (k.start - (SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL))/%d, SUM(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k GROUP BY 1, 2 ORDER BY 1, 2",
                        binNs
                    );
                    let (tlRows, err) = queryRows(db.clone(), sql);
                    if err == nil && tlRows.len() > 0 {
                        let mut sb = strings::Builder::new();
                        let _ = sb.WriteString(fmt::Sprintf!(
                            " \"timeline\": {\"binMs\": %s, \"series\": [",
                            jnum2((binNs as float64) / 1e6)
                        ));
                        let mut curDev: int = -1;
                        let mut nextBin: int = 0;
                        for (_, row) in range!(tlRows.clone()) {
                            let parts = strings::SplitN(row.clone(), "\t", 3);
                            if parts.len() < 3 {
                                continue;
                            }
                            let (dev, _) = strconv::Atoi(parts[0usize].clone());
                            let (bin, _) = strconv::Atoi(parts[1usize].clone());
                            let (busyNs, _) = strconv::Atoi(parts[2usize].clone());
                            if dev != curDev {
                                if curDev >= 0 {
                                    let _ = sb.WriteString("]}, ");
                                }
                                let _ = sb.WriteString(fmt::Sprintf!("{\"device\": %d, \"busy\": [", dev));
                                curDev = dev;
                                nextBin = 0;
                            }
                            if bin >= bins {
                                continue;
                            }
                            while nextBin < bin {
                                if nextBin > 0 {
                                    let _ = sb.WriteString(",");
                                }
                                let _ = sb.WriteString("0");
                                nextBin += 1;
                            }
                            let mut pct = (100 * busyNs) / binNs;
                            if pct > 100 {
                                pct = 100;
                            }
                            if nextBin > 0 {
                                let _ = sb.WriteString(",");
                            }
                            let _ = sb.WriteString(fmt::Sprintf!("%d", pct));
                            nextBin += 1;
                        }
                        if curDev >= 0 {
                            let _ = sb.WriteString("]}");
                        }
                        let _ = sb.WriteString("]},\n");
                        tlJSON = string(sb.String());
                    }
                }

                // detect the repeating layer block by name periodicity,
                // on simplified names: template arguments vary slightly
                // between the first layer and the rest (fused-residual
                // bools), which would shift the block boundary
                let mut names: goslice<string> = make!([]string, 0);
                for (_, n) in range!(nodes.clone()) {
                    names = append!(names.clone(), n.Name.clone());
                }
                let (unit, hasUnit) = profile::graph::DetectUnit(names);

                // flag state: the run's explicit args (vllm-args.txt)
                // resolved against the per-version flag catalog. The
                // resolved-config line is authoritative when present;
                // the catalog fills defaults for everything unset.
                let mut cfgVersion = string("");
                let mut cfgExplicit: goslice<profile::vllmcfg::KV> = make!([]profile::vllmcfg::KV, 0);
                let mut cfgResolved: goslice<profile::vllmcfg::KV> = make!([]profile::vllmcfg::KV, 0);
                let (argsData, argsErr) = os::ReadFile((path::Dir(input.clone())) + ("/vllm-args.txt"));
                if argsErr == nil {
                    for (_, line) in range!(strings::Split(string(argsData), "\n")) {
                        let nd = profile::vllmcfg::ParseNonDefault(line.clone());
                        if nd.len() > 0 {
                            cfgExplicit = nd;
                        }
                        let rs = profile::vllmcfg::ParseResolved(line.clone());
                        if rs.len() > 0 {
                            cfgResolved = rs;
                            let v = profile::vllmcfg::EngineVersion(line.clone());
                            if v != "" {
                                cfgVersion = v;
                            }
                        }
                    }
                }
                let (catalog, _) = profile::vllmflags::Load(cfgVersion.clone());
                // effective value per relevant key: resolved, then
                // explicit, then the catalog default
                let mut effective: goslice<profile::vllmcfg::KV> = make!([]profile::vllmcfg::KV, 0);
                for (_, key) in range!(profile::vllmcfg::RelevantKeys()) {
                    let mut val = string("");
                    for src in [&cfgResolved, &cfgExplicit, &catalog].iter() {
                        for (_, kv) in range!((*src).clone()) {
                            if kv.Key == key && val == "" {
                                val = kv.Value.clone();
                            }
                        }
                        if val != "" {
                            break;
                        }
                    }
                    if val != "" {
                        effective = append!(
                            effective.clone(),
                            profile::vllmcfg::KV { Key: key.clone(), Value: val, ..Default::default() }
                        );
                    }
                }

                let mut b = strings::Builder::new();
                let prov = if note != "" {
                    note.clone()
                } else {
                    fmt::Sprintf!("extracted from %s, graphId %d", input, primary.ID)
                };
                let _ = b.WriteString(fmt::Sprintf!("{\n \"provenance\": \"%s\",\n", jsonEsc(prov)));
                // environment fingerprint captured at collection time
                // (env.txt written next to the capture by profile run)
                let _ = b.WriteString(" \"env\": [");
                let (envData, envErr) = os::ReadFile((path::Dir(input.clone())) + ("/env.txt"));
                if envErr == nil {
                    let mut wrote: int = 0;
                    for (_, line) in range!(strings::Split(string(envData), "\n")) {
                        let l = strings::TrimSpace(line.clone());
                        if l == "" {
                            continue;
                        }
                        if wrote > 0 {
                            let _ = b.WriteString(", ");
                        }
                        let _ = b.WriteString(fmt::Sprintf!("\"%s\"", jsonEsc(l)));
                        wrote += 1;
                    }
                }
                let _ = b.WriteString("],\n");
                if gpuName != "" {
                    let _ = b.WriteString(fmt::Sprintf!(
                        " \"gpu\": {\"name\": \"%s\", \"sms\": %d, \"bwGBs\": %d},\n",
                        jsonEsc(gpuSpec.Name.clone()),
                        gpuSpec.SMs,
                        gpuSpec.BWGBs
                    ));
                } else {
                    let _ = b.WriteString(" \"gpu\": null,\n");
                }
                if cfgVersion != "" || cfgExplicit.len() > 0 {
                    let _ = b.WriteString(fmt::Sprintf!(" \"config\": {\"version\": \"%s\", \"explicit\": {", jsonEsc(cfgVersion.clone())));
                    for (i, kv) in range!(cfgExplicit.clone()) {
                        if i > 0 {
                            let _ = b.WriteString(", ");
                        }
                        let _ = b.WriteString(fmt::Sprintf!("\"%s\": \"%s\"", jsonEsc(kv.Key.clone()), jsonEsc(kv.Value.clone())));
                    }
                    let _ = b.WriteString("}, \"effective\": {");
                    for (i, kv) in range!(effective.clone()) {
                        if i > 0 {
                            let _ = b.WriteString(", ");
                        }
                        let _ = b.WriteString(fmt::Sprintf!("\"%s\": \"%s\"", jsonEsc(kv.Key.clone()), jsonEsc(kv.Value.clone())));
                    }
                    let _ = b.WriteString("}},\n");
                }
                let _ = b.WriteString(fmt::Sprintf!(
                    " \"window\": {\"wallS\": %s, \"gpuBusyS\": %s, \"busyPct\": %s, \"devices\": %d, \"verdict\": \"%s\"},\n",
                    jnum(wallS),
                    jnum(busyS),
                    jnum(busyPct),
                    devices,
                    jsonEsc(profile::graph::BusyVerdict(busyPct))
                ));
                if tlJSON != "" {
                    let _ = b.WriteString(tlJSON);
                }
                let _ = b.WriteString(" \"graphs\": [");
                for (i, g) in range!(graphs.clone()) {
                    if i >= 8 {
                        break;
                    }
                    if i > 0 {
                        let _ = b.WriteString(", ");
                    }
                    let _ = b.WriteString(fmt::Sprintf!(
                        "{\"id\": %d, \"nodes\": %d, \"replays\": %d, \"totalMs\": %s}",
                        g.ID, g.Nodes, g.Replays, jnum(g.TotalMs)
                    ));
                }
                let _ = b.WriteString("],\n");

                let recommendText;
                if hasUnit {
                    // fold the repeats into one averaged unit
                    let mut unitNodes: goslice<profile::graph::Node> = make!([]profile::graph::Node, 0);
                    let mut layerUs: float64 = 0.0;
                    let mut j: int = 0;
                    while j < unit.Period {
                        let mut sumPri: float64 = 0.0;
                        let mut sumCmp: float64 = 0.0;
                        let mut maxPri: float64 = 0.0;
                        // modal grid across the repeats
                        let mut grids: goslice<string> = make!([]string, 0);
                        let mut counts: goslice<int> = make!([]int, 0);
                        let mut blocksOf: goslice<int> = make!([]int, 0);
                        let mut r: int = 0;
                        while r < unit.Repeats {
                            let idx = (unit.Start + r * unit.Period + j) as usize;
                            sumPri += nodes[idx].AvgUs;
                            if nodes[idx].MaxUs > maxPri {
                                maxPri = nodes[idx].MaxUs;
                            }
                            if cmpID != 0 {
                                sumCmp += cmpNodes[idx].AvgUs;
                            }
                            let g = nodes[idx].Grid.clone();
                            let mut found = false;
                            for (gi, existing) in range!(grids.clone()) {
                                if existing == g {
                                    counts[gi as usize] += 1;
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                grids = append!(grids.clone(), g);
                                counts = append!(counts.clone(), 1);
                                blocksOf = append!(blocksOf.clone(), nodes[idx].Blocks);
                            }
                            r += 1;
                        }
                        let mut best: int = 0;
                        for (gi, c) in range!(counts.clone()) {
                            if c.clone() > counts[best as usize] {
                                best = gi;
                            }
                        }
                        let first = nodes[(unit.Start + j) as usize].clone();
                        let avg = sumPri / (unit.Repeats as float64);
                        let mut ratio: float64 = 0.0;
                        if cmpID != 0 && sumPri > 0.0 {
                            ratio = sumCmp / sumPri;
                        }
                        unitNodes = append!(
                            unitNodes.clone(),
                            profile::graph::Node {
                                Pos: j + 1,
                                Name: first.Name.clone(),
                                FullName: first.FullName.clone(),
                                Category: first.Category.clone(),
                                Grid: grids[best as usize].clone(),
                                Blocks: blocksOf[best as usize],
                                AvgUs: avg,
                                MaxUs: maxPri,
                                Ratio: ratio,
                                ..Default::default()
                            }
                        );
                        layerUs += avg;
                        j += 1;
                    }
                    let mut j: int = 0;
                    while j < unitNodes.len() as int {
                        if layerUs > 0.0 {
                            unitNodes[j as usize].PctOfLayer = 100.0 * unitNodes[j as usize].AvgUs / layerUs;
                        }
                        unitNodes[j as usize].Verdict = profile::graph::NodeVerdict(
                            unitNodes[j as usize].Blocks,
                            sms,
                            unitNodes[j as usize].AvgUs,
                            unitNodes[j as usize].Ratio,
                        );
                        j += 1;
                    }
                    // prologue and epilogue get verdicts and ratios too
                    let mut edges: goslice<profile::graph::Node> = make!([]profile::graph::Node, 0);
                    let epiStart = unit.Start + unit.Repeats * unit.Period;
                    let mut k: int = 0;
                    while k < nodes.len() as int {
                        if k >= unit.Start && k < epiStart {
                            k += 1;
                            continue;
                        }
                        let mut n = nodes[k as usize].clone();
                        if cmpID != 0 && n.AvgUs > 0.0 {
                            n.Ratio = cmpNodes[k as usize].AvgUs / n.AvgUs;
                        }
                        n.Verdict = profile::graph::NodeVerdict(n.Blocks, sms, n.AvgUs, n.Ratio);
                        edges = append!(edges.clone(), n);
                        k += 1;
                    }

                    let _ = b.WriteString(fmt::Sprintf!(
                        " \"chain\": {\"graphId\": %d, \"nodes\": %d, \"replays\": %d, \"replayUs\": %s, \"layerUs\": %s, \"unitRepeats\": %d, \"comparisonGraphId\": %d,\n  \"prologue\": [",
                        primary.ID, primary.Nodes, primary.Replays, jnum(replayUs), jnum(layerUs), unit.Repeats, cmpID
                    ));
                    let mut wrote: int = 0;
                    for (_, n) in range!(edges.clone()) {
                        if n.Pos - 1 >= unit.Start {
                            continue;
                        }
                        if wrote > 0 {
                            let _ = b.WriteString(", ");
                        }
                        writeGraphNode(&mut b, &n);
                        wrote += 1;
                    }
                    let _ = b.WriteString("],\n  \"unit\": [");
                    for (i, n) in range!(unitNodes.clone()) {
                        if i > 0 {
                            let _ = b.WriteString(", ");
                        }
                        writeGraphNode(&mut b, &n);
                    }
                    let _ = b.WriteString("],\n  \"epilogue\": [");
                    let mut wrote: int = 0;
                    for (_, n) in range!(edges.clone()) {
                        if n.Pos - 1 < epiStart {
                            continue;
                        }
                        if wrote > 0 {
                            let _ = b.WriteString(", ");
                        }
                        writeGraphNode(&mut b, &n);
                        wrote += 1;
                    }
                    let _ = b.WriteString("]},\n \"correlation\": [");
                    recommendText = writeCorrelation(&mut b, profile::graph::Correlate(unitNodes.clone()), &effective);
                    let _ = b.WriteString("],\n \"topNodes\": []\n}\n");
                    fmt::Printf!(
                        "graph %d: %d nodes = %d prologue + %d x %d-node block + %d epilogue; one replay = %s ms GPU\n",
                        primary.ID,
                        primary.Nodes,
                        unit.Start,
                        unit.Repeats,
                        unit.Period,
                        primary.Nodes - unit.Start - unit.Repeats * unit.Period,
                        jnum(replayUs / 1000.0)
                    );
                } else {
                    // no repeating block: report the heaviest nodes
                    let mut i: int = 1;
                    while i < nodes.len() as int {
                        let mut j = i;
                        while j > 0 && nodes[(j - 1) as usize].AvgUs < nodes[j as usize].AvgUs {
                            let tmp = nodes[(j - 1) as usize].clone();
                            nodes[(j - 1) as usize] = nodes[j as usize].clone();
                            nodes[j as usize] = tmp;
                            j -= 1;
                        }
                        i += 1;
                    }
                    let mut topNodes: goslice<profile::graph::Node> = make!([]profile::graph::Node, 0);
                    let mut k: int = 0;
                    while k < nodes.len() as int && k < 40 {
                        let mut n = nodes[k as usize].clone();
                        if replayUs > 0.0 {
                            n.PctOfLayer = 100.0 * n.AvgUs / replayUs;
                        }
                        n.Verdict = profile::graph::NodeVerdict(n.Blocks, sms, n.AvgUs, n.Ratio);
                        topNodes = append!(topNodes.clone(), n);
                        k += 1;
                    }
                    let _ = b.WriteString(" \"chain\": null,\n \"correlation\": [");
                    recommendText = writeCorrelation(&mut b, profile::graph::Correlate(topNodes.clone()), &effective);
                    let _ = b.WriteString("],\n \"topNodes\": [");
                    for (i, n) in range!(topNodes.clone()) {
                        if i > 0 {
                            let _ = b.WriteString(", ");
                        }
                        writeGraphNode(&mut b, &n);
                    }
                    let _ = b.WriteString("]\n}\n");
                    fmt::Printf!(
                        "graph %d: %d nodes, no repeating block detected; wrote top nodes by time\n",
                        primary.ID, primary.Nodes
                    );
                }

                let err = os::WriteFile(output.clone(), string(b.String()), 0o644);
                if err != nil {
                    return fmt::Errorf!("write %s: %v", output, err);
                }
                fmt::Printf!("%s\n", profile::graph::BusyVerdict(busyPct));
                if cmpID == 0 {
                    fmt::Printf!("only one batch size in this capture, so no batch-ratio verdicts; re-capture with: kvlm profile run nsys --probe\n");
                }
                if recommendText != "" {
                    fmt::Printf!("levers, by share of the layer:\n%s", recommendText);
                }
                fmt::Printf!("wrote %s\n", output);

                // folded stacks + flamegraph SVG next to the JSON: the
                // dashboard's GPU-time explorer is revision-scoped and
                // reads these, so every analyzed capture gets them
                let outDirF = path::Dir(output.clone());
                let err = writeFolded(db.clone(), outDirF.clone());
                if err != nil {
                    fmt::Printf!("folded stacks: %v (explorer will skip this run)\n", err);
                }
                nil.into()
        }
    }
}

// archiveCmd represents `kvlm profile archive <run-dir>`: pack one
// collected run into a portable .kvlm revision archive.
pub(crate) fn archiveCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("archive <run-dir>"),
        Short: string("Pack a collected run into a portable .kvlm archive"),
        Long: string(
            "Pack a run directory (profile-output/<run>) into a .kvlm file: a\n\
             gzipped tar with a manifest carrying the run name, provenance,\n\
             and environment fingerprint. kvlm profile import renders it back\n\
             into a revision anywhere.\n\
             \n\
             Everything the run collected is included, raw captures too\n\
             (.nsys-rep, .sqlite, torch traces). --basic packs only the\n\
             renderable data (a few KB) when the archive must travel light.",
        ),
        Args: Some(cobra::ExactArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: goslice<string>| -> error {
                let mut runDir = strings::TrimSuffix(args[0usize].clone(), "/");
                let (_, derr) = os::ReadDir(runDir.clone());
                if derr != nil {
                    // not a directory: resolve as a run reference
                    // (run7, 7, latest, -1)
                    let (resolved, rerr) = crate::cmd::run::ResolveRun(string("./profile-output"), runDir.clone());
                    if rerr != nil {
                        return rerr;
                    }
                    runDir = resolved;
                }
                let run = path::Base(runDir.clone());
                let (basic, _) = cmd.Flags().GetBool("basic");
                let (outFlag, _) = cmd.Flags().GetString("output");

                let (dirEntries, err) = os::ReadDir(runDir.clone());
                if err != nil {
                    return fmt::Errorf!("read %s: %v", runDir, err);
                }
                let mut entries: goslice<profile::archive::Entry> = make!([]profile::archive::Entry, 0);
                let mut skipped: int = 0;
                let mut prov = string("");
                let mut env: goslice<string> = make!([]string, 0);
                for e in dirEntries.iter() {
                    if e.IsDir() {
                        continue;
                    }
                    let name = e.Name();
                    if strings::HasSuffix(name.clone(), ".kvlm") {
                        continue;
                    }
                    let role = profile::archive::RoleOf(name.clone());
                    if basic && profile::archive::IsHeavy(role.clone()) {
                        skipped += 1;
                        continue;
                    }
                    let (data, err) = os::ReadFile((runDir.clone()) + ("/") + (name.clone()));
                    if err != nil {
                        return fmt::Errorf!("read %s/%s: %v", runDir, name, err);
                    }
                    if role == "graph" {
                        let doc = string(data.clone());
                        prov = profile::archive::JSONStringField(doc.clone(), string("provenance"));
                        env = profile::archive::JSONStringArray(doc, string("env"));
                    }
                    entries = append!(
                        entries.clone(),
                        profile::archive::Entry {
                            Name: name,
                            Data: data,
                            ..Default::default()
                        }
                    );
                }
                if entries.Len() == 0 {
                    return fmt::Errorf!("nothing to archive in %s", runDir);
                }
                // ship the version's flag catalog inside the archive
                // so the flag state reconstructs on any machine
                let ver = runVllmVersion(runDir.clone());
                if ver != "" {
                    let (cat, err) = os::ReadFile(profile::vllmflags::Path(ver.clone()));
                    if err == nil {
                        entries = append!(
                            entries.clone(),
                            profile::archive::Entry {
                                Name: profile::archive::CatalogEntryName(ver),
                                Data: cat,
                                ..Default::default()
                            }
                        );
                    }
                }
                let (data, err) = profile::archive::Pack(run.clone(), prov, env, entries.clone());
                if err != nil {
                    return err;
                }
                let mut out = outFlag.clone();
                if out == "" {
                    out = (runDir.clone()) + (".kvlm");
                }
                let err = os::WriteFile(out.clone(), data.clone(), 0o644);
                if err != nil {
                    return fmt::Errorf!("write %s: %v", out, err);
                }
                fmt::Printf!("wrote %s: run %s, %d files, %d bytes\n", out, run, entries.Len(), data.Len());
                if skipped > 0 {
                    fmt::Printf!("left out %d raw capture file(s); archive without --basic to include them\n", skipped);
                }
                nil.into()
            },
        )),
        ..Default::default()
    };
    c.Flags().StringP(string("output"), string("o"), string(""), string("output path (default <run-dir>.kvlm)"));
    let _ = c.Flags().Bool_flag(string("basic"), false, string("renderable data only, leave out raw captures (.nsys-rep, .sqlite, torch traces)"));
    c
}

// importCmd represents `kvlm profile import <file.kvlm>`: extract an
// archived run so the dashboard renders it as a revision.
pub(crate) fn importCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("import <file.kvlm>"),
        Short: string("Import a .kvlm archive as a revision"),
        Long: string(
            "Extract a .kvlm archive into profile-output/<run>/. The dashboard\n\
             lists the imported run in its revision picker like any locally\n\
             collected run, comparable against the others when it carries a\n\
             graph capture.",
        ),
        Args: Some(cobra::ExactArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: goslice<string>| -> error {
                let file = args[0usize].clone();
                let (outDir, _) = cmd.Flags().GetString("out");
                let (force, _) = cmd.Flags().GetBool("force");
                let (data, err) = os::ReadFile(file.clone());
                if err != nil {
                    return fmt::Errorf!("read %s: %v", file, err);
                }
                let (mut run, _, entries, err) = profile::archive::Unpack(data);
                if err != nil {
                    return err;
                }
                if run == "" {
                    run = strings::TrimSuffix(path::Base(file.clone()), ".kvlm");
                }
                let dest = (outDir.clone()) + ("/") + (run.clone());
                let (_, statErr) = os::ReadDir(dest.clone());
                if statErr == nil && !force {
                    return fmt::Errorf!("%s already exists; use --force to overwrite its files", dest);
                }
                let err = os::MkdirAll(dest.clone(), 0o755);
                if err != nil {
                    return err;
                }
                for (_, e) in range!(entries.clone()) {
                    let (ver, isCat) = profile::archive::CatalogVersion(e.Name.clone());
                    if isCat {
                        let (_, err) = os::Stat(profile::vllmflags::Path(ver.clone()));
                        if err != nil {
                            let _ = os::MkdirAll(string(profile::vllmflags::Dir), 0o755);
                            let _ = os::WriteFile(profile::vllmflags::Path(ver.clone()), e.Data.clone(), 0o644);
                            fmt::Printf!("restored the vLLM %s flag catalog from the archive\n", ver);
                        }
                        continue;
                    }
                    let err = os::WriteFile((dest.clone()) + ("/") + (e.Name.clone()), e.Data.clone(), 0o644);
                    if err != nil {
                        return fmt::Errorf!("write %s/%s: %v", dest, e.Name.clone(), err);
                    }
                }
                fmt::Printf!("imported run %s: %d files into %s; the dashboard lists it as a revision\n", run, entries.Len(), dest);
                nil.into()
            },
        )),
        ..Default::default()
    };
    c.Flags().StringP(string("out"), string("o"), string("./profile-output"), string("directory that holds run revisions"));
    let _ = c.Flags().Bool_flag(string("force"), false, string("overwrite files if the run directory already exists"));
    c
}

// transport runs commands on the target pod: over ssh (the path that
// works on RunPod, whose API has no exec) or through a platform
// driver's Exec/Download (k8s today).
pub(crate) struct transport {
    mode: string, // "ssh" or "driver"
    // ssh
    dest: string,
    port: string,
    identity: string, // -i key file, "" = ssh defaults
    // driver
    drv: Option<alloc::sync::Arc<dyn driver::Driver>>,
    creds: driver::Credentials,
    pod: string,
}

impl transport {
    // sshBase builds the shared ssh/scp options (the port flag is -p
    // for ssh, -P for scp).
    fn sshBase(&self, portFlag: string) -> goslice<string> {
        let mut args: goslice<string> = make!([]string, 0);
        args = append!(args.clone(), string("-o"));
        args = append!(args.clone(), string("BatchMode=yes"));
        // the deploy injects the ~/.runpod pubkey (driver publicKey
        // resolution), so with no explicit -I the matching private
        // key is the default; plain ssh does not offer it on its own
        let mut identity = self.identity.clone();
        if identity == "" {
            let (home, herr) = os::UserHomeDir();
            if herr == nil {
                let candidate = (home) + ("/.runpod/id_ed25519");
                let (_, serr) = os::ReadFile(candidate.clone());
                if serr == nil {
                    identity = candidate;
                }
            }
        }
        // ephemeral pods have fresh host keys every deploy; accept-new
        // records them on first contact and still fails hard if a
        // known host's key ever changes (strict checking would demand
        // a manual keyscan before every fresh pod)
        args = append!(args.clone(), string("-o"));
        args = append!(args.clone(), string("StrictHostKeyChecking=accept-new"));
        if identity != "" {
            args = append!(args.clone(), string("-i"));
            args = append!(args.clone(), identity);
        }
        args = append!(args.clone(), portFlag);
        args = append!(args.clone(), self.port.clone());
        args
    }

    pub(crate) fn exec(&self, cmd: string) -> (string, error) {
        if self.mode == "ssh" {
            // runCaptureRetry resolves the absolute ssh path pre-fork
            // (the "exit 127, empty stderr" failures were LookPath
            // breaking after the first fork, not ssh flaking)
            let mut args = self.sshBase(string("-p"));
            args = append!(args.clone(), self.dest.clone());
            args = append!(args.clone(), cmd.clone());
            let (out, errOut, err) = runCaptureRetry(string("ssh"), args);
            if err == nil {
                return (out, nil.into());
            }
            return (out, fmt::Errorf!("ssh: %v\nstderr: %s", err, errOut));
        }
        self.drv
            .as_ref()
            .unwrap()
            .Exec(&self.creds, self.pod.clone(), cmd)
    }

    // execWindow runs one remote capture-window command. The nsys
    // start line carries --gpu-metrics-devices, and GPU counters are
    // admin-restricted on some hosts (same driver policy that blocks
    // ncu); when that start fails it retries once without the
    // sampling flag so the capture itself never dies for it.
    fn execWindow(&self, line: string) -> (string, error) {
        let (out, err) = self.exec(line.clone());
        if err != nil && strings::Contains(line.clone(), "--gpu-metrics-devices") {
            let stripped = strings::ReplaceAll(line, " --gpu-metrics-devices=cuda-visible", "");
            fmt::Printf!("GPU metrics sampling refused by the target; retrying without it\n");
            return self.exec(stripped);
        }
        (out, err)
    }

    fn fetch(&self, remote: string, localDir: string) -> error {
        if self.mode == "ssh" {
            let mut args = self.sshBase(string("-P"));
            args = append!(args.clone(), (self.dest.clone()) + (":") + (remote));
            args = append!(args.clone(), localDir);
            let (_, errOut, err) = runCaptureRetry(string("scp"), args);
            if err != nil {
                return fmt::Errorf!("scp: %v\nstderr: %s", err, errOut);
            }
            return nil.into();
        }
        let local = (localDir) + ("/") + (path::Base(remote.clone()));
        self.drv
            .as_ref()
            .unwrap()
            .Download(&self.creds, self.pod.clone(), remote, local)
    }
}

// runCapture runs a local command, draining stdout here and stderr in
// a goroutine (sequential drains deadlock past one pipe buffer).
pub(crate) fn runCapture(name: string, args: goslice<string>) -> (string, string, error) {
    let mut command = exec::Command(name, args);
    let (mut stdoutReader, err) = command.StdoutPipe();
    if err != nil {
        return (string(""), string(""), err);
    }
    let (mut stderrReader, err) = command.StderrPipe();
    if err != nil {
        return (string(""), string(""), err);
    }
    let err = command.Start();
    if err != nil {
        return (string(""), string(""), err);
    }
    let stderrCh = make!(chan string, 1);
    {
        let ch = stderrCh.clone();
        go!(move || {
            let mut b = bytes::Buffer::new();
            let mut buf = make!([]types::byte, 4096);
            loop {
                let (n, err) = stderrReader.Read(&mut buf);
                if n > 0 {
                    b.Write(buf.slice(0, n));
                }
                if err != nil {
                    break;
                }
            }
            ch.Send(b.String());
        });
    }
    let mut outBuf = bytes::Buffer::new();
    let mut buf = make!([]types::byte, 4096);
    loop {
        let (n, err) = stdoutReader.Read(&mut buf);
        if n > 0 {
            outBuf.Write(buf.slice(0, n));
        }
        if err != nil {
            break;
        }
    }
    let (errStr, _) = stderrCh.Recv();
    let err = command.Wait();
    (outBuf.String(), errStr, err)
}

// runCmd represents `kvlm profile run <tool>`: execute a tool's
// capture window on a remote pod and fetch the artifacts.
fn runCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("run [tool]"),
        Short: string("Collect a profiling run from a remote pod"),
        Long: string(
            "With no tool named, collect everything the target is set up for\n\
             target with every default resolved: metrics before and after, an\n\
             nsys window (clean, first), a shorter torch profiler window,\n\
             GPU telemetry, the environment fingerprint, and the server log,\n\
             into an auto-numbered profile-output/runN, analyzed on the spot\n\
             when nsys is installed locally. The probe supplies load\n\
             automatically when the server is idle. Anything the server was\n\
             not launched for is skipped with its setup recipe printed.\n\
             \n\
             With a tool named, execute just that tool's capture window; the\n\
             commands executed are the same ones kvlm profile show prints.\n\
             \n\
             The server must already be running under the profiler (see the\n\
             launch phase in kvlm profile show); kvlm does not restart your\n\
             server. With no transport flag, the pod recorded by kvlm up\n\
             is the target. Otherwise: --ssh user@host[:port] for direct\n\
             access (RunPod), or --k8s-pod <name> with --driver k8s.\n\
             \n\
             --probe drives two concurrency plateaus on the target during\n\
             the window (low for the first half, high for the second), so\n\
             the trace always contains the decode graph at two batch sizes\n\
             and profile graph can compute its batch-ratio verdicts. Without\n\
             it, a steady load pins the batch to one capture bucket and the\n\
             ratios depend on luck. The probe needs curl on the target and\n\
             headroom under --max-num-seqs to change the batch.",
        ),
        Args: Some(cobra::MaximumNArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: goslice<string>| -> error {
                if args.Len() == 0 {
                    return runAll(cmd);
                }
                runProfile(cmd, args[0usize].clone())
            },
        )),
        ..Default::default()
    };
    addRunFlags(&mut c);
    c
}

// addRunFlags registers the collection flag set; `profile run` and
// `tune` share it, so a tune is a run plus a goal.
pub(crate) fn addRunFlags(c: &mut cobra::Command) {
    c.Flags().StringP(string("ssh"), string("S"), string(""), string("ssh target user@host[:port] (default: the pod kvlm up recorded)"));
    c.Flags().StringP(string("identity"), string("I"), string(""), string("ssh identity file (defaults to your ssh config)"));
    let _ = c.Flags().String_flag(string("k8s-pod"), string(""), string("Kubernetes pod name, used with --driver k8s"));
    c.Flags().StringP(string("session"), string("s"), string("kvlm"), string("profiler session name on the target"));
    c.Flags().IntP(string("window"), string("w"), 30, string("capture window in seconds"));
    // no shorthand: -r belongs to the root --runtime flag
    let _ = c.Flags().String_flag(string("remote-out"), string("/tmp/kvlm-profile/win"), string("output path prefix on the target"));
    c.Flags().StringP(string("out"), string("o"), string("./profile-output"), string("local directory for fetched artifacts"));
    let _ = c.Flags().String_flag(string("server-log"), string(""), string("path to the server log on the target (e.g. /workspace/vllm.log); fetches a filtered tail as server-log.txt"));
    let _ = c.Flags().Bool_flag(string("probe"), false, string("drive two concurrency plateaus during the window so the trace holds two batch sizes"));
    let _ = c.Flags().String_flag(string("probe-addr"), string("127.0.0.1:8000"), string("OpenAI-style endpoint on the target the probe talks to"));
    let _ = c.Flags().String_flag(string("probe-model"), string(""), string("model name for probe requests (default: first id from /v1/models)"));
    let _ = c.Flags().Int_flag(string("probe-low"), 4, string("probe concurrency for the first half of the window"));
    let _ = c.Flags().Int_flag(string("probe-high"), 24, string("probe concurrency for the second half of the window"));
}

// probeModel asks the target's /v1/models for the first model id, for
// probe requests when --probe-model is not given.
fn probeModel(tr: &transport, addr: string) -> (string, error) {
    let (out, err) = tr.exec(fmt::Sprintf!("curl -s -m 10 http://%s/v1/models", addr));
    if err != nil {
        return (string(""), err);
    }
    let marker = string("\"id\":\"");
    let idx = strings::Index(out.clone(), marker.clone());
    if idx < 0 {
        return (string(""), fmt::Errorf!("no model id in /v1/models response"));
    }
    let rest = out.slice(idx + marker.Len(), out.Len());
    let end = strings::Index(rest.clone(), string("\""));
    if end <= 0 {
        return (string(""), fmt::Errorf!("malformed /v1/models response"));
    }
    (rest.slice(0, end), nil.into())
}

// runProbePhases uploads the probe script and holds the two plateaus;
// the phases themselves consume the capture window.
// uploadProbe puts the probe script on the target (base64 over ssh,
// no quoting hazards).
fn uploadProbe(tr: &transport) -> error {
    let enc = base64::StdEncoding.EncodeToString(profile::probe::ProbeScript().as_bytes());
    let (_, err) = tr.exec(fmt::Sprintf!(
        "echo %s | base64 -d > /tmp/kvlm-probe.sh",
        enc
    ));
    if err != nil {
        return fmt::Errorf!("upload probe script: %v", err);
    }
    nil.into()
}

// runProbeAt drives one plateau: conc streams for secs seconds.
fn runProbeAt(tr: &transport, addr: string, model: string, conc: int, secs: int, maxtok: int) -> error {
    let (_, err) = tr.exec(fmt::Sprintf!(
        "sh /tmp/kvlm-probe.sh %s %s %d %d %d",
        addr.clone(),
        model.clone(),
        conc,
        secs,
        maxtok
    ));
    if err != nil {
        return fmt::Errorf!("probe at %d streams: %v", conc, err);
    }
    nil.into()
}

fn runProbePhases(tr: &transport, addr: string, model: string, low: int, high: int, window: int) -> error {
    let err = uploadProbe(tr);
    if err != nil {
        return err;
    }
    let firstHalf = window / 2;
    let secondHalf = window - firstHalf;
    for (conc, secs, label) in [
        (low, firstHalf, "low"),
        (high, secondHalf, "high"),
    ]
    .iter()
    {
        fmt::Printf!(
            "probe %s plateau: %d streams for %d s against %s (%s)\n",
            string(*label),
            *conc,
            *secs,
            addr.clone(),
            model.clone()
        );
        let err = runProbeAt(tr, addr.clone(), model.clone(), *conc, *secs, 96);
        if err != nil {
            return fmt::Errorf!("probe %s plateau: %v", string(*label), err);
        }
    }
    nil.into()
}

// fetchSnap grabs one /metrics snapshot from the target.
fn fetchSnap(tr: &transport, addr: string) -> (metrics::Snapshot, error) {
    let (txt, err) = tr.exec(fmt::Sprintf!("curl -s -m 10 http://%s/metrics", addr.clone()));
    if err != nil {
        return (Default::default(), err);
    }
    (metrics::Parse(txt), nil.into())
}

// runVllmVersion reads the vLLM version a run was collected against:
// the resolved-config line in vllm-args.txt first, the env.txt
// fingerprint line second.
fn runVllmVersion(runDir: string) -> string {
    let (data, err) = os::ReadFile((runDir.clone()) + ("/vllm-args.txt"));
    if err == nil {
        for (_, line) in range!(strings::Split(string(data), "\n")) {
            let v = profile::vllmcfg::EngineVersion(line.clone());
            if v != "" {
                return v;
            }
        }
    }
    let (envData, err) = os::ReadFile((runDir.clone()) + ("/env.txt"));
    if err == nil {
        for (_, line) in range!(strings::Split(string(envData), "\n")) {
            let l = strings::TrimSpace(line.clone());
            if strings::HasPrefix(l.clone(), string("vllm ")) {
                return strings::TrimPrefix(l, string("vllm "));
            }
        }
    }
    string("")
}

// effectiveMaxSeqs resolves the run's scheduler cap from the captured
// flag state: the resolved-config line first, explicit args second,
// the version catalog last.
fn effectiveMaxSeqs(outDir: string) -> (float64, bool) {
    let (data, err) = os::ReadFile((outDir.clone()) + ("/vllm-args.txt"));
    if err != nil {
        return (0.0, false);
    }
    let mut version = string("");
    let mut val = string("");
    for (_, line) in range!(strings::Split(string(data), "\n")) {
        let rs = profile::vllmcfg::ParseResolved(line.clone());
        let (v, ok) = profile::vllmcfg::Lookup(&rs, "max_num_seqs");
        if ok {
            val = v;
        }
        let ev = profile::vllmcfg::EngineVersion(line.clone());
        if ev != "" {
            version = ev;
        }
        if val == "" {
            let nd = profile::vllmcfg::ParseNonDefault(line.clone());
            let (v, ok) = profile::vllmcfg::Lookup(&nd, "max_num_seqs");
            if ok {
                val = v;
            }
        }
    }
    if val == "" && version != "" {
        let (catalog, ok) = profile::vllmflags::Load(version);
        if ok {
            let (v, ok) = profile::vllmcfg::Lookup(&catalog, "max_num_seqs");
            if ok {
                val = v;
            }
        }
    }
    if val == "" {
        return (0.0, false);
    }
    let (f, err) = strconv::ParseFloat(strings::TrimSpace(val), 64);
    if err != nil {
        return (0.0, false);
    }
    (f, true)
}

// runBench measures the two baseline characterizations a full run
// carries: the concurrency sweep (per-user and aggregate decode rate
// plus TTFT at each level) and a bounded KV pressure probe (peak
// running/waiting/KV usage and preemptions under heavy load). Only
// called for idle servers — the same guard as the auto-probe — so a
// production server is never load-tested by accident. Results land in
// sweep.json; every number is a measured metrics delta.
// padLeft right-aligns s in a field of width w (the fmt port has no
// width verbs).
pub(crate) fn padLeft(s: string, w: int) -> string {
    if s.Len() >= w {
        return s;
    }
    (strings::Repeat(" ", w - s.Len())) + (s)
}

fn runBench(tr: &transport, addr: string, model: string, outDir: string) -> error {
    let err = uploadProbe(tr);
    if err != nil {
        return err;
    }
    let mut b = strings::Builder::new();
    let _ = b.WriteString("{\n \"curve\": [");
    let sweepSecs: int = 18;
    let mut wrote: int = 0;
    // (concurrency, per-stream tok/s, aggregate tok/s, ttft ms) rows
    // for the terminal table once the sweep is done
    let mut curveRows: alloc::vec::Vec<(int, float64, float64, float64)> = alloc::vec::Vec::new();
    for conc in [1, 8, 32].iter() {
        let (before, err) = fetchSnap(tr, addr.clone());
        if err != nil {
            return err;
        }
        let started = time::Now();
        fmt::Printf!("sweep: %d streams for %d s\n", *conc, sweepSecs);
        let err = runProbeAt(tr, addr.clone(), model.clone(), *conc, sweepSecs, 256);
        if err != nil {
            return err;
        }
        let (after, err) = fetchSnap(tr, addr.clone());
        if err != nil {
            return err;
        }
        let elapsed = time::Since(started).Seconds();
        let gen = after.GenerationTokens - before.GenerationTokens;
        let tpotN = after.TPOTCount - before.TPOTCount;
        let ttftN = after.TTFTCount - before.TTFTCount;
        let mut perUser: float64 = 0.0;
        if tpotN > 0.0 && after.TPOTSum > before.TPOTSum {
            perUser = tpotN / (after.TPOTSum - before.TPOTSum);
        }
        let mut ttftMs: float64 = 0.0;
        if ttftN > 0.0 {
            ttftMs = 1000.0 * (after.TTFTSum - before.TTFTSum) / ttftN;
        }
        let mut aggregate: float64 = 0.0;
        if elapsed > 0.0 {
            aggregate = gen / elapsed;
        }
        if wrote > 0 {
            let _ = b.WriteString(", ");
        }
        let _ = b.WriteString(fmt::Sprintf!(
            "{\"concurrency\": %d, \"perUser\": %s, \"aggregate\": %s, \"ttftMs\": %s}",
            *conc,
            jnum2(math::Round(perUser * 10.0) / 10.0),
            jnum(math::Round(aggregate)),
            jnum(math::Round(ttftMs))
        ));
        curveRows.push((*conc, math::Round(perUser * 10.0) / 10.0, math::Round(aggregate), math::Round(ttftMs)));
        wrote += 1;
    }
    fmt::Println!("curve");
    fmt::Println!("  streams   per-stream     total      TTFT");
    for (conc, perUser, aggregate, ttftMs) in curveRows.iter() {
        fmt::Printf!(
            "  %s%s tok/s%s tok/s%s ms\n",
            padLeft(fmt::Sprintf!("%d", *conc), 7),
            padLeft(fmt::Sprintf!("%v", *perUser), 9),
            padLeft(fmt::Sprintf!("%v", *aggregate), 6),
            padLeft(fmt::Sprintf!("%v", *ttftMs), 6)
        );
    }
    let _ = b.WriteString("],\n \"pressure\": ");

    // pressure: launch the probe in the background on the target and
    // sample the scheduler gauges while it runs; peaks are the story
    let streams: int = 48;
    let pressSecs: int = 45;
    let maxtok: int = 2048;
    let (pb, err) = fetchSnap(tr, addr.clone());
    if err != nil {
        return err;
    }
    fmt::Printf!("pressure: %d streams, up to %d tokens each, %d s\n", streams, maxtok, pressSecs);
    let (_, err) = tr.exec(fmt::Sprintf!(
        "nohup sh /tmp/kvlm-probe.sh %s %s %d %d %d >/dev/null 2>&1 & echo started",
        addr.clone(),
        model.clone(),
        streams,
        pressSecs,
        maxtok
    ));
    if err != nil {
        return err;
    }
    let started = time::Now();
    let mut peakRun: float64 = 0.0;
    let mut peakWait: float64 = 0.0;
    let mut peakKv: float64 = 0.0;
    while time::Since(started).Seconds() < ((pressSecs as float64) + 3.0) {
        time::Sleep(time::Seconds(5));
        let (s, err) = fetchSnap(tr, addr.clone());
        if err != nil {
            continue;
        }
        if s.Running > peakRun {
            peakRun = s.Running;
        }
        if s.Waiting > peakWait {
            peakWait = s.Waiting;
        }
        if s.GPUCacheUsage > peakKv {
            peakKv = s.GPUCacheUsage;
        }
    }
    let (pa, err) = fetchSnap(tr, addr.clone());
    if err != nil {
        return err;
    }
    let preempt = pa.Preemptions - pb.Preemptions;
    let ttftN = pa.TTFTCount - pb.TTFTCount;
    let mut ttftMs: float64 = 0.0;
    if ttftN > 0.0 {
        ttftMs = 1000.0 * (pa.TTFTSum - pb.TTFTSum) / ttftN;
    }
    // name the regime against the actual flag value: pinned at the
    // scheduler cap, KV-bound, or neither saturated
    let (maxSeqs, haveMax) = effectiveMaxSeqs(outDir.clone());
    let mut boundBy = string("none");
    if haveMax && peakRun >= maxSeqs - 0.5 {
        boundBy = string("max-num-seqs");
    } else if preempt > 0.0 || peakKv >= 0.9 {
        boundBy = string("kv");
    }
    let _ = b.WriteString(fmt::Sprintf!(
        "{\"streams\": %d, \"seconds\": %d, \"maxTokens\": %d, \"peakRunning\": %s, \"peakWaiting\": %s, \"peakKvPct\": %s, \"preemptions\": %s, \"ttftMeanMs\": %s, \"maxNumSeqs\": %s, \"boundBy\": \"%s\"}\n}\n",
        streams,
        pressSecs,
        maxtok,
        jnum(peakRun),
        jnum(peakWait),
        jnum2(math::Round(1000.0 * peakKv) / 10.0),
        jnum(preempt),
        jnum(math::Round(ttftMs)),
        jnum(maxSeqs),
        boundBy
    ));
    // the regime, named with its evidence: these numbers are the whole
    // argument for the recommendation that follows
    if boundBy == "max-num-seqs" {
        fmt::Printf!(
            "verdict: bound by max-num-seqs: peak running %v equals the flag value %v (waiting %v, KV %v%%); raising --max-num-seqs admits more\n",
            jnum(peakRun),
            jnum(maxSeqs),
            jnum(peakWait),
            jnum2(math::Round(1000.0 * peakKv) / 10.0)
        );
    } else if boundBy == "kv" {
        fmt::Printf!(
            "verdict: bound by kv: KV peaked at %v%% with %v preemptions (running %v); more KV headroom admits more, a smaller KV footprint creates it\n",
            jnum2(math::Round(1000.0 * peakKv) / 10.0),
            jnum(preempt),
            jnum(peakRun)
        );
    } else {
        let mut capNote = string("");
        if haveMax {
            capNote = fmt::Sprintf!(" under the --max-num-seqs cap %v", jnum(maxSeqs));
        }
        fmt::Printf!(
            "verdict: bound by neither cap nor KV: peak running %v%s, KV %v%%; the offered load is the limit at this pressure\n",
            jnum(peakRun),
            capNote,
            jnum2(math::Round(1000.0 * peakKv) / 10.0)
        );
    }
    let err = os::WriteFile((outDir.clone()) + ("/sweep.json"), string(b.String()), 0o644);
    if err != nil {
        return err;
    }
    fmt::Printf!("wrote %s/sweep.json\n", outDir);
    nil.into()
}

// buildTransport wires the ssh or driver transport from run flags.
// sshTransport builds the ssh transport for a user@host[:port]
// destination; other commands (kvlm up's vLLM launch) reuse it.
pub(crate) fn sshTransport(ssh: string, identity: string) -> transport {
    let mut tr = transport {
        mode: string("ssh"),
        dest: string(""),
        port: string("22"),
        identity,
        drv: None,
        creds: Default::default(),
        pod: string(""),
    };
    let parts = strings::SplitN(ssh, ":", 2);
    tr.dest = parts[0usize].clone();
    if parts.Len() > 1 {
        tr.port = parts[1usize].clone();
    }
    tr
}

fn buildTransport(cmd: &mut cobra::Command, ssh: string, pod: string) -> (transport, error) {
    let (identity, _) = cmd.Flags().GetString("identity");
    let mut tr = transport {
        mode: string(""),
        dest: string(""),
        port: string("22"),
        identity: identity.clone(),
        drv: None,
        creds: Default::default(),
        pod: pod.clone(),
    };
    if ssh != "" {
        return (sshTransport(ssh, identity), nil.into());
    } else if pod != "" {
        tr.mode = string("driver");
        let (d, creds, err) = driver::FromCommand(cmd);
        if err != nil {
            return (tr, err);
        }
        tr.drv = d;
        tr.creds = creds;
    } else {
        // no explicit target: the pod kvlm up recorded is the target
        let (t, ok) = state::Current();
        if ok && t.SSH != "" {
            fmt::Printf!("target: %s pod %s (%s), recorded by kvlm up\n", t.Driver.clone(), t.Pod.clone(), t.SSH.clone());
            return (sshTransport(t.SSH.clone(), identity), nil.into());
        }
        if ok && t.SSH == "" {
            return (tr, fmt::Errorf!(
                "the recorded target (pod %s) has no ssh: production pods serve without it. Profile against a pod from kvlm up --mode profile, or pass --ssh user@host[:port]",
                t.Pod.clone()
            ));
        }
        return (tr, fmt::Errorf!("no target: kvlm up records the pod it starts; otherwise pass --ssh user@host[:port], or --k8s-pod <name> with --driver k8s"));
    }
    (tr, nil.into())
}

// collectEnv captures the environment fingerprint on the target
// (versions and hardware, measured not typed) and fetches it.
fn collectEnv(tr: &transport, remoteOut: string, outDir: string) {
    let envRemote = (path::Dir(remoteOut)) + ("/env.txt");
    let (_, err) = tr.exec(fmt::Sprintf!(
        "{ python3 -c 'import vllm; print(\"vllm\", vllm.__version__)' 2>/dev/null; python3 -c 'import torch; print(\"torch\", torch.__version__)' 2>/dev/null; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null; } > %s",
        envRemote.clone()
    ));
    if err == nil {
        fmt::Printf!("fetching %s -> %s/\n", envRemote.clone(), outDir.clone());
        let _ = tr.fetch(envRemote, outDir);
    }
}

// collectServerLog fetches the filtered server log tail: the startup
// config echo, the Maximum concurrency line, kv-cache-memory
// suggestion, graph capture summary, preemption warnings, errors.
fn collectServerLog(tr: &transport, serverLog: string, remoteOut: string, outDir: string) {
    if serverLog == "" {
        return;
    }
    let logRemote = (path::Dir(remoteOut)) + ("/server-log.txt");
    let (_, err) = tr.exec(fmt::Sprintf!(
        "{ grep -aE 'Maximum concurrency|preempt|kv-cache-memory|Graph capturing|startup complete|non-torch|non-default args|Initializing a V1 LLM engine|WARNING|ERROR' %s | tail -n 200; echo '--- last 60 lines ---'; tail -n 60 %s; } > %s",
        serverLog.clone(),
        serverLog.clone(),
        logRemote.clone()
    ));
    if err != nil {
        fmt::Printf!("server log %s not readable on the target: %v\n", serverLog, err);
        return;
    }
    fmt::Printf!("fetching %s -> %s/\n", logRemote.clone(), outDir.clone());
    let _ = tr.fetch(logRemote, outDir);
}

// collectVllmConfig gathers the per-run explicit flag state
// (vllm-args.txt: the serve argv plus vLLM's own non-default and
// resolved-config log lines) and, once per vLLM version, the full
// flag catalog into vllm-flags/<version>.json. A run carries only
// what was set; the version catalog supplies every other flag's
// default at analysis time.
fn collectVllmConfig(tr: &transport, serverLog: string, outDir: string) {
    let mut b = strings::Builder::new();
    let (argv, err) = tr.exec(string("tr '\\0' ' ' < /proc/$(pgrep -f '[v]llm serve' | head -1)/cmdline 2>/dev/null"));
    if err == nil && strings::TrimSpace(argv.clone()) != "" {
        let _ = b.WriteString(fmt::Sprintf!("argv: %s\n", strings::TrimSpace(argv)));
    }
    if serverLog != "" {
        for pat in ["non-default args", "Initializing a V1 LLM engine"].iter() {
            let (line, err) = tr.exec(fmt::Sprintf!("grep -a '%s' %s | tail -1", string(*pat), serverLog.clone()));
            if err == nil && strings::TrimSpace(line.clone()) != "" {
                let _ = b.WriteString(fmt::Sprintf!("%s\n", strings::TrimSpace(line)));
            }
        }
    }
    let doc = string(b.String());
    if doc != "" {
        let _ = os::WriteFile((outDir.clone()) + ("/vllm-args.txt"), doc, 0o644);
        fmt::Printf!("wrote %s/vllm-args.txt\n", outDir);
    }

    // the version catalog, collected once per vLLM version ever seen
    let (ver, err) = tr.exec(string("python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null"));
    if err != nil {
        return;
    }
    let ver = strings::TrimSpace(ver);
    if ver == "" {
        return;
    }
    let (_, have) = profile::vllmflags::Load(ver.clone());
    if have {
        return;
    }
    let (dump, err) = tr.exec(string(profile::vllmflags::DumpCommand));
    if err != nil {
        fmt::Printf!("flag catalog for vLLM %s not collectable on this target: %v\n", ver, err);
        return;
    }
    let err = profile::vllmflags::Save(
        ver.clone(),
        string("introspected dataclasses.fields(AsyncEngineArgs) on the profiled target"),
        strings::TrimSpace(dump),
    );
    if err != nil {
        fmt::Printf!("flag catalog for vLLM %s: %v\n", ver, err);
    } else {
        fmt::Printf!("collected the vLLM %s flag catalog into %s\n", ver.clone(), profile::vllmflags::Path(ver));
    }
}

// runAll is `profile run` with no tool argument: collect everything
// that preflights clean, with every default resolved. Sequential
// windows keep the nsys numbers clean of torch profiler overhead; the
// probe runs automatically when the server is idle; the run lands in
// an auto-numbered profile-output/runN and, when nsys is available
// locally, comes out fully analyzed.
pub(crate) fn runAll(cmd: &mut cobra::Command) -> error {
    let (ssh, _) = cmd.Flags().GetString("ssh");
    let (pod, _) = cmd.Flags().GetString("k8s-pod");
    let (session, _) = cmd.Flags().GetString("session");
    let (window, _) = cmd.Flags().GetInt("window");
    let (remoteOut, _) = cmd.Flags().GetString("remote-out");
    let (outFlag, _) = cmd.Flags().GetString("out");
    let (mut serverLog, _) = cmd.Flags().GetString("server-log");
    if serverLog == "" && ssh == "" && pod == "" {
        // profiling the recorded target: the launch script's log path
        // was recorded by kvlm up
        let (t, ok) = state::Current();
        if ok && t.ServerLog != "" {
            serverLog = t.ServerLog.clone();
        }
    }
    let (probeFlag, _) = cmd.Flags().GetBool("probe");
    let (probeAddr, _) = cmd.Flags().GetString("probe-addr");
    let (probeModelFlag, _) = cmd.Flags().GetString("probe-model");
    let (probeLow, _) = cmd.Flags().GetInt("probe-low");
    let (probeHigh, _) = cmd.Flags().GetInt("probe-high");

    let (tr, err) = buildTransport(cmd, ssh.clone(), pod.clone());
    if err != nil {
        return err;
    }
    let (_, err) = tr.exec(string("true"));
    if err != nil {
        return fmt::Errorf!("cannot reach the target: %v", err);
    }

    // preflight the serving endpoint before any window runs: a dead
    // server otherwise produces an empty capture that still costs the
    // full collection time (a common cause is engine init dying on a
    // host driver older than torch's CUDA build — the log names it)
    let (code, _) = tr.exec(fmt::Sprintf!(
        "curl -s -m 5 -o /dev/null -w '%%{http_code}' http://%s/v1/models",
        probeAddr.clone()
    ));
    if strings::TrimSpace(code.clone()) != "200" {
        return fmt::Errorf!(
            "vLLM is not answering on %s (got %q). If it just launched, wait for the model load; if it never comes up, check the server log for the cause (a host driver older than torch's CUDA build fails engine init with \"driver too old\"): tail -50 %s",
            probeAddr,
            strings::TrimSpace(code),
            if serverLog != "" { serverLog.clone() } else { string("/workspace/vllm.log") }
        );
    }

    // the run directory: -o names it exactly; the default parent gets
    // the next auto-numbered revision
    let mut outDir = outFlag.clone();
    let mut parentRun = string("");
    if !cmd.Flags().Changed("out") {
        let mut existing: goslice<string> = make!([]string, 0);
        let (entries, err) = os::ReadDir(outFlag.clone());
        if err == nil {
            for e in entries.iter() {
                if e.IsDir() {
                    existing = append!(existing.clone(), e.Name());
                }
            }
        }
        parentRun = profile::LatestRunName(existing.clone());
        outDir = (outFlag.clone()) + ("/") + (profile::NextRunName(existing));
    }
    let err = os::MkdirAll(outDir.clone(), 0o755);
    if err != nil {
        return err;
    }
    fmt::Printf!("collecting into %s\n", outDir.clone());
    // the phase lengths are fixed, so the wall time is knowable up
    // front: nsys window, torch window, three sweep levels, pressure,
    // plus fetches and the local analysis
    let etaMin = (window + 12 + 3 * 18 + 45 + 90 + 59) / 60;
    fmt::Printf!("expect roughly %d minutes end to end\n", etaMin);

    // flag state first: the explicit args this server runs with, and
    // the once-per-version flag catalog
    collectVllmConfig(&tr, serverLog.clone(), outDir.clone());

    // metrics before, and the idle check that decides the probe
    let (metricsBefore, err) = tr.exec(fmt::Sprintf!("curl -s -m 10 http://%s/metrics", probeAddr.clone()));
    if err != nil {
        return fmt::Errorf!("no vLLM server at http://%s on the target: %v", probeAddr, err);
    }
    let _ = os::WriteFile((outDir.clone()) + ("/metrics-before.txt"), metricsBefore.clone(), 0o644);
    let before = metrics::Parse(metricsBefore.clone());
    let startedAt = time::Now();
    let mut probe = probeFlag;
    if !cmd.Flags().Changed("probe") {
        probe = before.Running < 1.0;
        if probe {
            fmt::Printf!("server is idle: the probe will supply the load (two plateaus for batch ratios)\n");
        } else {
            fmt::Printf!("server is busy (%d running): capturing your real load, no probe\n", math::Round(before.Running) as int);
        }
    }
    let mut model = probeModelFlag.clone();
    if probe && model == "" {
        let (m, err) = probeModel(&tr, probeAddr.clone());
        if err != nil {
            return fmt::Errorf!("probe: cannot resolve model from http://%s/v1/models: %v (set --probe-model)", probeAddr, err);
        }
        model = m;
    }

    let mut collected: goslice<string> = make!([]string, 0);
    let mut skippedTools: goslice<string> = make!([]string, 0);

    // phase 1: nsys window, first so its numbers stay clean of torch
    // profiler overhead
    let (nsysTool, _) = profile::Find("nsys");
    let vars = slice!([]profile::Var{
        profile::Var{ Key: string("session"), Value: session.clone(), ..Default::default() },
        profile::Var{ Key: string("out"), Value: remoteOut.clone(), ..Default::default() },
        profile::Var{ Key: string("seconds"), Value: fmt::Sprintf!("%d", window), ..Default::default() },
        profile::Var{ Key: string("addr"), Value: probeAddr.clone(), ..Default::default() },
    });
    let mut nsysArmed = false;
    let (_, err) = tr.exec(string("command -v nsys"));
    if err == nil {
        let (sessions, err) = tr.exec(string("nsys sessions list"));
        nsysArmed = err != nil || strings::Contains(sessions.clone(), session.clone());
    }
    if nsysArmed {
        let (_, err) = tr.exec(("mkdir -p ") + (path::Dir(remoteOut.clone())));
        if err != nil {
            return err;
        }
        for (_, c) in range!(profile::Expand(nsysTool.Window.clone(), vars.clone())) {
            let first: &str = c.Argv[0usize].as_ref();
            if first == "sleep" {
                if probe {
                    let err = runProbePhases(&tr, probeAddr.clone(), model.clone(), probeLow, probeHigh, window);
                    if err != nil {
                        return err;
                    }
                } else {
                    fmt::Printf!("nsys capturing for %d s...\n", window);
                    time::Sleep(time::Seconds(window));
                }
                continue;
            }
            let line = profile::RenderCmd(&c);
            fmt::Printf!("$ %s\n", line.clone());
            let (_, err) = tr.execWindow(line);
            if err != nil {
                return err;
            }
        }
        for (_, a) in range!(nsysTool.Artifacts.clone()) {
            let remote = expandOne(a.clone(), &vars);
            fmt::Printf!("fetching %s -> %s/\n", remote.clone(), outDir.clone());
            let _ = tr.fetch(remote, outDir.clone());
        }
        collected = append!(collected.clone(), string("nsys"));
    } else {
        skippedTools = append!(skippedTools.clone(), string("nsys (no session on the target; see kvlm profile show nsys)"));
    }

    // phase 2: torchprof window, shorter (the trace is ~5 MB per
    // second under load); available = /start_profile answers 200
    let torchWindow: int = 12;
    let (code, _) = tr.exec(fmt::Sprintf!(
        "curl -s -m 5 -o /dev/null -w '%%{http_code}' -X POST http://%s/start_profile",
        probeAddr.clone()
    ));
    if strings::TrimSpace(code.clone()) == "200" {
        if probe {
            let err = runProbePhases(&tr, probeAddr.clone(), model.clone(), probeLow, probeHigh, torchWindow);
            if err != nil {
                return err;
            }
        } else {
            fmt::Printf!("torch profiler capturing for %d s...\n", torchWindow);
            time::Sleep(time::Seconds(torchWindow));
        }
        let (_, err) = tr.exec(fmt::Sprintf!("curl -s -m 120 -X POST http://%s/stop_profile", probeAddr.clone()));
        if err != nil {
            return err;
        }
        let (torchTool, _) = profile::Find("torchprof");
        for (_, a) in range!(torchTool.Artifacts.clone()) {
            let remote = expandOne(a.clone(), &vars);
            fmt::Printf!("fetching %s -> %s/\n", remote.clone(), outDir.clone());
            let _ = tr.fetch(remote, outDir.clone());
        }
        collected = append!(collected.clone(), string("torchprof"));
    } else {
        skippedTools = append!(skippedTools.clone(), string("torchprof (server not launched with --profiler-config; see kvlm profile show torchprof)"));
    }

    // phase 3: the baseline characterizations — concurrency sweep and
    // KV pressure probe. Idle servers only, same guard as the auto
    // probe: a busy production server is never load-tested by kvlm.
    if probe {
        let err = runBench(&tr, probeAddr.clone(), model.clone(), outDir.clone());
        if err != nil {
            fmt::Printf!("bench: %v (sweep and pressure skipped)\n", err);
        } else {
            collected = append!(collected.clone(), string("sweep + pressure"));
        }
    } else {
        skippedTools = append!(skippedTools.clone(), string("sweep + pressure (server has live traffic; only measured on idle servers)"));
    }

    // metrics after, derived over the measured wall time
    let (metricsAfter, err) = tr.exec(fmt::Sprintf!("curl -s -m 10 http://%s/metrics", probeAddr.clone()));
    if err == nil {
        let _ = os::WriteFile((outDir.clone()) + ("/metrics-after.txt"), metricsAfter.clone(), 0o644);
        let after = metrics::Parse(metricsAfter);
        let d = metrics::Derive(&before, &after, time::Since(startedAt).Seconds());
        metrics::PrintDerived(&d);
        collected = append!(collected.clone(), string("metrics"));
    }

    // telemetry snapshot, env fingerprint, server log
    let (telemetry, err) = tr.exec(string("nvidia-smi --query-gpu=timestamp,name,utilization.gpu,power.draw,memory.used,memory.total --format=csv"));
    if err == nil {
        let _ = os::WriteFile((outDir.clone()) + ("/gpu-telemetry.csv"), telemetry, 0o644);
        collected = append!(collected.clone(), string("telemetry"));
    }
    collectEnv(&tr, remoteOut.clone(), outDir.clone());
    collectServerLog(&tr, serverLog.clone(), remoteOut.clone(), outDir.clone());

    // local analysis: sqlite export + graph JSON, GPU read from the
    // measured fingerprint. A local nsys (PATH, KVLM_NSYS, or an
    // install root) exports here; with none, the export runs on the
    // target instead — nsys is there by construction, it took the
    // capture — and the sqlite is fetched. Either way the run ends
    // with a rendered graph.
    let rep = (outDir.clone()) + ("/win.nsys-rep");
    let (_, repErr) = os::Stat(rep.clone());
    if repErr == nil {
        let mut input = rep.clone();
        if nsysLocalPath() == "" {
            fmt::Printf!("no local nsys; exporting sqlite on the target instead\n");
            let (_, err) = tr.exec(fmt::Sprintf!(
                "nsys export --type sqlite --force-overwrite=true --output %s.sqlite %s.nsys-rep",
                remoteOut.clone(),
                remoteOut.clone()
            ));
            if err == nil {
                fmt::Printf!("fetching %s.sqlite -> %s/\n", remoteOut.clone(), outDir.clone());
                let ferr = tr.fetch((remoteOut.clone()) + (".sqlite"), outDir.clone());
                if ferr == nil {
                    input = (outDir.clone()) + ("/win.sqlite");
                } else {
                    input = string("");
                }
            } else {
                input = string("");
            }
            if input == "" {
                fmt::Printf!("remote sqlite export failed; analyze later with: kvlm profile graph -i %s\n", rep.clone());
            }
        }
        if input != "" {
            let mut gpuName = string("");
            let (envData, err) = os::ReadFile((outDir.clone()) + ("/env.txt"));
            if err == nil {
                for (_, line) in range!(strings::Split(string(envData), "\n")) {
                    let (spec, ok) = model::LookupGPU(line.clone());
                    if ok {
                        gpuName = spec.Name.clone();
                        break;
                    }
                }
            }
            let run = path::Base(outDir.clone());
            let err = generateGraph(
                input,
                (outDir.clone()) + ("/graph-structure.json"),
                gpuName,
                fmt::Sprintf!("collected by kvlm profile run, %s", run),
            );
            if err != nil {
                fmt::Printf!("graph analysis: %v (re-run later with kvlm profile graph)\n", err);
            } else {
                collected = append!(collected.clone(), string("graph analysis"));
            }
        }
    }

    printRecommendation(outDir.clone());
    writeRunMeta(outDir.clone(), parentRun, window);

    fmt::Printf!("\n%s: collected %s\n", outDir.clone(), strings::Join(collected, ", "));
    for (_, s) in range!(skippedTools.clone()) {
        fmt::Printf!("skipped %s\n", s);
    }
    fmt::Printf!("the dashboard lists this run as a revision; pack it with: kvlm profile archive %s\n", outDir);
    nil.into()
}

// printRecommendation is the doctrine's last step in the terminal:
// the measured regime joined with the run's own flag state, turned
// into concrete flag=value proposals with the arithmetic shown.
pub(crate) fn printRecommendation(outDir: string) {
    let h = crate::cmd::run::readHeadline(outDir.clone());
    if h.BoundBy == "" {
        return;
    }
    let (_, resolved) = crate::cmd::run::readConfig(outDir.clone());
    let (kvDtype, _) = profile::vllmcfg::Lookup(&resolved, "kv_cache_dtype");
    let mut worstFit: float64 = 0.0;
    let (logData, err) = os::ReadFile((outDir.clone()) + ("/server-log.txt"));
    if err == nil {
        for (_, line) in range!(strings::Split(string(logData), "\n")) {
            if strings::Contains(line.clone(), "Maximum concurrency") {
                let (v, ok) = logFloatAfter(line.clone(), "per request: ");
                if ok {
                    worstFit = v;
                }
            }
        }
    }
    let proposals = profile::propose::Propose(&profile::propose::Inputs {
        BoundBy: h.BoundBy.clone(),
        PeakRunning: h.PeakRunning,
        PeakKvPct: h.PeakKvPct,
        Preemptions: h.Preemptions,
        MaxNumSeqs: h.MaxNumSeqs,
        WorstCaseFit: worstFit,
        KvCacheDtype: kvDtype,
    });
    if proposals.Len() == 0 {
        return;
    }
    fmt::Println!("recommend");
    for (_, p) in range!(proposals) {
        if p.Flag == "" {
            fmt::Printf!("  %s\n", p.Why.clone());
            continue;
        }
        let flag = strings::TrimPrefix(profile::vllmcfg::FlagFor(p.Flag.clone()), string("--"));
        fmt::Printf!("  kvlm apply %s=%s\n", flag, p.Value.clone());
        fmt::Printf!("    why: %s\n", p.Why.clone());
        if p.Risk != "" {
            fmt::Printf!("    risk: %s\n", p.Risk.clone());
        }
    }
}

// writeRunMeta records the run's identity header (run.json): where it
// was collected, what was serving, and which run it descends from.
// Structured facts only; ls/show/diff compute their display from it.
fn writeRunMeta(outDir: string, parentRun: string, window: int) {
    let mut obj: map<string, json::Value> = map::new_no_zero();
    obj.Set(string("run"), json::Value::String(path::Base(outDir.clone())));
    obj.Set(string("parent"), json::Value::String(parentRun));
    obj.Set(string("windowS"), json::Value::Number(float64(window)));
    obj.Set(string("collected"), json::Value::String(time::Now().Format(string(time::RFC3339))));
    let (t, ok) = state::Current();
    if ok {
        obj.Set(string("driver"), json::Value::String(t.Driver.clone()));
        obj.Set(string("pod"), json::Value::String(t.Pod.clone()));
        obj.Set(string("model"), json::Value::String(t.Model.clone()));
        obj.Set(string("variant"), json::Value::String(t.Variant.clone()));
        obj.Set(string("mode"), json::Value::String(t.Mode.clone()));
        obj.Set(string("gpuType"), json::Value::String(t.GPUType.clone()));
        obj.Set(string("gpuCount"), json::Value::Number(float64(t.GPUCount)));
        obj.Set(string("vllmVersion"), json::Value::String(t.VllmVersion.clone()));
    }
    // a flag delta staged by kvlm apply belongs to the next collected
    // run: consume it here so the chain records what changed
    let pendingPath = (path::Dir(outDir.clone())) + ("/pending.json");
    let (data, err) = os::ReadFile(pendingPath.clone());
    if err == nil {
        let mut v = json::Value::Null;
        let perr = json::Unmarshal(data.as_ref(), &mut v);
        if perr == nil {
            if let Some(p) = v.AsObject() {
                let (delta, has) = p.Get("delta");
                if has {
                    obj.Set(string("delta"), delta);
                }
                let (par, has) = p.Get("parent");
                if has {
                    if let Some(s) = par.AsString() {
                        if s.clone() != "" {
                            obj.Set(string("parent"), json::Value::String(s.clone()));
                        }
                    }
                }
            }
        }
        let _ = os::Remove(pendingPath);
    }
    let v = json::Value::Object(obj);
    let (out, err) = json::MarshalIndent(&v, "", "  ");
    if err != nil {
        return;
    }
    let werr = os::WriteFile((outDir.clone()) + ("/run.json"), out, 0o644);
    if werr == nil {
        fmt::Printf!("wrote %s/run.json\n", outDir);
    }
}

// runProfile is the body of `profile run`.
fn runProfile(cmd: &mut cobra::Command, toolName: string) -> error {
    let (t, ok) = profile::Find(toolName.clone());
    if !ok {
        return fmt::Errorf!("unknown tool %q (have: %s)", toolName, profile::Names());
    }
    if t.Window.Len() == 0 {
        return fmt::Errorf!(
            "%s has no automated capture recipe; kvlm profile show %s prints the manual one",
            t.Name.clone(),
            t.Name.clone()
        );
    }

    let (ssh, _) = cmd.Flags().GetString("ssh");
    let (pod, _) = cmd.Flags().GetString("k8s-pod");
    let (session, _) = cmd.Flags().GetString("session");
    let (window, _) = cmd.Flags().GetInt("window");
    let (remoteOut, _) = cmd.Flags().GetString("remote-out");
    let (outDir, _) = cmd.Flags().GetString("out");

    let (serverLog, _) = cmd.Flags().GetString("server-log");
    let (probe, _) = cmd.Flags().GetBool("probe");
    let (probeAddr, _) = cmd.Flags().GetString("probe-addr");
    let (probeModelFlag, _) = cmd.Flags().GetString("probe-model");
    let (probeLow, _) = cmd.Flags().GetInt("probe-low");
    let (probeHigh, _) = cmd.Flags().GetInt("probe-high");

    let (tr, err) = buildTransport(cmd, ssh.clone(), pod.clone());
    if err != nil {
        return err;
    }

    let vars = slice!([]profile::Var{
        profile::Var{ Key: string("session"), Value: session.clone(), ..Default::default() },
        profile::Var{ Key: string("out"), Value: remoteOut.clone(), ..Default::default() },
        profile::Var{ Key: string("seconds"), Value: fmt::Sprintf!("%d", window), ..Default::default() },
        profile::Var{ Key: string("addr"), Value: probeAddr.clone(), ..Default::default() },
    });

    // 0. can we reach the target at all? Without this, an ssh auth
    // failure would masquerade as "tool not installed".
    let (_, err) = tr.exec(string("true"));
    if err != nil {
        return fmt::Errorf!("cannot reach the target: %v", err);
    }

    // 1. tool-specific preflights. nsys: binary installed and the
    // launch session alive. torchprof and anything endpoint-driven:
    // the server answers at {addr} (the profiler flag was set at
    // launch, kvlm cannot arm it after the fact).
    let toolIs: &str = t.Name.as_ref();
    if toolIs == "nsys" {
        let (_, err) = tr.exec(string("command -v nsys"));
        if err != nil {
            printPhase("nsys is not installed on the target. Install:", profile::Expand(t.Install.clone(), vars.clone()));
            return fmt::Errorf!("nsys not found on the target (probe: %v)", err);
        }
        // best effort: older nsys lacks `sessions list`; when the
        // subcommand itself fails we proceed and let start report the
        // real state
        let (sessions, err) = tr.exec(string("nsys sessions list"));
        if err == nil && !strings::Contains(sessions.clone(), session.clone()) {
            printPhase(
                "No profiler session on the target. Launch the server under nsys first\n(kvlm does not restart your server):",
                profile::Expand(t.Setup.clone(), vars.clone()),
            );
            return fmt::Errorf!("nsys session %q not found on the target", session);
        }
    } else {
        let (_, err) = tr.exec(fmt::Sprintf!("curl -s -m 5 -o /dev/null http://%s/v1/models", probeAddr.clone()));
        if err != nil {
            printPhase(
                "No server answering on the target. Launch it with the profiler flag set\n(kvlm does not restart your server):",
                profile::Expand(t.Setup.clone(), vars.clone()),
            );
            return fmt::Errorf!("no server at http://%s on the target: %v", probeAddr, err);
        }
    }

    // 2. every artifact's directory must exist; nsys silently writes
    // a fallback report in /tmp when its -o directory is missing.
    let (_, err) = tr.exec(("mkdir -p ") + (path::Dir(remoteOut.clone())));
    if err != nil {
        return err;
    }
    for (_, a) in range!(t.Artifacts.clone()) {
        let dir = path::Dir(expandOne(a.clone(), &vars));
        let (_, err) = tr.exec(("mkdir -p ") + (dir));
        if err != nil {
            return err;
        }
    }

    // 2c. probe preflight: resolve the model name before the window
    // opens so a bad endpoint fails fast, not mid-capture.
    let mut model = probeModelFlag.clone();
    if probe && model == "" {
        let (m, err) = probeModel(&tr, probeAddr.clone());
        if err != nil {
            return fmt::Errorf!("probe: cannot resolve model from http://%s/v1/models: %v (set --probe-model)", probeAddr, err);
        }
        model = m;
    }

    // 3. the capture window: same data `profile show` renders. The
    // time between start and stop is either a local sleep or, with
    // --probe, the two concurrency plateaus running on the target.
    let win = profile::Expand(t.Window.clone(), vars.clone());
    for (_, c) in range!(win.clone()) {
        let first: &str = c.Argv[0usize].as_ref();
        if first == "sleep" {
            if probe {
                let err = runProbePhases(&tr, probeAddr.clone(), model.clone(), probeLow, probeHigh, window);
                if err != nil {
                    return err;
                }
                continue;
            }
            fmt::Printf!("capturing for %d s...\n", window);
            time::Sleep(time::Seconds(window));
            continue;
        }
        let line = profile::RenderCmd(&c);
        fmt::Printf!("$ %s\n", line.clone());
        let (out, err) = tr.execWindow(line);
        if err != nil {
            return err;
        }
        let trimmed = strings::TrimSpace(out);
        if trimmed != "" {
            fmt::Printf!("%s\n", trimmed);
        }
    }

    // 4. fetch artifacts
    let err = os::MkdirAll(outDir.clone(), 0o755);
    if err != nil {
        return err;
    }
    let mut localRep = string("");
    for (_, a) in range!(t.Artifacts.clone()) {
        let remote = expandOne(a.clone(), &vars);
        fmt::Printf!("fetching %s -> %s/\n", remote.clone(), outDir.clone());
        let err = tr.fetch(remote.clone(), outDir.clone());
        if err != nil {
            return err;
        }
        localRep = (outDir.clone()) + ("/") + (path::Base(remote));
    }

    collectEnv(&tr, remoteOut.clone(), outDir.clone());
    collectServerLog(&tr, serverLog.clone(), remoteOut.clone(), outDir.clone());

    // 5. analyze locally when nsys is installed here; print the
    // commands either way.
    let localVars = slice!([]profile::Var{
        profile::Var{ Key: string("out"), Value: strings::TrimSuffix(localRep.clone(), ".nsys-rep"), ..Default::default() },
    });
    let analyze = profile::Expand(t.Analyze.clone(), localVars);
    if toolIs == "nsys" && nsysLocalPath() == "" {
        printPhase("no local nsys found (PATH, KVLM_NSYS, install roots). To analyze:", analyze);
        return nil.into();
    }
    for (_, c) in range!(analyze.clone()) {
        fmt::Printf!("$ %s\n", profile::RenderCmd(&c));
        let argv = c.Argv.clone();
        let name = argv[0usize].clone();
        let rest = argv.slice(1, argv.Len());
        let (out, errOut, err) = runCaptureRetry(name, rest);
        if err != nil {
            fmt::Printf!("%s\n", errOut);
            return err;
        }
        fmt::Printf!("%s\n", out);
    }
    nil.into()
}

// ─── dashboard ───────────────────────────────────────────────────────

// Go: //go:embed dashboard.html
goish::embed! {
    #[embed("dashboard.html")]
    static dashboardHTML: string;
}

// vendored UI library (htm + preact standalone ESM), served by the
// dashboard itself so the page stays fully self-contained offline
goish::embed! {
    #[embed("preact-htm.mjs")]
    static preactHtmJS: string;
}

// live state for the dashboard's /api/live endpoint: the vLLM address
// and the previous sample to derive window statistics against.
static dashVllm: Lazy<sync::Mutex<string>> = Lazy::new(|| sync::Mutex::new(string("")));
static dashLast: Lazy<sync::Mutex<Option<(metrics::Snapshot, time::Time)>>> =
    Lazy::new(|| sync::Mutex::new(None));

// jsonEsc escapes a string for embedding in a JSON literal.
fn jsonEsc(s: string) -> string {
    let mut e = strings::ReplaceAll(s, "\\", "\\\\");
    e = strings::ReplaceAll(e, "\"", "\\\"");
    e = strings::ReplaceAll(e, "\n", " ");
    e
}

fn jnum(v: float64) -> string {
    fmt::Sprintf!("%v", math::Round(v * 10.0) / 10.0)
}

// summaryJSON builds /api/summary from the live registries: the
// measured reference run, the tool matrix, and the catalog snapshot.
fn summaryJSON() -> string {
    let mut b = strings::Builder::new();
    let r = profile::measured::Reference();

    let _ = b.WriteString("{\"measured\":{\"provenance\":\"");
    let _ = b.WriteString(jsonEsc(r.Provenance.clone()));
    let _ = b.WriteString("\",\"curve\":[");
    for (i, p) in range!(r.Curve.clone()) {
        if i > 0 {
            let _ = b.WriteString(",");
        }
        let _ = b.WriteString(fmt::Sprintf!(
            "{\"concurrency\":%d,\"perUser\":%s,\"aggregate\":%s,\"ttftMs\":%s,\"tpotMs\":%s}",
            p.Concurrency,
            jnum(p.PerUserTokS),
            jnum(p.AggregateTokS),
            jnum(p.TTFTMeanMs),
            jnum(p.TPOTMeanMs)
        ));
    }
    let _ = b.WriteString("],\"kernels\":[");
    for (i, k) in range!(r.Kernels.clone()) {
        if i > 0 {
            let _ = b.WriteString(",");
        }
        let _ = b.WriteString(fmt::Sprintf!(
            "{\"name\":\"%s\",\"pct\":%s}",
            jsonEsc(k.Name.clone()),
            jnum(k.Pct)
        ));
    }
    let _ = b.WriteString("],\"metrics\":[");
    writeMetricArray(&mut b, r.Metrics.clone());
    let _ = b.WriteString("],\"saturation\":[");
    writeMetricArray(&mut b, r.Saturation.clone());
    let _ = b.WriteString("],\"graphMetrics\":[");
    writeMetricArray(&mut b, r.GraphMetrics.clone());
    let _ = b.WriteString("],\"cudaGraphs\":[");
    for (i, f) in range!(r.CudaGraphs.clone()) {
        if i > 0 {
            let _ = b.WriteString(",");
        }
        let _ = b.WriteString(fmt::Sprintf!(
            "{\"label\":\"%s\",\"value\":\"%s\"}",
            jsonEsc(f.Label.clone()),
            jsonEsc(f.Value.clone())
        ));
    }
    let _ = b.WriteString("],\"neverCaptured\":[");
    for (i, n) in range!(r.NeverCaptured.clone()) {
        if i > 0 {
            let _ = b.WriteString(",");
        }
        let _ = b.WriteString(fmt::Sprintf!("\"%s\"", jsonEsc(n.clone())));
    }
    let _ = b.WriteString("]},\"artifacts\":[");
    let arts = listArtifacts();
    for (i, a) in range!(arts.clone()) {
        if i > 0 {
            let _ = b.WriteString(",");
        }
        let _ = b.WriteString(fmt::Sprintf!("\"%s\"", jsonEsc(a.clone())));
    }
    let _ = b.WriteString("],\"tools\":[");
    let mut first = true;
    for t in profile::ToolsSorted().iter() {
        if !first {
            let _ = b.WriteString(",");
        }
        first = false;
        let _ = b.WriteString(fmt::Sprintf!(
            "{\"name\":\"%s\",\"summary\":\"%s\",\"support\":[",
            jsonEsc(t.Name.clone()),
            jsonEsc(t.Summary.clone())
        ));
        for (i, s) in range!(t.Support.clone()) {
            if i > 0 {
                let _ = b.WriteString(",");
            }
            let _ = b.WriteString(fmt::Sprintf!(
                "{\"env\":\"%s\",\"status\":\"%s\",\"reason\":\"%s\",\"verified\":\"%s\"}",
                jsonEsc(s.Env.clone()),
                jsonEsc(s.Status.clone()),
                jsonEsc(s.Reason.clone()),
                jsonEsc(s.Verified.clone())
            ));
        }
        let _ = b.WriteString("]}");
    }
    let _ = b.WriteString("]}");
    b.String()
}

// liveJSON builds /api/live: one fresh sample derived against the
// previous one.
fn liveJSON() -> string {
    let addr = dashVllm.Lock().clone();
    if addr == "" {
        return string("{\"enabled\":false}");
    }
    let (snap, err) = metrics::Fetch(addr.clone());
    if err != nil {
        return fmt::Sprintf!(
            "{\"enabled\":true,\"error\":\"%s\"}",
            jsonEsc(fmt::Sprintf!("%v", err))
        );
    }
    let now = time::Now();
    let mut guard = dashLast.Lock();
    let prev = guard.clone();
    *guard = Some((snap.clone(), now));
    drop(guard);
    match prev {
        None => string("{\"enabled\":true,\"error\":\"first sample, refresh shortly\"}"),
        Some((last, t0)) => {
            let d = metrics::Derive(&last, &snap, time::Since(t0).Seconds());
            fmt::Sprintf!(
                "{\"enabled\":true,\"running\":%d,\"waiting\":%d,\"kvUsagePct\":%s,\"genTokS\":%d,\"ttftMeanMs\":%d,\"tpotMeanMs\":%s,\"queueMeanS\":%s,\"prefillMeanS\":%s,\"decodeMeanS\":%s,\"preemptions\":%d,\"requests\":%d,\"windowS\":%d}",
                math::Round(d.Running) as int,
                math::Round(d.Waiting) as int,
                jnum(d.KVUsage * 100.0),
                math::Round(d.GenTokS) as int,
                math::Round(d.TTFTMeanMs) as int,
                jnum(d.TPOTMeanMs),
                jnum2(d.QueueMeanS),
                jnum2(d.PrefillMeanS),
                jnum2(d.DecodeMeanS),
                math::Round(d.Preemptions) as int,
                math::Round(d.Requests) as int,
                math::Round(d.ElapsedS) as int
            )
        }
    }
}

// logFloatBefore returns the number that ends right before marker in
// line ("16.21 GiB for weight" with marker " GiB for weight" -> 16.21).
// Thousands commas are tolerated.
fn logFloatBefore(line: string, marker: &'static str) -> (float64, bool) {
    let idx = strings::Index(line.clone(), string(marker));
    if idx <= 0 {
        return (0.0, false);
    }
    let mut start = idx;
    while start > 0 {
        let c = line[(start - 1) as usize];
        if (c >= b'0' && c <= b'9') || c == b'.' || c == b',' {
            start -= 1;
        } else {
            break;
        }
    }
    if start == idx {
        return (0.0, false);
    }
    let nums = strings::ReplaceAll(line.slice(start, idx), ",", "");
    let (v, err) = strconv::ParseFloat(nums, 64);
    if err != nil {
        return (0.0, false);
    }
    (v, true)
}

// logFloatAfter returns the number that starts right after marker.
fn logFloatAfter(line: string, marker: &'static str) -> (float64, bool) {
    let idx = strings::Index(line.clone(), string(marker));
    if idx < 0 {
        return (0.0, false);
    }
    let i = idx + (string(marker)).Len();
    let mut end = i;
    while end < line.Len() {
        let c = line[end as usize];
        if (c >= b'0' && c <= b'9') || c == b'.' || c == b',' {
            end += 1;
        } else {
            break;
        }
    }
    if end == i {
        return (0.0, false);
    }
    let nums = strings::ReplaceAll(line.slice(i, end), ",", "");
    let (v, err) = strconv::ParseFloat(nums, 64);
    if err != nil {
        return (0.0, false);
    }
    (v, true)
}

// baselineJSON derives one run's baseline characterization from its
// own artifacts: the vLLM startup lines in server-log.txt (KV pool,
// weight and graph memory, max concurrency) and the counter deltas
// between metrics-before and metrics-after. Cards that need data the
// run did not collect are simply absent — nothing is estimated.
fn baselineJSON(run: string) -> string {
    let mut ms: goslice<profile::measured::Metric> = make!([]profile::measured::Metric, 0);
    if strings::Contains(run.clone(), "/") || strings::Contains(run.clone(), "..") || run == "" {
        let mut b = strings::Builder::new();
        let _ = b.WriteString("{\"metrics\": []}");
        return string(b.String());
    }
    let dir = ("profile-output/") + (run.clone());

    let (logData, err) = os::ReadFile((dir.clone()) + ("/server-log.txt"));
    if err == nil {
        let mut haveConc = false;
        let mut haveMem = false;
        for (_, line) in range!(strings::Split(string(logData), "\n")) {
            if !haveConc && strings::Contains(line.clone(), "Maximum concurrency") {
                let (v, ok) = logFloatAfter(line.clone(), "per request: ");
                let (ctx, _) = logFloatAfter(line.clone(), "Maximum concurrency for ");
                if ok {
                    haveConc = true;
                    ms = append!(ms.clone(), profile::measured::Metric{
                        Key: string("max_concurrency"), Label: string("max concurrency"),
                        Value: v, Unit: string("x"), Max: 0.0,
                        Note: fmt::Sprintf!("vLLM's own line at %d-token requests", ctx as int),
                        ..Default::default()
                    });
                }
            }
            if !haveMem && strings::Contains(line.clone(), " GiB for weight") {
                haveMem = true;
                let (w, ok) = logFloatBefore(line.clone(), " GiB for weight");
                if ok {
                    ms = append!(ms.clone(), profile::measured::Metric{
                        Key: string("weight_mem"), Label: string("model weights per GPU"),
                        Value: w, Unit: string("GiB"), Max: 0.0, Note: string("from the startup memory report"),
                        ..Default::default()
                    });
                }
                let (g, ok) = logFloatBefore(line.clone(), " GiB for CUDAGraph");
                if ok {
                    ms = append!(ms.clone(), profile::measured::Metric{
                        Key: string("graph_mem"), Label: string("CUDA graph memory per GPU"),
                        Value: g, Unit: string("GiB"), Max: 0.0, Note: string(""),
                        ..Default::default()
                    });
                }
                let (kv, ok) = logFloatAfter(line.clone(), "kv cache memory in use is ");
                if ok {
                    ms = append!(ms.clone(), profile::measured::Metric{
                        Key: string("kv_pool"), Label: string("KV pool per GPU"),
                        Value: kv, Unit: string("GiB"), Max: 0.0, Note: string("from the startup memory report"),
                        ..Default::default()
                    });
                }
            }
        }
    }

    let (mb, errB) = os::ReadFile((dir.clone()) + ("/metrics-before.txt"));
    let (ma, errA) = os::ReadFile((dir.clone()) + ("/metrics-after.txt"));
    if errB == nil && errA == nil {
        let before = metrics::Parse(string(mb));
        let after = metrics::Parse(string(ma));
        let reqs = after.RequestSuccess - before.RequestSuccess;
        if reqs > 0.0 {
            ms = append!(ms.clone(), profile::measured::Metric{
                Key: string("requests"), Label: string("requests completed in the window"),
                Value: reqs, Unit: string(""), Max: 0.0, Note: string(""), ..Default::default()
            });
        }
        let gen = after.GenerationTokens - before.GenerationTokens;
        if gen > 0.0 {
            ms = append!(ms.clone(), profile::measured::Metric{
                Key: string("gen_tokens"), Label: string("tokens generated in the window"),
                Value: math::Round(gen), Unit: string(""), Max: 0.0, Note: string(""), ..Default::default()
            });
        }
        let q = after.PrefixCacheQueries - before.PrefixCacheQueries;
        let h = after.PrefixCacheHits - before.PrefixCacheHits;
        if q > 0.0 {
            ms = append!(ms.clone(), profile::measured::Metric{
                Key: string("prefix_hit"), Label: string("prefix cache hit rate"),
                Value: math::Round(1000.0 * h / q) / 10.0, Unit: string("%"), Max: 100.0,
                Note: string("over the collection window"), ..Default::default()
            });
        }
        let pre = after.Preemptions - before.Preemptions;
        ms = append!(ms.clone(), profile::measured::Metric{
            Key: string("preemptions"), Label: string("preemptions in the window"),
            Value: pre, Unit: string(""), Max: 0.0,
            Note: if pre > 0.0 { string("KV pressure: raise --gpu-memory-utilization or lower --max-num-seqs") } else { string("no KV pressure") },
            ..Default::default()
        });
    }

    let mut b = strings::Builder::new();
    let _ = b.WriteString("{\n \"metrics\": [");
    writeMetricArray(&mut b, ms);
    let _ = b.WriteString("]\n}\n");
    string(b.String())
}

// baselineHandler answers /api/baseline/<run> with the run-derived
// baseline; the registry characterization stays in /api/summary.
struct baselineHandler {}

impl http::Handler for baselineHandler {
    fn ServeHTTP(&self, w: &(dyn http::ResponseWriter + Send + Sync + 'static), r: &http::Request) {
        let run = r.URL.Path.clone();
        w.Header().Set(string("Content-Type"), string("application/json"));
        let _ = w.Write(bytes(baselineJSON(run)));
    }
}

// importArchive extracts one .kvlm from profile-output/ into its run
// directory: the dashboard explorer's "load archive" action. Names are
// flat (no path separators) and the unpack itself rejects traversal.
fn importArchive(name: string, force: bool) -> (string, error) {
    if strings::Contains(name.clone(), "/") || strings::Contains(name.clone(), "..") || !strings::HasSuffix(name.clone(), ".kvlm") {
        return (string(""), fmt::Errorf!("not a flat .kvlm name: %q", name));
    }
    let (data, err) = os::ReadFile(("profile-output/") + (name.clone()));
    if err != nil {
        return (string(""), fmt::Errorf!("read %s: %v", name, err));
    }
    let (mut run, _, entries, err) = profile::archive::Unpack(data);
    if err != nil {
        return (string(""), err);
    }
    if run == "" {
        run = strings::TrimSuffix(name.clone(), ".kvlm");
    }
    let dest = ("profile-output/") + (run.clone());
    let (_, statErr) = os::ReadDir(dest.clone());
    if statErr == nil && !force {
        return (run.clone(), fmt::Errorf!("run %s already exists; import again with force to overwrite its files", run));
    }
    let err = os::MkdirAll(dest.clone(), 0o755);
    if err != nil {
        return (string(""), err);
    }
    for (_, e) in range!(entries.clone()) {
        // a bundled flag catalog restores the shared per-version
        // store, never the run dir; an already-collected catalog for
        // that version is kept as is
        let (ver, isCat) = profile::archive::CatalogVersion(e.Name.clone());
        if isCat {
            let (_, err) = os::Stat(profile::vllmflags::Path(ver.clone()));
            if err != nil {
                let _ = os::MkdirAll(string(profile::vllmflags::Dir), 0o755);
                let _ = os::WriteFile(profile::vllmflags::Path(ver.clone()), e.Data.clone(), 0o644);
                fmt::Printf!("restored the vLLM %s flag catalog from the archive\n", ver);
            }
            continue;
        }
        let err = os::WriteFile((dest.clone()) + ("/") + (e.Name.clone()), e.Data.clone(), 0o644);
        if err != nil {
            return (string(""), fmt::Errorf!("write %s/%s: %v", dest, e.Name.clone(), err));
        }
    }
    (run, nil.into())
}

// importHandler answers POST /api/import?archive=<name>.kvlm[&force=1].
struct importHandler {}

impl http::Handler for importHandler {
    fn ServeHTTP(&self, w: &(dyn http::ResponseWriter + Send + Sync + 'static), r: &http::Request) {
        w.Header().Set(string("Content-Type"), string("application/json"));
        if r.Method != "POST" {
            w.WriteHeader(405);
            let _ = w.Write(bytes("{\"error\": \"POST only\"}"));
            return;
        }
        let q = r.URL.RawQuery.clone();
        let mut name = string("");
        let mut force = false;
        for (_, kv) in range!(strings::Split(q, "&")) {
            if strings::HasPrefix(kv.clone(), string("archive=")) {
                name = strings::TrimPrefix(kv.clone(), string("archive="));
            }
            if kv == "force=1" {
                force = true;
            }
        }
        let (run, err) = importArchive(name, force);
        if err != nil {
            w.WriteHeader(400);
            let _ = w.Write(bytes(fmt::Sprintf!("{\"error\": \"%s\"}", jsonEsc(fmt::Sprintf!("%v", err)))));
            return;
        }
        let _ = w.Write(bytes(fmt::Sprintf!("{\"run\": \"%s\"}", jsonEsc(run))));
    }
}

// artifactHandler serves files under profile-output/ read-only, for
// the dashboard's flamegraph embed and report downloads.
struct artifactHandler {}

impl http::Handler for artifactHandler {
    fn ServeHTTP(&self, w: &(dyn http::ResponseWriter + Send + Sync + 'static), r: &http::Request) {
        let p = r.URL.Path.clone();
        if strings::Contains(p.clone(), "..") || strings::HasPrefix(p.clone(), "/") {
            w.WriteHeader(400);
            let _ = w.Write(bytes("bad path"));
            return;
        }
        let full = ("profile-output/") + (p.clone());
        let (data, err) = os::ReadFile(full);
        if err != nil {
            w.WriteHeader(404);
            let _ = w.Write(bytes("not found"));
            return;
        }
        w.Header().Set(string("Content-Type"), artifactContentType(p));
        let _ = w.Write(data);
    }
}

// listArtifacts walks profile-output/ (two levels) and returns the
// relative paths of files worth surfacing on the dashboard.
fn listArtifacts() -> slice<string> {
    let mut out: slice<string> = make!([]string, 0);
    let (entries, err) = os::ReadDir("profile-output");
    if err != nil {
        return out;
    }
    for e in entries.iter() {
        if e.IsDir() {
            let sub = ("profile-output/") + (e.Name());
            let (subEntries, err) = os::ReadDir(sub.clone());
            if err != nil {
                continue;
            }
            for s in subEntries.iter() {
                if !s.IsDir() {
                    out = append!(out.clone(), (e.Name()) + ("/") + (s.Name()));
                }
            }
        } else {
            out = append!(out.clone(), e.Name());
        }
    }
    out
}

// artifactContentType maps a filename to the Content-Type it should be
// served with; SVGs render inline, reports download.
fn artifactContentType(name: string) -> string {
    if strings::HasSuffix(name.clone(), ".svg") {
        return string("image/svg+xml");
    }
    if strings::HasSuffix(name.clone(), ".md") || strings::HasSuffix(name.clone(), ".txt") || strings::HasSuffix(name.clone(), ".csv") || strings::HasSuffix(name.clone(), ".log") {
        return string("text/plain; charset=utf-8");
    }
    if strings::HasSuffix(name.clone(), ".json") {
        return string("application/json");
    }
    if strings::HasSuffix(name.clone(), ".kvlm") {
        return string("application/gzip");
    }
    string("application/octet-stream")
}

// dashboardCmd represents `kvlm profile dashboard`: serve the local
// performance dashboard.
pub(crate) fn dashboardCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("dashboard"),
        Short: string("Serve the local performance dashboard"),
        Long: string(
            "Serve a local web dashboard: the measured H100 reference run\n\
             (concurrency curve, kernel breakdown, saturation behavior), the\n\
             profiling capability matrix, and the catalog snapshot, all from\n\
             the same registries the CLI uses. With --vllm, a live panel\n\
             polls that server's /metrics every few seconds.",
        ),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, _args: goslice<string>| -> error {
                let (addr, _) = cmd.Flags().GetString("addr");
                let (vllm, _) = cmd.Flags().GetString("vllm");
                *dashVllm.Lock() = vllm.clone();

                let mux = http::ServeMux::new();
                mux.HandleFunc(string("/"), |w, _r| {
                    w.Header().Set(string("Content-Type"), string("text/html; charset=utf-8"));
                    let _ = w.Write(bytes(dashboardHTML.clone()));
                });
                mux.HandleFunc(string("/api/summary"), |w, _r| {
                    w.Header().Set(string("Content-Type"), string("application/json"));
                    let _ = w.Write(bytes(summaryJSON()));
                });
                mux.HandleFunc(string("/api/live"), |w, _r| {
                    w.Header().Set(string("Content-Type"), string("application/json"));
                    let _ = w.Write(bytes(liveJSON()));
                });
                mux.Handle(string("/artifacts/"), http::StripPrefix(string("/artifacts/"), artifactHandler{}));
                mux.Handle(string("/api/baseline/"), http::StripPrefix(string("/api/baseline/"), baselineHandler{}));
                mux.Handle(string("/api/import"), importHandler{});
                mux.HandleFunc(string("/lib/preact-htm.mjs"), |w, _r| {
                    w.Header().Set(string("Content-Type"), string("text/javascript"));
                    let _ = w.Write(bytes(preactHtmJS.clone()));
                });
                let handler: alloc::sync::Arc<dyn http::Handler> = alloc::sync::Arc::new(mux);
                fmt::Printf!("kvlm dashboard on http://%s/\n", addr.clone());
                if vllm != "" {
                    fmt::Printf!("live panel polling %s\n", vllm);
                } else {
                    fmt::Printf!("no --vllm target; live panel disabled\n");
                }
                http::ListenAndServe(addr, handler)
            },
        )),
        ..Default::default()
    };
    c.Flags().StringP(
        string("addr"),
        string("a"),
        string("127.0.0.1:7788"),
        string("address to serve the dashboard on"),
    );
    let _ = c.Flags().String_flag(
        string("vllm"),
        string(""),
        string("vLLM server host:port for the live panel"),
    );
    c
}

// Go: func init() { profileCmd.AddCommand(...); rootCmd.AddCommand(profileCmd) }
#[goish::init]
fn init() {
    let mut p = profileCmd();
    p.AddCommand(lsCmd());
    p.AddCommand(showCmd());
    p.AddCommand(metricsCmd());
    p.AddCommand(flamegraphCmd());
    p.AddCommand(graphCmd());
    p.AddCommand(archiveCmd());
    p.AddCommand(importCmd());
    p.AddCommand(runCmd());
    p.AddCommand(dashboardCmd());
    rootCmd.Lock().AddCommand(p);

    // the same machinery, loop-facing at the top level: the dashboard
    // is the loop's second face, live metrics answer "what is
    // happening right now", and the tool registry reads better as
    // profiler (it lists profilers, not profiles)
    let mut dash = dashboardCmd();
    dash.Use = string("dash");
    rootCmd.Lock().AddCommand(dash);
    rootCmd.Lock().AddCommand(metricsCmd());
    let mut prof = cobra::Command {
        Use: string("profiler"),
        Short: string("The profiling tool registry: what runs where, and how"),
        ..Default::default()
    };
    prof.AddCommand(lsCmd());
    prof.AddCommand(showCmd());
    rootCmd.Lock().AddCommand(prof);
}
