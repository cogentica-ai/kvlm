// run group: the durable side of the CLI. A run is one collected
// revision under profile-output/; these commands list them, show one,
// diff two, re-analyze captures, and move them as .kvlm archives.
// Every subcommand accepts a run reference: run7, bare 7, latest,
// -1 (one before latest), or a directory path.
#![allow(non_snake_case)]

use goish::encoding::json;
use goish::fmt;
use goish::gomap::map;
use goish::math;
use goish::os;
use goish::path;
use goish::strconv;
use goish::string;
use goish::strings;
use goish::text::tabwriter;
use goish::goslice::slice;
use goish::errors::error;
use goish::{append, float64, int, make, nil, range, slice as slicemac};

use spf13_cobra as cobra;

use crate::cmd::profile as profcmd;
use crate::cmd::rootCmd;
use crate::flamegraph;
use crate::model;
use crate::profile;

const defaultRoot: &str = "./profile-output";

// runNumber parses the N out of runN (-1 when the name is not runN).
fn runNumber(name: string) -> int {
    if !strings::HasPrefix(name.clone(), string("run")) {
        return -1;
    }
    let digits = strings::TrimPrefix(name, string("run"));
    let (n, err) = strconv::Atoi(digits);
    if err != nil {
        return -1;
    }
    n
}

// runNames lists runN directories under root, ascending by number.
fn runNames(root: string) -> slice<string> {
    let mut names: slice<string> = make!([]string, 0);
    let (entries, err) = os::ReadDir(root);
    if err != nil {
        return names;
    }
    let mut nums: alloc::vec::Vec<int> = alloc::vec::Vec::new();
    for e in entries.iter() {
        if !e.IsDir() {
            continue;
        }
        let n = runNumber(e.Name());
        if n >= 0 {
            nums.push(n);
        }
    }
    nums.sort();
    for n in nums.iter() {
        names = append!(names.clone(), fmt::Sprintf!("run%d", *n));
    }
    names
}

// ResolveRun turns a user-typed reference into a run directory:
// run7, bare 7, latest/last, -1 for one back, or a path.
pub(crate) fn ResolveRun(root: string, reference: string) -> (string, error) {
    // a path (contains a slash, or names an existing directory) wins
    if strings::Contains(reference.clone(), string("/")) {
        let (_, err) = os::ReadDir(reference.clone());
        if err == nil {
            return (strings::TrimSuffix(reference, "/"), nil.into());
        }
        return (string(""), fmt::Errorf!("%s is not a run directory", reference));
    }
    let names = runNames(root.clone());
    if names.Len() == 0 {
        return (string(""), fmt::Errorf!("no runs under %s yet; kvlm profile run collects the first one", root));
    }
    let mut name = reference.clone();
    if reference == "latest" || reference == "last" || reference == "" {
        name = names[(names.Len() - 1) as usize].clone();
    } else if strings::HasPrefix(reference.clone(), string("-")) {
        let (back, err) = strconv::Atoi(strings::TrimPrefix(reference.clone(), string("-")));
        if err != nil || back < 1 {
            return (string(""), fmt::Errorf!("bad run reference %q", reference));
        }
        let idx = names.Len() - 1 - back;
        if idx < 0 {
            return (string(""), fmt::Errorf!("only %d runs exist; %q reaches past the first", names.Len(), reference));
        }
        name = names[idx as usize].clone();
    } else if runNumber(reference.clone()) < 0 {
        let (n, err) = strconv::Atoi(reference.clone());
        if err != nil {
            return (string(""), fmt::Errorf!("bad run reference %q (want runN, N, latest, -1, or a path)", reference));
        }
        name = fmt::Sprintf!("run%d", n);
    }
    let dir = (root.clone()) + ("/") + (name.clone());
    let (_, err) = os::ReadDir(dir.clone());
    if err != nil {
        return (string(""), fmt::Errorf!("no run %s under %s (kvlm run ls lists them)", name, root));
    }
    (dir, nil.into())
}

// ---- JSON walking (local copies; each module stays standalone) ----

fn loadJSON(pathStr: string) -> (json::Value, bool) {
    let (data, err) = os::ReadFile(pathStr);
    if err != nil {
        return (json::Value::Null, false);
    }
    let mut v = json::Value::Null;
    let perr = json::Unmarshal(data.as_ref(), &mut v);
    if perr != nil {
        return (json::Value::Null, false);
    }
    (v, true)
}

fn jstr(v: &json::Value, key: &str) -> string {
    if let Some(obj) = v.AsObject() {
        let (val, _) = obj.Get(key);
        if let Some(s) = val.AsString() {
            return s.clone();
        }
    }
    string("")
}

fn jnum(v: &json::Value, key: &str) -> float64 {
    if let Some(obj) = v.AsObject() {
        let (val, _) = obj.Get(key);
        if let Some(n) = val.AsNumber() {
            return n;
        }
    }
    0.0
}

fn jget(v: &json::Value, key: &str) -> json::Value {
    if let Some(obj) = v.AsObject() {
        let (val, _) = obj.Get(key);
        return val;
    }
    json::Value::Null
}

// ---- per-run readers ----

// loadRunMeta reads a run's run.json identity header.
pub(crate) fn loadRunMeta(dir: string) -> (json::Value, bool) {
    loadJSON((dir) + ("/run.json"))
}

// metaParent is the parent run name recorded in run.json ("" if none).
pub(crate) fn metaParent(meta: &json::Value) -> string {
    jstr(meta, "parent")
}

// Headline is what a run measured, for tables and diffs. Zero values
// mean the artifact is absent.
#[derive(Clone, Default)]
pub(crate) struct Headline {
    pub(crate) Curve: alloc::vec::Vec<(int, float64, float64, float64)>, // concurrency, per-stream, total, ttft ms
    pub(crate) PeakRunning: float64,
    pub(crate) PeakKvPct: float64,
    pub(crate) Preemptions: float64,
    pub(crate) MaxNumSeqs: float64,
    pub(crate) BoundBy: string,
}

pub(crate) fn readHeadline(dir: string) -> Headline {
    let mut h: Headline = Default::default();
    let (v, ok) = loadJSON((dir) + ("/sweep.json"));
    if !ok {
        return h;
    }
    if let Some(curve) = jget(&v, "curve").AsArray() {
        for (_, c) in range!(curve.clone()) {
            h.Curve.push((
                int(jnum(&c, "concurrency")),
                jnum(&c, "perUser"),
                jnum(&c, "aggregate"),
                jnum(&c, "ttftMs"),
            ));
        }
    }
    let p = jget(&v, "pressure");
    h.PeakRunning = jnum(&p, "peakRunning");
    h.PeakKvPct = jnum(&p, "peakKvPct");
    h.Preemptions = jnum(&p, "preemptions");
    h.MaxNumSeqs = jnum(&p, "maxNumSeqs");
    h.BoundBy = jstr(&p, "boundBy");
    h
}

// readConfig returns (vllm version, resolved config KVs) from the
// run's captured vllm-args.txt (empty when the run predates capture).
pub(crate) fn readConfig(dir: string) -> (string, slice<profile::vllmcfg::KV>) {
    let empty: slice<profile::vllmcfg::KV> = make!([]profile::vllmcfg::KV, 0);
    let (data, err) = os::ReadFile((dir) + ("/vllm-args.txt"));
    if err != nil {
        return (string(""), empty);
    }
    let mut ver = string("");
    let mut resolved = empty.clone();
    let mut explicit = empty.clone();
    for (_, line) in range!(strings::Split(string(data), "\n")) {
        if strings::Contains(line.clone(), string("Initializing a V1 LLM engine")) {
            ver = profile::vllmcfg::EngineVersion(line.clone());
            resolved = profile::vllmcfg::ParseResolved(line.clone());
        }
        if strings::Contains(line.clone(), string("non-default args")) {
            explicit = profile::vllmcfg::ParseNonDefault(line.clone());
        }
    }
    // the resolved line wins; explicit flags fill keys it lacks
    // (e.g. max_num_seqs appears only in the non-default dict)
    for (_, kv) in range!(explicit) {
        let mut have = false;
        for (_, r) in range!(resolved.clone()) {
            if r.Key == kv.Key {
                have = true;
                break;
            }
        }
        if !have {
            resolved = append!(resolved.clone(), kv.clone());
        }
    }
    (ver, resolved)
}

// deltaString renders run.json's delta object as "k=v k=v".
fn deltaString(meta: &json::Value) -> string {
    let d = jget(meta, "delta");
    if let Some(obj) = d.AsObject() {
        let mut parts: slice<string> = make!([]string, 0);
        for (_, k) in range!(obj.Keys()) {
            let (val, _) = obj.Get(k.clone());
            let mut vs = string("");
            if let Some(s) = val.AsString() {
                vs = s.clone();
            }
            parts = append!(parts.clone(), fmt::Sprintf!("%s=%s", k, vs));
        }
        if parts.Len() > 0 {
            return strings::Join(parts, " ");
        }
    }
    string("")
}

fn fmtNum(x: float64) -> string {
    fmt::Sprintf!("%v", x)
}

fn lsCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("ls"),
        Aliases: slicemac!([]string{"list"}),
        Short: string("List collected runs with their headline numbers"),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, _args: slice<string>| -> error {
                let (root, _) = cmd.Flags().GetString("root");
                let names = runNames(root.clone());
                if names.Len() == 0 {
                    fmt::Printf!("no runs under %s yet; kvlm profile run collects the first one\n", root);
                    return nil.into();
                }
                let mut tw = tabwriter::NewWriter(os::Stdout(), 8, 1, 3, b' ', 0);
                fmt::Fprintf!(tw, "RUN\tPARENT\tDELTA\tMODEL\tVLLM\tPER-STREAM@32\tTOTAL@32\tBOUND BY\n");
                for (_, name) in range!(names) {
                    let dir = (root.clone()) + ("/") + (name.clone());
                    let (meta, _) = loadJSON((dir.clone()) + ("/run.json"));
                    let h = readHeadline(dir.clone());
                    let (ver, _) = readConfig(dir.clone());
                    let mut parent = jstr(&meta, "parent");
                    if parent == "" {
                        parent = string("-");
                    }
                    let mut delta = deltaString(&meta);
                    if delta == "" {
                        delta = string("-");
                    }
                    let mut modelCol = jstr(&meta, "model");
                    let variant = jstr(&meta, "variant");
                    if modelCol != "" && variant != "" {
                        modelCol = (modelCol) + (" ") + (variant);
                    }
                    if modelCol == "" {
                        modelCol = string("-");
                    }
                    let mut verCol = ver;
                    if verCol == "" {
                        verCol = string("-");
                    }
                    let mut per32 = string("-");
                    let mut tot32 = string("-");
                    for (conc, perUser, aggregate, _) in h.Curve.iter() {
                        if *conc == 32 {
                            per32 = fmtNum(*perUser);
                            tot32 = fmtNum(*aggregate);
                        }
                    }
                    let mut bound = h.BoundBy.clone();
                    if bound == "" {
                        bound = string("-");
                    }
                    fmt::Fprintf!(
                        tw,
                        "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                        name,
                        parent,
                        delta,
                        modelCol,
                        verCol,
                        per32,
                        tot32,
                        bound
                    );
                }
                let _ = tw.Flush();
                nil.into()
            },
        )),
        ..Default::default()
    };
    let _ = c.Flags().String_flag(string("root"), string(defaultRoot), string("directory holding the runN revisions"));
    c
}

fn showCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("show [run]"),
        Short: string("Show one run: identity, measurements, verdicts"),
        Args: Some(cobra::MaximumNArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (root, _) = cmd.Flags().GetString("root");
                let mut reference = string("latest");
                if args.Len() > 0 {
                    reference = args[0usize].clone();
                }
                let (dir, err) = ResolveRun(root, reference);
                if err != nil {
                    return err;
                }
                let name = path::Base(dir.clone());
                let (meta, haveMeta) = loadJSON((dir.clone()) + ("/run.json"));

                let mut header = name.clone();
                if haveMeta {
                    let parent = jstr(&meta, "parent");
                    let delta = deltaString(&meta);
                    if parent != "" && delta != "" {
                        header = fmt::Sprintf!("%s  (parent %s, + %s)", name.clone(), parent, delta);
                    } else if parent != "" {
                        header = fmt::Sprintf!("%s  (parent %s)", name.clone(), parent);
                    }
                }
                fmt::Println!(header);
                if haveMeta {
                    let mut served = jstr(&meta, "model");
                    let variant = jstr(&meta, "variant");
                    if served != "" && variant != "" {
                        served = (served) + (" ") + (variant);
                    }
                    if served != "" {
                        fmt::Printf!(
                            "served    %s on %vx %s (%s driver, %s mode), collected %s\n",
                            served,
                            jnum(&meta, "gpuCount"),
                            jstr(&meta, "gpuType"),
                            jstr(&meta, "driver"),
                            jstr(&meta, "mode"),
                            jstr(&meta, "collected")
                        );
                    }
                }
                let (ver, resolved) = readConfig(dir.clone());
                if ver != "" {
                    let mut flags: slice<string> = make!([]string, 0);
                    for (_, key) in range!(profile::vllmcfg::RelevantKeys()) {
                        for (_, kv) in range!(resolved.clone()) {
                            if kv.Key == key {
                                flags = append!(flags.clone(), fmt::Sprintf!("%s=%s", kv.Key.clone(), kv.Value.clone()));
                                break;
                            }
                        }
                    }
                    fmt::Printf!("config    vLLM %s: %s\n", ver, strings::Join(flags, " "));
                }
                let h = readHeadline(dir.clone());
                if h.Curve.len() > 0 {
                    fmt::Println!("curve");
                    fmt::Println!("  streams   per-stream     total      TTFT");
                    for (conc, perUser, aggregate, ttft) in h.Curve.iter() {
                        fmt::Printf!(
                            "  %s%s tok/s%s tok/s%s ms\n",
                            profcmd::padLeft(fmt::Sprintf!("%d", *conc), 7),
                            profcmd::padLeft(fmtNum(*perUser), 9),
                            profcmd::padLeft(fmtNum(*aggregate), 6),
                            profcmd::padLeft(fmtNum(*ttft), 6)
                        );
                    }
                }
                if h.BoundBy != "" {
                    fmt::Printf!(
                        "pressure  peak running %s, KV %s%%, preemptions %s, bound by %s (cap %s)\n",
                        fmtNum(h.PeakRunning),
                        fmtNum(h.PeakKvPct),
                        fmtNum(h.Preemptions),
                        h.BoundBy.clone(),
                        fmtNum(h.MaxNumSeqs)
                    );
                }
                let (g, haveGraph) = loadJSON((dir.clone()) + ("/graph-structure.json"));
                if haveGraph {
                    let chain = jget(&g, "chain");
                    let window = jget(&g, "window");
                    if !chain.IsNull() {
                        fmt::Printf!(
                            "graph     %v nodes, one replay %v ms, GPU busy %v%% of the window\n",
                            jnum(&chain, "nodes"),
                            math::Round(jnum(&chain, "replayUs") / 100.0) / 10.0,
                            jnum(&window, "busyPct")
                        );
                    }
                }
                profcmd::printRecommendation(dir.clone());
                fmt::Printf!("artifacts %s\n", dir);
                nil.into()
            },
        )),
        ..Default::default()
    };
    let _ = c.Flags().String_flag(string("root"), string(defaultRoot), string("directory holding the runN revisions"));
    c
}

fn diffCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("diff <run> <run>"),
        Short: string("Diff two runs: flags first, then measurements"),
        Long: string(
            "Diff two revisions. Flags come first because they are the cause:\n\
             every relevant config key whose resolved value differs. Then the\n\
             measured effect: the concurrency curve side by side and the\n\
             pressure verdicts.",
        ),
        Args: Some(cobra::ExactArgs(2)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (root, _) = cmd.Flags().GetString("root");
                let (dirA, err) = ResolveRun(root.clone(), args[0usize].clone());
                if err != nil {
                    return err;
                }
                let (dirB, err) = ResolveRun(root, args[1usize].clone());
                if err != nil {
                    return err;
                }
                let nameA = path::Base(dirA.clone());
                let nameB = path::Base(dirB.clone());

                let (verA, resA) = readConfig(dirA.clone());
                let (verB, resB) = readConfig(dirB.clone());
                let mut flagRows: slice<string> = make!([]string, 0);
                if verA != "" && verB != "" {
                    if verA != verB {
                        flagRows = append!(flagRows.clone(), fmt::Sprintf!("  vllm: %s -> %s", verA.clone(), verB.clone()));
                    }
                    for (_, key) in range!(profile::vllmcfg::RelevantKeys()) {
                        let mut a = string("");
                        let mut b = string("");
                        for (_, kv) in range!(resA.clone()) {
                            if kv.Key == key {
                                a = kv.Value.clone();
                                break;
                            }
                        }
                        for (_, kv) in range!(resB.clone()) {
                            if kv.Key == key {
                                b = kv.Value.clone();
                                break;
                            }
                        }
                        if a != b {
                            flagRows = append!(flagRows.clone(), fmt::Sprintf!("  %s: %s -> %s", key, a, b));
                        }
                    }
                }
                if flagRows.Len() > 0 {
                    fmt::Printf!("flags, %s -> %s\n", nameA.clone(), nameB.clone());
                    for (_, r) in range!(flagRows) {
                        fmt::Println!(r);
                    }
                } else if verA != "" && verB != "" {
                    fmt::Printf!("flags: no relevant differences between %s and %s\n", nameA.clone(), nameB.clone());
                } else {
                    fmt::Println!("flags: config not captured in one of the runs");
                }

                let ha = readHeadline(dirA);
                let hb = readHeadline(dirB);
                if ha.Curve.len() > 0 && hb.Curve.len() > 0 {
                    fmt::Println!("curve");
                    fmt::Printf!(
                        "  streams   %s per-stream   %s per-stream     delta\n",
                        nameA.clone(),
                        nameB.clone()
                    );
                    for (conc, perA, _, _) in ha.Curve.iter() {
                        for (concB, perB, _, _) in hb.Curve.iter() {
                            if *conc != *concB {
                                continue;
                            }
                            let mut deltaCol = string("flat");
                            if *perA > 0.0 {
                                let pct = math::Round(1000.0 * (*perB - *perA) / *perA) / 10.0;
                                if pct >= 1.0 {
                                    deltaCol = fmt::Sprintf!("+%v%%", pct);
                                } else if pct <= -1.0 {
                                    deltaCol = fmt::Sprintf!("%v%%", pct);
                                }
                            }
                            fmt::Printf!(
                                "  %s%s tok/s%s tok/s%s\n",
                                profcmd::padLeft(fmt::Sprintf!("%d", *conc), 7),
                                profcmd::padLeft(fmtNum(*perA), 12),
                                profcmd::padLeft(fmtNum(*perB), 15),
                                profcmd::padLeft(deltaCol, 10)
                            );
                        }
                    }
                }
                if ha.BoundBy != "" && hb.BoundBy != "" {
                    fmt::Printf!(
                        "pressure  %s: bound by %s (running %s, KV %s%%); %s: bound by %s (running %s, KV %s%%)\n",
                        nameA,
                        ha.BoundBy.clone(),
                        fmtNum(ha.PeakRunning),
                        fmtNum(ha.PeakKvPct),
                        nameB,
                        hb.BoundBy.clone(),
                        fmtNum(hb.PeakRunning),
                        fmtNum(hb.PeakKvPct)
                    );
                }
                nil.into()
            },
        )),
        ..Default::default()
    };
    let _ = c.Flags().String_flag(string("root"), string(defaultRoot), string("directory holding the runN revisions"));
    c
}

fn graphCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("graph [run]"),
        Short: string("Re-analyze a run's capture into graph-structure.json"),
        Args: Some(cobra::MaximumNArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (root, _) = cmd.Flags().GetString("root");
                let mut reference = string("latest");
                if args.Len() > 0 {
                    reference = args[0usize].clone();
                }
                let (dir, err) = ResolveRun(root, reference);
                if err != nil {
                    return err;
                }
                let mut input = (dir.clone()) + ("/win.sqlite");
                let (_, serr) = os::ReadFile(input.clone());
                if serr != nil {
                    input = (dir.clone()) + ("/win.nsys-rep");
                    let (_, rerr) = os::ReadFile(input.clone());
                    if rerr != nil {
                        return fmt::Errorf!("no capture in %s (win.sqlite or win.nsys-rep); the run may have been packed with --basic", dir);
                    }
                }
                let mut gpuName = string("");
                let (envData, err) = os::ReadFile((dir.clone()) + ("/env.txt"));
                if err == nil {
                    for (_, line) in range!(strings::Split(string(envData), "\n")) {
                        let (spec, ok) = model::LookupGPU(line.clone());
                        if ok {
                            gpuName = spec.Name.clone();
                            break;
                        }
                    }
                }
                profcmd::generateGraph(
                    input,
                    (dir.clone()) + ("/graph-structure.json"),
                    gpuName,
                    fmt::Sprintf!("re-analyzed by kvlm run graph, %s", path::Base(dir.clone())),
                )
            },
        )),
        ..Default::default()
    };
    let _ = c.Flags().String_flag(string("root"), string(defaultRoot), string("directory holding the runN revisions"));
    c
}

fn flamegraphCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("flamegraph [run]"),
        Short: string("Render the run's GPU-time flamegraph SVG"),
        Args: Some(cobra::MaximumNArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (root, _) = cmd.Flags().GetString("root");
                let mut reference = string("latest");
                if args.Len() > 0 {
                    reference = args[0usize].clone();
                }
                let (dir, err) = ResolveRun(root, reference);
                if err != nil {
                    return err;
                }
                let folded = (dir.clone()) + ("/gpu-folded.txt");
                let (data, err) = os::ReadFile(folded.clone());
                if err != nil {
                    return fmt::Errorf!("no folded stacks in %s; kvlm run graph %s produces them from the capture", dir, path::Base(dir.clone()));
                }
                let (fg, err) = flamegraph::ParseFolded(string(data));
                if err != nil {
                    return err;
                }
                let mut fg = fg;
                fg.Title = fmt::Sprintf!("GPU time, %s", path::Base(dir.clone()));
                let out = (dir.clone()) + ("/gpu-time.svg");
                let err = os::WriteFile(out.clone(), flamegraph::ToSVG(&fg), 0o644);
                if err != nil {
                    return fmt::Errorf!("write %s: %v", out, err);
                }
                fmt::Printf!("wrote %s (%d samples)\n", out, fg.Total);
                nil.into()
            },
        )),
        ..Default::default()
    };
    let _ = c.Flags().String_flag(string("root"), string(defaultRoot), string("directory holding the runN revisions"));
    c
}

fn rmCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("rm <run>..."),
        Short: string("Delete run directories"),
        Args: Some(cobra::MinimumNArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (root, _) = cmd.Flags().GetString("root");
                for (_, reference) in range!(args) {
                    let (dir, err) = ResolveRun(root.clone(), reference.clone());
                    if err != nil {
                        return err;
                    }
                    let err = os::RemoveAll(dir.clone());
                    if err != nil {
                        return fmt::Errorf!("remove %s: %v", dir, err);
                    }
                    fmt::Printf!("removed %s\n", dir);
                }
                nil.into()
            },
        )),
        ..Default::default()
    };
    let _ = c.Flags().String_flag(string("root"), string(defaultRoot), string("directory holding the runN revisions"));
    c
}

// shipArgv extracts the serve argv a run actually ran with, minus the
// profiler flag: the tokens after "serve" in the captured argv line.
fn shipArgv(dir: string) -> (slice<string>, bool) {
    let empty: slice<string> = make!([]string, 0);
    let (data, err) = os::ReadFile((dir) + ("/vllm-args.txt"));
    if err != nil {
        return (empty, false);
    }
    for (_, line) in range!(strings::Split(string(data), "\n")) {
        if !strings::HasPrefix(line.clone(), string("argv: ")) {
            continue;
        }
        let mut out: slice<string> = make!([]string, 0);
        let mut started = false;
        let mut skippingProfiler = false;
        for (_, tok) in range!(strings::Fields(strings::TrimPrefix(line.clone(), string("argv: ")))) {
            if !started {
                if tok == "serve" {
                    started = true;
                }
                continue;
            }
            if tok == "--profiler-config" {
                skippingProfiler = true;
                continue;
            }
            if skippingProfiler {
                // the profiler value is json that spaces split apart;
                // it ends where the next --flag begins
                if !strings::HasPrefix(tok.clone(), string("--")) {
                    continue;
                }
                skippingProfiler = false;
            }
            out = append!(out.clone(), tok.clone());
        }
        if out.Len() > 0 {
            return (out, true);
        }
    }
    (empty, false)
}

// deltaOrigins walks a run's ancestry and maps each flag override to
// the run whose delta introduced it (later runs win).
fn deltaOrigins(root: string, name: string) -> map<string, string> {
    let mut origins: map<string, string> = map::new_no_zero();
    // chain from oldest to newest so newer deltas overwrite
    let mut chain: slice<string> = make!([]string, 0);
    let mut cur = name;
    let mut hops = 0;
    while cur != "" && hops < 100 {
        chain = append!(chain.clone(), cur.clone());
        let (meta, ok) = loadRunMeta((root.clone()) + ("/") + (cur.clone()));
        if !ok {
            break;
        }
        cur = metaParent(&meta);
        hops += 1;
    }
    let mut i = chain.Len() - 1;
    while i >= 0 {
        let rn = chain[i as usize].clone();
        let (meta, ok) = loadRunMeta((root.clone()) + ("/") + (rn.clone()));
        if ok {
            let d = jget(&meta, "delta");
            if let Some(obj) = d.AsObject() {
                for (_, k) in range!(obj.Keys()) {
                    origins.Set(("--") + (k.clone()), rn.clone());
                }
            }
        }
        i -= 1;
    }
    origins
}

fn whyCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("why [run]"),
        Short: string("Show the evidence behind a run's verdict, number by number"),
        Args: Some(cobra::MaximumNArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (root, _) = cmd.Flags().GetString("root");
                let mut reference = string("latest");
                if args.Len() > 0 {
                    reference = args[0usize].clone();
                }
                let (dir, err) = ResolveRun(root, reference);
                if err != nil {
                    return err;
                }
                let name = path::Base(dir.clone());
                let h = readHeadline(dir.clone());
                if h.BoundBy == "" {
                    return fmt::Errorf!("%s has no pressure data (sweep.json); a full kvlm profile run collects it", name);
                }
                fmt::Printf!("%s verdict: bound by %s\n\n", name, h.BoundBy.clone());
                fmt::Printf!("  peak running   %s\tsweep.json pressure.peakRunning\n", fmtNum(h.PeakRunning));
                fmt::Printf!("  max_num_seqs   %s\tvllm-args.txt, resolved config\n", fmtNum(h.MaxNumSeqs));
                fmt::Printf!("  peak KV        %s%%\tsweep.json pressure.peakKvPct\n", fmtNum(h.PeakKvPct));
                fmt::Printf!("  preemptions    %s\tmetrics-after minus metrics-before\n\n", fmtNum(h.Preemptions));
                if h.BoundBy == "max-num-seqs" {
                    fmt::Println!("  running equals the cap: the scheduler admitted everything it is allowed to; KV had room, so the flag is the wall");
                } else if h.BoundBy == "kv" {
                    fmt::Println!("  running stayed under the cap while KV crossed 90% or requests were preempted: the pool is the wall, not the scheduler");
                } else {
                    fmt::Println!("  neither the cap nor KV saturated: the offered load ran out before the server did");
                }
                let (logData, lerr) = os::ReadFile((dir.clone()) + ("/server-log.txt"));
                if lerr == nil {
                    for (_, line) in range!(strings::Split(string(logData), "\n")) {
                        if strings::Contains(line.clone(), "Maximum concurrency") {
                            fmt::Printf!("\n  the pool's own ceiling (server-log.txt): %s\n", strings::TrimSpace(line));
                            break;
                        }
                    }
                }
                nil.into()
            },
        )),
        ..Default::default()
    };
    let _ = c.Flags().String_flag(string("root"), string(defaultRoot), string("directory holding the runN revisions"));
    c
}

fn shipCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("ship [run]"),
        Short: string("Print the serve command a run earned; --up deploys it"),
        Long: string(
            "Print the vllm serve command the named run (default: latest)\n\
             actually ran with, profiler flag stripped: the flags the tuning\n\
             loop arrived at, ready for production. Flags introduced by a\n\
             kvlm apply are annotated with the run that measured them.\n\
             --up deploys a fresh production pod serving exactly this.",
        ),
        Args: Some(cobra::MaximumNArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (root, _) = cmd.Flags().GetString("root");
                let mut reference = string("latest");
                if args.Len() > 0 {
                    reference = args[0usize].clone();
                }
                let (dir, err) = ResolveRun(root.clone(), reference);
                if err != nil {
                    return err;
                }
                let name = path::Base(dir.clone());
                let (argv, ok) = shipArgv(dir.clone());
                if !ok {
                    return fmt::Errorf!("%s carries no captured argv (vllm-args.txt); runs collected by kvlm profile run have it", name);
                }
                let origins = deltaOrigins(root, name.clone());

                fmt::Printf!("%s served:\n\n  vllm serve", name);
                let mut i: int = 0;
                while i < argv.Len() {
                    let tok = argv[i as usize].clone();
                    if strings::HasPrefix(tok.clone(), string("--")) {
                        let mut lineOut = fmt::Sprintf!(" \\\n    %s", tok.clone());
                        if i + 1 < argv.Len() && !strings::HasPrefix(argv[(i + 1) as usize].clone(), string("--")) {
                            lineOut = (lineOut) + (" ") + (argv[(i + 1) as usize].clone());
                            i += 1;
                        }
                        if origins.Has(tok.clone()) {
                            let (from, _) = origins.Get(tok.clone());
                            lineOut = (lineOut) + (fmt::Sprintf!("   # measured in %s", from));
                        }
                        fmt::Print!(lineOut);
                    } else {
                        fmt::Print!((" ") + (tok));
                    }
                    i += 1;
                }
                fmt::Println!("");

                let (up, _) = cmd.Flags().GetBool("up");
                if !up {
                    fmt::Println!("\ndeploy it in production mode: kvlm ship --up");
                    return nil.into();
                }
                let (meta, haveMeta) = loadRunMeta(dir);
                if !haveMeta {
                    return fmt::Errorf!("%s has no run.json, so the pod shape is unknown; deploy by hand with kvlm up and the flags above", name);
                }
                let modelName = jstr(&meta, "model");
                let variantName = jstr(&meta, "variant");
                let (_, m, mok) = model::Find(modelName.clone());
                if !mok {
                    return fmt::Errorf!("run.json names model %q, which is not in the catalog", modelName);
                }
                let mut image = string("");
                for (_, v) in range!(m.Variants.clone()) {
                    if v.Name == variantName {
                        image = v.Image.clone();
                        break;
                    }
                }
                if image == "" {
                    return fmt::Errorf!("no production image for %s %s in the catalog", modelName, variantName);
                }
                let (d, creds, err) = crate::driver::FromCommand(cmd);
                if err != nil {
                    return err;
                }
                let mut serveArgv: slice<string> = make!([]string, 0);
                serveArgv = append!(serveArgv.clone(), string("vllm"));
                serveArgv = append!(serveArgv.clone(), string("serve"));
                for (_, a) in range!(argv.clone()) {
                    serveArgv = append!(serveArgv.clone(), a.clone());
                }
                let mut cudas: slice<string> = make!([]string, 0);
                cudas = append!(cudas.clone(), string("13.0"));
                let (volumeID, _) = cmd.Flags().GetString("volume");
                let opts = crate::driver::Options {
                    Model: modelName,
                    Runtime: string("vllm"),
                    Image: image,
                    GPUType: jstr(&meta, "gpuType"),
                    GPUCount: int(jnum(&meta, "gpuCount")),
                    CudaVersions: cudas,
                    ServeArgv: serveArgv,
                    VolumeID: volumeID,
                    Variant: variantName,
                    Mode: string("production"),
                    VllmVersion: jstr(&meta, "vllmVersion"),
                    ..Default::default()
                };
                let (_, err) = d.unwrap().Up(&creds, &opts);
                err
            },
        )),
        ..Default::default()
    };
    let _ = c.Flags().String_flag(string("root"), string(defaultRoot), string("directory holding the runN revisions"));
    let _ = c.Flags().Bool_flag(string("up"), false, string("deploy a production pod serving exactly this"));
    let _ = c.Flags().String_flag(string("volume"), string(""), string("shared network volume id for the deploy"));
    c
}

fn runGroup() -> cobra::Command {
    cobra::Command {
        Use: string("run"),
        Short: string("Work with collected runs: list, show, diff, pack, import"),
        Long: string(
            "A run is one collected profiling revision under profile-output/.\n\
             These commands work with runs as durable objects: list them with\n\
             their headline numbers, show one, diff two (flags first, then\n\
             measurements), re-analyze captures, and pack or import .kvlm\n\
             archives. Run references: run7, bare 7, latest, -1 for one\n\
             back, or a directory path.",
        ),
        ..Default::default()
    }
}

// Go: func init() { rootCmd.AddCommand(runCmd) }
#[goish::init]
fn init() {
    let mut g = runGroup();
    g.AddCommand(lsCmd());
    g.AddCommand(showCmd());
    g.AddCommand(diffCmd());
    g.AddCommand(graphCmd());
    g.AddCommand(flamegraphCmd());
    let mut pack = profcmd::archiveCmd();
    pack.Use = string("pack <run>");
    pack.Aliases = slicemac!([]string{"archive", "export"});
    g.AddCommand(pack);
    let mut imp = profcmd::importCmd();
    imp.Use = string("import <file.kvlm>");
    g.AddCommand(imp);
    g.AddCommand(rmCmd());
    rootCmd.Lock().AddCommand(g);
    rootCmd.Lock().AddCommand(whyCmd());
    rootCmd.Lock().AddCommand(shipCmd());
}
