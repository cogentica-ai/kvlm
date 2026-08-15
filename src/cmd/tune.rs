// tune: one iteration of the loop, as one verb. A tune is a full
// profile run plus a goal: collect everything, then say where the
// goal stands against the parent revision. The recommendation block
// the collection prints is the next move; kvlm apply takes it.
#![allow(non_snake_case)]

use goish::encoding::json;
use goish::fmt;
use goish::math;
use goish::os;
use goish::strconv;
use goish::string;
use goish::strings;
use goish::gomap::map;
use goish::goslice::slice;
use goish::errors::error;
use goish::{float64, int, nil, range};

use spf13_cobra as cobra;

use crate::cmd::profile as profcmd;
use crate::cmd::rootCmd;
use crate::cmd::run as runcmd;

// Goal is "metric@concurrency": total@32 or per-stream@32. Floor adds
// "=value": per-stream@32=20.
#[derive(Clone, Default)]
struct goal {
    Metric: string, // "total" or "per-stream"
    Conc: int,
    Value: float64, // floors only
}

fn parseGoal(s: string, withValue: bool) -> (goal, error) {
    let mut g: goal = Default::default();
    let mut body = s.clone();
    if withValue {
        let eq = strings::Index(body.clone(), string("="));
        if eq <= 0 {
            return (g, fmt::Errorf!("floor %q needs =value (e.g. per-stream@32=20)", s));
        }
        let (v, err) = strconv::ParseFloat(body.slice(eq + 1, body.Len()), 64);
        if err != nil {
            return (g, fmt::Errorf!("floor value in %q is not a number", s));
        }
        g.Value = v;
        body = body.slice(0, eq);
    }
    let at = strings::Index(body.clone(), string("@"));
    if at <= 0 {
        return (g, fmt::Errorf!("%q is not metric@concurrency (total@32, per-stream@32)", s));
    }
    g.Metric = body.slice(0, at);
    if g.Metric != "total" && g.Metric != "per-stream" {
        let m = g.Metric.clone();
        return (g, fmt::Errorf!("metric %q must be total or per-stream", m));
    }
    let (conc, err) = strconv::Atoi(body.slice(at + 1, body.Len()));
    if err != nil || conc < 1 {
        return (g, fmt::Errorf!("concurrency in %q is not a positive integer", s));
    }
    g.Conc = conc;
    (g, nil.into())
}

fn goalString(g: &goal) -> string {
    fmt::Sprintf!("%s@%d", g.Metric.clone(), g.Conc)
}

// measure reads the goal metric out of a run's curve (ok=false when
// the run has no row at that concurrency).
fn measure(dir: string, g: &goal) -> (float64, bool) {
    let h = runcmd::readHeadline(dir);
    for (conc, perUser, aggregate, _) in h.Curve.iter() {
        if *conc != g.Conc {
            continue;
        }
        if g.Metric == "total" {
            return (*aggregate, true);
        }
        return (*perUser, true);
    }
    (0.0, false)
}

const sessionPath: &str = "./profile-output/session.json";

fn loadSession() -> (string, string) {
    let (data, err) = os::ReadFile(string(sessionPath));
    if err != nil {
        return (string(""), string(""));
    }
    let mut v = json::Value::Null;
    let perr = json::Unmarshal(data.as_ref(), &mut v);
    if perr != nil {
        return (string(""), string(""));
    }
    let mut g = string("");
    let mut f = string("");
    if let Some(obj) = v.AsObject() {
        let (gv, _) = obj.Get("goal");
        if let Some(s) = gv.AsString() {
            g = s.clone();
        }
        let (fv, _) = obj.Get("floor");
        if let Some(s) = fv.AsString() {
            f = s.clone();
        }
    }
    (g, f)
}

fn saveSession(goalStr: string, floorStr: string) {
    let mut obj: map<string, json::Value> = map::new_no_zero();
    obj.Set(string("goal"), json::Value::String(goalStr));
    obj.Set(string("floor"), json::Value::String(floorStr));
    let v = json::Value::Object(obj);
    let (out, err) = json::MarshalIndent(&v, "", "  ");
    if err != nil {
        return;
    }
    let _ = os::MkdirAll(string("./profile-output"), 0o755);
    let _ = os::WriteFile(string(sessionPath), out, 0o644);
}

fn tuneCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("tune"),
        Short: string("One tuning iteration: collect, verdict, recommendation"),
        Long: string(
            "One iteration of the tuning loop against the recorded target:\n\
             collect a full profiling run (the same collection kvlm profile\n\
             run does), then judge the goal metric against the parent\n\
             revision. The goal persists in profile-output/session.json, so\n\
             it is set once: kvlm tune --goal total@32 --floor\n\
             per-stream@32=20, then just kvlm tune after each kvlm apply.",
        ),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, _args: slice<string>| -> error {
                let (goalFlag, _) = cmd.Flags().GetString("goal");
                let (floorFlag, _) = cmd.Flags().GetString("floor");
                let (savedGoal, savedFloor) = loadSession();
                let mut goalStr = goalFlag.clone();
                if goalStr == "" {
                    goalStr = savedGoal.clone();
                }
                let mut floorStr = floorFlag.clone();
                if floorStr == "" {
                    floorStr = savedFloor.clone();
                }
                if goalStr == "" {
                    return fmt::Errorf!("no goal set: kvlm tune --goal total@32 (add --floor per-stream@32=20 to guard latency); it persists for later tunes");
                }
                let (g, err) = parseGoal(goalStr.clone(), false);
                if err != nil {
                    return err;
                }
                let mut floor: goal = Default::default();
                let mut haveFloor = false;
                if floorStr != "" {
                    let (f, err) = parseGoal(floorStr.clone(), true);
                    if err != nil {
                        return err;
                    }
                    floor = f;
                    haveFloor = true;
                }
                if goalFlag != "" && savedGoal != "" && goalFlag != savedGoal {
                    fmt::Printf!("goal changed: %s (was %s)\n", goalStr.clone(), savedGoal.clone());
                }
                saveSession(goalStr.clone(), floorStr.clone());

                let err = profcmd::runAll(cmd);
                if err != nil {
                    return err;
                }

                // judge the goal on the run that just landed, against
                // its recorded parent
                let (root, _) = cmd.Flags().GetString("out");
                let (dir, rerr) = runcmd::ResolveRun(root.clone(), string("latest"));
                if rerr != nil {
                    return nil.into();
                }
                let (val, ok) = measure(dir.clone(), &g);
                if !ok {
                    fmt::Printf!("goal %s: the sweep has no %d-stream level, so the goal was not measured this run\n", goalString(&g), g.Conc);
                    return nil.into();
                }
                let mut vs = string("");
                let (meta, haveMeta) = runcmd::loadRunMeta(dir.clone());
                if haveMeta {
                    let parent = runcmd::metaParent(&meta);
                    if parent != "" {
                        let (pdir, perr) = runcmd::ResolveRun(root, parent.clone());
                        if perr == nil {
                            let (pval, pok) = measure(pdir, &g);
                            if pok && pval > 0.0 {
                                let pct = math::Round(1000.0 * (val - pval) / pval) / 10.0;
                                if pct >= 1.0 {
                                    vs = fmt::Sprintf!(", up %v%% from %s (%v)", pct, parent, pval);
                                } else if pct <= -1.0 {
                                    vs = fmt::Sprintf!(", down %v%% from %s (%v)", math::Abs(pct), parent, pval);
                                } else {
                                    vs = fmt::Sprintf!(", flat against %s (%v)", parent, pval);
                                }
                            }
                        }
                    }
                }
                fmt::Printf!("goal      %s = %v tok/s%s\n", goalString(&g), val, vs);
                if haveFloor {
                    let (fval, fok) = measure(dir, &floor);
                    if fok {
                        if fval >= floor.Value {
                            fmt::Printf!("floor     %s = %v, ok (need %v)\n", goalString(&floor), fval, floor.Value);
                        } else {
                            fmt::Printf!("floor     %s = %v, VIOLATED (need %v); the last apply traded too much per-stream speed\n", goalString(&floor), fval, floor.Value);
                        }
                    }
                }
                nil.into()
            },
        )),
        ..Default::default()
    };
    profcmd::addRunFlags(&mut c);
    let _ = c.Flags().String_flag(string("goal"), string(""), string("objective as metric@concurrency (total@32, per-stream@8); persists in the session"));
    let _ = c.Flags().String_flag(string("floor"), string(""), string("constraint as metric@concurrency=value (per-stream@32=20); a run below it is called out"));
    c
}

// Go: func init() { rootCmd.AddCommand(tuneCmd) }
#[goish::init]
fn init() {
    rootCmd.Lock().AddCommand(tuneCmd());
}
