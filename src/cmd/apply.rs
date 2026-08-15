// apply: the loop's fourth step. Restart the server kvlm launched
// with changed vLLM flags, recording the delta so the next collected
// run chains to its parent. Only ever touches a server kvlm itself
// launched in profile mode; anything else is refused, because a
// restart it did not arm is a restart it cannot vouch for.
#![allow(non_snake_case)]

use goish::encoding::base64;
use goish::encoding::json;
use goish::fmt;
use goish::os;
use goish::string;
use goish::strings;
use goish::time;
use goish::gomap::map;
use goish::goslice::slice;
use goish::errors::error;
use goish::{append, int, make, nil, range};

use spf13_cobra as cobra;

use crate::cmd::profile as profcmd;
use crate::cmd::rootCmd;
use crate::model;
use crate::profile;
use crate::state;

// parseDelta turns k=v args into dash-form flag overrides, validated
// against the pinned version's flag catalog when one was collected.
fn parseDelta(args: &slice<string>, vllmVersion: string) -> (map<string, string>, error) {
    let mut delta: map<string, string> = map::new_no_zero();
    let (catalog, haveCatalog) = profile::vllmflags::Load(vllmVersion);
    for (_, a) in range!(args.clone()) {
        let idx = strings::Index(a.clone(), string("="));
        if idx <= 0 {
            return (delta, fmt::Errorf!("%q is not flag=value (e.g. max-num-seqs=64)", a.clone()));
        }
        let mut key = a.slice(0, idx);
        let value = a.slice(idx + 1, a.Len());
        key = strings::TrimPrefix(key, string("--"));
        let dashed = strings::ReplaceAll(key.clone(), "_", "-");
        if dashed == "tp" {
            delta.Set(string("tensor-parallel-size"), value);
            continue;
        }
        if haveCatalog {
            let underscored = strings::ReplaceAll(dashed.clone(), "-", "_");
            let mut known = false;
            for (_, f) in range!(catalog.clone()) {
                if f.Key == underscored {
                    known = true;
                    break;
                }
            }
            if !known {
                return (
                    delta,
                    fmt::Errorf!(
                        "the collected vLLM catalog has no flag --%s (every flag it does have is in %s/)",
                        dashed,
                        string(profile::vllmflags::Dir)
                    ),
                );
            }
        }
        delta.Set(dashed, value);
    }
    if delta.Len() == 0 {
        return (delta, fmt::Errorf!("nothing to apply: pass flag=value pairs (kvlm profile run prints proposals)"));
    }
    (delta, nil.into())
}

// serveLineFor recomposes the serve line for the target: the catalog
// recipe, the overrides already applied, then the new delta.
fn serveLineFor(t: &state::Target, delta: &map<string, string>) -> (string, error) {
    let (_, m, ok) = model::Find(t.Model.clone());
    if !ok {
        return (string(""), fmt::Errorf!("the target serves %q, which is not in the catalog", t.Model.clone()));
    }
    let mut spec: model::ServeSpec = Default::default();
    let mut have = false;
    for (_, v) in range!(m.Variants.clone()) {
        if v.Name != t.Variant {
            continue;
        }
        let (s, ok) = model::DefaultServe(&v);
        if ok {
            spec = s;
            have = true;
        }
        break;
    }
    if !have {
        return (string(""), fmt::Errorf!("no serve recipe for %s %s in the catalog", t.Model.clone(), t.Variant.clone()));
    }
    for (_, k) in range!(t.Applied.Keys()) {
        let (v, _) = t.Applied.Get(k.clone());
        spec = profile::launch::WithFlag(&spec, ("--") + (k.clone()), v);
    }
    for (_, k) in range!(delta.Keys()) {
        let (v, _) = delta.Get(k.clone());
        spec = profile::launch::WithFlag(&spec, ("--") + (k.clone()), v);
    }
    (profile::launch::ServeLine(&spec, t.GPUCount, true), nil.into())
}

// tailOf fetches the last n lines of a remote file for failure
// evidence, so an error names its cause instead of pointing at a
// command the user has to run next.
fn tailOf(tr: &profcmd::transport, path: string, n: int) -> string {
    let (out, err) = tr.exec(fmt::Sprintf!("tail -%d %s 2>/dev/null", n, path));
    if err != nil {
        return string("");
    }
    strings::TrimSpace(out)
}

// lastLogLine is the server log's most recent non-empty line, tail
// truncated, for progress reporting during the engine re-init.
fn lastLogLine(tr: &profcmd::transport) -> string {
    let (out, err) = tr.exec(fmt::Sprintf!(
        "tail -c 4000 %s 2>/dev/null | grep -a . | tail -1",
        string(profile::launch::ServerLogPath)
    ));
    if err != nil {
        return string("");
    }
    let line = strings::TrimSpace(out);
    if line.Len() > 140 {
        return line.slice(line.Len() - 140, line.Len());
    }
    line
}

// waitServe polls the restarted server until it answers. Weights are
// cached, but a flag change makes the engine re-profile memory and
// recapture CUDA graphs, so the honest budget is minutes (a 400 s
// cap was measured expiring on a healthy restart). While waiting it
// distinguishes the ways a restart actually fails: engine init
// failure, the process dying (a lingering nsys session was one
// measured cause), and the pod itself going away.
fn waitServe(tr: &profcmd::transport) -> error {
    let mut sshFails = 0;
    let mut i = 0;
    while i < 90 {
        time::Sleep(time::Seconds(10));
        let (code, err) = tr.exec(string(
            "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null",
        ));
        if err != nil {
            // a streak means the pod may be gone — but a ~2 min sshd
            // bounce on a healthy pod was also measured, so require
            // several minutes of silence before calling it
            sshFails += 1;
            if sshFails >= 18 {
                return fmt::Errorf!(
                    "ssh to the pod stopped answering for minutes during the restart; the pod may have been terminated externally. kvlm ps reconciles against the platform; if it still shows RUNNING, re-run the same kvlm apply"
                );
            }
            i += 1;
            continue;
        }
        sshFails = 0;
        if strings::TrimSpace(code) == "200" {
            return nil.into();
        }
        let (probe, err) = tr.exec(fmt::Sprintf!(
            "grep -q 'Engine core initialization failed' %s 2>/dev/null && echo ENGINE_FAILED; grep -q RESTARTED %s 2>/dev/null && ! pgrep -f '[v]llm serve' >/dev/null && echo SERVER_DEAD; true",
            string(profile::launch::ServerLogPath),
            string(profile::launch::RestartLogPath)
        ));
        if err == nil {
            if strings::Contains(probe.clone(), "ENGINE_FAILED") {
                return fmt::Errorf!(
                    "the restarted server failed engine init:\n%s\nRoll the flag back with kvlm apply --reset",
                    tailOf(tr, string(profile::launch::ServerLogPath), 8)
                );
            }
            // i >= 2 gives the relaunch a grace window: right after
            // pkill there is legitimately no serve process yet
            if i >= 2 && strings::Contains(probe, "SERVER_DEAD") {
                return fmt::Errorf!(
                    "the server process exited during the restart.\nserver log:\n%s\nrestart script:\n%s\nRetry the same kvlm apply, or roll back with kvlm apply --reset",
                    tailOf(tr, string(profile::launch::ServerLogPath), 8),
                    tailOf(tr, string(profile::launch::RestartLogPath), 5)
                );
            }
        }
        if i > 0 && i % 6 == 0 {
            let line = lastLogLine(tr);
            if line != "" {
                fmt::Printf!("  still starting (%d s): %s\n", i * 10, line);
            }
        }
        i += 1;
    }
    fmt::Errorf!(
        "server not answering 15 minutes after the restart:\n%s\nIf it comes up later, re-run the same kvlm apply so the recorded flag state matches the server",
        tailOf(tr, string(profile::launch::ServerLogPath), 8)
    )
}

// stagePending records the delta for the next collected run's chain.
fn stagePending(delta: &map<string, string>) {
    let mut existing: slice<string> = make!([]string, 0);
    let (entries, err) = os::ReadDir(string("./profile-output"));
    if err == nil {
        for e in entries.iter() {
            if e.IsDir() {
                existing = append!(existing.clone(), e.Name());
            }
        }
    }
    let parent = profile::LatestRunName(existing);
    let mut dm: map<string, json::Value> = map::new_no_zero();
    for (_, k) in range!(delta.Keys()) {
        let (v, _) = delta.Get(k.clone());
        dm.Set(k.clone(), json::Value::String(v));
    }
    let mut obj: map<string, json::Value> = map::new_no_zero();
    obj.Set(string("parent"), json::Value::String(parent));
    obj.Set(string("delta"), json::Value::Object(dm));
    let v = json::Value::Object(obj);
    let (out, err) = json::MarshalIndent(&v, "", "  ");
    if err != nil {
        return;
    }
    let _ = os::MkdirAll(string("./profile-output"), 0o755);
    let _ = os::WriteFile(string("./profile-output/pending.json"), out, 0o644);
}

fn applyCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("apply [flag=value...]"),
        Short: string("Restart the target server with changed vLLM flags"),
        Long: string(
            "Restart the server kvlm launched, with vLLM flags changed on top\n\
             of the catalog recipe: kvlm apply max-num-seqs=64. The delta is\n\
             recorded so the next kvlm profile run chains to its parent with\n\
             the change named. Weights stay cached on the pod; the engine\n\
             still re-profiles memory and recaptures CUDA graphs, so the\n\
             cycle is a few minutes.\n\
             \n\
             Only a server kvlm launched in profile mode is ever touched;\n\
             production pods and foreign servers are refused. --reset drops\n\
             every applied override and returns to the catalog recipe.\n\
             --dry-run prints the serve line and changes nothing.",
        ),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (t, ok) = state::Current();
                if !ok {
                    return fmt::Errorf!("no target: kvlm apply restarts the pod kvlm up started (none is recorded)");
                }
                if t.Mode != "profile" || t.SSH == "" {
                    return fmt::Errorf!(
                        "the target (pod %s, %s mode) was not launched by kvlm over ssh; apply only restarts servers kvlm itself launched. Redeploy tunable with: kvlm up %s --mode profile",
                        t.Pod.clone(),
                        t.Mode.clone(),
                        t.Model.clone()
                    );
                }

                let (reset, _) = cmd.Flags().GetBool("reset");
                let mut delta: map<string, string> = map::new_no_zero();
                if reset {
                    if args.Len() > 0 {
                        return fmt::Errorf!("--reset takes no flag=value arguments");
                    }
                } else {
                    let (d, err) = parseDelta(&args, t.VllmVersion.clone());
                    if err != nil {
                        return err;
                    }
                    delta = d;
                }

                let mut base = t.clone();
                if reset {
                    base.Applied = map::new_no_zero();
                }
                let (serveLine, err) = serveLineFor(&base, &delta);
                if err != nil {
                    return err;
                }

                let mut deltaParts: slice<string> = make!([]string, 0);
                for (_, k) in range!(delta.Keys()) {
                    let (v, _) = delta.Get(k.clone());
                    deltaParts = append!(deltaParts.clone(), fmt::Sprintf!("--%s %s", k, v));
                }
                if reset {
                    fmt::Printf!("resetting to the catalog recipe on pod %s\n", t.Pod.clone());
                } else {
                    fmt::Printf!("pod %s: current flags + %s\n", t.Pod.clone(), strings::Join(deltaParts, " "));
                }
                fmt::Printf!("  %s\n", strings::Replace(serveLine.clone(), "exec ", "", 1));

                let (dryRun, _) = cmd.Flags().GetBool("dry-run");
                if dryRun {
                    fmt::Println!("dry run: nothing restarted");
                    return nil.into();
                }

                let (identity, _) = cmd.Flags().GetString("identity");
                let tr = profcmd::sshTransport(t.SSH.clone(), identity);
                let script = profile::launch::RestartScript(serveLine);
                let enc = base64::StdEncoding.EncodeToString(script.as_bytes());
                let (_, err) = tr.exec(fmt::Sprintf!(
                    "echo %s | base64 -d > /workspace/kvlm-restart.sh && chmod +x /workspace/kvlm-restart.sh && nohup /workspace/kvlm-restart.sh >/dev/null 2>&1 & echo staged",
                    enc
                ));
                if err != nil {
                    return fmt::Errorf!("upload restart script to %s: %v", t.SSH.clone(), err);
                }
                fmt::Println!("restarting under the profilers (weights are cached, but the engine re-profiles memory and recaptures CUDA graphs; expect a few minutes)...");
                let err = waitServe(&tr);
                if err != nil {
                    return err;
                }

                // the restart held: record the new flag state and stage
                // the delta for the next run's chain
                let mut updated = t.clone();
                if reset {
                    updated.Applied = map::new_no_zero();
                } else {
                    for (_, k) in range!(delta.Keys()) {
                        let (v, _) = delta.Get(k.clone());
                        updated.Applied.Set(k.clone(), v);
                    }
                }
                state::Update(&updated);
                if !reset {
                    stagePending(&delta);
                }
                fmt::Println!("server answering; the next kvlm profile run records this change as the new revision's delta");
                nil.into()
            },
        )),
        ..Default::default()
    };
    let _ = c.Flags().Bool_flag(string("dry-run"), false, string("print the resulting serve line without restarting"));
    let _ = c.Flags().Bool_flag(string("reset"), false, string("drop every applied override and restart with the catalog recipe"));
    c.Flags().StringP(string("identity"), string("I"), string(""), string("ssh identity file (defaults to your ssh config)"));
    c
}

// Go: func init() { rootCmd.AddCommand(applyCmd) }
#[goish::init]
fn init() {
    rootCmd.Lock().AddCommand(applyCmd());
}
