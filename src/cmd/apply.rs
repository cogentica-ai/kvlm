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
use goish::{append, make, nil, range};

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

// waitServe polls the restarted server until it answers, watching the
// log for engine-init failures. Weights are cached, so the wait is
// short compared to a fresh launch.
fn waitServe(tr: &profcmd::transport) -> error {
    let mut i = 0;
    while i < 40 {
        time::Sleep(time::Seconds(10));
        let (code, _) = tr.exec(string(
            "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null",
        ));
        if strings::TrimSpace(code) == "200" {
            return nil.into();
        }
        let (fails, err) = tr.exec(fmt::Sprintf!(
            "grep -c 'Engine core initialization failed' %s 2>/dev/null; true",
            string(profile::launch::ServerLogPath)
        ));
        if err == nil {
            let f = strings::TrimSpace(fails);
            if f != "" && f != "0" {
                return fmt::Errorf!(
                    "the restarted server failed engine init; kvlm logs shows the cause. Roll the flag back with kvlm apply --reset"
                );
            }
        }
        i += 1;
    }
    fmt::Errorf!("server not answering 400 s after the restart; kvlm logs shows where it is stuck")
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
             the change named. Weights stay cached on the pod, so the cycle\n\
             is about 90 seconds.\n\
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
                fmt::Println!("restarting under the profilers (weights are cached; expect about 90 s)...");
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
