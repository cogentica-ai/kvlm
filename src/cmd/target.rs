// Target commands: the live-pod side of the CLI. ps lists what is
// running (and what it costs), status describes the current target,
// use switches it, logs and ssh reach into it. All of them resolve
// the target from ~/.kvlm/state.json, which kvlm up writes, so none
// of them need an address retyped from scrollback.
#![allow(non_snake_case)]

use goish::fmt;
use goish::net::http;
use goish::os;
use goish::string;
use goish::strings;
use goish::text::tabwriter;
use goish::time;
use goish::goslice::slice;
use goish::errors::error;
use goish::{float64, int, nil, range};

use spf13_cobra as cobra;

use crate::cmd::profile as profcmd;
use crate::cmd::rootCmd;
use crate::driver;
use crate::state;

// fmtUptime renders seconds as 3d4h / 4h12m / 12m / 45s.
fn fmtUptime(secs: int) -> string {
    if secs <= 0 {
        return string("-");
    }
    let d = secs / 86400;
    let h = (secs % 86400) / 3600;
    let m = (secs % 3600) / 60;
    if d > 0 {
        return fmt::Sprintf!("%dd%dh", d, h);
    }
    if h > 0 {
        return fmt::Sprintf!("%dh%dm", h, m);
    }
    if m > 0 {
        return fmt::Sprintf!("%dm", m);
    }
    fmt::Sprintf!("%ds", secs)
}

// round2 pre-rounds a dollar figure to cents (the fmt port has no
// precision verbs).
fn round2(x: float64) -> float64 {
    float64(int(x * 100.0 + 0.5)) / 100.0
}

// modelFor joins a listed pod with the state file: what kvlm recorded
// it as serving, "-" for pods kvlm did not start.
fn modelFor(podID: string) -> (string, string) {
    let (t, ok) = state::Find(podID);
    if !ok || t.Model == "" {
        return (string("-"), t.Mode.clone());
    }
    let mut m = t.Model.clone();
    if t.Variant != "" {
        m = (m) + (" ") + (t.Variant.clone());
    }
    (m, t.Mode.clone())
}

// reconcile drops state entries for this driver whose pods the
// platform no longer reports: the API is the truth, the file is a
// cache of it.
fn reconcile(driverName: string, pods: &slice<driver::PodInfo>) {
    let (targets, _) = state::Load();
    for (_, t) in range!(targets) {
        if t.Driver != driverName {
            continue;
        }
        let mut alive = false;
        for (_, p) in range!(pods.clone()) {
            if p.ID == t.Pod {
                alive = true;
                break;
            }
        }
        if !alive {
            state::Remove(t.Driver.clone(), t.Pod.clone());
        }
    }
}

fn psCmd() -> cobra::Command {
    cobra::Command {
        Use: string("ps"),
        Short: string("List live pods on the platform, with cost"),
        Long: string(
            "List the pods running on the platform selected with --driver/-d,\n\
             with GPU shape, uptime, and cost per hour. The pod marked with *\n\
             is the current target. Pods kvlm did not start show '-' in the\n\
             MODEL column. The recorded state in ~/.kvlm/state.json is\n\
             reconciled against this list: the platform is the truth.",
        ),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, _args: slice<string>| -> error {
                let (d, creds, err) = driver::FromCommand(cmd);
                if err != nil {
                    return err;
                }
                let d = d.unwrap();
                let (pods, err) = d.List(&creds);
                if err != nil {
                    return err;
                }
                reconcile(d.Name(), &pods);
                if pods.Len() == 0 {
                    fmt::Println!("no pods running");
                    return nil.into();
                }
                let (cur, _) = state::Current();
                let mut tw = tabwriter::NewWriter(os::Stdout(), 8, 1, 3, b' ', 0);
                fmt::Fprintf!(tw, "POD\tMODEL\tMODE\tGPU\tSTATUS\tUPTIME\t$/HR\n");
                let mut total: float64 = 0.0;
                for (_, p) in range!(pods) {
                    let (m, mode) = modelFor(p.ID.clone());
                    let mut modeCol = mode;
                    if modeCol == "" {
                        modeCol = string("-");
                    }
                    let mut id = p.ID.clone();
                    if p.ID == cur.Pod {
                        id = (id) + (" *");
                    }
                    let gpu = fmt::Sprintf!("%dx %s", p.GPUCount, p.GPUType.clone());
                    total += p.CostPerHr;
                    fmt::Fprintf!(
                        tw,
                        "%s\t%s\t%s\t%s\t%s\t%s\t%v\n",
                        id,
                        m,
                        modeCol,
                        gpu,
                        p.Status.clone(),
                        fmtUptime(p.UptimeSeconds),
                        round2(p.CostPerHr)
                    );
                }
                let _ = tw.Flush();
                fmt::Printf!("total burn: $%v/hr\n", round2(total));
                nil.into()
            },
        )),
        ..Default::default()
    }
}

// reachable answers whether the target's server responds, over its
// endpoint for production pods and over ssh for profile pods.
fn reachable(t: &state::Target, identity: string) -> (bool, string) {
    if t.Endpoint != "" {
        let client = http::Client::default();
        let (resp, err) = client.Get((t.Endpoint.clone()) + ("/v1/models"));
        if err != nil {
            return (false, fmt::Sprintf!("endpoint not answering: %v", err));
        }
        if resp.StatusCode == 200 {
            return (true, (string("serving at ")) + (t.Endpoint.clone()) + ("/v1"));
        }
        return (false, fmt::Sprintf!("endpoint answered %d (still loading, or the server exited)", resp.StatusCode));
    }
    if t.SSH != "" {
        let tr = profcmd::sshTransport(t.SSH.clone(), identity);
        let (code, err) = tr.exec(string(
            "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null",
        ));
        if err != nil {
            return (false, fmt::Sprintf!("ssh to %s failed: %v", t.SSH.clone(), err));
        }
        if strings::TrimSpace(code) == "200" {
            return (true, string("serving on the pod at 127.0.0.1:8000"));
        }
        return (false, string("pod reachable, server not answering (still loading, or not launched)"));
    }
    (false, string("no endpoint or ssh recorded yet"))
}

fn statusCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("status [pod]"),
        Short: string("Describe the current target pod and its server"),
        Long: string(
            "Describe the current target: which pod, what it serves, whether\n\
             the server answers, and what to run next. Name a pod id or a\n\
             model to describe another recorded target.",
        ),
        Args: Some(cobra::MaximumNArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (t, ok);
                if args.Len() > 0 {
                    let (found, fok) = state::Find(args[0usize].clone());
                    if !fok {
                        return fmt::Errorf!("no recorded target matches %q (kvlm ps lists live pods)", args[0usize].clone());
                    }
                    t = found;
                    ok = fok;
                } else {
                    let (found, fok) = state::Current();
                    t = found;
                    ok = fok;
                }
                if !ok {
                    fmt::Println!("no target: kvlm up records the pod it starts here");
                    return nil.into();
                }

                let mut age = string("");
                let (created, perr) = time::Parse(string(time::RFC3339), t.Created.clone());
                if perr == nil {
                    age = fmt::Sprintf!(", up %s", fmtUptime(int(time::Since(created).Seconds())));
                }
                let mut cost = string("");
                if t.CostPerHr > 0.0 {
                    cost = fmt::Sprintf!(", $%v/hr", round2(t.CostPerHr));
                }
                fmt::Printf!(
                    "target   %s pod %s, %dx %s%s%s\n",
                    t.Driver.clone(),
                    t.Pod.clone(),
                    t.GPUCount,
                    t.GPUType.clone(),
                    age,
                    cost
                );
                let mut served = t.Model.clone();
                if served == "" {
                    served = string("no model (bare pod)");
                } else if t.Variant != "" {
                    served = (served) + (" ") + (t.Variant.clone());
                }
                let mut ver = string("");
                if t.VllmVersion != "" {
                    ver = (string(", vLLM ")) + (t.VllmVersion.clone());
                }
                fmt::Printf!("server   %s%s, %s mode\n", served, ver, t.Mode.clone());
                if t.SSH != "" {
                    fmt::Printf!("ssh      %s\n", t.SSH.clone());
                }

                let (identity, _) = cmd.Flags().GetString("identity");
                let (up, detail) = reachable(&t, identity);
                fmt::Printf!("health   %s\n", detail);
                if up && t.Mode == "profile" {
                    fmt::Println!("next     kvlm profile run");
                }
                nil.into()
            },
        )),
        ..Default::default()
    };
    c.Flags().StringP(string("identity"), string("I"), string(""), string("ssh identity file (defaults to your ssh config)"));
    c
}

fn useCmd() -> cobra::Command {
    cobra::Command {
        Use: string("use <pod>"),
        Short: string("Switch the current target pod"),
        Long: string(
            "Switch which recorded pod the target-scoped commands (status,\n\
             logs, ssh, profile run, down) act on. Accepts a pod id or a\n\
             model name from the state file.",
        ),
        Args: Some(cobra::ExactArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |_cmd: &mut cobra::Command, args: slice<string>| -> error {
                let reference = args[0usize].clone();
                if !state::SetCurrent(reference.clone()) {
                    return fmt::Errorf!("no recorded target matches %q (kvlm ps lists live pods)", reference);
                }
                let (t, _) = state::Current();
                fmt::Printf!("current target: %s pod %s (%s)\n", t.Driver.clone(), t.Pod.clone(), t.Model.clone());
                nil.into()
            },
        )),
        ..Default::default()
    }
}

fn logsCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("logs"),
        Short: string("Show the server log from the current target"),
        Long: string(
            "Show the tail of the vLLM server log on the current target over\n\
             ssh. Production pods run without ssh; their logs live in the\n\
             platform console.",
        ),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, _args: slice<string>| -> error {
                let (t, ok) = state::Current();
                if !ok {
                    return fmt::Errorf!("no target: kvlm up records the pod it starts (kvlm ps lists live pods)");
                }
                if t.SSH == "" {
                    return fmt::Errorf!(
                        "pod %s has no ssh (production pods serve without it); its logs are in the %s console",
                        t.Pod.clone(),
                        t.Driver.clone()
                    );
                }
                let mut logPath = t.ServerLog.clone();
                if logPath == "" {
                    logPath = string(crate::profile::launch::ServerLogPath);
                }
                let (lines, _) = cmd.Flags().GetInt("lines");
                let (identity, _) = cmd.Flags().GetString("identity");
                let tr = profcmd::sshTransport(t.SSH.clone(), identity);
                let (out, err) = tr.exec(fmt::Sprintf!("tail -n %d %s", lines, logPath.clone()));
                if err != nil {
                    return fmt::Errorf!("read %s on %s: %v", logPath, t.SSH.clone(), err);
                }
                fmt::Print!(out);
                nil.into()
            },
        )),
        ..Default::default()
    };
    c.Flags().IntP(string("lines"), string("n"), 100, string("lines of log tail to show"));
    c.Flags().StringP(string("identity"), string("I"), string(""), string("ssh identity file (defaults to your ssh config)"));
    c
}

fn sshCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("ssh [command]"),
        Short: string("Run a command on the current target over ssh"),
        Long: string(
            "Run one command on the current target and print its output.\n\
             With no command, print the ssh invocation for an interactive\n\
             shell.",
        ),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (t, ok) = state::Current();
                if !ok {
                    return fmt::Errorf!("no target: kvlm up records the pod it starts (kvlm ps lists live pods)");
                }
                if t.SSH == "" {
                    return fmt::Errorf!("pod %s has no ssh (production pods serve without it)", t.Pod.clone());
                }
                if args.Len() == 0 {
                    // an interactive shell needs a tty this process
                    // does not own; print the exact invocation instead
                    let host = t.SSH.clone();
                    let mut dest = host.clone();
                    let mut port = string("22");
                    let idx = strings::LastIndex(host.clone(), string(":"));
                    if idx > 0 {
                        dest = host.slice(0, idx);
                        port = host.slice(idx + 1, host.Len());
                    }
                    fmt::Printf!("ssh -p %s %s\n", port, dest);
                    return nil.into();
                }
                let (identity, _) = cmd.Flags().GetString("identity");
                let tr = profcmd::sshTransport(t.SSH.clone(), identity);
                let (out, err) = tr.exec(strings::Join(args, " "));
                if err != nil {
                    return err;
                }
                fmt::Print!(out);
                nil.into()
            },
        )),
        ..Default::default()
    };
    c.Flags().StringP(string("identity"), string("I"), string(""), string("ssh identity file (defaults to your ssh config)"));
    c
}

// Go: func init() { rootCmd.AddCommand(...) }
#[goish::init]
fn init() {
    rootCmd.Lock().AddCommand(psCmd());
    rootCmd.Lock().AddCommand(statusCmd());
    rootCmd.Lock().AddCommand(useCmd());
    rootCmd.Lock().AddCommand(logsCmd());
    rootCmd.Lock().AddCommand(sshCmd());
}
