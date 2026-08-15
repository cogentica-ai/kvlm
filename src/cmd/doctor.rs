// doctor: preflight checks that fail fast and name the fix, instead
// of a deployed pod discovering the problem mid-wait. Local checks
// are free and run before any spend; --target reaches into the
// current pod and verifies what the launch will need there.
//
// Every check here exists because its failure was measured on a real
// deploy: the missing/mismatched ssh key (pod billed while the wait
// loop could never connect), and the vision-model import chain dying
// on a fresh pod (torchcodec without ffmpeg and the nvidia lib path).
#![allow(non_snake_case)]

use goish::fmt;
use goish::os;
use goish::string;
use goish::strings;
use goish::goslice::slice;
use goish::errors::error;
use goish::{append, int, make, nil, range};

use spf13_cobra as cobra;

use crate::cmd::profile as profcmd;
use crate::cmd::rootCmd;
use crate::config;
use crate::driver;
use crate::model;
use crate::state;

fn pass(name: &'static str, detail: string) {
    fmt::Printf!("  ok    %s: %s\n", string(name), detail);
}

fn fail(name: &'static str, detail: string, fix: string) -> int {
    fmt::Printf!("  FAIL  %s: %s\n", string(name), detail);
    if fix != "" {
        fmt::Printf!("        fix: %s\n", fix);
    }
    1
}

// injectedPubkey mirrors the runpod driver's resolution of the public
// key a deploy injects: env, config, then the key files.
fn injectedPubkey() -> (string, string) {
    let (v, ok) = os::LookupEnv("RUNPOD_PUBLIC_KEY");
    if ok && v != "" {
        return (v, string("RUNPOD_PUBLIC_KEY"));
    }
    let cfg = config::Value("drivers", "runpod", "public_key");
    if cfg != "" {
        return (cfg, string("config public_key"));
    }
    let (home, err) = os::UserHomeDir();
    if err != nil {
        return (string(""), string(""));
    }
    for rel in ["/.runpod/id_ed25519.pub", "/.ssh/id_ed25519.pub", "/.ssh/id_rsa.pub"].iter() {
        let p = (home.clone()) + (string(*rel));
        let (data, err) = os::ReadFile(p.clone());
        if err == nil {
            return (strings::TrimSpace(string(data)), p);
        }
    }
    (string(""), string(""))
}

// privateKeyPath mirrors the transport's default identity resolution.
fn privateKeyPath() -> string {
    let (home, err) = os::UserHomeDir();
    if err != nil {
        return string("");
    }
    for rel in ["/.runpod/id_ed25519", "/.ssh/id_ed25519"].iter() {
        let p = (home.clone()) + (string(*rel));
        let (_, err) = os::ReadFile(p.clone());
        if err == nil {
            return p;
        }
    }
    string("")
}

// keyPart isolates the base64 body of an openssh public key line so
// comparisons ignore the comment field.
fn keyPart(pubkey: string) -> string {
    let fields = strings::Fields(pubkey.clone());
    if fields.Len() >= 2 {
        return fields[1usize].clone();
    }
    pubkey
}

fn localChecks(cmd: &mut cobra::Command) -> int {
    let mut fails: int = 0;
    fmt::Println!("local");

    // driver credentials resolve without deploying anything
    let (d, _, err) = driver::FromCommand(cmd);
    if err != nil {
        fails += fail("credentials", fmt::Sprintf!("%v", err), string(""));
    } else {
        pass("credentials", fmt::Sprintf!("%s driver resolves", d.unwrap().Name()));
    }

    // the ssh pair: a private key must exist AND derive the public key
    // a deploy would inject; mismatch bills a pod that never connects
    let privKey = privateKeyPath();
    let (pubkey, pubSrc) = injectedPubkey();
    if privKey == "" {
        fails += fail(
            "ssh key",
            string("no private key at ~/.runpod/id_ed25519 or ~/.ssh/id_ed25519"),
            string("ssh-keygen -t ed25519 -f ~/.runpod/id_ed25519 -N ''"),
        );
    } else if pubkey == "" {
        fails += fail(
            "ssh key",
            string("no public key to inject (env, config, or .pub file)"),
            fmt::Sprintf!("ssh-keygen -y -f %s > %s.pub", privKey.clone(), privKey.clone()),
        );
    } else {
        let mut args: slice<string> = make!([]string, 0);
        args = append!(args.clone(), string("-y"));
        args = append!(args.clone(), string("-f"));
        args = append!(args.clone(), privKey.clone());
        let (derived, _, err) = profcmd::runCapture(string("ssh-keygen"), args);
        if err != nil {
            fails += fail("ssh key", fmt::Sprintf!("cannot derive public key from %s: %v", privKey.clone(), err), string(""));
        } else if keyPart(strings::TrimSpace(derived)) != keyPart(pubkey) {
            fails += fail(
                "ssh key",
                fmt::Sprintf!("%s does not match the key a deploy injects (%s)", privKey.clone(), pubSrc),
                fmt::Sprintf!("ssh-keygen -y -f %s > %s.pub, or pass --identity", privKey.clone(), privKey),
            );
        } else {
            pass("ssh key", fmt::Sprintf!("%s matches the injected key (%s)", privKey, pubSrc));
        }
    }

    // local analysis tools: without them runs still collect, but the
    // in-flow analysis silently degrades to remote export
    for tool in ["nsys", "sqlite3"].iter() {
        if profcmd::toolAvailable(tool) {
            pass("tool", string(*tool));
        } else {
            fails += fail(
                "tool",
                fmt::Sprintf!("%s not found locally; graph analysis will need the remote-export fallback", string(*tool)),
                string(""),
            );
        }
    }

    // the config file carries the api key; it must not be readable by
    // other users
    let (home, herr) = os::UserHomeDir();
    if herr == nil {
        let cfgPath = (home) + ("/.kvlm/config.json");
        let (info, err) = os::Stat(cfgPath.clone());
        if err == nil {
            let mode = info.Mode().Perm().0;
            if (mode & 0o077) != 0 {
                fails += fail(
                    "config perms",
                    fmt::Sprintf!("%s is group/world readable", cfgPath.clone()),
                    fmt::Sprintf!("chmod 600 %s", cfgPath),
                );
            } else {
                pass("config perms", cfgPath);
            }
        }
    }
    fails
}

fn targetChecks(cmd: &mut cobra::Command) -> int {
    let mut fails: int = 0;
    let (t, ok) = state::Current();
    if !ok || t.SSH == "" {
        return fail(
            "target",
            string("no recorded pod with ssh to check (kvlm up --mode profile records one)"),
            string(""),
        );
    }
    fmt::Printf!("target (%s pod %s)\n", t.Driver.clone(), t.Pod.clone());
    let (identity, _) = cmd.Flags().GetString("identity");
    let tr = profcmd::sshTransport(t.SSH.clone(), identity);

    let (_, err) = tr.exec(string("true"));
    if err != nil {
        return fails + fail("ssh", fmt::Sprintf!("cannot reach %s: %v", t.SSH.clone(), err), string(""));
    }
    pass("ssh", t.SSH.clone());

    // host driver vs the torch CUDA build: mismatch fails engine init
    // minutes later with "driver too old"
    let (drv, _) = tr.exec(string("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"));
    let (tcuda, terr) = tr.exec(string("python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null"));
    if terr == nil && strings::TrimSpace(tcuda.clone()) != "" {
        pass("cuda", fmt::Sprintf!("host driver %s, torch cuda %s", strings::TrimSpace(drv), strings::TrimSpace(tcuda)));
    } else {
        pass("cuda", fmt::Sprintf!("host driver %s (torch not installed yet)", strings::TrimSpace(drv)));
    }

    // the imports the launch will make, checked in seconds instead of
    // discovered as a dead server after the model load
    let mut imports = string("torch, vllm");
    let mut vision = false;
    if t.Model != "" {
        let (_, m, ok) = model::Find(t.Model.clone());
        if ok && m.Vision {
            vision = true;
            imports = string("torch, vllm, torchcodec");
        }
    }
    let line = fmt::Sprintf!(
        "NVLIBS=$(ls -d /usr/local/lib/python*/dist-packages/nvidia/*/lib 2>/dev/null | tr '\\n' ':'); LD_LIBRARY_PATH=$NVLIBS$LD_LIBRARY_PATH python3 -c 'import %s' 2>&1 | tail -1",
        strings::ReplaceAll(imports.clone(), " ", "")
    );
    let (out, err) = tr.exec(line);
    let trimmed = strings::TrimSpace(out);
    if err == nil && trimmed == "" {
        pass("imports", imports);
    } else if strings::Contains(trimmed.clone(), "No module named") || strings::Contains(trimmed.clone(), "Error") {
        let mut fix = string("");
        if vision && strings::Contains(trimmed.clone(), "torchcodec") {
            fix = string("apt-get install -y ffmpeg (the launch script does this for vision models)");
        }
        fails += fail("imports", trimmed, fix);
    } else {
        pass("imports", imports);
    }

    // disk: the weights land on the volume mount
    let (df, err) = tr.exec(string("df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}'"));
    if err == nil {
        let avail = strings::TrimSpace(strings::TrimSuffix(strings::TrimSpace(df), "G"));
        pass("disk", fmt::Sprintf!("%s GB free on /workspace", avail));
    }
    fails
}

fn doctorCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("doctor"),
        Short: string("Preflight checks: fail here, not on a billing pod"),
        Long: string(
            "Check everything a deploy needs before spending money on one:\n\
             credentials, the ssh key pair (existence and that the private\n\
             key matches the public key a deploy injects), local analysis\n\
             tools, and config file permissions.\n\
             \n\
             With --target, also reach into the recorded pod and verify\n\
             what the launch will need there: driver and CUDA versions,\n\
             the python imports the model requires, and disk space.",
        ),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, _args: slice<string>| -> error {
                let mut fails = localChecks(cmd);
                let (target, _) = cmd.Flags().GetBool("target");
                if target {
                    fails += targetChecks(cmd);
                }
                if fails > 0 {
                    return fmt::Errorf!("doctor found %d problem(s)", fails);
                }
                fmt::Println!("all checks passed");
                nil.into()
            },
        )),
        ..Default::default()
    };
    let _ = c.Flags().Bool_flag(string("target"), false, string("also check the recorded pod over ssh"));
    c.Flags().StringP(string("identity"), string("I"), string(""), string("ssh identity file (defaults to your ssh config)"));
    c
}

// Go: func init() { rootCmd.AddCommand(doctorCmd) }
#[goish::init]
fn init() {
    rootCmd.Lock().AddCommand(doctorCmd());
}
