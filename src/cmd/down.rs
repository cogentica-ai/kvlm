// cmd/down.go equivalent: the down command, self-registered via init().
#![allow(non_snake_case)]

use goish::fmt;
use goish::string;
use goish::goslice::slice;
use goish::errors::error;
use goish::{nil, range};

use spf13_cobra as cobra;

use crate::cmd::rootCmd;
use crate::driver;
use crate::state;

// downCmd represents the down command. (Go: var downCmd = &cobra.Command{...})
fn downCmd() -> cobra::Command {
    cobra::Command {
        Use: string("down [pod]"),
        Short: string("Stop a kvlm pod"),
        Long: string(
            "Stop a kvlm pod. With no argument, the pod kvlm up recorded in\n\
             ~/.kvlm/state.json is terminated. Name a pod id or a model to\n\
             pick one, or pass --all to stop every pod on the platform.\n\
             kvlm ps lists what is running.",
        ),
        Args: Some(cobra::MaximumNArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                // an idempotent down: with nothing named and nothing
                // recorded, there is nothing to stop and that is fine
                let (allFlag, _) = cmd.Flags().GetBool("all");
                let (podFlag, _) = cmd.Flags().GetString("pod");
                if !allFlag && podFlag == "" && args.Len() == 0 {
                    let (_, ok) = state::Current();
                    if !ok {
                        fmt::Println!("nothing to stop: no pod is recorded (kvlm ps lists live ones)");
                        return nil.into();
                    }
                }
                let (d, creds, err) = driver::FromCommand(cmd);
                if err != nil {
                    return err;
                }
                let d = d.unwrap();

                let (all, _) = cmd.Flags().GetBool("all");
                if all {
                    let (pods, err) = d.List(&creds);
                    if err != nil {
                        return err;
                    }
                    if pods.Len() == 0 {
                        fmt::Println!("kvlm: no pods running");
                        return nil.into();
                    }
                    for (_, p) in range!(pods) {
                        let opts = driver::Options {
                            PodID: p.ID.clone(),
                            ..Default::default()
                        };
                        let err = d.Down(&creds, &opts);
                        if err != nil {
                            return err;
                        }
                    }
                    return nil.into();
                }

                // pod resolution: --pod flag, then the positional (a pod
                // id or a model name from the state file), then the
                // current target
                let (mut podID, _) = cmd.Flags().GetString("pod");
                if podID == "" && args.Len() > 0 {
                    let reference = args[0usize].clone();
                    let (t, ok) = state::Find(reference.clone());
                    if ok {
                        podID = t.Pod.clone();
                    } else {
                        // not in the state file: treat it as a raw
                        // platform pod id
                        podID = reference;
                    }
                }
                if podID == "" {
                    let (t, ok) = state::Current();
                    if ok {
                        podID = t.Pod.clone();
                    }
                }
                if podID == "" {
                    return fmt::Errorf!(
                        "no pod to stop: kvlm has no recorded target. List what is running with kvlm ps, then: kvlm down <pod-id>"
                    );
                }
                let opts = driver::Options {
                    PodID: podID,
                    ..Default::default()
                };
                d.Down(&creds, &opts)
            },
        )),
        ..Default::default()
    }
}

// Go: func init() { rootCmd.AddCommand(downCmd) }
#[goish::init]
fn init() {
    let mut c = downCmd();
    let _ = c.Flags().String_flag(
        string("pod"),
        string(""),
        string("platform pod id to terminate (default: the recorded target)"),
    );
    let _ = c.Flags().Bool_flag(string("all"), false, string("terminate every pod on the platform"));
    rootCmd.Lock().AddCommand(c);
}
