// cmd/root.go equivalent: the package-level rootCmd and Execute().
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub mod apply;
pub mod doctor;
pub mod down;
pub mod model;
pub mod profile;
pub mod run;
pub mod target;
pub mod tune;
pub mod up;

// Go: every file in package cmd self-registers through func init() {
// rootCmd.AddCommand(...) }. goish::import! wires each module's init()
// into .init_array, which #[goish::main] runs before main — same ordering
// as Go package inits. (Aliased so the emitted `use` doesn't collide with
// the `pub mod` names above.)
goish::import! {
    crate::cmd::up as __init_up,
    crate::cmd::apply as __init_apply,
    crate::cmd::doctor as __init_doctor,
    crate::cmd::down as __init_down,
    crate::cmd::model as __init_model,
    crate::cmd::profile as __init_profile,
    crate::cmd::run as __init_run,
    crate::cmd::target as __init_target,
    crate::cmd::tune as __init_tune,
}

use goish::lazy::Lazy;
use goish::os;
use goish::string;
use goish::sync;
use goish::{nil};

use spf13_cobra as cobra;

// rootCmd represents the base command when called without any subcommands.
// (Go: var rootCmd = &cobra.Command{...} — package-level mutable value,
// lowered to the goish Lazy<Mutex<..>> pattern for settable globals.)
pub static rootCmd: Lazy<sync::Mutex<cobra::Command>> = Lazy::new(|| {
    let mut cmd = cobra::Command {
        Use: string("kvlm"),
        Short: string("kvlm deploys and serves open-weight LLMs"),
        Long: string(
            "kvlm deploys open-weight LLMs.\n\
             \n\
             It keeps serving recipes for the top model families (kvlm model ls,\n\
             kvlm model show), works out KV cache and VRAM from context size and\n\
             each model's attention layout (kvlm model vram), and starts or stops\n\
             serving pods on RunPod, Vast.ai, or Kubernetes with the vLLM,\n\
             SGLang, or llama.cpp runtimes.\n\
             \n\
             Built in Goish Rust on a verbatim port of spf13/cobra.",
        ),
        // Errors from RunE carry actionable messages; don't drown them
        // in a usage dump.
        SilenceUsage: true,
        Version: string("0.1.2"),
        ..Default::default()
    };
    // Go: rootCmd.PersistentFlags().StringVarP(&driver, "driver", "d", "", ...)
    cmd.PersistentFlags().StringP(
        string("driver"),
        string("d"),
        string(""),
        string("target platform driver: runpod, vastai, or k8s"),
    );
    // Go: rootCmd.PersistentFlags().StringVarP(&runtime, "runtime", "r", "vllm", ...)
    cmd.PersistentFlags().StringP(
        string("runtime"),
        string("r"),
        string("vllm"),
        string("serving runtime: vllm, sglang, or llama.cpp"),
    );
    sync::Mutex::new(cmd)
});

// Execute runs the root command. This is called by main.main(). It only
// needs to happen once to the rootCmd.
pub fn Execute() {
    let err = rootCmd.Lock().Execute();
    if err != nil {
        os::Exit(1);
    }
}
