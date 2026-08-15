// driver package: platform drivers for kvlm (runpod, vastai, k8s).
//
// Modeled on Go's database/sql driver registry: each driver module
// self-registers in its init() (same .init_array pattern as the cmd
// package), and commands look drivers up by name at run time.
//
// Credential resolution (per driver, highest wins):
//   1. environment variables    — RUNPOD_API_KEY / VAST_API_KEY /
//                                 KUBECONFIG, KUBE_CONTEXT,
//                                 KUBE_NAMESPACE
//   2. ~/.kvlm/config.json      — {"drivers": {"<name>": {...}}}
//   3. platform defaults        — k8s falls back to ~/.kube/config
//
// Driver selection (highest wins): --driver/-d flag → KVLM_DRIVER env →
// "driver" key in ~/.kvlm/config.json.
//
// Secrets are never taken from command-line flags — flags leak into
// shell history and the process list.
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub mod k8s;
pub mod runpod;
pub mod vastai;

// Each driver file self-registers via #[goish::init]; import! wires those
// init()s into .init_array (aliased to avoid colliding with the `pub mod`
// names above).
goish::import! {
    crate::driver::runpod as __init_runpod,
    crate::driver::vastai as __init_vastai,
    crate::driver::k8s as __init_k8s,
}

use goish::fmt;
use goish::lazy::Lazy;
use goish::os;
use goish::strings;
use goish::sync;
use goish::errors::error;
use goish::gomap::map;
use goish::string;
use goish::slice;
use goish::{append, float64, int, make, nil};

use spf13_cobra as cobra;

use crate::config;

// Credentials carries everything a driver needs to reach its platform.
// Each driver fills only the fields it uses.
#[derive(Clone, Default)]
pub struct Credentials {
    pub APIKey: string,     // runpod, vastai
    pub Kubeconfig: string, // k8s: path to kubeconfig
    pub Context: string,    // k8s: kubeconfig context ("" = current-context)
    pub Namespace: string,  // k8s: target namespace
}

// Options carries launch parameters resolved by the command layer
// (runtime selection today; model/variant later).
#[derive(Clone, Default)]
pub struct Options {
    pub Model: string, // catalog model name ("" = none selected yet)
    pub Runtime: string,
    pub Image: string,
    // GPU shape of the pod. Defaults live in the command layer:
    // 1x H100 SXM, CUDA >= 13.0. The CUDA floor exists because a
    // host driver older than the runtime's torch CUDA build fails
    // engine init ("driver too old").
    pub GPUType: string,
    pub GPUCount: int,
    pub CudaVersions: slice<string>,
    // PodID targets an existing pod (down / status).
    pub PodID: string,
    // ServeArgv is the container command for production serving
    // (["vllm", "serve", <model>, flags...]); the platform runs it
    // natively (Docker command on RunPod, pod spec on K8s). Empty
    // means a bare pod (or a profile-mode launch over ssh).
    pub ServeArgv: slice<string>,
    // VolumeID attaches a shared network volume (model weights and HF
    // cache persist across pods and are shared between them).
    pub VolumeID: string,
    // Identity fields recorded into ~/.kvlm/state.json when the pod is
    // created, so ps/status/down can name what a pod serves without
    // asking the user to remember it.
    pub Variant: string,
    pub Mode: string, // "production" or "profile"
    pub VllmVersion: string,
}

// PodInfo is one live pod as the platform reports it (kvlm ps).
#[derive(Clone, Default)]
pub struct PodInfo {
    pub ID: string,
    pub Name: string,
    pub Status: string,
    pub GPUType: string,
    pub GPUCount: int,
    pub Image: string,
    pub CostPerHr: float64,
    pub UptimeSeconds: int,
    pub SSH: string,      // user@host:port when a public ssh port is up
    pub Endpoint: string, // https base URL when the platform proxies one
}

// Driver starts and stops a kvlm pod on one platform.
pub trait Driver: Send + Sync {
    fn Name(&self) -> string;
    // ResolveCredentials walks env → ~/.kvlm/config.json → platform
    // defaults, and errors with an actionable message when something
    // required is missing.
    fn ResolveCredentials(&self) -> (Credentials, error);
    // Up starts a pod and returns the ssh destination
    // ("user@host:port") when the platform exposes one, "" otherwise.
    fn Up(&self, creds: &Credentials, opts: &Options) -> (string, error);
    fn Down(&self, creds: &Credentials, opts: &Options) -> error;

    // List returns the live pods on the platform. The default is an
    // explicit refusal so drivers without an inventory API stay honest.
    fn List(&self, creds: &Credentials) -> (slice<PodInfo>, error) {
        let _ = creds;
        let empty: slice<PodInfo> = make!([]PodInfo, 0);
        (
            empty,
            fmt::Errorf!("%s: pod listing not implemented", self.Name()),
        )
    }

    // Exec runs a command on a remote pod and returns stdout.
    // podId is the platform-specific pod identifier (k8s pod name,
    // runpod pod id, vastai instance id).
    fn Exec(&self, creds: &Credentials, podId: string, cmd: string) -> (string, error);

    // Download copies a file from the remote pod to a local path.
    fn Download(
        &self,
        creds: &Credentials,
        podId: string,
        remotePath: string,
        localPath: string,
    ) -> error;
}

// Registry — mirrors database/sql: drivers.Register in each init().
static drivers: Lazy<sync::Mutex<map<string, alloc::sync::Arc<dyn Driver>>>> =
    Lazy::new(|| sync::Mutex::new(map::new_no_zero()));

pub fn Register<S: Into<string>>(name: S, d: alloc::sync::Arc<dyn Driver>) {
    drivers.Lock().Set(name.into(), d);
}

// Names returns the registered driver names, sorted, for error messages.
pub fn Names() -> string {
    let keys = drivers.Lock().Keys();
    let mut sorted_: alloc::vec::Vec<string> = alloc::vec::Vec::new();
    for (_, k) in goish::range!(keys) {
        sorted_.push(k.clone());
    }
    sorted_.sort();
    let mut names: slice<string> = make!([]string, 0);
    for k in sorted_.iter() {
        names = append!(names.clone(), k.clone());
    }
    strings::Join(names, ", ")
}

// FromCommand picks the driver (--driver/-d → KVLM_DRIVER → config
// "driver") and resolves its credentials.
pub fn FromCommand(
    cmd: &mut cobra::Command,
) -> (Option<alloc::sync::Arc<dyn Driver>>, Credentials, error) {
    let (mut name, _) = cmd.Flags().GetString("driver");
    if name == "" {
        name = os::Getenv("KVLM_DRIVER");
    }
    if name == "" {
        name = config::TopLevel("driver");
    }
    if name == "" {
        // the current target in ~/.kvlm/state.json knows its driver,
        // so commands aimed at a pod kvlm started need no -d at all
        let (t, ok) = crate::state::Current();
        if ok {
            name = t.Driver.clone();
        }
    }
    if name == "" {
        return (
            None,
            Default::default(),
            fmt::Errorf!(
                "no driver selected (available: %s): pass --driver/-d, set KVLM_DRIVER, or set \"driver\" in ~/.kvlm/config.json",
                Names()
            ),
        );
    }

    let available = Names();
    let m = drivers.Lock();
    if !m.Has(name.clone()) {
        drop(m);
        return (
            None,
            Default::default(),
            fmt::Errorf!("unknown driver %q (available: %s)", name, available),
        );
    }
    let (d, _) = m.Get(name);
    drop(m);

    let (creds, err) = d.ResolveCredentials();
    if err != nil {
        return (None, Default::default(), err);
    }
    (Some(d), creds, nil.into())
}

// resolveString: env var (if set and non-empty) → config drivers.<name>.<key>.
pub(crate) fn resolveString<S: Into<string>>(envVar: &str, driverName: S, cfgKey: &str) -> string {
    let (v, ok) = os::LookupEnv(envVar);
    if ok && v != "" {
        return v;
    }
    config::Value("drivers", driverName, cfgKey)
}

// modelOrDash renders Options.Model for log lines ("-" when unset).
pub(crate) fn modelOrDash(opts: &Options) -> string {
    if opts.Model == "" {
        return string("-");
    }
    opts.Model.clone()
}

// Mask renders a secret as ****<last 4> for log output.
pub(crate) fn Mask(key: &string) -> string {
    if key.Len() <= 4 {
        return string("****");
    }
    ("****") + (key.slice(key.Len() - 4, key.Len()))
}
