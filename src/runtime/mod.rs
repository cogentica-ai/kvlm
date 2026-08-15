// runtime package: serving engines kvlm can launch inside a pod —
// vllm (default), sglang, llama.cpp.
//
// Same registry idiom as the driver package: each runtime module
// self-registers in init(). Selection (highest wins): --runtime/-r flag
// (only when explicitly set) → KVLM_RUNTIME env → "runtime" key in
// ~/.kvlm/config.json → "vllm".
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub mod llamacpp;
pub mod sglang;
pub mod vllm;

goish::import! {
    crate::runtime::vllm as __init_vllm,
    crate::runtime::sglang as __init_sglang,
    crate::runtime::llamacpp as __init_llamacpp,
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
use goish::{append, make, nil};

use spf13_cobra as cobra;

use crate::config;

pub const DefaultRuntime: &str = "vllm";

// Runtime is one serving engine: it knows its container image and, later,
// how to turn a model recipe into a launch command.
pub trait Runtime: Send + Sync {
    fn Name(&self) -> string;
    fn Image(&self) -> string;
}

static runtimes: Lazy<sync::Mutex<map<string, alloc::sync::Arc<dyn Runtime>>>> =
    Lazy::new(|| sync::Mutex::new(map::new_no_zero()));

pub fn Register<S: Into<string>>(name: S, r: alloc::sync::Arc<dyn Runtime>) {
    runtimes.Lock().Set(name.into(), r);
}

// Names returns the registered runtime names, sorted, for error messages.
pub fn Names() -> string {
    let keys = runtimes.Lock().Keys();
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

// FromCommand resolves the runtime: explicit --runtime/-r flag →
// KVLM_RUNTIME → config "runtime" → DefaultRuntime. The flag carries
// "vllm" as its pflag default, so only an explicitly Changed flag
// short-circuits the chain.
pub fn FromCommand(cmd: &mut cobra::Command) -> (Option<alloc::sync::Arc<dyn Runtime>>, error) {
    let mut name = string("");
    if cmd.Flags().Changed("runtime") {
        let (v, _) = cmd.Flags().GetString("runtime");
        name = v;
    }
    if name == "" {
        name = os::Getenv("KVLM_RUNTIME");
    }
    if name == "" {
        name = config::TopLevel("runtime");
    }
    if name == "" {
        name = string(DefaultRuntime);
    }

    let available = Names();
    let m = runtimes.Lock();
    if !m.Has(name.clone()) {
        drop(m);
        return (
            None,
            fmt::Errorf!("unknown runtime %q (available: %s)", name, available),
        );
    }
    let (r, _) = m.Get(name);
    drop(m);
    (Some(r), nil.into())
}
