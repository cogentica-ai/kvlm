// llama.cpp runtime — GGUF serving (single GPU / CPU / edge).
// TODO: pin a dated tag before prod (:server is mutable).
#![allow(non_snake_case)]

use goish::string;

use crate::runtime::{Register, Runtime};

struct LlamaCpp {}

impl Runtime for LlamaCpp {
    fn Name(&self) -> string {
        string("llama.cpp")
    }
    fn Image(&self) -> string {
        string("ghcr.io/ggml-org/llama.cpp:server")
    }
}

// Go: func init() { runtime.Register("llama.cpp", &LlamaCpp{}) }
#[goish::init]
fn init() {
    Register("llama.cpp", alloc::sync::Arc::new(LlamaCpp {}));
}
