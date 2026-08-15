// vLLM runtime — the default. Image per the Aug 2026 recipe research;
// swap to -cu129-ubuntu2404 on CUDA 12.9 drivers.
#![allow(non_snake_case)]

use goish::string;

use crate::runtime::{Register, Runtime};

struct VLLM {}

impl Runtime for VLLM {
    fn Name(&self) -> string {
        string("vllm")
    }
    fn Image(&self) -> string {
        string("vllm/vllm-openai:v0.26.0")
    }
}

// Go: func init() { runtime.Register("vllm", &VLLM{}) }
#[goish::init]
fn init() {
    Register("vllm", alloc::sync::Arc::new(VLLM {}));
}
