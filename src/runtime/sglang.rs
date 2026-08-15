// SGLang runtime — reference implementation for the DeepSeek family.
// TODO: pin a version tag before prod (:latest is mutable).
#![allow(non_snake_case)]

use goish::string;

use crate::runtime::{Register, Runtime};

struct SGLang {}

impl Runtime for SGLang {
    fn Name(&self) -> string {
        string("sglang")
    }
    fn Image(&self) -> string {
        string("lmsysorg/sglang:latest")
    }
}

// Go: func init() { runtime.Register("sglang", &SGLang{}) }
#[goish::init]
fn init() {
    Register("sglang", alloc::sync::Arc::new(SGLang {}));
}
