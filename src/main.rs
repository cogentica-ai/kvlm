// kvlm — a key-value CLI in Goish Rust, structured like a cobra-generator
// application (main.go → main.rs, packages in the kvlm lib crate).
#![no_std]
#![no_main]
#![allow(non_snake_case)]

extern crate alloc;

#[goish::main]
fn main() {
    kvlm::cmd::Execute();
}
