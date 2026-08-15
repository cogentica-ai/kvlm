// kvlm library crate — the packages behind the kvlm binary, exposed so
// the examples/ test targets can exercise them (ported-crates pattern:
// lib + goish testing examples).
#![no_std]
#![allow(non_snake_case)]

extern crate alloc;

pub mod cmd;
pub mod config;
pub mod driver;
pub mod flamegraph;
pub mod metrics;
pub mod model;
pub mod profile;
pub mod runtime;
pub mod state;
