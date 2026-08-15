// MiMo — Xiaomi family (composite rank #8, Aug 2026). Recipe data from
// the Aug 2026 deployment research: weights/GPU floors/KV budgets per
// variant; ServeCmd holds the researched vLLM command where one exists.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::model::{Family, KVArch, Model, Register, Variant};

#[goish::init]
fn init() {
    Register(Family {
        Rank: 8,
        Name: string("mimo"),
        Models: slice!([]Model{
            Model {
                Name: string("mimo-v2.5"),
                Size: string("310B/15B"),
                MoE: true,
                Vision: true,
                NativeContext: 1048576,
                KV: KVArch {
                    Kind: string("flat"),
                    PerTokenB: 150000,
                    Est: true,
                    MaxContext: 1048576,
                    ..Default::default()
                },
                License: string("MIT"),
                Engines: slice!([]string{string("vLLM"), string("SGLang")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 310.0,
                        MinGPU: string("4xH200 / 8xH100"),
                        ProdGPU: string("4xH200"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 175.0,
                        MinGPU: string("2xH200"),
                        ProdGPU: string("2xH200"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("mimo-v2.5-pro"),
                Size: string("1.02T/42B"),
                MoE: true,
                NativeContext: 1048576,
                KV: KVArch {
                    Kind: string("flat"),
                    PerTokenB: 175000,
                    Est: true,
                    MaxContext: 1048576,
                    ..Default::default()
                },
                License: string("MIT"),
                Engines: slice!([]string{string("vLLM"), string("SGLang")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 1000.0,
                        MinGPU: string("8xH200 / 16xH100"),
                        ProdGPU: string("16xH100"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
        }),
    });
}
