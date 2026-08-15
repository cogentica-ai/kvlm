// Kimi — Moonshot AI family (composite rank #7, Aug 2026). Recipe data from
// the Aug 2026 deployment research: weights/GPU floors/KV budgets per
// variant; ServeCmd holds the researched vLLM command where one exists.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::model::{Family, KVArch, Model, Register, Variant};

#[goish::init]
fn init() {
    Register(Family {
        Rank: 7,
        Name: string("kimi"),
        Models: slice!([]Model{
            Model {
                Name: string("kimi-k2"),
                Size: string("1T/32B"),
                MoE: true,
                Attention: string("MLA"),
                NativeContext: 131072,
                KV: KVArch {
                    Kind: string("mla"),
                    Layers: 61,
                    LatentDim: 576,
                    MaxContext: 262144,
                    ..Default::default()
                },
                License: string("Modified MIT"),
                Engines: slice!([]string{string("SGLang"), string("vLLM")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 1000.0,
                        MinGPU: string("16xH100 / 8xH200"),
                        ProdGPU: string("16xH100"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 594.0,
                        MinGPU: string("8xH100"),
                        ProdGPU: string("8xH200"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("1.8-bit"),
                        WeightsGB: 245.0,
                        MinGPU: string("4xH100"),
                        ProdGPU: string("4xH100"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "MLA: a 1T-param model with smaller KV than a dense 32B.",
                    "Modified MIT: attribution clause above a usage threshold.",
                    "Community quants of huge MoE vary 20%+ between packagers (mixed-precision choices).",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("kimi-k3"),
                Size: string("2.8T"),
                MoE: true,
                Attention: string("MLA"),
                NativeContext: 1048576,
                KV: KVArch {
                    Kind: string("flat"),
                    PerTokenB: 60000,
                    Est: true,
                    MaxContext: 1048576,
                    ..Default::default()
                },
                License: string("Modified MIT"),
                Engines: slice!([]string{string("SGLang"), string("vLLM")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 2800.0,
                        MinGPU: string("multi-node only"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "Largest open-weight model ever released (weights July 27 2026).",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
        }),
    });
}
