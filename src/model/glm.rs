// GLM — Zhipu / Z.ai family (composite rank #6, Aug 2026). Recipe data from
// the Aug 2026 deployment research: weights/GPU floors/KV budgets per
// variant; ServeCmd holds the researched vLLM command where one exists.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::model::{Family, KVArch, Model, Register, Variant};

#[goish::init]
fn init() {
    Register(Family {
        Rank: 6,
        Name: string("glm"),
        Models: slice!([]Model{
            Model {
                Name: string("glm-4.5-air"),
                Size: string("106B/12B"),
                MoE: true,
                NativeContext: 131072,
                KV: KVArch {
                    Kind: string("gqa"),
                    Layers: 46,
                    KVHeads: 8,
                    HeadDim: 128,
                    MaxContext: 131072,
                    ..Default::default()
                },
                License: string("MIT"),
                Engines: slice!([]string{string("vLLM"), string("SGLang")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 106.0,
                        MinGPU: string("1xH200 / 2xH100"),
                        ProdGPU: string("2xH100"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 60.0,
                        MinGPU: string("1xH100"),
                        ProdGPU: string("2xH100"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("glm-4.6"),
                Size: string("357B/32B"),
                MoE: true,
                NativeContext: 204800,
                KV: KVArch {
                    Kind: string("gqa"),
                    Layers: 92,
                    KVHeads: 8,
                    HeadDim: 128,
                    MaxContext: 204800,
                    ..Default::default()
                },
                License: string("MIT"),
                Engines: slice!([]string{string("vLLM"), string("SGLang")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 357.0,
                        MinGPU: string("4xH200 / 8xH100"),
                        ProdGPU: string("4xH200"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 200.0,
                        MinGPU: string("2xH200 / 4xH100"),
                        ProdGPU: string("4xH100"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "Plain GQA at 200K costs 36 GB KV/seq; architecture, not params, drive long-context cost.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("glm-5"),
                Size: string("744B/40B"),
                MoE: true,
                NativeContext: 1048576,
                KV: KVArch {
                    Kind: string("flat"),
                    PerTokenB: 225000,
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
                        WeightsGB: 744.0,
                        MinGPU: string("8xH200"),
                        ProdGPU: string("8xH200"),
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
