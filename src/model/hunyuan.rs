// Hunyuan — Tencent family (composite rank #9, Aug 2026). Recipe data from
// the Aug 2026 deployment research: weights/GPU floors/KV budgets per
// variant; ServeCmd holds the researched vLLM command where one exists.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::model::{Family, KVArch, Model, Register, Variant};

#[goish::init]
fn init() {
    Register(Family {
        Rank: 9,
        Name: string("hunyuan"),
        Models: slice!([]Model{
            Model {
                Name: string("hunyuan-a13b"),
                Size: string("80B/13B"),
                MoE: true,
                NativeContext: 262144,
                KV: KVArch {
                    Kind: string("gqa"),
                    Layers: 32,
                    KVHeads: 8,
                    HeadDim: 128,
                    MaxContext: 262144,
                    ..Default::default()
                },
                License: string("Apache 2.0"),
                Engines: slice!([]string{string("vLLM"), string("SGLang"), string("llama.cpp")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 80.0,
                        MinGPU: string("1xH200 / 2xH100"),
                        ProdGPU: string("2xH100"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 45.0,
                        MinGPU: string("1xL40S"),
                        ProdGPU: string("1xH100"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("hunyuan-hy3"),
                Size: string("295B/21B"),
                MoE: true,
                NativeContext: 262144,
                KV: KVArch {
                    Kind: string("flat"),
                    PerTokenB: 125000,
                    Est: true,
                    MaxContext: 262144,
                    ..Default::default()
                },
                License: string("Apache 2.0"),
                LicenseNote: string("April preview license excluded EU/UK/KR"),
                Engines: slice!([]string{string("vLLM"), string("SGLang"), string("llama.cpp")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 295.0,
                        MinGPU: string("4xH200 / 8xH100"),
                        ProdGPU: string("4xH200"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 165.0,
                        MinGPU: string("2xH200"),
                        ProdGPU: string("2xH200"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("1-bit"),
                        WeightsGB: 40.0,
                        MinGPU: string("1xH100"),
                        ProdGPU: string("1xH100"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "Official 1-/4-bit GGUF quants exist for single-GPU llama.cpp serving.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
        }),
    });
}
