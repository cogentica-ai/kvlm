// Llama — Meta family (composite rank #3, Aug 2026). Recipe data from
// the Aug 2026 deployment research: weights/GPU floors/KV budgets per
// variant; ServeCmd holds the researched vLLM command where one exists.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::model::{Family, KVArch, Model, Register, Variant};

#[goish::init]
fn init() {
    Register(Family {
        Rank: 3,
        Name: string("llama"),
        Models: slice!([]Model{
            Model {
                Name: string("llama3.1-8b"),
                Size: string("8B"),
                NativeContext: 131072,
                KV: KVArch {
                    Kind: string("gqa"),
                    Layers: 32,
                    KVHeads: 8,
                    HeadDim: 128,
                    MaxContext: 131072,
                    ..Default::default()
                },
                License: string("Llama Community License"),
                LicenseNote: string("700M-MAU clause + attribution"),
                Engines: slice!([]string{string("vLLM"), string("TensorRT-LLM"), string("llama.cpp")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 8.0,
                        MinGPU: string("1xL4"),
                        ProdGPU: string("1xL4"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 4.7,
                        MinGPU: string("1xT4"),
                        ProdGPU: string("1xL4"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "License: 700M-MAU clause + naming/attribution requirements; legal review before prod.",
                    "Plain GQA pays KV linearly: at 128K an 8B needs 8 GB KV; as much as its weights.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("llama3.3-70b"),
                Size: string("70B"),
                NativeContext: 131072,
                KV: KVArch {
                    Kind: string("gqa"),
                    Layers: 80,
                    KVHeads: 8,
                    HeadDim: 128,
                    MaxContext: 131072,
                    ..Default::default()
                },
                License: string("Llama Community License"),
                LicenseNote: string("700M-MAU clause"),
                Engines: slice!([]string{string("vLLM"), string("TensorRT-LLM"), string("llama.cpp")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("bf16"),
                        WeightsGB: 140.0,
                        MinGPU: string("2xH100"),
                        ProdGPU: string("2xH200"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 70.0,
                        MinGPU: string("1xH200"),
                        ProdGPU: string("2xH100"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 40.0,
                        MinGPU: string("1xL40S"),
                        ProdGPU: string("1xH100"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "License: 700M-MAU clause; legal review before prod.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("llama4-scout"),
                Size: string("109B/17B"),
                MoE: true,
                NativeContext: 10485760,
                KV: KVArch {
                    Kind: string("flat"),
                    PerTokenB: 96000,
                    Est: true,
                    MaxContext: 10485760,
                    ..Default::default()
                },
                License: string("Llama Community License"),
                LicenseNote: string("700M-MAU clause"),
                Engines: slice!([]string{string("vLLM"), string("TensorRT-LLM")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 109.0,
                        MinGPU: string("1xH200 / 2xH100"),
                        ProdGPU: string("2xH100"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "10M context is claimed, not a practical serving target; budget KV for real traffic.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
        }),
    });
}
