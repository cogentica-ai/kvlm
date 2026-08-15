// Gemma — Google family (composite rank #4, Aug 2026). Recipe data from
// the Aug 2026 deployment research: weights/GPU floors/KV budgets per
// variant; ServeCmd holds the researched vLLM command where one exists.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::model::{Family, KVArch, Model, Register, Variant};

#[goish::init]
fn init() {
    Register(Family {
        Rank: 4,
        Name: string("gemma"),
        Models: slice!([]Model{
            Model {
                Name: string("gemma3-4b"),
                Size: string("4B"),
                Attention: string("SWA"),
                NativeContext: 131072,
                KV: KVArch {
                    Kind: string("swa"),
                    Layers: 34,
                    KVHeads: 4,
                    HeadDim: 256,
                    Window: 1024,
                    FullEvery: 6,
                    MaxContext: 131072,
                    ..Default::default()
                },
                License: string("Gemma Terms"),
                Engines: slice!([]string{string("vLLM"), string("llama.cpp"), string("Ollama")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 2.6,
                        MinGPU: string("CPU / phone-class"),
                        ProdGPU: string("1xL4"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "Sliding-window attention (5/6 layers at 1K window) keeps KV tiny.",
                    "KV figures computed from the published config (5:1 local:global, window 1024); supersede the earlier rough estimates.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("gemma3-12b"),
                Size: string("12B"),
                Attention: string("SWA"),
                NativeContext: 131072,
                KV: KVArch {
                    Kind: string("swa"),
                    Layers: 48,
                    KVHeads: 8,
                    HeadDim: 256,
                    Window: 1024,
                    FullEvery: 6,
                    MaxContext: 131072,
                    ..Default::default()
                },
                License: string("Gemma Terms"),
                Engines: slice!([]string{string("vLLM"), string("llama.cpp"), string("Ollama")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 7.0,
                        MinGPU: string("1xL4"),
                        ProdGPU: string("1xL4"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("gemma3-27b"),
                Size: string("27B"),
                Attention: string("SWA"),
                NativeContext: 131072,
                KV: KVArch {
                    Kind: string("swa"),
                    Layers: 62,
                    KVHeads: 16,
                    HeadDim: 128,
                    Window: 1024,
                    FullEvery: 6,
                    MaxContext: 131072,
                    ..Default::default()
                },
                License: string("Gemma Terms"),
                LicenseNote: string("Gemma 4: Apache 2.0"),
                Engines: slice!([]string{string("vLLM"), string("llama.cpp"), string("Ollama")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("bf16"),
                        WeightsGB: 54.0,
                        MinGPU: string("1xH100"),
                        ProdGPU: string("1xH100"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("int4-qat"),
                        WeightsGB: 15.0,
                        MinGPU: string("1xA10G"),
                        ProdGPU: string("1xL40S"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "Official INT4 QAT checkpoints exist; prefer them over community quants.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
        }),
    });
}
