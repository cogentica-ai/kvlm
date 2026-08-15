// gpt-oss — OpenAI family (composite rank #5, Aug 2026). Recipe data from
// the Aug 2026 deployment research: weights/GPU floors/KV budgets per
// variant; ServeCmd holds the researched vLLM command where one exists.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::model::{Family, KVArch, Model, Register, Variant};

#[goish::init]
fn init() {
    Register(Family {
        Rank: 5,
        Name: string("gpt-oss"),
        Models: slice!([]Model{
            Model {
                Name: string("gpt-oss-20b"),
                Size: string("21B/3.6B"),
                MoE: true,
                Attention: string("SWA"),
                NativeContext: 131072,
                KV: KVArch {
                    Kind: string("swa"),
                    Layers: 24,
                    KVHeads: 8,
                    HeadDim: 64,
                    Window: 128,
                    FullEvery: 2,
                    MaxContext: 131072,
                    ..Default::default()
                },
                License: string("Apache 2.0"),
                Engines: slice!([]string{string("vLLM"), string("Ollama"), string("llama.cpp"), string("TensorRT-LLM")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("mxfp4"),
                        WeightsGB: 12.8,
                        MinGPU: string("1xA10G / 16GB RAM"),
                        ProdGPU: string("1xL40S"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "Ships MXFP4-native; no separate quantization step needed.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("gpt-oss-120b"),
                Size: string("117B/5.1B"),
                MoE: true,
                Attention: string("SWA"),
                NativeContext: 131072,
                KV: KVArch {
                    Kind: string("swa"),
                    Layers: 36,
                    KVHeads: 8,
                    HeadDim: 64,
                    Window: 128,
                    FullEvery: 2,
                    MaxContext: 131072,
                    ..Default::default()
                },
                License: string("Apache 2.0"),
                Engines: slice!([]string{string("vLLM"), string("Ollama"), string("llama.cpp"), string("TensorRT-LLM")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("mxfp4"),
                        WeightsGB: 61.0,
                        MinGPU: string("1xH100"),
                        ProdGPU: string("1xH100"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("bf16"),
                        WeightsGB: 235.0,
                        MinGPU: string("4xH100"),
                        ProdGPU: string("4xH100"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "Full 128K on one H100 with 2.4 GB KV; less than a Llama 8B; interleaved sliding-window attention.",
                    "Best long-context concurrency per dollar in the top 10.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
        }),
    });
}
