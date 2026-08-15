// MiniMax family (composite rank #10, Aug 2026). Recipe data from
// the Aug 2026 deployment research: weights/GPU floors/KV budgets per
// variant; ServeCmd holds the researched vLLM command where one exists.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::model::{Family, KVArch, Model, Register, Variant};

#[goish::init]
fn init() {
    Register(Family {
        Rank: 10,
        Name: string("minimax"),
        Models: slice!([]Model{
            Model {
                Name: string("minimax-m2"),
                Size: string("230B/10B"),
                MoE: true,
                NativeContext: 204800,
                KV: KVArch {
                    Kind: string("flat"),
                    PerTokenB: 100000,
                    Est: true,
                    MaxContext: 204800,
                    ..Default::default()
                },
                License: string("Mixed"),
                LicenseNote: string("some releases commercially restricted"),
                Engines: slice!([]string{string("vLLM"), string("SGLang")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 230.0,
                        MinGPU: string("2xH200 / 4xH100"),
                        ProdGPU: string("4xH100"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 130.0,
                        MinGPU: string("2xH200"),
                        ProdGPU: string("2xH200"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "Read the per-checkpoint license; some MiniMax weights are commercially restricted.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("minimax-m3"),
                Size: string("456B"),
                MoE: true,
                Vision: true,
                Attention: string("MSA"),
                NativeContext: 1048576,
                KV: KVArch {
                    Kind: string("flat"),
                    PerTokenB: 175000,
                    Est: true,
                    MaxContext: 1048576,
                    ..Default::default()
                },
                License: string("Mixed"),
                LicenseNote: string("read per-checkpoint"),
                Engines: slice!([]string{string("vLLM"), string("SGLang")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 456.0,
                        MinGPU: string("8xH100 / 4xH200"),
                        ProdGPU: string("8xH100"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "First open-weight model combining native multimodality, 1M context, and computer use.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
        }),
    });
}
