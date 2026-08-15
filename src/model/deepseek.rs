// DeepSeek family (composite rank #2, Aug 2026). Recipe data from
// the Aug 2026 deployment research: weights/GPU floors/KV budgets per
// variant; ServeCmd holds the researched vLLM command where one exists.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::model::{Family, KVArch, Model, Register, Variant};

#[goish::init]
fn init() {
    Register(Family {
        Rank: 2,
        Name: string("deepseek"),
        Models: slice!([]Model{
            Model {
                Name: string("deepseek-v4-flash"),
                Size: string("284B/13B"),
                MoE: true,
                Attention: string("MLA"),
                NativeContext: 1048576,
                KV: KVArch {
                    Kind: string("flat"),
                    PerTokenB: 35000,
                    Est: true,
                    MaxContext: 1048576,
                    ..Default::default()
                },
                License: string("MIT"),
                Engines: slice!([]string{string("SGLang"), string("vLLM")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 284.0,
                        MinGPU: string("4xH200 / 8xH100"),
                        ProdGPU: string("4xH200"),
                        ..Default::default()
                    },
                    Variant {
                        Name: string("q4"),
                        WeightsGB: 160.0,
                        MinGPU: string("2xH200"),
                        ProdGPU: string("2xH200"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "MLA compresses KV to ~35 KB/token regardless of size; why 1M context is viable.",
                    "MoE floor is set by TOTAL params (284 GB in memory), active 13B buys throughput only.",
                    "vLLM v0.20.0/v0.20.1 were largely DeepSeek V4 stabilization releases (hang + KV-allocation fixes).",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("deepseek-v4-pro"),
                Size: string("1.6T/49B"),
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
                License: string("MIT"),
                Engines: slice!([]string{string("SGLang")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 1600.0,
                        MinGPU: string("16-24xH200 multi-node"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "Multi-node only; expert parallelism required.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
            Model {
                Name: string("deepseek-r1"),
                Size: string("671B/37B"),
                MoE: true,
                Attention: string("MLA"),
                NativeContext: 131072,
                KV: KVArch {
                    Kind: string("mla"),
                    Layers: 61,
                    LatentDim: 576,
                    MaxContext: 131072,
                    ..Default::default()
                },
                License: string("MIT"),
                Engines: slice!([]string{string("SGLang"), string("vLLM")}),
                Image: string("vllm/vllm-openai:v0.26.0"),
                Variants: slice!([]Variant{
                    Variant {
                        Name: string("fp8"),
                        WeightsGB: 671.0,
                        MinGPU: string("8xH200"),
                        ProdGPU: string("8xH200"),
                        ..Default::default()
                    },
                }),
                Notes: slice!([]string{
                    "Most-liked open LLM on HF; superseded for inference by V4.",
                    "vLLM flag recipe pending; hardware/KV facts only; Qwen was researched first.",
                }),
                ..Default::default()
            },
        }),
    });
}
