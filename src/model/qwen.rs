// Qwen — Alibaba family (composite rank #1, Aug 2026).
//
// Layout: one `fn <model>() -> Model` per checkpoint, registered in
// init() at the bottom. Serve invocations are structured (ServeSpec):
// drivers consume Argv() for the real platform; `kvlm model show`
// derives the shell form via Render().
//
// Provenance: Aug 2026 deployment research. qwen3-32b flags verified
// 2026-08-07 against docs.vllm.ai (reasoning_outputs, tool_calling,
// quantized_kvcache, awq_marlin) and the Qwen3-32B-FP8 model card; see
// that model's Notes for what the verification changed.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;

use crate::model::{Family, Flag, Hardware, KVArch, Model, Quant, Register, ServeSpec, Variant};

// Family-wide default image. Per-variant Image fields add the version
// floor that particular weight format needs; per-target tags live in
// each variant's Hardware table.
const IMG_V0260: &str = "vllm/vllm-openai:v0.26.0";
const ROCM_NOTE: &str = "bundles ROCm 7.2.2 (>=v0.21; 7.2.1 for v0.19-v0.20); Python 3.12, glibc >= 2.35";

// flag / boolFlag — literal-noise reducers for ServeSpec tables.
fn flag(name: &'static str, value: &'static str) -> Flag {
    Flag {
        Name: string(name),
        Value: string(value),
    }
}
fn boolFlag(name: &'static str) -> Flag {
    Flag {
        Name: string(name),
        Value: string(""),
    }
}

// ── qwen3-32b — dense 32B ──────────────────────────────────────────────

fn qwen3_32b() -> Model {
    Model {
        Name: string("qwen3-32b"),
        Size: string("32B"),
        NativeContext: 32768,
        ContextMethod: string("YaRN"),
        KV: KVArch {
            Kind: string("gqa"),
            Layers: 64,
            KVHeads: 8,
            HeadDim: 128,
            MaxContext: 131072,
            ..Default::default()
        },
        License: string("Apache 2.0"),
        Engines: slice!([]string{
            string("vLLM"), string("SGLang"), string("LMDeploy"), string("TensorRT-LLM"),
        }),
        Image: string(IMG_V0260),
        Variants: slice!([]Variant{
            Variant {
                Name: string("bf16"),
                Image: string("vllm/vllm-openai:v0.26.0"),
                WeightsGB: 64.0,
                MinGPU: string("1xH100"),
                ProdGPU: string("2xH100"),
                Serve: slice!([]ServeSpec{
                    ServeSpec {
                        Model: string("Qwen/Qwen3-32B"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3-32b"),
                            flag("--max-model-len", "16384"),
                            flag("--max-num-seqs", "16"),
                            flag("--max-num-batched-tokens", "8192"),
                            flag("--gpu-memory-utilization", "0.92"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                }),
                Hardware: slice!([]Hardware{
                    Hardware {
                        Target: string("hopper/ada (CUDA 13)"),
                        Key: string("cuda"),
                        Image: string("vllm/vllm-openai:v0.26.0"),
                        Floor: string("v0.8.5"),
                        Note: string("default image is CUDA 13 since v0.20.0; check nvidia-smi"),
                        ..Default::default()
                    },
                    Hardware {
                        Target: string("cuda 12.9 drivers"),
                        Image: string("vllm/vllm-openai:v0.26.0-cu129-ubuntu2404"),
                        Floor: string("v0.8.5"),
                        Note: string("-"),
                        ..Default::default()
                    },
                    Hardware {
                        Target: string("grace (aarch64)"),
                        Key: string("arm"),
                        Image: string("vllm/vllm-openai:v0.26.0-aarch64-cu129-ubuntu2404"),
                        Floor: string("v0.8.5"),
                        Note: string("Grace-Hopper / Grace-Blackwell ARM hosts"),
                        ..Default::default()
                    },
                    Hardware {
                        Target: string("mi300x/mi355x (ROCm)"),
                        Key: string("rocm"),
                        Image: string("vllm/vllm-openai-rocm:v0.26.0"),
                        Floor: string("v0.8.5"),
                        Note: string(ROCM_NOTE),
                        ..Default::default()
                    },
                }),
                ..Default::default()
            },
            Variant {
                Name: string("fp8"),
                Image: string("vllm/vllm-openai:v0.26.0"),
                WeightsGB: 32.0,
                MinGPU: string("1xL40S"),
                ProdGPU: string("1xH100"),
                Serve: slice!([]ServeSpec{
                    ServeSpec {
                        Model: string("Qwen/Qwen3-32B-FP8"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3-32b"),
                            flag("--max-model-len", "32768"),
                            flag("--max-num-seqs", "32"),
                            flag("--max-num-batched-tokens", "8192"),
                            flag("--gpu-memory-utilization", "0.90"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-auto-tool-choice"),
                            flag("--tool-call-parser", "hermes"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                    // vLLM tuning docs + the official Qwen3-32B recipe
                    // page: concurrency is KV-pool arithmetic. 2xH100 at
                    // 0.95 utilization minus fp8 weights leaves ~117 GB;
                    // at 16K each sequence costs 2.15 GB, so ~54 fit at
                    // full length (more in practice, most requests run
                    // shorter). Chunked prefill is on by default in V1
                    // and already decode-prioritizing.
                    ServeSpec {
                        Profile: string("high concurrency"),
                        Note: string("2xH100 at 0.95 util with fp8 KV leaves ~117 GB pool, ~54 seqs at full 16K; cap max-model-len at real traffic length, it is admission control"),
                        Model: string("Qwen/Qwen3-32B-FP8"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3-32b"),
                            flag("-tp", "2"),
                            flag("--max-model-len", "16384"),
                            flag("--max-num-seqs", "128"),
                            flag("--max-num-batched-tokens", "16384"),
                            flag("--gpu-memory-utilization", "0.95"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-auto-tool-choice"),
                            flag("--tool-call-parser", "hermes"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                }),
                Hardware: slice!([]Hardware{
                    Hardware {
                        Target: string("hopper/ada (CUDA 13)"),
                        Key: string("cuda"),
                        Image: string("vllm/vllm-openai:v0.26.0"),
                        Floor: string("v0.18.0"),
                        FloorRef: string("#35656"),
                        Note: string("native FP8 tensor cores; real speedup, not just memory"),
                        ..Default::default()
                    },
                    Hardware {
                        Target: string("blackwell (B200)"),
                        Key: string("blackwell"),
                        Image: string("vllm/vllm-openai:v0.26.0"),
                        Floor: string("v0.19.0"),
                        FloorRef: string("hard floor"),
                        Note: string("FP8-KV accuracy silently degraded through v0.18.x (#37618/#38083); eval fp8 KV vs auto"),
                        ..Default::default()
                    },
                    Hardware {
                        Target: string("mi300x (ROCm)"),
                        Key: string("rocm"),
                        Image: string("vllm/vllm-openai-rocm:v0.26.0"),
                        Floor: string("v0.18.0"),
                        Note: string("MI300X-native FP8 (e4m3fnuz)"),
                        ..Default::default()
                    },
                }),
                ..Default::default()
            },
            Variant {
                Name: string("awq-q4"),
                Image: string("vllm/vllm-openai:v0.26.0"),
                WeightsGB: 19.0,
                MinGPU: string("1xA10G"),
                ProdGPU: string("1xL40S"),
                Serve: slice!([]ServeSpec{
                    ServeSpec {
                        Model: string("Qwen/Qwen3-32B-AWQ"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3-32b"),
                            flag("--quantization", "awq_marlin"),
                            flag("--max-model-len", "16384"),
                            flag("--max-num-seqs", "8"),
                            flag("--max-num-batched-tokens", "4096"),
                            flag("--gpu-memory-utilization", "0.94"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                }),
                Hardware: slice!([]Hardware{
                    Hardware {
                        Target: string("ampere+ (CUDA)"),
                        Key: string("cuda"),
                        Image: string("vllm/vllm-openai:v0.26.0"),
                        Floor: string("v0.19.0"),
                        Note: string("awq_marlin needs SM80+ (A10G/A100 up)"),
                        ..Default::default()
                    },
                    Hardware {
                        Target: string("mi300x (ROCm)"),
                        Key: string("rocm"),
                        Image: string("vllm/vllm-openai-rocm:v0.26.0"),
                        Floor: string("v0.19.0"),
                        FloorRef: string("#36505"),
                        Note: string(ROCM_NOTE),
                        ..Default::default()
                    },
                }),
                ..Default::default()
            },
            Variant {
                Name: string("nvfp4"),
                Image: string("vllm/vllm-openai:v0.26.0"),
                WeightsGB: 18.0,
                MinGPU: string("1xRTX Pro 6000 / B200"),
                ProdGPU: string("1xB200"),
                Serve: slice!([]ServeSpec{
                    ServeSpec {
                        Note: string("quantization is auto-detected from the checkpoint; do not pass --quantization, a mismatch with the compressed-tensors config errors out"),
                        Model: string("RedHatAI/Qwen3-32B-NVFP4"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3-32b"),
                            flag("--max-model-len", "32768"),
                            flag("--max-num-seqs", "32"),
                            flag("--max-num-batched-tokens", "8192"),
                            flag("--gpu-memory-utilization", "0.90"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-auto-tool-choice"),
                            flag("--tool-call-parser", "hermes"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                    ServeSpec {
                        Profile: string("a16 fallback for pre-Blackwell"),
                        Note: string("FP4 weights with 16-bit activations; the memory savings without FP4 tensor cores, so it runs on Hopper/Ada"),
                        Model: string("RedHatAI/Qwen3-32B-NVFP4A16"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3-32b"),
                            flag("--max-model-len", "32768"),
                            flag("--max-num-seqs", "16"),
                            flag("--max-num-batched-tokens", "8192"),
                            flag("--gpu-memory-utilization", "0.90"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-auto-tool-choice"),
                            flag("--tool-call-parser", "hermes"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                }),
                Hardware: slice!([]Hardware{
                    Hardware {
                        Target: string("blackwell (B200 / RTX Pro 6000)"),
                        Key: string("blackwell"),
                        Image: string("vllm/vllm-openai:v0.26.0"),
                        Floor: string("v0.24.0"),
                        FloorRef: string("NVFP4 W4A4 gemm"),
                        Note: string("serve RedHatAI/Qwen3-32B-NVFP4 (llm-compressor); nvidia/Qwen3-32B-NVFP4 is ModelOpt format, TRT-LLM-first"),
                        ..Default::default()
                    },
                    Hardware {
                        Target: string("hopper/ada (NVFP4A16)"),
                        Key: string("cuda"),
                        Profile: string("a16 fallback for pre-Blackwell"),
                        Image: string("vllm/vllm-openai:v0.26.0"),
                        Floor: string("v0.9.1"),
                        Note: string("weight-only path: FP4 weights, 16-bit activations, no FP4 tensor cores needed"),
                        ..Default::default()
                    },
                }),
                ..Default::default()
            },
        }),
        Quants: qwen3_32b_quants(),
        Notes: slice!([]string{
            "Reasoning parser qwen3; tool-call parser hermes (Qwen3 generation).",
            "bf16 on 1xH100 is KV-starved (~4 concurrent at 16K); prefer fp8 or add -tp 2.",
            "For q4 use AWQ with the awq_marlin kernel (10.9x vs plain awq); do not serve GGUF from vLLM.",
            "INT4 is the wrong choice for reasoning-heavy/coding work (visible quality loss).",
            "NVFP4 is near-lossless on evals (99.8% OpenLLM V1 recovery; AIME 0.80 vs 0.81 bf16), making it the q4 pick on Blackwell over AWQ.",
            "--calculate-kv-scales was removed in vLLM v0.19: uncalibrated FP8 KV now runs at scale 1.0. For accuracy-sensitive work use an llm-compressor KV-calibrated checkpoint, or drop --kv-cache-dtype fp8.",
            "HF model card still shows the v0.8.5-era '--enable-reasoning --reasoning-parser deepseek_r1'; superseded by --reasoning-parser qwen3 (vLLM >= 0.9; --enable-reasoning no longer exists).",
            "ROCm images swap the ROCm runtime under you across vLLM version boundaries; treat upgrades as host-driver compat checks.",
        }),
        ..Default::default()
    }
}

// Full published-quantization inventory (researched 2026-08-07; see the
// HF quantized-models tree for Qwen/Qwen3-32B).
fn qwen3_32b_quants() -> goish::slice<Quant> {
    slice!([]Quant{
        Quant {
            Format: string("bf16"),
            Bits: string("16"),
            Repo: string("Qwen/Qwen3-32B"),
            Provenance: string("official"),
            Engines: slice!([]string{string("vLLM"), string("SGLang"), string("TRT-LLM"), string("llama.cpp")}),
            Note: string("reference precision"),
            ..Default::default()
        },
        Quant {
            Format: string("fp8"),
            Bits: string("8"),
            Repo: string("Qwen/Qwen3-32B-FP8"),
            Provenance: string("official"),
            Engines: slice!([]string{string("vLLM"), string("SGLang")}),
            Note: string("e4m3, block-128 scales; Hopper/Ada+"),
            ..Default::default()
        },
        Quant {
            Format: string("fp8-dynamic"),
            Bits: string("8"),
            Repo: string("RedHatAI/Qwen3-32B-FP8-dynamic"),
            Engines: slice!([]string{string("vLLM")}),
            Note: string("per-token scales, no calibration"),
            ..Default::default()
        },
        Quant {
            Format: string("int8-w8a8"),
            Bits: string("8"),
            Repo: string("Bedovyy/Qwen3-32B.w8a8"),
            Provenance: string("community"),
            Engines: slice!([]string{string("vLLM")}),
            Kernel: string("compressed-tensors"),
            Note: string("Ampere-friendly (no FP8 HW needed)"),
            ..Default::default()
        },
        Quant {
            Format: string("gptq-int8"),
            Bits: string("8"),
            Repo: string("JunHowie/Qwen3-32B-GPTQ-Int8"),
            Provenance: string("community"),
            Engines: slice!([]string{string("vLLM")}),
            ..Default::default()
        },
        Quant {
            Format: string("awq-int4"),
            Bits: string("4"),
            Repo: string("Qwen/Qwen3-32B-AWQ"),
            Provenance: string("official"),
            Engines: slice!([]string{string("vLLM")}),
            Kernel: string("awq_marlin"),
            Note: string("the recipe q4 pick"),
            ..Default::default()
        },
        Quant {
            Format: string("gptq-int4"),
            Bits: string("4"),
            Repo: string("JunHowie/Qwen3-32B-GPTQ-Int4"),
            Provenance: string("community"),
            Engines: slice!([]string{string("vLLM")}),
            Kernel: string("gptq_marlin"),
            ..Default::default()
        },
        Quant {
            Format: string("int4-w4a16"),
            Bits: string("4"),
            Repo: string("RedHatAI/Qwen3-32B-quantized.w4a16"),
            Engines: slice!([]string{string("vLLM")}),
            Kernel: string("compressed-tensors"),
            Note: string("llm-compressor"),
            ..Default::default()
        },
        Quant {
            Format: string("nvfp4"),
            Bits: string("4"),
            Repo: string("nvidia/Qwen3-32B-NVFP4"),
            Provenance: string("official"),
            Engines: slice!([]string{string("vLLM"), string("TRT-LLM")}),
            Note: string("Blackwell only"),
            ..Default::default()
        },
        Quant {
            Format: string("nvfp4a16"),
            Bits: string("4"),
            Repo: string("RedHatAI/Qwen3-32B-NVFP4A16"),
            Engines: slice!([]string{string("vLLM")}),
            Note: string("FP4 weights/16-bit act; runs pre-Blackwell"),
            ..Default::default()
        },
        Quant {
            Format: string("bnb-4bit"),
            Bits: string("4"),
            Repo: string("unsloth/Qwen3-32B-unsloth-bnb-4bit"),
            Provenance: string("community"),
            Engines: slice!([]string{string("transformers"), string("unsloth")}),
            Note: string("fine-tuning, not serving"),
            ..Default::default()
        },
        Quant {
            Format: string("gguf"),
            Bits: string("1.7-8"),
            Repo: string("Qwen, unsloth, bartowski, lmstudio-community"),
            Engines: slice!([]string{string("llama.cpp"), string("Ollama"), string("LM Studio")}),
            Note: string("NOT vLLM (~8x slower there); unsloth has 128K-YaRN repo"),
            ..Default::default()
        },
        Quant {
            Format: string("mlx"),
            Bits: string("4/8"),
            Repo: string("mlx-community, lmstudio-community"),
            Engines: slice!([]string{string("MLX")}),
            Note: string("Apple Silicon"),
            ..Default::default()
        },
        Quant {
            Format: string("exl2"),
            Bits: string("4bpw"),
            Repo: string("Jellon/Qwen3-32B-exl2-4bpw"),
            Provenance: string("community"),
            Engines: slice!([]string{string("exllamav2"), string("TabbyAPI")}),
            ..Default::default()
        },
        Quant {
            Format: string("autoround"),
            Bits: string("2"),
            Repo: string("kaitchup/Qwen3-32B-autoround-2bit-gptq"),
            Provenance: string("community"),
            Engines: slice!([]string{string("vLLM")}),
            Kernel: string("gptq loader"),
            Note: string("extreme compression, visible quality loss"),
            ..Default::default()
        },
    })
}

// ── qwen3-30b-a3b — MoE 30B/3B ─────────────────────────────────────────

fn qwen3_30b_a3b() -> Model {
    Model {
        Name: string("qwen3-30b-a3b"),
        Size: string("30B/3B"),
        MoE: true,
        NativeContext: 262144,
        KV: KVArch {
            Kind: string("gqa"),
            Layers: 48,
            KVHeads: 4,
            HeadDim: 128,
            MaxContext: 1048576,
            ..Default::default()
        },
        License: string("Apache 2.0"),
        Engines: slice!([]string{
            string("vLLM"), string("SGLang"), string("LMDeploy"), string("TensorRT-LLM"),
        }),
        Image: string(IMG_V0260),
        Variants: slice!([]Variant{
            Variant {
                Name: string("fp8"),
                WeightsGB: 31.0,
                MinGPU: string("1xL40S"),
                ProdGPU: string("1xH100"),
                Serve: slice!([]ServeSpec{
                    ServeSpec {
                        Model: string("Qwen/Qwen3-30B-A3B-FP8"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3-30b-a3b"),
                            flag("--max-model-len", "65536"),
                            flag("--max-num-seqs", "64"),
                            flag("--max-num-batched-tokens", "16384"),
                            flag("--gpu-memory-utilization", "0.90"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-auto-tool-choice"),
                            flag("--tool-call-parser", "hermes"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                }),
                ..Default::default()
            },
            Variant {
                Name: string("awq-q4"),
                WeightsGB: 17.0,
                MinGPU: string("1xA10G"),
                ProdGPU: string("1xL40S"),
                Serve: slice!([]ServeSpec{
                    ServeSpec {
                        Model: string("Qwen/Qwen3-30B-A3B-AWQ"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3-30b-a3b"),
                            flag("--quantization", "moe_wna16"),
                            flag("--max-model-len", "32768"),
                            flag("--max-num-seqs", "24"),
                            flag("--max-num-batched-tokens", "8192"),
                            flag("--gpu-memory-utilization", "0.94"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                }),
                ..Default::default()
            },
        }),
        Notes: slice!([]string{
            "Best single-GPU config in the family: KV is 2.7x cheaper per token than dense 32B.",
            "Quantized MoE uses the moe_wna16 kernel, not awq_marlin (that is dense-only).",
            "--enable-expert-parallel does nothing on a single GPU; add it only at -tp 2+.",
            "Requires vLLM >= v0.20.0 (MoE double gate-call fix, free perf win).",
        }),
        ..Default::default()
    }
}

// ── qwen3.8-27b — dense 27B, GDN hybrid, multimodal ────────────────────
//
// Provenance: 2026-08-15 release research. Architecture from the
// official model card (64 layers as 16 x (3 x GatedDeltaNet + 1 x
// GatedAttention), GQA 4 KV heads x 256 dim on the full layers, native
// 262K context, 1M via rope override). Serve flags from the vLLM
// recipes page for this checkpoint (vLLM >= 0.17 for the GDN hybrid
// class, so the pinned image clears it).

fn qwen3_8_27b() -> Model {
    Model {
        Name: string("qwen3.8-27b"),
        Size: string("27B"),
        Vision: true,
        NativeContext: 262144,
        ContextMethod: string("YaRN"),
        // GDN hybrid: KV grows only in the 16 full-attention layers
        // (1 in 4); the 48 linear layers hold a constant delta-rule
        // state (est. 48 v-heads x 128x128 state x 2B per layer).
        KV: KVArch {
            Kind: string("kda"),
            Layers: 64,
            KVHeads: 4,
            HeadDim: 256,
            FullEvery: 4,
            StateB: 1572864,
            Est: true,
            MaxContext: 1000000,
            ..Default::default()
        },
        License: string("Apache 2.0"),
        Engines: slice!([]string{
            string("vLLM"), string("SGLang"), string("TokenSpeed"),
        }),
        Image: string(IMG_V0260),
        Variants: slice!([]Variant{
            Variant {
                Name: string("fp8"),
                Image: string(IMG_V0260),
                WeightsGB: 28.0,
                MinGPU: string("1xL40S"),
                ProdGPU: string("1xH100"),
                MeasuredTokS: 79,
                Serve: slice!([]ServeSpec{
                    ServeSpec {
                        Model: string("Qwen/Qwen3.8-27B-FP8"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3.8-27b"),
                            flag("--max-model-len", "262144"),
                            flag("--max-num-seqs", "32"),
                            flag("--max-num-batched-tokens", "8192"),
                            flag("--gpu-memory-utilization", "0.90"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-auto-tool-choice"),
                            flag("--tool-call-parser", "qwen3_coder"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                }),
                ..Default::default()
            },
            Variant {
                Name: string("bf16"),
                Image: string(IMG_V0260),
                WeightsGB: 55.0,
                MinGPU: string("1xH100"),
                ProdGPU: string("1xH100"),
                Serve: slice!([]ServeSpec{
                    ServeSpec {
                        Note: string("bf16 weights leave ~15 GB of KV pool on one H100; max-model-len capped to keep admission honest"),
                        Model: string("Qwen/Qwen3.8-27B"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3.8-27b"),
                            flag("--max-model-len", "131072"),
                            flag("--max-num-seqs", "16"),
                            flag("--max-num-batched-tokens", "8192"),
                            flag("--gpu-memory-utilization", "0.90"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-auto-tool-choice"),
                            flag("--tool-call-parser", "qwen3_coder"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                }),
                ..Default::default()
            },
            Variant {
                Name: string("nvfp4"),
                Image: string(IMG_V0260),
                WeightsGB: 25.0,
                MinGPU: string("1xRTX Pro 6000 / B200"),
                ProdGPU: string("1xB200"),
                Serve: slice!([]ServeSpec{
                    ServeSpec {
                        Note: string("quantization auto-detected from the checkpoint; MXFP4 does not load on NVIDIA devices in vLLM, NVFP4 is the working 4-bit path"),
                        Model: string("Inferact/Qwen3.8-27B-NVFP4"),
                        Flags: slice!([]Flag{
                            flag("--served-model-name", "qwen3.8-27b"),
                            flag("--max-model-len", "262144"),
                            flag("--max-num-seqs", "32"),
                            flag("--max-num-batched-tokens", "8192"),
                            flag("--gpu-memory-utilization", "0.90"),
                            flag("--kv-cache-dtype", "fp8"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-auto-tool-choice"),
                            flag("--tool-call-parser", "qwen3_coder"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                }),
                ..Default::default()
            },
        }),
        Quants: slice!([]Quant{
            Quant {
                Format: string("bf16"),
                Bits: string("16"),
                Repo: string("Qwen/Qwen3.8-27B"),
                Provenance: string("official"),
                Engines: slice!([]string{string("vLLM"), string("SGLang"), string("llama.cpp")}),
                Note: string("reference precision, vision included"),
                ..Default::default()
            },
            Quant {
                Format: string("fp8"),
                Bits: string("8"),
                Repo: string("Qwen/Qwen3.8-27B-FP8"),
                Provenance: string("official"),
                Engines: slice!([]string{string("vLLM"), string("SGLang")}),
                Note: string("fine-grained, block-128 scales; near-identical metrics per the card"),
                ..Default::default()
            },
            Quant {
                Format: string("nvfp4"),
                Bits: string("4"),
                Repo: string("Inferact/Qwen3.8-27B-NVFP4"),
                Provenance: string("community"),
                Engines: slice!([]string{string("vLLM")}),
                Note: string("Blackwell; 24.6 GiB per the recipes page"),
                ..Default::default()
            },
            Quant {
                Format: string("nvfp4a16"),
                Bits: string("4"),
                Repo: string("huginnfork/Qwen3.8-27B-NVFP4A16"),
                Provenance: string("community"),
                Engines: slice!([]string{string("vLLM")}),
                Note: string("FP4 weights/16-bit act; runs pre-Blackwell"),
                ..Default::default()
            },
            Quant {
                Format: string("gguf"),
                Bits: string("2-8"),
                Repo: string("unsloth, lmstudio-community"),
                Engines: slice!([]string{string("llama.cpp"), string("Ollama"), string("LM Studio")}),
                Note: string("NOT vLLM"),
                ..Default::default()
            },
        }),
        Notes: slice!([]string{
            "Measured 2026-08-15, fp8 on RunPod 1xH100 (vLLM 0.26.0, recipe flags, via the platform http proxy): single-seq decode 79 tok/s (streaming, median of 3 x 512 tok); 8 streams 446 tok/s aggregate at 56 per stream; 32 streams 1,442 tok/s aggregate at 47 per stream; prefill 7,900 tok/s (30K-token prompt, cold), 3.7x faster warm via prefix cache; TTFT 0.6-1.4 s proxy-inclusive.",
            "Decode beats the dense bandwidth estimate (~56) by 40%: the 48 linear layers read no per-token KV, so per-stream speed holds nearly flat from batch 1 (79) to batch 32 (47) and aggregate scales to 1.4x the dense-32B figure on the same GPU.",
            "GDN hybrid: 48 of 64 layers are linear attention, so KV per token is ~4x smaller than a dense-attention 27B; long context is this model's home turf.",
            "Tool-call parser is qwen3_coder for the 3.8 generation (parser drifts per generation; 32B-era hermes does not apply).",
            "Thinking mode is on by default and controlled per request (reasoning_effort); no serve flag needed.",
            "Native vision-language; text-only serving can skip the vision encoder for KV headroom.",
            "Past 262K context: VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 plus the card's rope_parameters override, up to 1M.",
            "MXFP4 checkpoints do not load on NVIDIA devices in vLLM (missing linear method); use NVFP4.",
        }),
        ..Default::default()
    }
}

// ── qwen3.5-397b-a17b — MoE 397B/17B, multimodal ───────────────────────

fn qwen3_5_397b_a17b() -> Model {
    Model {
        Name: string("qwen3.5-397b-a17b"),
        Size: string("397B/17B"),
        MoE: true,
        Vision: true,
        NativeContext: 1048576,
        // GDN (Gated DeltaNet) hybrid — the KDA memory model: only the
        // full-attention layers (est. 1 in 4) grow with context; the
        // linear layers hold a constant delta-rule state. Split is
        // unpublished (est) — replaces the earlier flat 200 KB/token
        // estimate, which wrongly extrapolated linearly to 1M.
        KV: KVArch {
            Kind: string("kda"),
            Layers: 64,
            KVHeads: 8,
            HeadDim: 128,
            FullEvery: 4,
            StateB: 8388608,
            Est: true,
            MaxContext: 1048576,
            ..Default::default()
        },
        License: string("Apache 2.0"),
        Engines: slice!([]string{string("vLLM"), string("SGLang")}),
        Image: string(IMG_V0260),
        Variants: slice!([]Variant{
            Variant {
                Name: string("fp8"),
                WeightsGB: 397.0,
                MinGPU: string("8xH100 / 4xH200"),
                ProdGPU: string("8xH100"),
                Serve: slice!([]ServeSpec{
                    ServeSpec {
                        Profile: string("throughput"),
                        Note: string("verified on 8xH200 / 8xMI300X"),
                        Model: string("Qwen/Qwen3.5-397B-A17B-FP8"),
                        Flags: slice!([]Flag{
                            flag("-dp", "8"),
                            boolFlag("--enable-expert-parallel"),
                            boolFlag("--language-model-only"),
                            flag("--reasoning-parser", "qwen3"),
                            boolFlag("--enable-prefix-caching"),
                        }),
                        ..Default::default()
                    },
                    ServeSpec {
                        Profile: string("latency"),
                        Note: string("low-concurrency alternative; do not mix with the throughput flags"),
                        Model: string("Qwen/Qwen3.5-397B-A17B-FP8"),
                        Flags: slice!([]Flag{
                            flag("--tensor-parallel-size", "8"),
                            flag(
                                "--speculative-config",
                                r#"{"method": "mtp", "num_speculative_tokens": 1}"#,
                            ),
                            flag("--reasoning-parser", "qwen3"),
                        }),
                        ..Default::default()
                    },
                }),
                ..Default::default()
            },
        }),
        Notes: slice!([]string{
            "NEVER run on vLLM < v0.19.0: FP8-KV accuracy silently degraded on Blackwell through v0.18.x.",
            "Tool-call parser is qwen3_coder here, NOT hermes (drifts per generation; 3.6 uses qwen3_xml).",
            "--language-model-only skips the vision encoder; free KV headroom for text-only serving.",
            "Past 262K context: --hf-overrides rope_parameters YaRN; pick factor for real traffic, not max.",
            "GDN/KDA hybrid: KV grows only in the full-attention layers (~1/4), so 1M-context KV is ~10x smaller than a linear extrapolation suggests; the mamba-style state also drives the causal_conv1d startup failure mode (reduce --max-cudagraph-capture-size).",
        }),
        ..Default::default()
    }
}

// Go: func init() { model.Register(Family{...}) }
#[goish::init]
fn init() {
    Register(Family {
        Rank: 1,
        Name: string("qwen"),
        Models: slice!([]Model{
            qwen3_32b(),
            qwen3_30b_a3b(),
            qwen3_8_27b(),
            qwen3_5_397b_a17b(),
        }),
    });
}
