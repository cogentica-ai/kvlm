// KV-cache / VRAM math regression tests: pin the per-kind formulas
// (gqa / mla / swa / kda / flat) to hand-computed figures, including the
// capacity-planning worked example from the Aug 2026 recipe research.
#![no_std]
#![no_main]
#![allow(non_snake_case)]

extern crate alloc;

use goish::fmt;
use goish::math;
use goish::syscall;
use goish::testing;
use goish::string;
use goish::{int32};

use kvlm::model;
use kvlm::model::{
    KVArch, KVCacheBytes, LookupGPU, MemProfile, ParseGPUs, PickHardware, TokSPerSeq,
    VRAMEstimateGB,
};

fn test_gqa_per_token(t: &mut testing::T) {
    // qwen3-32b: 2 x 64 layers x 8 KV heads x 128 head_dim = 131072 B/token
    let (_, m, ok) = model::Find("qwen3-32b");
    if !ok {
        t.Fatal(string("qwen3-32b not in catalog"));
    }
    let got = m.KV.PerTokenBytes();
    if got != 131072 {
        t.Fatal(fmt::Sprintf!("per-token: got %d, want 131072", got));
    }
}

fn test_gqa_clamps_at_cap(t: &mut testing::T) {
    // 200K request must clamp to the 131072-token cap: 131072^2 bytes.
    let (_, m, _) = model::Find("qwen3-32b");
    let (bytes, clamped, known) = KVCacheBytes(&m, 204800);
    if !known {
        t.Fatal(string("expected KVArch"));
    }
    if clamped != 131072 {
        t.Fatal(fmt::Sprintf!("clamped: got %d, want 131072", clamped));
    }
    if bytes != 131072 * 131072 {
        t.Fatal(fmt::Sprintf!("bytes: got %d, want %d", bytes, 131072 as goish::int * 131072));
    }
}

fn test_mla_per_token(t: &mut testing::T) {
    // deepseek-r1: 61 layers x 576 latent = 35136 B/token, size-independent.
    let (_, m, _) = model::Find("deepseek-r1");
    let got = m.KV.PerTokenBytes();
    if got != 61 * 576 {
        t.Fatal(fmt::Sprintf!("mla per-token: got %d, want %d", got, 61 * 576));
    }
}

fn test_swa_at_full_context(t: &mut testing::T) {
    // gpt-oss-120b: 36L, 1/2 full, window 128, per-layer 2x8x64 = 1024 B.
    // full: 18 x 1024 x 131072; window: 18 x 1024 x 128.
    let (_, m, _) = model::Find("gpt-oss-120b");
    let want: goish::int = 18 * 1024 * 131072 + 18 * 1024 * 128;
    let got = m.KV.BytesAt(131072);
    if got != want {
        t.Fatal(fmt::Sprintf!("swa bytes: got %d, want %d", got, want));
    }
}

fn test_swa_gemma27b_matches_measurement(t: &mut testing::T) {
    // gemma3-27b from published config (62L, 16KV, 128d, 5:1, window
    // 1024) must land near the researched ~5.5 GB @128K figure.
    let (_, m, _) = model::Find("gemma3-27b");
    let got = m.KV.BytesAt(131072);
    let gb = (got as goish::float64) / 1e9;
    if gb < 5.3 || gb > 5.9 {
        t.Fatal(fmt::Sprintf!("gemma3-27b @128K: got %v GB, want ~5.6", gb));
    }
}

fn test_kda_state_is_context_independent(t: &mut testing::T) {
    // The linear layers' state term must not grow with ctx: the delta
    // between two context sizes is exactly the full-attention term.
    let (_, m, _) = model::Find("qwen3.5-397b-a17b");
    let full = m.KV.Layers / m.KV.FullEvery;
    let perLayer: goish::int = 2 * m.KV.KVHeads * m.KV.HeadDim;
    let d = m.KV.BytesAt(2000) - m.KV.BytesAt(1000);
    if d != full * perLayer * 1000 {
        t.Fatal(fmt::Sprintf!("kda delta: got %d, want %d", d, full * perLayer * 1000));
    }
}

fn test_kda_full_kind_mla(t: &mut testing::T) {
    // Kimi-Linear-style hybrid: KDA linear layers + MLA full layers —
    // the ctx-growing term must use LatentDim, not 2*KVHeads*HeadDim.
    let a = KVArch {
        Kind: string("kda"),
        Layers: 64,
        KVHeads: 8,
        HeadDim: 128,
        LatentDim: 576,
        FullEvery: 4,
        StateB: 1000000,
        FullKind: string("mla"),
        MaxContext: 1048576,
        ..Default::default()
    };
    // per-token (asymptotic): 16 full layers x 576
    if a.PerTokenBytes() != 16 * 576 {
        t.Fatal(fmt::Sprintf!("kda+mla per-token: got %d, want %d", a.PerTokenBytes(), 16 * 576));
    }
    // at ctx: full term + constant state for the 48 linear layers
    let want: goish::int = 16 * 576 * 10000 + 48 * 1000000;
    if a.BytesAt(10000) != want {
        t.Fatal(fmt::Sprintf!("kda+mla bytes: got %d, want %d", a.BytesAt(10000), want));
    }
}

fn test_vram_worked_example(t: &mut testing::T) {
    // qwen3-32b fp8, 32K ctx, 16 seqs: KV/seq 4.295 GB, total = 32
    // weights + 68.72 KV + 3 runtime reserve = 103.72 GB. The reserve
    // is flat, not a percentage: vLLM's non-KV memory is measured at
    // startup and does not scale with weight size.
    let (_, m, _) = model::Find("qwen3-32b");
    let mut fp8 = Default::default();
    for (_, v) in goish::range!(m.Variants.clone()) {
        if v.Name == "fp8" {
            fp8 = v.clone();
        }
    }
    let (total, kvPerSeq, ok) = VRAMEstimateGB(&m, &fp8, 32768, 16);
    if !ok {
        t.Fatal(string("expected numeric weights + KVArch"));
    }
    if math::Abs(kvPerSeq - 4.294967296) > 0.001 {
        t.Fatal(fmt::Sprintf!("kv/seq: got %v, want 4.295", kvPerSeq));
    }
    if math::Abs(total - 103.71947673600001) > 0.01 {
        t.Fatal(fmt::Sprintf!("total: got %v, want ~103.72", total));
    }
}

fn test_parse_gpus(t: &mut testing::T) {
    // "NxNAME" with optional VRAM suffix; longest name wins (L40S vs L4).
    let (n, g, ok) = ParseGPUs("2xH100");
    if !ok || n != 2 || g.BWGBs != 3350 {
        t.Fatal(fmt::Sprintf!("2xH100: got n=%d bw=%d ok=%v", n, g.BWGBs, ok));
    }
    let (n, g, ok) = ParseGPUs("1xL40S 48GB");
    if !ok || n != 1 || g.BWGBs != 864 {
        t.Fatal(fmt::Sprintf!("1xL40S: got n=%d bw=%d ok=%v", n, g.BWGBs, ok));
    }
    // no leading count, or no recognizable GPU: not parseable
    let (_, _, ok) = ParseGPUs("CPU / phone-class");
    if ok {
        t.Fatal(string("CPU / phone-class should not parse"));
    }
    let (_, _, ok) = ParseGPUs("multi-node only (32xH200+)");
    if ok {
        t.Fatal(string("multi-node prose should not parse"));
    }
}

fn test_toks_dense(t: &mut testing::T) {
    // qwen3-32b fp8 on 1xH100: 32 GB / (0.55 x 3350 GB/s) = 17.37 ms
    // + 64 layers x 40 us = 19.93 ms -> 50 tok/s. Anchor: the pytorch
    // Qwen3-32B-FP8 model card measures 49.6 batch-1 on H100.
    let (_, m, _) = model::Find("qwen3-32b");
    let mut fp8 = Default::default();
    for (_, v) in goish::range!(m.Variants.clone()) {
        if v.Name == "fp8" {
            fp8 = v.clone();
        }
    }
    let (tokS, ok) = TokSPerSeq(&m, &fp8);
    if !ok || tokS != 50 {
        t.Fatal(fmt::Sprintf!("dense tok/s: got %d ok=%v, want 50", tokS, ok));
    }
}

fn test_toks_moe_uses_active_params(t: &mut testing::T) {
    // gpt-oss-120b mxfp4 on 1xH100: only 5.1B of 117B params are read
    // per token, so 61 GB x (5.1/117) = 2.66 GB active -> 347 tok/s
    // (published single-stream ~313). A dense read of all 61 GB would
    // give ~30; MoE must not do that.
    let (_, m, _) = model::Find("gpt-oss-120b");
    let mut v = Default::default();
    for (_, cand) in goish::range!(m.Variants.clone()) {
        if cand.Name == "mxfp4" {
            v = cand.clone();
        }
    }
    let (tokS, ok) = TokSPerSeq(&m, &v);
    if !ok || tokS != 347 {
        t.Fatal(fmt::Sprintf!("moe tok/s: got %d ok=%v, want 347", tokS, ok));
    }
}

fn test_mem_profile(t: &mut testing::T) {
    // qwen3-32b fp8 on 1xH100 "80 GB" (= 80 GiB = 85.9 decimal GB):
    // recipe utilization 0.90 x 85.899 = 77.31 usable, minus 32 GB
    // weights + 2 GB/GPU runtime reserve = 34 loaded -> 43.31 GB KV
    // pool; at 4.295 GB/seq (32K) that holds 10 sequences.
    let (_, m, _) = model::Find("qwen3-32b");
    let mut fp8 = Default::default();
    for (_, v) in goish::range!(m.Variants.clone()) {
        if v.Name == "fp8" {
            fp8 = v.clone();
        }
    }
    let (cap, loaded, pool, seqs, ok) = MemProfile(&m, &fp8);
    if !ok {
        t.Fatal(string("expected a profile for fp8"));
    }
    if cap != 80.0 || math::Abs(loaded - 34.0) > 0.001 || math::Abs(pool - 43.309411328) > 0.001 {
        t.Fatal(fmt::Sprintf!("profile: cap=%v loaded=%v pool=%v", cap, loaded, pool));
    }
    if seqs != 10 {
        t.Fatal(fmt::Sprintf!("seqs at 32K: got %d, want 10", seqs));
    }
}

fn test_recipe_caps(t: &mut testing::T) {
    // The bf16 recipe pins utilization 0.92, admits 16 sequences, and
    // serves at most 16384 tokens; a variant with no recipe falls back
    // to utilization 0.90 and reports no caps.
    let (_, m, _) = model::Find("qwen3-32b");
    let bf16 = m.Variants[0usize].clone();
    if math::Abs(model::RecipeUtil(&bf16) - 0.92) > 0.0001 {
        t.Fatal(fmt::Sprintf!("bf16 util: got %v, want 0.92", model::RecipeUtil(&bf16)));
    }
    let (s, ok) = model::RecipeMaxSeqs(&bf16);
    if !ok || s != 16 {
        t.Fatal(fmt::Sprintf!("bf16 max seqs: got %d ok=%v, want 16", s, ok));
    }
    let (l, ok) = model::RecipeMaxLen(&bf16);
    if !ok || l != 16384 {
        t.Fatal(fmt::Sprintf!("bf16 max len: got %d ok=%v, want 16384", l, ok));
    }
    let (_, m2, _) = model::Find("kimi-k2");
    let noRecipe = m2.Variants[0usize].clone();
    if math::Abs(model::RecipeUtil(&noRecipe) - 0.90) > 0.0001 {
        t.Fatal(string("no-recipe variant must fall back to 0.90"));
    }
    let (_, ok) = model::RecipeMaxSeqs(&noRecipe);
    if ok {
        t.Fatal(string("no-recipe variant must report no seq cap"));
    }
}

fn test_pick_hardware(t: &mut testing::T) {
    // qwen3-32b: bf16 has no blackwell row, so class blackwell must
    // fall back to the cuda row; nvfp4 on cuda must land on the a16
    // fallback profile; nothing falls back to rocm.
    let (_, m, _) = model::Find("qwen3-32b");
    let mut bf16 = Default::default();
    let mut nvfp4 = Default::default();
    for (_, v) in goish::range!(m.Variants.clone()) {
        if v.Name == "bf16" {
            bf16 = v.clone();
        }
        if v.Name == "nvfp4" {
            nvfp4 = v.clone();
        }
    }
    let (h, ok) = PickHardware(&bf16, string("blackwell"));
    if !ok || h.Key != "cuda" {
        t.Fatal(fmt::Sprintf!("bf16/blackwell: got key %q ok=%v, want cuda fallback", h.Key.clone(), ok));
    }
    let (h, ok) = PickHardware(&nvfp4, string("cuda"));
    if !ok || h.Profile != "a16 fallback for pre-Blackwell" {
        t.Fatal(fmt::Sprintf!("nvfp4/cuda: got profile %q ok=%v, want a16", h.Profile.clone(), ok));
    }
    let (_, ok) = PickHardware(&nvfp4, string("rocm"));
    if ok {
        t.Fatal(string("nvfp4/rocm should not resolve"));
    }
    let (g, ok) = LookupGPU("b200");
    if !ok || g.Class != "blackwell" {
        t.Fatal(fmt::Sprintf!("LookupGPU(b200): got class %q ok=%v", g.Class.clone(), ok));
    }
}

fn test_flat_is_est(t: &mut testing::T) {
    // Every flat-kind entry must be marked Est — unpublished figures
    // must never present as measured.
    for f in model::Families().iter() {
        for (_, m) in goish::range!(f.Models.clone()) {
            let kind: &str = m.KV.Kind.as_ref();
            if kind == "flat" && !m.KV.Est {
                t.Fatal(fmt::Sprintf!("%s: flat KVArch without Est", m.Name.clone()));
            }
            if kind == "" {
                t.Fatal(fmt::Sprintf!("%s: missing KVArch", m.Name.clone()));
            }
        }
    }
}

fn test_format_weights(t: &mut testing::T) {
    // Clean multiples of 10 render without tilde; others get tilde.
    let (_, m, _) = model::Find("qwen3-32b");
    // bf16: 64.0 GB -> "64 GB"
    let bf16 = m.Variants[0usize].clone();
    let w = bf16.FormatWeights();
    if w != "64 GB" {
        t.Fatal(fmt::Sprintf!("bf16 weights: got %q, want \"64 GB\"", w.clone()));
    }
    // fp8: 32.0 GB -> "32 GB"
    let fp8 = m.Variants[1usize].clone();
    let w = fp8.FormatWeights();
    if w != "32 GB" {
        t.Fatal(fmt::Sprintf!("fp8 weights: got %q, want \"32 GB\"", w.clone()));
    }
    // awq-q4: 19.0 GB -> "19 GB" (clean integer)
    let awq = m.Variants[2usize].clone();
    let w = awq.FormatWeights();
    if w != "19 GB" {
        t.Fatal(fmt::Sprintf!("awq weights: got %q, want \"19 GB\"", w.clone()));
    }
    // llama3.1-8b q4: 4.7 GB -> "~5 GB" (non-clean integer)
    let (_, m2, _) = model::Find("llama3.1-8b");
    let mut q4 = Default::default();
    for (_, v) in goish::range!(m2.Variants.clone()) {
        if v.Name == "q4" {
            q4 = v.clone();
        }
    }
    let w = q4.FormatWeights();
    if w != "~5 GB" {
        t.Fatal(fmt::Sprintf!("llama q4 weights: got %q, want \"~5 GB\"", w.clone()));
    }
}

fn test_format_context(t: &mut testing::T) {
    // qwen3-32b: 32K native, 131K max (YaRN)
    let (_, m, _) = model::Find("qwen3-32b");
    let ctx = m.FormatContext();
    if ctx != "32K native, 128K max (YaRN)" {
        t.Fatal(fmt::Sprintf!("qwen3-32b context: got %q", ctx.clone()));
    }
    // deepseek-r1: 128K native (max == native, no method)
    let (_, m, _) = model::Find("deepseek-r1");
    let ctx = m.FormatContext();
    if ctx != "128K native" {
        t.Fatal(fmt::Sprintf!("deepseek-r1 context: got %q", ctx.clone()));
    }
}

fn test_format_arch(t: &mut testing::T) {
    // qwen3-32b: dense (no MoE, no attention suffix, no vision)
    let (_, m, _) = model::Find("qwen3-32b");
    let arch = m.FormatArch();
    if arch != "dense" {
        t.Fatal(fmt::Sprintf!("qwen3-32b arch: got %q, want \"dense\"", arch.clone()));
    }
    // deepseek-r1: MoE, MLA
    let (_, m, _) = model::Find("deepseek-r1");
    let arch = m.FormatArch();
    if arch != "MoE, MLA" {
        t.Fatal(fmt::Sprintf!("deepseek-r1 arch: got %q, want \"MoE, MLA\"", arch.clone()));
    }
    // qwen3.5-397b: MoE, multimodal
    let (_, m, _) = model::Find("qwen3.5-397b-a17b");
    let arch = m.FormatArch();
    if arch != "MoE, multimodal" {
        t.Fatal(fmt::Sprintf!("qwen3.5 arch: got %q, want \"MoE, multimodal\"", arch.clone()));
    }
}

#[goish::main]
fn main() {
    let tests: &[(&str, testing::TestFn)] = &[
        ("TestGQAPerToken", test_gqa_per_token),
        ("TestGQAClampsAtCap", test_gqa_clamps_at_cap),
        ("TestMLAPerToken", test_mla_per_token),
        ("TestSWAAtFullContext", test_swa_at_full_context),
        ("TestSWAGemma27BMatchesMeasurement", test_swa_gemma27b_matches_measurement),
        ("TestKDAStateIsContextIndependent", test_kda_state_is_context_independent),
        ("TestKDAFullKindMLA", test_kda_full_kind_mla),
        ("TestVRAMWorkedExample", test_vram_worked_example),
        ("TestParseGPUs", test_parse_gpus),
        ("TestTokSDense", test_toks_dense),
        ("TestTokSMoEUsesActiveParams", test_toks_moe_uses_active_params),
        ("TestMemProfile", test_mem_profile),
        ("TestRecipeCaps", test_recipe_caps),
        ("TestPickHardware", test_pick_hardware),
        ("TestFlatIsEst", test_flat_is_est),
        ("TestFormatWeights", test_format_weights),
        ("TestFormatContext", test_format_context),
        ("TestFormatArch", test_format_arch),
    ];
    let code = testing::Main(tests);
    syscall::Exit(int32(code));
}
