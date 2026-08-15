// model package: the catalog of model families kvlm can run.
//
// One file per family (qwen.rs, deepseek.rs, ...), each self-registering
// its Family in init() — same registry idiom as the driver package. A
// family file is where that family's serving recipe will grow (per-model
// vLLM image pins, flags, GPU floors), so "1 Family -> many models" stays
// self-contained per file.
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub mod deepseek;
pub mod gemma;
pub mod glm;
pub mod gpt_oss;
pub mod hunyuan;
pub mod kimi;
pub mod llama;
pub mod mimo;
pub mod minimax;
pub mod qwen;

goish::import! {
    crate::model::qwen as __init_qwen,
    crate::model::deepseek as __init_deepseek,
    crate::model::llama as __init_llama,
    crate::model::gemma as __init_gemma,
    crate::model::gpt_oss as __init_gpt_oss,
    crate::model::glm as __init_glm,
    crate::model::kimi as __init_kimi,
    crate::model::mimo as __init_mimo,
    crate::model::hunyuan as __init_hunyuan,
    crate::model::minimax as __init_minimax,
}

use goish::fmt;
use goish::lazy::Lazy;
use goish::os;
use goish::strconv;
use goish::strings;
use goish::sync;
use goish::text::tabwriter;
use goish::string;
use goish::{append, make, nil, slice};

// KVArch describes a model's attention memory layout — enough to
// compute KV-cache bytes for any context length (FP8 KV, 1 byte/elem):
//   gqa:  2 × Layers × KVHeads × HeadDim per token
//   mla:  Layers × LatentDim per token (constant, tiny — DeepSeek/Kimi)
//   swa:  full-attention layers pay ctx, sliding-window layers pay
//         min(ctx, Window); full layers = Layers / FullEvery
//   kda:  linear-attention hybrid (Kimi Delta Attention / Gated
//         DeltaNet): full layers (1 per FullEvery) pay per-token KV,
//         linear layers keep a CONSTANT recurrent state of StateB
//         bytes per layer per sequence — context-independent. The full
//         layers are GQA-shaped unless FullKind == "mla" (Kimi Linear
//         interleaves KDA with MLA full layers — then they pay
//         LatentDim per token instead)
//   flat: PerTokenB directly (unpublished architectures, reverse-derived
//         from measured figures — always Est)
// MaxContext is the hard ceiling in tokens; Est marks estimates.
#[derive(Clone, Default)]
pub struct KVArch {
    pub Kind: string,
    pub Layers: goish::int,
    pub KVHeads: goish::int,
    pub HeadDim: goish::int,
    pub LatentDim: goish::int,
    pub Window: goish::int,
    pub FullEvery: goish::int,
    pub PerTokenB: goish::int,
    pub StateB: goish::int,
    pub FullKind: string,
    pub Est: bool,
    pub MaxContext: goish::int,
}

impl KVArch {
    // PerTokenBytes returns KV bytes per token at FP8 (long-context
    // upper bound for swa — window layers amortize to ~0 there).
    pub fn PerTokenBytes(&self) -> goish::int {
        let kind: &str = self.Kind.as_ref();
        match kind {
            "gqa" => 2 * self.Layers * self.KVHeads * self.HeadDim,
            "mla" => self.Layers * self.LatentDim,
            "swa" => {
                let full = self.Layers / self.FullEvery;
                2 * full * self.KVHeads * self.HeadDim
            }
            // kda: state amortizes to ~0 per token at long context —
            // only the full-attention layers grow with ctx.
            "kda" => {
                let full = self.Layers / self.FullEvery;
                full * self.fullLayerPerToken()
            }
            "flat" => self.PerTokenB,
            _ => 0,
        }
    }

    // fullLayerPerToken: per-token bytes of ONE full-attention layer in
    // a kda hybrid — GQA-shaped by default, latent when FullKind "mla".
    fn fullLayerPerToken(&self) -> goish::int {
        let fk: &str = self.FullKind.as_ref();
        if fk == "mla" {
            return self.LatentDim;
        }
        2 * self.KVHeads * self.HeadDim
    }

    // BytesAt returns KV-cache bytes for ONE sequence at ctx tokens
    // (unclamped — callers clamp against MaxContext).
    pub fn BytesAt(&self, ctx: goish::int) -> goish::int {
        let kind: &str = self.Kind.as_ref();
        if kind == "swa" {
            let perLayer = 2 * self.KVHeads * self.HeadDim;
            let full = self.Layers / self.FullEvery;
            let mut w = ctx;
            if self.Window < w {
                w = self.Window;
            }
            return full * perLayer * ctx + (self.Layers - full) * perLayer * w;
        }
        if kind == "kda" {
            let perLayer = self.fullLayerPerToken();
            let full = self.Layers / self.FullEvery;
            // linear layers: constant delta-rule state, no ctx term
            return full * perLayer * ctx + (self.Layers - full) * self.StateB;
        }
        self.PerTokenBytes() * ctx
    }
}

// KVCacheBytes computes one sequence's KV size at ctx tokens, clamping
// to the model's context cap. Returns (bytes, clampedCtx, known).
pub fn KVCacheBytes(m: &Model, ctx: goish::int) -> (goish::int, goish::int, bool) {
    if m.KV.Kind == "" {
        return (0, ctx, false);
    }
    let mut c = ctx;
    if m.KV.MaxContext > 0 && c > m.KV.MaxContext {
        c = m.KV.MaxContext;
    }
    (m.KV.BytesAt(c), c, true)
}

// VRAMEstimateGB: weights + KV(ctx) × seqs + a flat ~3 GB runtime
// reserve (activations and CUDA graphs — vLLM measures this at startup
// and it does not scale with weight size, so a percentage would charge
// large models for overhead that does not exist). Returns (totalGB,
// kvPerSeqGB, ok) — ok is false when the variant has no numeric
// weights or the model no KVArch.
pub fn VRAMEstimateGB(
    m: &Model,
    v: &Variant,
    ctx: goish::int,
    seqs: goish::int,
) -> (goish::float64, goish::float64, bool) {
    let (kvBytes, _, known) = KVCacheBytes(m, ctx);
    if !known || v.WeightsGB <= 0.0 {
        return (0.0, 0.0, false);
    }
    let kvPerSeq = (kvBytes as goish::float64) / 1e9;
    let total = v.WeightsGB + kvPerSeq * (seqs as goish::float64) + 3.0;
    (total, kvPerSeq, true)
}

// fmtGB / fmtTokens — display helpers for computed KV figures.
pub(crate) fn fmtGB(bytes: goish::int) -> string {
    // goish fmt has no precision verbs — round to one decimal by hand.
    let v = goish::math::Round((bytes as goish::float64) / 1e8) / 10.0;
    fmt::Sprintf!("%v GB", v)
}
// fmtTokens renders a token count exactly: integer division would let
// a header label a point it did not compute (1.25M shown as 1M).
pub(crate) fn fmtTokens(t: goish::int) -> string {
    if t >= 1048576 {
        let r = goish::math::Round((t as goish::float64) / 1048576.0 * 10.0) / 10.0;
        let ri = r as goish::int;
        if r == (ri as goish::float64) {
            return fmt::Sprintf!("%dM", ri);
        }
        return fmt::Sprintf!("%vM", r);
    }
    let r = goish::math::Round((t as goish::float64) / 1024.0 * 10.0) / 10.0;
    let ri = r as goish::int;
    if r == (ri as goish::float64) {
        return fmt::Sprintf!("%dK", ri);
    }
    fmt::Sprintf!("%vK", r)
}

// kvLine renders the model-level "KV/seq" header line: per-sequence KV
// at the model's own native and max context (one figure when they
// coincide). KV does not vary by variant, so it lives in the header,
// not the variants table.
pub(crate) fn kvLine(m: &Model) -> string {
    if m.KV.Kind == "" {
        return string("");
    }
    let kind: &str = m.KV.Kind.as_ref();
    let mut tag = string("fp8 KV");
    if kind == "mla" {
        tag = string("MLA, fp8 KV");
    }
    if kind == "kda" {
        tag = string("KDA, fp8 KV");
    }
    // estimates must disclose themselves on the primary display, not
    // only in `model vram`
    if m.KV.Est {
        tag = (tag) + ("; estimated, architecture unpublished");
    }
    let native = m.NativeContext;
    let max = m.KV.MaxContext;
    if native > 0 && max > 0 && max != native {
        let (nb, _, _) = KVCacheBytes(m, native);
        let (mb, _, _) = KVCacheBytes(m, max);
        return fmt::Sprintf!(
            "%s at %s, %s at %s (%s)",
            fmtGB(nb),
            fmtTokens(native),
            fmtGB(mb),
            fmtTokens(max),
            tag
        );
    }
    let mut at = native;
    if at <= 0 {
        at = max;
    }
    if at <= 0 {
        return string("");
    }
    let (b, _, _) = KVCacheBytes(m, at);
    fmt::Sprintf!("%s at %s (%s)", fmtGB(b), fmtTokens(at), tag)
}

// GPUSpec is one GPU type the catalog can budget against: usable VRAM,
// memory bandwidth (decode speed is bandwidth-bound, so BWGBs is what
// TOK/S estimates run on), and the hardware class that Hardware.Key
// rows match against ("cuda", "blackwell", "rocm").
#[derive(Clone, Default)]
pub struct GPUSpec {
    pub Name: string,
    pub VRAMGB: goish::int,
    pub BWGBs: goish::int,
    pub Class: string,
    // SMs is the streaming-multiprocessor count (compute units for
    // rocm); a kernel whose grid has fewer blocks than SMs cannot fill
    // the GPU, which is the latency-bound heuristic in profile graph.
    pub SMs: goish::int,
}

// gpuSpecs: bandwidth in GB/s, SXM figures where PCIe differs. Order
// matters for the substring match in ParseGPUs: names that contain a
// shorter name ("L40S" vs "L4") must come first.
fn gpuSpecs() -> slice<GPUSpec> {
    slice!([]GPUSpec{
        GPUSpec{ Name: string("RTX Pro 6000"), VRAMGB: 96, BWGBs: 1792, Class: string("blackwell"), SMs: 188, ..Default::default() },
        GPUSpec{ Name: string("RTX 4090"), VRAMGB: 24, BWGBs: 1008, Class: string("cuda"), SMs: 128, ..Default::default() },
        GPUSpec{ Name: string("MI300X"), VRAMGB: 192, BWGBs: 5300, Class: string("rocm"), SMs: 304, ..Default::default() },
        GPUSpec{ Name: string("L40S"), VRAMGB: 48, BWGBs: 864, Class: string("cuda"), SMs: 142, ..Default::default() },
        GPUSpec{ Name: string("A100"), VRAMGB: 80, BWGBs: 2039, Class: string("cuda"), SMs: 108, ..Default::default() },
        GPUSpec{ Name: string("A10G"), VRAMGB: 24, BWGBs: 600, Class: string("cuda"), SMs: 80, ..Default::default() },
        GPUSpec{ Name: string("B200"), VRAMGB: 192, BWGBs: 8000, Class: string("blackwell"), SMs: 148, ..Default::default() },
        GPUSpec{ Name: string("H100"), VRAMGB: 80, BWGBs: 3350, Class: string("cuda"), SMs: 132, ..Default::default() },
        GPUSpec{ Name: string("H200"), VRAMGB: 141, BWGBs: 4800, Class: string("cuda"), SMs: 132, ..Default::default() },
        GPUSpec{ Name: string("L4"), VRAMGB: 24, BWGBs: 300, Class: string("cuda"), SMs: 58, ..Default::default() },
        GPUSpec{ Name: string("T4"), VRAMGB: 16, BWGBs: 320, Class: string("cuda"), SMs: 40, ..Default::default() },
    })
}

// LookupGPU finds a spec by bare GPU name ("h100", "B200"), matched
// case-insensitively as a substring so "h100 80GB" also works.
pub fn LookupGPU<S: Into<string>>(name: S) -> (GPUSpec, bool) {
    let name = strings::ToUpper(name.into());
    for (_, g) in goish::range!(gpuSpecs()) {
        if strings::Contains(name.clone(), strings::ToUpper(g.Name.clone())) {
            return (g.clone(), true);
        }
    }
    (Default::default(), false)
}

// ParseGPUs parses a GPU config in the catalog's "2xH100" / "1xL40S
// 48GB" form into a count and the matching spec.
pub fn ParseGPUs<S: Into<string>>(s: S) -> (goish::int, GPUSpec, bool) {
    let s = s.into();
    let parts = strings::SplitN(s.clone(), "x", 2);
    if parts.Len() != 2 {
        return (0, Default::default(), false);
    }
    let (n, err) = strconv::Atoi(strings::TrimSpace(parts[0usize].clone()));
    if err != nil || n <= 0 {
        return (0, Default::default(), false);
    }
    for (_, g) in goish::range!(gpuSpecs()) {
        if strings::Contains(parts[1usize].clone(), g.Name.clone()) {
            return (n, g.clone(), true);
        }
    }
    (0, Default::default(), false)
}

// parseParams reads a param-count token: "284" or "284B" -> 284,
// "1.02T" -> 1020, unparseable -> 0.
fn parseParams(s: string) -> goish::float64 {
    let mut mult = 1.0;
    let mut num = strings::TrimSpace(s);
    if strings::HasSuffix(num.clone(), "T") {
        mult = 1000.0;
        num = strings::TrimSuffix(num.clone(), "T");
    } else if strings::HasSuffix(num.clone(), "B") {
        num = strings::TrimSuffix(num.clone(), "B");
    }
    let (v, err) = strconv::ParseFloat(num, 64);
    if err != nil {
        return 0.0;
    }
    v * mult
}

// activeWeightFrac: the fraction of the weights each decoded token
// actually reads — active/total params for MoE (Size "397B/17B"), 1.0
// for dense (Size "32B").
fn activeWeightFrac(size: string) -> goish::float64 {
    let parts = strings::Split(size, "/");
    if parts.Len() != 2 {
        return 1.0;
    }
    let total = parseParams(parts[0usize].clone());
    let active = parseParams(parts[1usize].clone());
    if total <= 0.0 || active <= 0.0 {
        return 1.0;
    }
    active / total
}

// TokSPerSeq estimates single-sequence decode speed on the variant's
// ProdGPU config. Each token reads the active weights once from HBM at
// an effective bandwidth, plus a per-layer kernel and TP-collective
// cost that grows with GPU count. Calibrated against measured
// single-request decode: Qwen3-32B-FP8 on H100 49.6 tok/s (pytorch
// model card), gpt-oss-120b on H100 ~313 tok/s (Clarifai benchmark).
// The TP8 branch remains optimistic for large MoE (deepseek-r1 real
// figures run ~30-40 on 8xH200 vs ~46 here).
pub fn TokSPerSeq(m: &Model, v: &Variant) -> (goish::int, bool) {
    let (n, g, ok) = ParseGPUs(v.ProdGPU.clone());
    if !ok || v.WeightsGB <= 0.0 || m.KV.Layers <= 0 {
        return (0, false);
    }
    let activeGB = v.WeightsGB * activeWeightFrac(m.Size.clone());
    let mut eff = 0.55;
    if n >= 2 {
        eff = 0.50;
    }
    if n >= 4 {
        eff = 0.45;
    }
    if n >= 8 {
        eff = 0.40;
    }
    if n >= 16 {
        eff = 0.35;
    }
    let bw = eff * ((n * g.BWGBs) as goish::float64);
    // ms per token: weight reads + 40 us per layer per GPU of kernel
    // launch, routing, and all-reduce cost
    let ms = activeGB / bw * 1000.0
        + (m.KV.Layers as goish::float64) * 0.04 * (n as goish::float64);
    (goish::math::Round(1000.0 / ms) as goish::int, true)
}

// round2sig caps estimates at two significant figures; three-figure
// output (~137) would claim precision the model does not have.
fn round2sig(v: goish::int) -> goish::int {
    let mut mag: goish::int = 1;
    while v / mag >= 100 {
        mag *= 10;
    }
    ((v + mag / 2) / mag) * mag
}

// defaultSpec returns the variant's default-profile recipe, if any.
// ParseGPURef splits the catalog's "2xH100" GPU references into a
// count and the short type name.
pub fn ParseGPURef<S: Into<string>>(s: S) -> (goish::int, string, bool) {
    let t = strings::TrimSpace(s.into());
    let idx = strings::Index(t.clone(), string("x"));
    if idx <= 0 {
        return (0, string(""), false);
    }
    let (n, err) = strconv::Atoi(t.slice(0, idx));
    if err != nil || n <= 0 {
        return (0, string(""), false);
    }
    (n, t.slice(idx + 1, t.Len()), true)
}

// DefaultServe returns a variant's default-profile serve spec, for
// callers outside the package (kvlm up's vLLM launch).
pub fn DefaultServe(v: &Variant) -> (ServeSpec, bool) {
    defaultSpec(v)
}

fn defaultSpec(v: &Variant) -> (ServeSpec, bool) {
    for (_, s) in goish::range!(v.Serve.clone()) {
        if s.Profile == "" {
            return (s.clone(), true);
        }
    }
    (Default::default(), false)
}

// RecipeUtil returns the --gpu-memory-utilization the variant's default
// recipe pins, or vLLM's default 0.90 when no recipe sets it. The
// utilization is per-variant data, never a global constant.
pub fn RecipeUtil(v: &Variant) -> goish::float64 {
    let (s, ok) = defaultSpec(v);
    if ok {
        let (val, has) = s.FlagValue("--gpu-memory-utilization");
        if has {
            let (u, err) = strconv::ParseFloat(val, 64);
            if err == nil && u > 0.0 && u <= 1.0 {
                return u;
            }
        }
    }
    0.90
}

// RecipeMaxSeqs returns the --max-num-seqs the default recipe admits;
// ok is false when no recipe pins it.
pub fn RecipeMaxSeqs(v: &Variant) -> (goish::int, bool) {
    let (s, ok) = defaultSpec(v);
    if !ok {
        return (0, false);
    }
    let (val, has) = s.FlagValue("--max-num-seqs");
    if !has {
        return (0, false);
    }
    let (n, err) = strconv::Atoi(val);
    if err != nil {
        return (0, false);
    }
    (n, true)
}

// RecipeMaxLen returns the --max-model-len the default recipe serves;
// requests longer than this are rejected outright.
pub fn RecipeMaxLen(v: &Variant) -> (goish::int, bool) {
    let (s, ok) = defaultSpec(v);
    if !ok {
        return (0, false);
    }
    let (val, has) = s.FlagValue("--max-model-len");
    if !has {
        return (0, false);
    }
    let (n, err) = strconv::Atoi(val);
    if err != nil {
        return (0, false);
    }
    (n, true)
}

// gibToGB: marketed VRAM sizes are GiB ("80 GB" H100 = 85.9e9 bytes);
// the KV math runs in decimal GB, so capacities convert before use.
const gibToGB: goish::float64 = 1.073741824;

// MemProfile describes how a variant's memory lands on its ProdGPU
// config: total VRAM (as marketed), the loaded footprint (weights plus
// a 2 GB/GPU runtime reserve for activations and CUDA graphs — vLLM
// measures this at startup and it does not scale with weight size),
// the KV pool left under the recipe's --gpu-memory-utilization, and
// the sequences that pool holds at 32K context.
// Returns (capGB, loadedGB, poolGB, seqs, ok).
pub fn MemProfile(
    m: &Model,
    v: &Variant,
) -> (goish::float64, goish::float64, goish::float64, goish::int, bool) {
    let (n, g, ok) = ParseGPUs(v.ProdGPU.clone());
    if !ok || v.WeightsGB <= 0.0 {
        return (0.0, 0.0, 0.0, 0, false);
    }
    let (kvBytes, _, known) = KVCacheBytes(m, 32768);
    if !known || kvBytes <= 0 {
        return (0.0, 0.0, 0.0, 0, false);
    }
    let cap = ((n * g.VRAMGB) as goish::float64);
    let loaded = v.WeightsGB + 2.0 * (n as goish::float64);
    let pool = cap * gibToGB * RecipeUtil(v) - loaded;
    if pool <= 0.0 {
        return (cap, loaded, 0.0, 0, true);
    }
    let kvPerSeq = (kvBytes as goish::float64) / 1e9;
    let seqs = (pool / kvPerSeq) as goish::int;
    (cap, loaded, pool, seqs, true)
}

// Hardware is one supported target for a variant: the exact image tag
// for that platform, the vLLM version floor it needs, and its caveats.
// The recipe axes are weight format (Variant) × platform (Hardware) —
// some combinations are invalid (nvfp4 needs Blackwell; awq_marlin
// needs SM80+ or ROCm >= v0.19).
//
// Key is the machine-matchable class ("cuda", "blackwell", "rocm",
// "arm") that --gpu resolution and the support matrix run on; "" marks
// an informational row (an alternative image, not a distinct target).
// Profile names the ServeSpec that applies on this hardware; "" means
// the default profile.
#[derive(Clone, Default)]
pub struct Hardware {
    pub Target: string,
    pub Key: string,
    pub Profile: string,
    pub Image: string,
    pub Floor: string,
    pub FloorRef: string,
    pub Note: string,
}

// PickHardware resolves the hardware row for a GPU class. Blackwell
// GPUs run plain CUDA recipes, so they fall back to "cuda" rows when a
// variant has no dedicated "blackwell" row; nothing falls back to or
// from "rocm".
pub fn PickHardware(v: &Variant, class: string) -> (Hardware, bool) {
    for (_, h) in goish::range!(v.Hardware.clone()) {
        if h.Key == class {
            return (h.clone(), true);
        }
    }
    if class == "blackwell" {
        for (_, h) in goish::range!(v.Hardware.clone()) {
            if h.Key == "cuda" {
                return (h.clone(), true);
            }
        }
    }
    (Default::default(), false)
}

// Flag is one CLI flag as data. Name carries its dashes verbatim
// ("--max-model-len", "-dp"); Value "" means a boolean switch.
#[derive(Clone, Default)]
pub struct Flag {
    pub Name: string,
    pub Value: string,
}

// ServeSpec is one serve invocation as structured data — the source of
// truth both for drivers (Argv → container command/args on the real
// platform) and for humans (Render → the shell form `model show`
// prints). Multi-profile variants (throughput vs latency) carry one
// spec per profile.
#[derive(Clone, Default)]
pub struct ServeSpec {
    pub Profile: string, // "" for the single default profile
    pub Note: string,    // rendered as a leading # comment
    pub Env: slice<string>, // "KEY=VALUE" entries for the container env
    pub Model: string,   // HF repo the runtime loads
    pub Flags: slice<Flag>,
}

impl ServeSpec {
    // Argv returns the exec-style container command for platform
    // drivers: ["vllm", "serve", <model>, "--flag", "value", ...].
    // No shell, no quoting — values are passed verbatim.
    pub fn Argv(&self) -> slice<string> {
        let mut argv: slice<string> = make!([]string, 0);
        argv = append!(argv.clone(), string("vllm"));
        argv = append!(argv.clone(), string("serve"));
        argv = append!(argv.clone(), self.Model.clone());
        for (_, f) in goish::range!(self.Flags.clone()) {
            argv = append!(argv.clone(), f.Name.clone());
            if f.Value != "" {
                argv = append!(argv.clone(), f.Value.clone());
            }
        }
        argv
    }

    // FlagValue returns the value of one flag in the spec; ok reports
    // whether the flag is present (bool switches carry value "").
    pub fn FlagValue<S: Into<string>>(&self, name: S) -> (string, bool) {
        let name = name.into();
        for (_, f) in goish::range!(self.Flags.clone()) {
            if f.Name == name {
                return (f.Value.clone(), true);
            }
        }
        (string(""), false)
    }

    // Render returns the human copy-pasteable shell form: one flag per
    // continuation line, values shell-quoted only when they need it.
    pub fn Render(&self) -> string {
        let mut b = strings::Builder::new();
        if self.Note != "" {
            let _ = b.WriteString(("# ") + (self.Note.clone()) + ("\n"));
        }
        for (_, e) in goish::range!(self.Env.clone()) {
            let _ = b.WriteString(e.clone() + (" "));
        }
        let _ = b.WriteString(("vllm serve ") + (self.Model.clone()));
        for (_, f) in goish::range!(self.Flags.clone()) {
            let _ = b.WriteString(" \\\n  ");
            let _ = b.WriteString(f.Name.clone());
            if f.Value != "" {
                let mut v = f.Value.clone();
                if strings::Contains(v.clone(), " ") || strings::Contains(v.clone(), "\"") {
                    v = ("'") + (v) + ("'");
                }
                let _ = b.WriteString((" ") + (v));
            }
        }
        b.String()
    }
}

// Variant is one weight format of a checkpoint — the recipe unit. KV
// figures are computed from KVArch for any context length. Serve is
// empty while a recipe is pending. Image is the default (CUDA
// Hopper/Ada) pin; Hardware lists per-platform pins.
#[derive(Clone, Default)]
pub struct Variant {
    pub Name: string,
    pub Image: string,
    pub WeightsGB: goish::float64,
    pub MinGPU: string,
    // ProdGPU is the config you would actually serve on (weights plus a
    // real KV pool, with headroom); TOK/S in `model show` computes off
    // it via TokSPerSeq. "" when there is no sane single-node answer.
    pub ProdGPU: string,
    // MeasuredTokS is measured single-sequence decode on the ProdGPU
    // config (0 = unmeasured; the display then falls back to the
    // TokSPerSeq bandwidth estimate). Measured values print without
    // the ~ estimate marker.
    pub MeasuredTokS: goish::int,
    pub Serve: slice<ServeSpec>,
    pub Hardware: slice<Hardware>,
}

impl Variant {
    // FormatWeights renders the weight size from WeightsGB: "64 GB",
    // "~19 GB", "~1.6 TB".
    pub fn FormatWeights(&self) -> string {
        if self.WeightsGB <= 0.0 {
            return string("");
        }
        if self.WeightsGB >= 1000.0 {
            let tb = goish::math::Round(self.WeightsGB / 100.0) / 10.0;
            return fmt::Sprintf!("~%v TB", tb);
        }
        let rounded = goish::math::Round(self.WeightsGB);
        if goish::math::Abs(self.WeightsGB - rounded) < 0.01 {
            return fmt::Sprintf!("%v GB", rounded as goish::int);
        }
        fmt::Sprintf!("~%v GB", rounded as goish::int)
    }
}

// Quant is one published quantization of a checkpoint — the full
// inventory, wider than Variants (which are the curated serving recipes):
// it includes formats that only run outside vLLM (GGUF, MLX, EXL2) and
// fine-tuning-only formats (bitsandbytes).
#[derive(Clone, Default)]
pub struct Quant {
    pub Format: string,
    pub Bits: string,
    pub Repo: string,
    pub Provenance: string,
    pub Engines: slice<string>,
    pub Kernel: string,
    pub Note: string,
}

// Model is one runnable checkpoint. Size is total params, with "/active"
// for MoE (capacity plans on total, latency on active).
#[derive(Clone, Default)]
pub struct Model {
    pub Name: string,
    pub Size: string,
    pub MoE: bool,
    pub Attention: string,
    pub Vision: bool,
    pub NativeContext: goish::int,
    pub ContextMethod: string,
    pub KV: KVArch,
    pub License: string,
    pub LicenseNote: string,
    pub Engines: slice<string>,
    pub Image: string,
    pub Variants: slice<Variant>,
    pub Quants: slice<Quant>,
    pub Notes: slice<string>,
}

impl Model {
    // FormatContext renders the context line: "32K native, 131K max (YaRN)".
    pub fn FormatContext(&self) -> string {
        let native = fmtTokens(self.NativeContext);
        let max = self.KV.MaxContext;
        if max <= 0 || max == self.NativeContext {
            if self.ContextMethod != "" {
                return (native.clone()) + (" (") + (self.ContextMethod.clone()) + (")");
            }
            return (native.clone()) + (" native");
        }
        let mut s = (native.clone()) + (" native, ") + (fmtTokens(max)) + (" max");
        if self.ContextMethod != "" {
            s = (s.clone()) + (" (") + (self.ContextMethod.clone()) + (")");
        }
        s
    }

    // FormatArch renders the architecture line: "MoE, MLA" or "dense".
    pub fn FormatArch(&self) -> string {
        let mut s = string("dense");
        if self.MoE {
            s = string("MoE");
        }
        if self.Attention != "" {
            s = (s.clone()) + (", ") + (self.Attention.clone());
        }
        if self.Vision {
            s = (s.clone()) + (", multimodal");
        }
        s
    }

    // FormatLicense renders the license with optional caveat.
    pub fn FormatLicense(&self) -> string {
        if self.LicenseNote != "" {
            return (self.License.clone()) + (" (") + (self.LicenseNote.clone()) + (")");
        }
        self.License.clone()
    }
}

// Family groups the models of one lab/line, ranked by the composite
// popularity ranking (Aug 2026).
#[derive(Clone, Default)]
pub struct Family {
    pub Rank: goish::int,
    pub Name: string,
    pub Models: slice<Model>,
}

static families: Lazy<sync::Mutex<alloc::vec::Vec<Family>>> =
    Lazy::new(|| sync::Mutex::new(alloc::vec::Vec::new()));

pub fn Register(f: Family) {
    families.Lock().push(f);
}

// Families returns the catalog sorted by rank (init_array registration
// order is link-order, not rank order).
pub fn Families() -> alloc::vec::Vec<Family> {
    let mut list = families.Lock().clone();
    list.sort_by(|a, b| a.Rank.cmp(&b.Rank));
    list
}

// Find looks a model up by name across all families.
pub fn Find<S: Into<string>>(name: S) -> (Family, Model, bool) {
    let name = name.into();
    for f in Families().iter() {
        for (_, m) in goish::range!(f.Models.clone()) {
            if m.Name == name {
                return (f.clone(), m.clone(), true);
            }
        }
    }
    (Default::default(), Default::default(), false)
}

// Show prints the serving recipe for one model. With variantName it
// scopes the serve/hardware detail to that variant; with gpuName it
// resolves each variant against that GPU instead. Bare Show prints the
// overview plus a compact hardware support matrix.
pub fn Show<S1: Into<string>, S2: Into<string>, S3: Into<string>>(
    name: S1,
    variantName: S2,
    gpuName: S3,
) -> goish::errors::error {
    let name = name.into();
    let variantName = variantName.into();
    let gpuName = gpuName.into();
    let (f, m, ok) = Find(name.clone());
    if !ok {
        return fmt::Errorf!("unknown model %q (see 'kvlm model ls')", name);
    }

    fmt::Printf!("Model:    %s\n", m.Name.clone());
    fmt::Printf!("Family:   %s (rank #%d)\n", f.Name.clone(), f.Rank);
    fmt::Printf!("Size:     %s (%s)\n", m.Size.clone(), m.FormatArch());
    fmt::Printf!("Context:  %s\n", m.FormatContext());
    let kv = kvLine(&m);
    if kv != "" {
        fmt::Printf!("KV/seq:   %s\n", kv);
    }
    fmt::Printf!("License:  %s\n", m.FormatLicense());
    fmt::Printf!("Engines:  %s\n", strings::Join(m.Engines.clone(), ", "));
    fmt::Printf!("Image:    %s\n", m.Image.clone());

    if m.Variants.Len() > 0 {
        fmt::Printf!(
            "\nVariants on PROD GPU (fp8 KV; pool = recipe utilization x VRAM minus\nweights and runtime reserve; tok/s is single-sequence decode; SEQS =\nsequences resident at once if all run at that context; a plain SEQS\nnumber is the recipe's --max-num-seqs cap, '-' exceeds its max-model-len):\n"
        );
        // concurrency ladder anchored to THIS model's native context:
        // native/8, native/4, native/2, native — so a 32K model reads
        // 4K..32K and a 1M model reads 128K..1M.
        let mut base = m.NativeContext;
        if base <= 0 {
            base = m.KV.MaxContext;
        }
        let points: slice<goish::int> =
            slice!([]goish::int{base / 8, base / 4, base / 2, base});
        let mut tw = tabwriter::NewWriter(os::Stdout(), 10, 1, 3, b' ', 0);
        fmt::Fprintf!(
            tw,
            "VARIANT\tWEIGHTS\tMIN GPU\tPROD GPU\tVRAM\tKV POOL\tTOK/S\tSEQS @ %s/%s/%s/%s\n",
            fmtTokens(points[0usize]),
            fmtTokens(points[1usize]),
            fmtTokens(points[2usize]),
            fmtTokens(points[3usize])
        );
        for (_, v) in goish::range!(m.Variants.clone()) {
            let mut tokS = string("-");
            if v.MeasuredTokS > 0 {
                tokS = fmt::Sprintf!("%d", v.MeasuredTokS);
            } else {
                let (t, ok) = TokSPerSeq(&m, &v);
                if ok {
                    tokS = fmt::Sprintf!("~%d", round2sig(t));
                }
            }
            let mut vramCell = string("-");
            let mut poolCell = string("-");
            let mut seqCell = string("-");
            let (cap, _, pool, _, ok) = MemProfile(&m, &v);
            if ok {
                vramCell = fmt::Sprintf!("%v GB", goish::math::Round(cap));
                poolCell = string("does not fit");
                if pool > 0.0 {
                    poolCell = fmt::Sprintf!("%v GB", goish::math::Round(pool));
                    // SEQS reconciles two limits: what the pool holds
                    // (estimate, ~) and what the recipe admits via
                    // --max-num-seqs (config, plain). Ladder points the
                    // recipe's --max-model-len rejects show "-".
                    let (maxSeqs, hasSeqCap) = RecipeMaxSeqs(&v);
                    let (maxLen, hasLenCap) = RecipeMaxLen(&v);
                    seqCell = string("");
                    for (i, p) in goish::range!(points.clone()) {
                        let mut cell = string("-");
                        if *p > 0 && !(hasLenCap && *p > maxLen) {
                            let (kvB, _, _) = KVCacheBytes(&m, *p);
                            if kvB > 0 {
                                let s = (pool / ((kvB as goish::float64) / 1e9)) as goish::int;
                                if hasSeqCap && s > maxSeqs {
                                    cell = fmt::Sprintf!("%d", maxSeqs);
                                } else if s <= 0 {
                                    cell = string("0");
                                } else {
                                    cell = fmt::Sprintf!("~%d", round2sig(s));
                                }
                            }
                        }
                        if i > 0 {
                            seqCell = (seqCell) + (" / ");
                        }
                        seqCell = (seqCell) + (cell);
                    }
                }
            }
            fmt::Fprintf!(
                tw,
                "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                v.Name.clone(),
                v.FormatWeights(),
                v.MinGPU.clone(),
                v.ProdGPU.clone(),
                vramCell,
                poolCell,
                tokS,
                seqCell
            );
        }
        let _ = tw.Flush();
    }

    if m.Quants.Len() > 0 {
        fmt::Printf!("\nQuantizations (all published formats):\n");
        let mut tw = tabwriter::NewWriter(os::Stdout(), 10, 1, 3, b' ', 0);
        fmt::Fprintf!(tw, "FORMAT\tBITS\tREPO\tSOURCE\tENGINES\n");
        for (_, q) in goish::range!(m.Quants.clone()) {
            fmt::Fprintf!(
                tw,
                "%s\t%s\t%s\t%s\t%s\n",
                q.Format.clone(),
                q.Bits.clone(),
                q.Repo.clone(),
                q.Provenance.clone(),
                strings::Join(q.Engines.clone(), ", ")
            );
        }
        let _ = tw.Flush();
    }

    if gpuName != "" {
        let err = showResolved(&m, gpuName.clone(), variantName.clone());
        if err != nil {
            return err;
        }
    } else if variantName != "" {
        let mut found = false;
        for (_, v) in goish::range!(m.Variants.clone()) {
            if v.Name == variantName {
                showVariantDetail(&m, &v);
                found = true;
            }
        }
        if !found {
            let mut names: slice<string> = make!([]string, 0);
            for (_, v) in goish::range!(m.Variants.clone()) {
                names = append!(names.clone(), v.Name.clone());
            }
            return fmt::Errorf!(
                "unknown variant %q for %q (have: %s)",
                variantName,
                name,
                strings::Join(names, ", ")
            );
        }
    } else {
        showMatrix(&m);
        fmt::Printf!(
            "\nServe commands: kvlm model show %s <variant>, or --gpu <type> for one GPU\n",
            m.Name.clone()
        );
    }

    if m.Notes.Len() > 0 {
        fmt::Printf!("\nNotes:\n");
        for (_, n) in goish::range!(m.Notes.clone()) {
            fmt::Printf!("  - %s\n", n.clone());
        }
    }

    nil.into()
}

// showVariantDetail prints one variant's serve commands and hardware
// table (the detail view behind `model show <model> <variant>`).
fn showVariantDetail(m: &Model, v: &Variant) {
    let _ = m;
    for (_, spec) in goish::range!(v.Serve.clone()) {
        let mut label = v.Name.clone();
        if spec.Profile != "" {
            label = (label) + (", ") + (spec.Profile.clone());
        }
        if v.Image != "" {
            fmt::Printf!("\nServe (%s), image %s:\n", label, v.Image.clone());
        } else {
            fmt::Printf!("\nServe (%s):\n", label);
        }
        for (_, line) in goish::range!(strings::Split(spec.Render(), "\n")) {
            fmt::Printf!("  %s\n", line.clone());
        }
    }
    if v.Serve.Len() == 0 {
        fmt::Printf!("\nServe (%s): recipe pending\n", v.Name.clone());
    }
    if v.Hardware.Len() > 0 {
        fmt::Printf!("\nHardware (%s):\n", v.Name.clone());
        let mut tw = tabwriter::NewWriter(os::Stdout(), 10, 1, 3, b' ', 0);
        fmt::Fprintf!(tw, "TARGET\tIMAGE\tFLOOR\tNOTE\n");
        for (_, h) in goish::range!(v.Hardware.clone()) {
            let mut floor = h.Floor.clone();
            if h.FloorRef != "" {
                floor = (floor.clone()) + (" (") + (h.FloorRef.clone()) + (")");
            }
            fmt::Fprintf!(
                tw,
                "%s\t%s\t%s\t%s\n",
                h.Target.clone(),
                h.Image.clone(),
                floor,
                h.Note.clone()
            );
        }
        let _ = tw.Flush();
    }
}

// showMatrix prints the variant x target support matrix: each cell is
// the vLLM floor on that combination, "-" where unsupported. Rows come
// from the distinct Hardware.Key values in first-seen order; keyless
// rows (alternative images) are skipped.
fn showMatrix(m: &Model) {
    let mut keys: slice<string> = make!([]string, 0);
    let mut labels: slice<string> = make!([]string, 0);
    for (_, v) in goish::range!(m.Variants.clone()) {
        for (_, h) in goish::range!(v.Hardware.clone()) {
            if h.Key == "" {
                continue;
            }
            let mut seen = false;
            for (_, k) in goish::range!(keys.clone()) {
                if *k == h.Key {
                    seen = true;
                }
            }
            if !seen {
                keys = append!(keys.clone(), h.Key.clone());
                labels = append!(labels.clone(), h.Target.clone());
            }
        }
    }
    if keys.Len() == 0 {
        return;
    }
    fmt::Printf!("\nHardware support (vLLM floor per variant x target):\n");
    let mut tw = tabwriter::NewWriter(os::Stdout(), 10, 1, 3, b' ', 0);
    let mut head = string("TARGET");
    for (_, v) in goish::range!(m.Variants.clone()) {
        head = (head) + ("\t") + (v.Name.clone());
    }
    fmt::Fprintf!(tw, "%s\n", head);
    for (i, k) in goish::range!(keys.clone()) {
        let mut row = labels[i as usize].clone();
        for (_, v) in goish::range!(m.Variants.clone()) {
            // same resolution rule as --gpu (blackwell falls back to
            // cuda rows), so the matrix never contradicts the resolver
            let mut cell = string("-");
            let (h, ok) = PickHardware(&v, k.clone());
            if ok {
                cell = h.Floor.clone();
                if h.FloorRef != "" {
                    cell = (cell.clone()) + (" (") + (h.FloorRef.clone()) + (")");
                }
                if h.Profile != "" {
                    cell = (cell.clone()) + (" [") + (h.Profile.clone()) + ("]");
                }
            }
            row = (row) + ("\t") + (cell);
        }
        fmt::Fprintf!(tw, "%s\n", row);
    }
    let _ = tw.Flush();
}

// showResolved prints, for each variant (or just variantFilter), the
// recipe as it lands on one concrete GPU type: the matching hardware
// row's image and floor, plus the serve profile that applies there.
fn showResolved(m: &Model, gpuName: string, variantFilter: string) -> goish::errors::error {
    let (g, ok) = LookupGPU(gpuName.clone());
    if !ok {
        return fmt::Errorf!("unknown GPU %q (know: h100, h200, b200, a100, l40s, l4, a10g, t4, rtx 4090, rtx pro 6000, mi300x)", gpuName);
    }
    fmt::Printf!("\nResolved for %s (%s, %d GB, %d GB/s):\n", g.Name.clone(), g.Class.clone(), g.VRAMGB, g.BWGBs);
    for (_, v) in goish::range!(m.Variants.clone()) {
        if variantFilter != "" && v.Name != variantFilter {
            continue;
        }
        if v.Hardware.Len() == 0 {
            fmt::Printf!("\n%s: no hardware data yet\n", v.Name.clone());
            continue;
        }
        let (h, ok) = PickHardware(&v, g.Class.clone());
        if !ok {
            fmt::Printf!("\n%s: no %s support\n", v.Name.clone(), g.Class.clone());
            continue;
        }
        fmt::Printf!("\n%s on %s (%s):\n", v.Name.clone(), g.Name.clone(), h.Target.clone());
        let mut floor = h.Floor.clone();
        if h.FloorRef != "" {
            floor = (floor.clone()) + (" (") + (h.FloorRef.clone()) + (")");
        }
        fmt::Printf!("  Image:  %s (floor %s)\n", h.Image.clone(), floor);
        if h.Note != "" {
            fmt::Printf!("  Note:   %s\n", h.Note.clone());
        }
        let mut served = false;
        for (_, spec) in goish::range!(v.Serve.clone()) {
            if spec.Profile != h.Profile {
                continue;
            }
            served = true;
            for (_, line) in goish::range!(strings::Split(spec.Render(), "\n")) {
                fmt::Printf!("  %s\n", line.clone());
            }
        }
        if !served {
            fmt::Printf!("  (recipe pending)\n");
        }
    }
    nil.into()
}

// VRAM prints the context -> KV cache -> VRAM pipeline for one model +
// variant (first variant when unnamed).
pub fn VRAM<S1: Into<string>, S2: Into<string>>(
    name: S1,
    variantName: S2,
    ctx: goish::int,
    seqs: goish::int,
) -> goish::errors::error {
    let name = name.into();
    let variantName = variantName.into();
    let (_, m, ok) = Find(name.clone());
    if !ok {
        return fmt::Errorf!("unknown model %q (see 'kvlm model ls')", name);
    }
    if m.KV.Kind == "" {
        return fmt::Errorf!("model %q has no attention-architecture data yet", name);
    }
    if m.Variants.Len() == 0 {
        return fmt::Errorf!("model %q has no variants", name);
    }

    let mut v = m.Variants[0usize].clone();
    if variantName != "" {
        let mut found = false;
        let mut names: slice<string> = make!([]string, 0);
        for (_, cand) in goish::range!(m.Variants.clone()) {
            names = append!(names.clone(), cand.Name.clone());
            if cand.Name == variantName {
                v = cand.clone();
                found = true;
            }
        }
        if !found {
            return fmt::Errorf!(
                "unknown variant %q for %q (have: %s)",
                variantName,
                name,
                strings::Join(names, ", ")
            );
        }
    }

    let (kvBytes, clamped, _) = KVCacheBytes(&m, ctx);
    let perTok = m.KV.PerTokenBytes();

    let mut archDesc = m.KV.Kind.clone();
    let kind: &str = m.KV.Kind.as_ref();
    match kind {
        "gqa" => {
            archDesc = fmt::Sprintf!(
                "GQA %dL x %dKV x %dd",
                m.KV.Layers,
                m.KV.KVHeads,
                m.KV.HeadDim
            );
        }
        "mla" => {
            archDesc = fmt::Sprintf!("MLA %dL x %d latent", m.KV.Layers, m.KV.LatentDim);
        }
        "swa" => {
            archDesc = fmt::Sprintf!(
                "sliding-window %dL (1/%d full, window %d)",
                m.KV.Layers,
                m.KV.FullEvery,
                m.KV.Window
            );
        }
        "kda" => {
            archDesc = fmt::Sprintf!(
                "KDA hybrid %dL (1/%d full attn, %d MB state/linear layer)",
                m.KV.Layers,
                m.KV.FullEvery,
                m.KV.StateB / 1000000
            );
        }
        _ => {
            archDesc = string("unpublished (reverse-derived estimate)");
        }
    }

    fmt::Printf!("Model:      %s (%s)\n", m.Name.clone(), v.Name.clone());
    fmt::Printf!("Arch:       %s, %s/token (fp8 KV)\n", archDesc, fmtKB(perTok));
    if clamped < ctx {
        fmt::Printf!(
            "Context:    %d tokens requested, clamped to the %d cap (%s)\n",
            ctx,
            clamped,
            fmtTokens(clamped)
        );
    } else {
        fmt::Printf!(
            "Context:    %d tokens (cap %s)\n",
            ctx,
            fmtTokens(m.KV.MaxContext)
        );
    }
    fmt::Printf!("KV/seq:     %s\n", fmtGB(kvBytes));
    fmt::Printf!("KV pool:    %s (%d seqs)\n", fmtGB(kvBytes * seqs), seqs);
    if v.WeightsGB > 0.0 {
        let (total, _, _) = VRAMEstimateGB(&m, &v, ctx, seqs);
        fmt::Printf!("Weights:    %s (%s)\n", v.FormatWeights(), v.Name.clone());
        let totalRounded = goish::math::Round(total) as goish::int;
        fmt::Printf!("VRAM est:   ~%d GB (weights + KV pool + runtime reserve)\n", totalRounded);
        fmt::Printf!("Min GPU:    %s\n", v.MinGPU.clone());
        if v.ProdGPU != "" {
            if v.MeasuredTokS > 0 {
                fmt::Printf!("Prod GPU:   %s (%d tok/s decode per seq, measured)\n", v.ProdGPU.clone(), v.MeasuredTokS);
            } else {
                let (tokS, tok_ok) = TokSPerSeq(&m, &v);
                if tok_ok {
                    fmt::Printf!("Prod GPU:   %s (~%d tok/s decode per seq)\n", v.ProdGPU.clone(), tokS);
                } else {
                    fmt::Printf!("Prod GPU:   %s\n", v.ProdGPU.clone());
                }
            }
        }
    } else {
        fmt::Printf!("Weights:    unknown; no numeric size, VRAM total unavailable\n");
    }
    if m.KV.Est {
        fmt::Printf!("Note:       architecture unpublished; figures are estimates\n");
    }
    nil.into()
}

// fmtKB renders per-token bytes ("128.0 KB").
pub(crate) fn fmtKB(bytes: goish::int) -> string {
    let v = goish::math::Round((bytes as goish::float64) / 100.0) / 10.0;
    fmt::Sprintf!("%v KB", v)
}

// List prints the catalog docker-style (same tabwriter geometry as
// `docker images`: minwidth 20, tabwidth 1, padding 3).
pub fn List() {
    let mut tw = tabwriter::NewWriter(os::Stdout(), 20, 1, 3, b' ', 0);
    fmt::Fprintf!(tw, "MODEL\tFAMILY\tSIZE\n");
    for f in Families().iter() {
        for (_, m) in goish::range!(f.Models.clone()) {
            fmt::Fprintf!(tw, "%s\t%s\t%s\n", m.Name.clone(), f.Name.clone(), m.Size.clone());
        }
    }
    let _ = tw.Flush();
}
