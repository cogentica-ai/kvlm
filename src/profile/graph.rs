// graph: pure analysis for `kvlm profile graph`. Takes the per-node
// kernel aggregates extracted from a node-mode nsys capture (any
// model, any GPU) and finds the structure and the bottlenecks:
//
//   - DetectUnit finds the repeating layer block by periodicity, so a
//     48-layer 11-node model works the same as a 64x15 one.
//   - SimplifyKernelName / ClassifyKernel turn demangled C++ kernel
//     names into readable labels and coarse categories.
//   - NodeVerdict / BusyVerdict apply the bottleneck heuristics:
//     grid blocks < SMs = latency bound; time flat across two batch
//     sizes = memory (weights-streaming) bound; time scaling with
//     batch = compute or data bound; low GPU-busy fraction = host
//     bound. All trace-derived, no model math needed.
//
// The sqlite extraction and JSON serialization live in cmd/profile.rs;
// everything here is pure so it can be pinned by tests.
#![allow(non_snake_case)]

use goish::fmt;
use goish::math;
use goish::strings;
use goish::string;
use goish::{append, float64, int, make, slice};

// Node is one kernel node of the executable graph, averaged across
// every replay in the capture window. Ratio compares this node's time
// in a second captured graph of a different batch size (0 = no second
// graph to compare against).
#[derive(Clone, Default)]
pub struct Node {
    pub Pos: int,
    pub Name: string,
    pub FullName: string,
    pub Category: string,
    pub Grid: string,
    pub Blocks: int,
    pub AvgUs: float64,
    // worst single execution across all replays; far above AvgUs on a
    // flat kernel it means stalls, not work (e.g. an all-reduce
    // waiting for the peer GPU)
    pub MaxUs: float64,
    pub PctOfLayer: float64,
    pub Ratio: float64,
    pub Verdict: string,
}

// GraphInfo is one executable CUDA graph seen in the capture.
#[derive(Clone, Default)]
pub struct GraphInfo {
    pub ID: int,
    pub Nodes: int,
    pub Replays: int,
    pub TotalMs: float64,
}

// Unit is the detected repeating block: Start/Period are node indexes
// into the full chain (0-based), Repeats is how many times it runs.
#[derive(Clone, Default)]
pub struct Unit {
    pub Start: int,
    pub Period: int,
    pub Repeats: int,
}

// DetectUnit finds the longest periodic run in the node-name sequence.
// For every candidate period p it marks positions where names[i] ==
// names[i+p] and takes the longest consecutive run; the best (longest)
// run wins, and shorter periods win ties naturally because the same
// region gives them a longer run. Requires at least 3 repeats.
pub fn DetectUnit(names: slice<string>) -> (Unit, bool) {
    let n = names.len() as int;
    let mut best = Unit::default();
    let mut bestRun: int = 0;
    let mut p: int = 1;
    while p <= n / 2 {
        let mut run: int = 0;
        let mut i: int = 0;
        while i + p < n {
            if names[i as usize] == names[(i + p) as usize] {
                run += 1;
                if run > bestRun {
                    bestRun = run;
                    best = Unit {
                        Start: i - run + 1,
                        Period: p,
                        Repeats: run / p + 1,
                        ..Default::default()
                    };
                }
            } else {
                run = 0;
            }
            i += 1;
        }
        p += 1;
    }
    if best.Repeats < 3 {
        return (Unit::default(), false);
    }
    (best, true)
}

// SimplifyKernelName turns a demangled C++ kernel name into a short
// label: strip "void " and argument lists, keep the last path
// component, and descend through generic launcher templates
// (cutlass::device_kernel<Inner>, flash::enable_sm90_or_later<Inner>)
// to the kernel they actually launch. Plain names (triton_poi_fused_0)
// pass through.
pub fn SimplifyKernelName<S: Into<string>>(name: S) -> string {
    let mut s: string = name.into();
    s = strings::TrimPrefix(s, string("void "));
    loop {
        // the name ends at the first template, arg list, or separator
        let mut cut = s.Len();
        let mut atTemplate = false;
        for stop in ["<", "(", ",", ">"].iter() {
            let idx = strings::Index(s.clone(), string(*stop));
            if idx >= 0 && idx < cut {
                cut = idx;
                atTemplate = *stop == "<";
            }
        }
        let base = lastPathPart(strings::TrimSpace(s.slice(0, cut)));
        if atTemplate && isWrapper(base.clone()) {
            s = strings::TrimSpace(s.slice(cut + 1, s.Len()));
            continue;
        }
        return base;
    }
}

// isWrapper reports whether a kernel-name component is a generic
// launcher whose first template argument is the real kernel.
fn isWrapper(base: string) -> bool {
    base == "device_kernel"
        || base == "kernel"
        || base == "Kernel2"
        || strings::HasPrefix(base.clone(), string("enable_"))
}

fn lastPathPart(s: string) -> string {
    let idx = strings::LastIndex(s.clone(), string("::"));
    if idx >= 0 {
        return s.slice(idx + 2, s.Len());
    }
    s
}

// ClassifyKernel maps a demangled kernel name to a coarse category.
// Order matters: attention kernels mention cutlass in their template
// arguments, cache kernels mention flash, so the more specific checks
// run first. Unrecognized names return "".
pub fn ClassifyKernel<S: Into<string>>(name: S) -> string {
    let s = strings::ToLower(name.into());
    fn has_in(s: &string, sub: &'static str) -> bool {
        strings::Contains(s.clone(), string(sub))
    }
    let has = |sub: &'static str| has_in(&s, sub);
    if has("cache") {
        return string("kv cache");
    }
    if has("attention") || has("fmha") || has("flash") || has("mha_") {
        return string("attention");
    }
    if has("embedding") || has("embed_") {
        return string("embedding");
    }
    if has("gemm") || has("nvjet") || has("matmul") || has("cutlass") || has("wgmma") {
        return string("matrix multiply");
    }
    if has("rotary") || has("rope") {
        return string("rotary embedding");
    }
    if has("norm") {
        return string("normalization");
    }
    if has("softmax") {
        return string("softmax");
    }
    if has("silu") || has("gelu") || has("swiglu") || has("relu") {
        return string("activation");
    }
    if has("quant") {
        return string("quantization");
    }
    if has("sampl") || has("topk") || has("top_k") || has("top_p") {
        return string("sampling");
    }
    if has("reduce") || has("red_fused") {
        return string("reduction");
    }
    if has("elementwise") || has("poi_fused") || has("copy") || has("cat_") || has("fill") {
        return string("elementwise");
    }
    string("")
}

// NodeVerdict applies the per-node heuristics, strongest evidence
// first. ratio is this node's average time in a second captured graph
// of another batch size divided by its time in the primary graph (0 =
// no comparison available): flat time on a long kernel means it is
// re-reading the same weights whatever the batch (memory bound); time
// moving with batch means the work itself scales (compute or data
// bound). Only for short kernels does a grid smaller than the SM count
// mean the kernel is pure launch-and-drain latency; a long kernel with
// a small grid gets a low-occupancy note instead. blocks and sms gate
// the grid checks (either 0 skips them).
pub fn NodeVerdict(blocks: int, sms: int, avgUs: float64, ratio: float64) -> string {
    if ratio > 0.0 && avgUs >= 10.0 && ratio >= 0.85 && ratio <= 1.15 {
        return string("waiting on memory: same time at any batch size (re-reads the weights)");
    }
    if ratio > 0.0 && (ratio >= 1.25 || ratio <= 0.8) {
        return string("scales with load: takes longer as the batch grows");
    }
    if blocks > 0 && sms > 0 && blocks < sms {
        if avgUs < 10.0 {
            return fmt::Sprintf!("too small to fill the GPU: %d blocks on %d SMs", blocks, sms);
        }
        return fmt::Sprintf!("uses only part of the GPU: %d blocks on %d SMs", blocks, sms);
    }
    string("")
}

// Lever is one concrete knob: the serve flag or format choice, and
// what it does to this bottleneck class.
#[derive(Clone, Default)]
pub struct Lever {
    pub Param: string,
    pub Effect: string,
}

// Correlation is one row of the measurement-to-action table: all the
// unit nodes sharing a bottleneck class, the share of layer time they
// own together, and the levers that actually move that class. The
// point of the split: a flag tweak only helps the classes marked with
// one, and kernel work only pays where the time actually is.
#[derive(Clone, Default)]
pub struct Correlation {
    pub Class: string,
    pub PctOfLayer: float64,
    pub NodeCount: int,
    pub Levers: slice<Lever>,
    pub KernelPaths: slice<string>,
}

// VerdictClass maps a node verdict back to its bottleneck class.
pub fn VerdictClass<S: Into<string>>(v: S) -> string {
    let v: string = v.into();
    if strings::HasPrefix(v.clone(), string("waiting on memory")) {
        return string("waiting on memory");
    }
    if strings::HasPrefix(v.clone(), string("scales with load")) {
        return string("scales with load");
    }
    if strings::HasPrefix(v.clone(), string("too small"))
        || strings::HasPrefix(v.clone(), string("uses only part"))
    {
        return string("too small to fill the GPU");
    }
    string("no clear verdict")
}

// leversFor: which serve flags and format choices move each
// bottleneck class, as discrete knobs.
fn leversFor(class: string) -> slice<Lever> {
    fn lever(param: &'static str, effect: &'static str) -> Lever {
        Lever {
            Param: string(param),
            Effect: string(effect),
        }
    }
    if class == "waiting on memory" {
        return slice!([]Lever{
            lever("nvfp4 / w4a8 weights", "halves the streamed bytes again; the biggest single lever"),
            lever("-tp 2", "splits the weight streaming across two GPUs"),
            lever("--max-num-seqs", "raise it: batch rides along free while these nodes stay flat"),
        });
    }
    if class == "scales with load" {
        return slice!([]Lever{
            lever("--kv-cache-dtype fp8", "halves the KV bytes read every step"),
            lever("--enable-prefix-caching", "skips re-reading shared prefixes; measured +47%"),
            lever("--max-model-len", "cap it to bound the KV each step can touch"),
            lever("--max-num-seqs", "lower it: trades throughput for per-user speed"),
        });
    }
    if class == "too small to fill the GPU" {
        return slice!([]Lever{
            lever("no flag helps", "launch cost is already amortized by CUDA graphs; keep the capture sizes covering your real batch range"),
        });
    }
    make!([]Lever, 0)
}

// kernelPath: where each kernel category lives in the vLLM tree, for
// anyone going after the code itself.
fn kernelPath(category: string) -> string {
    if category == "matrix multiply" {
        return string("csrc/quantization cutlass w8a8 blockwise GEMM");
    }
    if category == "attention" {
        return string("vllm-flash-attn FA3");
    }
    if category == "kv cache" {
        return string("csrc/cache_kernels.cu");
    }
    if category == "normalization" {
        return string("csrc/layernorm quant kernels");
    }
    if category == "quantization" {
        return string("csrc/quantization per-token and group quant");
    }
    if category == "activation" {
        return string("csrc/activation_kernels.cu");
    }
    if category == "rotary embedding" {
        return string("csrc/pos_encoding_kernels.cu");
    }
    if category == "elementwise" || category == "reduction" {
        return string("torch.compile triton fusions (regenerated, not hand-edited)");
    }
    string("")
}

// Correlate groups the unit nodes by bottleneck class, heaviest class
// first, and attaches the levers. Works on any node list (the layer
// unit, or the top-nodes fallback).
pub fn Correlate(nodes: slice<Node>) -> slice<Correlation> {
    let mut out: slice<Correlation> = make!([]Correlation, 0);
    for (_, n) in goish::range!(nodes.clone()) {
        let class = VerdictClass(n.Verdict.clone());
        let path = kernelPath(n.Category.clone());
        let mut found = false;
        let mut i: int = 0;
        while i < out.len() as int {
            if out[i as usize].Class == class {
                out[i as usize].PctOfLayer += n.PctOfLayer;
                out[i as usize].NodeCount += 1;
                if path != "" {
                    let mut seen = false;
                    for (_, p) in goish::range!(out[i as usize].KernelPaths.clone()) {
                        if p == path {
                            seen = true;
                            break;
                        }
                    }
                    if !seen {
                        out[i as usize].KernelPaths = append!(out[i as usize].KernelPaths.clone(), path.clone());
                    }
                }
                found = true;
                break;
            }
            i += 1;
        }
        if !found {
            let mut paths: slice<string> = make!([]string, 0);
            if path != "" {
                paths = append!(paths.clone(), path);
            }
            out = append!(
                out.clone(),
                Correlation {
                    Class: class.clone(),
                    PctOfLayer: n.PctOfLayer,
                    NodeCount: 1,
                    Levers: leversFor(class),
                    KernelPaths: paths,
                    ..Default::default()
                }
            );
        }
    }
    // heaviest class first
    let mut i: int = 1;
    while i < out.len() as int {
        let mut j = i;
        while j > 0 && out[(j - 1) as usize].PctOfLayer < out[j as usize].PctOfLayer {
            let tmp = out[(j - 1) as usize].clone();
            out[(j - 1) as usize] = out[j as usize].clone();
            out[j as usize] = tmp;
            j -= 1;
        }
        i += 1;
    }
    out
}

// BusyVerdict turns the GPU-busy fraction of the capture window into
// the host-bound heuristic.
pub fn BusyVerdict(busyPct: float64) -> string {
    let pct = math::Round(busyPct * 10.0) / 10.0;
    if busyPct >= 95.0 {
        return fmt::Sprintf!(
            "the GPU sets the pace, not the CPU: GPU busy %v%% of the window",
            pct
        );
    }
    if busyPct >= 85.0 {
        return fmt::Sprintf!(
            "mostly GPU limited, some CPU overhead: GPU busy %v%% of the window",
            pct
        );
    }
    fmt::Sprintf!(
        "the CPU is holding the GPU back: GPU idle %v%% of the window (scheduler, sampling, python)",
        math::Round((100.0 - busyPct) * 10.0) / 10.0
    )
}
