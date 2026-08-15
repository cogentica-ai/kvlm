// profile::graph regression tests: the pure analysis behind `kvlm
// profile graph`. Pins the period detection that finds the repeating
// layer block, the demangled-name simplifier (wrapper-template descent
// verified against real vLLM 0.26 kernel names from the 2026-08-08
// H100 capture), and the bottleneck-verdict heuristics with the
// measured numbers that motivated them.
#![no_std]
#![no_main]
#![allow(non_snake_case)]

extern crate alloc;

use goish::fmt;
use goish::strings;
use goish::syscall;
use goish::testing;
use goish::string;
use goish::{append, int32, make, slice};

use kvlm::profile::graph;

fn names(list: &[&'static str]) -> goish::slice<string> {
    let mut out: goish::slice<string> = make!([]string, 0);
    for n in list.iter() {
        out = append!(out.clone(), string(*n));
    }
    out
}

fn test_detect_unit_transformer_shape(t: &mut testing::T) {
    // 1 prologue + 4 repeats of a 3-node block + 1 epilogue
    let seq = names(&[
        "embed", "norm", "gemm", "act", "norm", "gemm", "act", "norm", "gemm",
        "act", "norm", "gemm", "act", "head",
    ]);
    let (u, ok) = graph::DetectUnit(seq);
    if !ok {
        t.Fatal(string("expected a unit"));
    }
    if u.Start != 1 || u.Period != 3 || u.Repeats != 4 {
        t.Fatal(fmt::Sprintf!(
            "got start=%d period=%d repeats=%d, want 1/3/4",
            u.Start, u.Period, u.Repeats
        ));
    }
}

fn test_detect_unit_prefers_smallest_period(t: &mut testing::T) {
    // 6 repeats of a 2-node block also matches at period 4; the
    // smallest period must win
    let seq = names(&[
        "a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b",
    ]);
    let (u, ok) = graph::DetectUnit(seq);
    if !ok || u.Period != 2 {
        t.Fatal(fmt::Sprintf!("got period=%d, want 2", u.Period));
    }
    if u.Repeats != 6 || u.Start != 0 {
        t.Fatal(fmt::Sprintf!("got start=%d repeats=%d, want 0/6", u.Start, u.Repeats));
    }
}

fn test_detect_unit_none(t: &mut testing::T) {
    let seq = names(&["a", "b", "c", "d", "e", "f"]);
    let (_, ok) = graph::DetectUnit(seq);
    if ok {
        t.Fatal(string("no repetition, expected ok=false"));
    }
}

fn test_simplify_plain_name(t: &mut testing::T) {
    let got = graph::SimplifyKernelName("triton_poi_fused_0");
    if got != "triton_poi_fused_0" {
        t.Fatal(fmt::Sprintf!("got %q", got));
    }
}

fn test_simplify_template_kernel(t: &mut testing::T) {
    let got = graph::SimplifyKernelName(
        "void vllm::rms_norm_per_block_quant_kernel<c10::BFloat16, c10::Float8_e4m3fn, (bool)0, (bool)1, (int)128>(T2 *)",
    );
    if got != "rms_norm_per_block_quant_kernel" {
        t.Fatal(fmt::Sprintf!("got %q", got));
    }
}

fn test_simplify_unwraps_device_kernel(t: &mut testing::T) {
    let got = graph::SimplifyKernelName(
        "void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, (int)1, (int)128>>(T1)",
    );
    if got != "cutlass_3x_gemm_fp8_blockwise" {
        t.Fatal(fmt::Sprintf!("got %q", got));
    }
}

fn test_simplify_descends_nested_wrappers(t: &mut testing::T) {
    // the real FlashAttention name: two wrapper levels deep
    let got = graph::SimplifyKernelName(
        "void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<int>>>>(T1)",
    );
    if got != "FlashAttnFwdSm90" {
        t.Fatal(fmt::Sprintf!("got %q", got));
    }
}

fn test_classify_order_matters(t: &mut testing::T) {
    // cache before attention: this kernel mentions flash but writes KV
    let cases: &[(&'static str, &'static str)] = &[
        ("reshape_and_cache_flash_kernel", "kv cache"),
        ("FlashAttnFwdSm90", "attention"),
        ("cutlass_3x_gemm_fp8_blockwise", "matrix multiply"),
        ("rms_norm_per_block_quant_kernel", "normalization"),
        ("silu_and_mul_per_block_quant_kernel", "activation"),
        ("triton_poi_fused__to_copy_embedding_0", "embedding"),
        ("triton_red_fused_1", "reduction"),
        ("per_token_group_quant_8bit_kernel", "quantization"),
        ("totally_unknown_thing", ""),
    ];
    for (name, want) in cases.iter() {
        let got = graph::ClassifyKernel(*name);
        if got != *want {
            t.Fatal(fmt::Sprintf!("%s: got %q want %q", string(*name), got, string(*want)));
        }
    }
}

fn test_node_verdicts(t: &mut testing::T) {
    // the measured MLP down GEMM: long, small grid, flat across batch
    // sizes; flatness must win over the grid note
    let v = graph::NodeVerdict(40, 132, 72.8, 1.0);
    if !strings::Contains(v.clone(), "waiting on memory") {
        t.Fatal(fmt::Sprintf!("down GEMM: got %q", v));
    }
    // the measured FlashAttention: time drops with the smaller batch
    let v = graph::NodeVerdict(132, 132, 39.9, 0.7);
    if !strings::Contains(v.clone(), "scales with load") {
        t.Fatal(fmt::Sprintf!("attention: got %q", v));
    }
    // the measured rms-norm: short, 24 blocks on 132 SMs
    let v = graph::NodeVerdict(24, 132, 7.4, 1.01);
    if !strings::Contains(v.clone(), "too small to fill the GPU") {
        t.Fatal(fmt::Sprintf!("norm: got %q", v));
    }
    // long kernel, small grid, no comparison: an occupancy note, not a
    // latency-bound claim
    let v = graph::NodeVerdict(40, 132, 72.8, 0.0);
    if !strings::Contains(v.clone(), "uses only part of the GPU") {
        t.Fatal(fmt::Sprintf!("no-ratio GEMM: got %q", v));
    }
    // big grid, short, ratio in the ambiguous band: no verdict
    let v = graph::NodeVerdict(432, 132, 2.2, 0.93);
    if v != "" {
        t.Fatal(fmt::Sprintf!("ambiguous: got %q", v));
    }
    // no GPU spec: grid heuristics disabled
    let v = graph::NodeVerdict(24, 0, 7.4, 0.0);
    if v != "" {
        t.Fatal(fmt::Sprintf!("no-gpu: got %q", v));
    }
}

fn test_busy_verdicts(t: &mut testing::T) {
    let v = graph::BusyVerdict(97.9);
    if !strings::Contains(v.clone(), "GPU sets the pace") {
        t.Fatal(fmt::Sprintf!("busy: got %q", v));
    }
    let v = graph::BusyVerdict(70.0);
    if !strings::Contains(v.clone(), "CPU is holding the GPU back") {
        t.Fatal(fmt::Sprintf!("idle: got %q", v));
    }
}

fn test_verdict_class(t: &mut testing::T) {
    let cases: &[(&'static str, &'static str)] = &[
        ("waiting on memory: same time at any batch size (re-reads the weights)", "waiting on memory"),
        ("scales with load: takes longer as the batch grows", "scales with load"),
        ("too small to fill the GPU: 24 blocks on 132 SMs", "too small to fill the GPU"),
        ("uses only part of the GPU: 40 blocks on 132 SMs", "too small to fill the GPU"),
        ("", "no clear verdict"),
    ];
    for (v, want) in cases.iter() {
        let got = graph::VerdictClass(*v);
        if got != *want {
            t.Fatal(fmt::Sprintf!("%q: got %q want %q", string(*v), got, string(*want)));
        }
    }
}

fn test_correlate_groups_and_sorts(t: &mut testing::T) {
    let mut nodes: goish::slice<graph::Node> = make!([]graph::Node, 0);
    let mk = |cat: &'static str, verdict: &'static str, pct: f64| graph::Node {
        Category: string(cat),
        Verdict: string(verdict),
        PctOfLayer: pct,
        ..Default::default()
    };
    nodes = append!(nodes.clone(), mk("matrix multiply", "waiting on memory: flat", 34.2));
    nodes = append!(nodes.clone(), mk("matrix multiply", "waiting on memory: flat", 23.6));
    nodes = append!(nodes.clone(), mk("attention", "scales with load: grows", 12.9));
    nodes = append!(nodes.clone(), mk("normalization", "too small to fill the GPU: 24 blocks", 2.4));
    let corr = graph::Correlate(nodes);
    if corr.Len() != 3 {
        t.Fatal(fmt::Sprintf!("got %d classes, want 3", corr.Len()));
    }
    // heaviest first: the two GEMMs merged into one class
    let top = corr[0usize].clone();
    if top.Class != "waiting on memory" || top.NodeCount != 2 {
        t.Fatal(fmt::Sprintf!("top class %q count %d", top.Class, top.NodeCount));
    }
    if top.PctOfLayer < 57.7 || top.PctOfLayer > 57.9 {
        t.Fatal(fmt::Sprintf!("top pct %v", top.PctOfLayer));
    }
    let mut hasSeqsLever = false;
    for (_, l) in goish::range!(top.Levers.clone()) {
        if strings::Contains(l.Param.clone(), "--max-num-seqs") {
            hasSeqsLever = true;
        }
    }
    if !hasSeqsLever {
        t.Fatal(string("memory levers missing --max-num-seqs"));
    }
    let mut hasCutlass = false;
    for (_, p) in goish::range!(top.KernelPaths.clone()) {
        if strings::Contains(p.clone(), "cutlass") {
            hasCutlass = true;
        }
    }
    if !hasCutlass {
        t.Fatal(string("memory kernels missing cutlass path"));
    }
    if corr[1usize].Class != "scales with load" {
        t.Fatal(fmt::Sprintf!("second class %q", corr[1usize].Class.clone()));
    }
}

#[goish::main]
fn main() {
    let tests: &[(&str, testing::TestFn)] = &[
        ("TestDetectUnitTransformerShape", test_detect_unit_transformer_shape),
        ("TestDetectUnitPrefersSmallestPeriod", test_detect_unit_prefers_smallest_period),
        ("TestDetectUnitNone", test_detect_unit_none),
        ("TestSimplifyPlainName", test_simplify_plain_name),
        ("TestSimplifyTemplateKernel", test_simplify_template_kernel),
        ("TestSimplifyUnwrapsDeviceKernel", test_simplify_unwraps_device_kernel),
        ("TestSimplifyDescendsNestedWrappers", test_simplify_descends_nested_wrappers),
        ("TestClassifyOrderMatters", test_classify_order_matters),
        ("TestNodeVerdicts", test_node_verdicts),
        ("TestBusyVerdicts", test_busy_verdicts),
        ("TestVerdictClass", test_verdict_class),
        ("TestCorrelateGroupsAndSorts", test_correlate_groups_and_sorts),
    ];
    let code = testing::Main(tests);
    syscall::Exit(int32(code));
}
