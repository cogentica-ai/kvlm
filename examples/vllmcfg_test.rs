// profile::vllmcfg + vllmflags regression tests: the python-repr
// tokenizer, the two startup-line parsers, the flag-name mapping, and
// the per-version catalog round trip. Fixture lines follow the exact
// format vLLM 0.26.0 logs (nested ProfilerConfig, dicts, lists).
#![no_std]
#![no_main]
#![allow(non_snake_case)]

extern crate alloc;

use goish::fmt;
use goish::strings;
use goish::syscall;
use goish::testing;
use goish::string;
use goish::{int32, range};

use kvlm::profile::vllmcfg;
use kvlm::profile::vllmflags;

const NONDEFAULT: &str = "(APIServer pid=885) INFO 08-09 01:40:15 [api_utils.py:273] non-default args: {'model_tag': 'Qwen/Qwen3-32B-FP8', 'max_model_len': 32768, 'served_model_name': ['qwen3-32b'], 'gpu_memory_utilization': 0.9, 'kv_cache_dtype': 'fp8', 'enable_prefix_caching': True, 'max_num_seqs': 32, 'profiler_config': ProfilerConfig(profiler='torch', torch_profiler_dir='/tmp/kvlm-profile/torch')}";

const RESOLVED: &str = "(EngineCore pid=1557) INFO 08-09 01:40:23 [core.py:116] Initializing a V1 LLM engine (v0.26.0) with config: model='Qwen/Qwen3-32B-FP8', speculative_config=None, tensor_parallel_size=1, max_num_seqs=32, enable_prefix_caching=True, compilation_config={'mode': 3, 'cudagraph_capture_sizes': [1, 2, 4]}, kernel_config=KernelConfig(a=['x, y'], b=2)";

fn test_split_top_level(t: &mut testing::T) {
    let parts = vllmcfg::SplitTopLevel(string("a=1, b=(x, y), c={'k': [1, 2]}, d='q, r'"));
    if parts.Len() != 4 {
        t.Fatal(fmt::Sprintf!("got %d parts", parts.Len()));
    }
    if parts[1usize] != "b=(x, y)" || parts[2usize] != "c={'k': [1, 2]}" || parts[3usize] != "d='q, r'" {
        t.Fatal(fmt::Sprintf!("nested split broke: %q %q %q", parts[1usize].clone(), parts[2usize].clone(), parts[3usize].clone()));
    }
}

fn test_parse_non_default(t: &mut testing::T) {
    let kvs = vllmcfg::ParseNonDefault(string(NONDEFAULT));
    let (v, ok) = vllmcfg::Lookup(&kvs, "max_num_seqs");
    if !ok || v != "32" {
        t.Fatal(fmt::Sprintf!("max_num_seqs: %q %v", v, ok));
    }
    let (v, ok) = vllmcfg::Lookup(&kvs, "kv_cache_dtype");
    if !ok || v != "'fp8'" {
        t.Fatal(fmt::Sprintf!("kv_cache_dtype: %q", v));
    }
    // the nested ProfilerConfig value survives with its inner comma
    let (v, ok) = vllmcfg::Lookup(&kvs, "profiler_config");
    if !ok || !strings::Contains(v.clone(), "torch_profiler_dir") {
        t.Fatal(fmt::Sprintf!("profiler_config lost nesting: %q", v));
    }
}

fn test_parse_resolved(t: &mut testing::T) {
    let kvs = vllmcfg::ParseResolved(string(RESOLVED));
    let (v, ok) = vllmcfg::Lookup(&kvs, "tensor_parallel_size");
    if !ok || v != "1" {
        t.Fatal(fmt::Sprintf!("tensor_parallel_size: %q %v", v, ok));
    }
    let (v, ok) = vllmcfg::Lookup(&kvs, "enable_prefix_caching");
    if !ok || v != "True" {
        t.Fatal(fmt::Sprintf!("enable_prefix_caching: %q", v));
    }
    let (v, ok) = vllmcfg::Lookup(&kvs, "compilation_config");
    if !ok || !strings::Contains(v.clone(), "cudagraph_capture_sizes") {
        t.Fatal(fmt::Sprintf!("compilation_config lost nesting: %q", v));
    }
}

fn test_engine_version(t: &mut testing::T) {
    let v = vllmcfg::EngineVersion(string(RESOLVED));
    if v != "0.26.0" {
        t.Fatal(fmt::Sprintf!("version: %q", v));
    }
    if vllmcfg::EngineVersion(string("no version here")) != "" {
        t.Fatal(string("phantom version"));
    }
}

fn test_flag_for(t: &mut testing::T) {
    if vllmcfg::FlagFor("max_num_seqs") != "--max-num-seqs" {
        t.Fatal(string("max_num_seqs mapping"));
    }
    if vllmcfg::FlagFor("gpu_memory_utilization") != "--gpu-memory-utilization" {
        t.Fatal(string("gpu_memory_utilization mapping"));
    }
    if !strings::Contains(vllmcfg::FlagFor("model"), "positional") {
        t.Fatal(string("model must map to the positional arg"));
    }
}

fn test_relevant_keys(t: &mut testing::T) {
    // the keys the lever registry and the pressure verdict rely on
    let keys = vllmcfg::RelevantKeys();
    for want in ["max_num_seqs", "kv_cache_dtype", "gpu_memory_utilization", "tensor_parallel_size", "enable_prefix_caching", "max_num_batched_tokens"].iter() {
        let mut found = false;
        for (_, k) in range!(keys.clone()) {
            if k == *want {
                found = true;
            }
        }
        if !found {
            t.Fatal(fmt::Sprintf!("RelevantKeys lost %q", string(*want)));
        }
    }
}

fn test_catalog_round_trip(t: &mut testing::T) {
    let dump = string("{\"max_num_seqs\": \"32\", \"kv_cache_dtype\": \"'auto'\", \"served_model_name\": \"['a', 'b']\"}");
    let (doc, err) = vllmflags::Render(string("0.26.0"), string("test"), dump);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("Render: %v", err));
    }
    if !strings::Contains(doc.clone(), "\"vllm\": \"0.26.0\"") {
        t.Fatal(string("version lost"));
    }
    let flags = vllmflags::ParseFlags(doc);
    let (v, ok) = vllmcfg::Lookup(&flags, "max_num_seqs");
    if !ok || v != "32" {
        t.Fatal(fmt::Sprintf!("round trip max_num_seqs: %q %v", v, ok));
    }
    let (v, ok) = vllmcfg::Lookup(&flags, "kv_cache_dtype");
    if !ok || v != "'auto'" {
        t.Fatal(fmt::Sprintf!("round trip kv_cache_dtype: %q", v));
    }
}

#[goish::main]
fn main() {
    let tests: &[(&str, testing::TestFn)] = &[
        ("TestSplitTopLevel", test_split_top_level),
        ("TestParseNonDefault", test_parse_non_default),
        ("TestParseResolved", test_parse_resolved),
        ("TestEngineVersion", test_engine_version),
        ("TestFlagFor", test_flag_for),
        ("TestRelevantKeys", test_relevant_keys),
        ("TestCatalogRoundTrip", test_catalog_round_trip),
    ];
    let code = testing::Main(tests);
    syscall::Exit(int32(code));
}
