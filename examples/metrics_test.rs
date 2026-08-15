// metrics package regression tests, pinned to REAL vLLM 0.26.0 output
// captured 2026-08-08 (qwen3-32b fp8 on 1xH100 SXM, the 180 s load of
// 961 requests). Fixtures: examples/testdata/metrics-{before,after}.txt,
// byte-for-byte copies of profile-output/. Run from the repo root.
#![no_std]
#![no_main]
#![allow(non_snake_case)]

extern crate alloc;

use goish::fmt;
use goish::math;
use goish::os;
use goish::syscall;
use goish::testing;
use goish::string;
use goish::{int32, nil};

use kvlm::metrics;

fn loadFixture(t: &mut testing::T, name: &'static str) -> metrics::Snapshot {
    let path = ("examples/testdata/") + (string(name));
    let (data, err) = os::ReadFile(path.clone());
    if err != nil {
        t.Fatal(fmt::Sprintf!("read %s: %v (run tests from the repo root)", path, err));
    }
    metrics::Parse(string(data))
}

fn test_parse_real_after_snapshot(t: &mut testing::T) {
    let s = loadFixture(t, "metrics-after.txt");
    if math::Abs(s.TTFTSum - 87.95222401618958) > 1e-9 || s.TTFTCount != 961.0 {
        t.Fatal(fmt::Sprintf!("TTFT: sum=%v count=%v", s.TTFTSum, s.TTFTCount));
    }
    // TPOT arrives under the 0.26 name request_time_per_output_token
    if math::Abs(s.TPOTSum - 20.152015517908307) > 1e-9 || s.TPOTCount != 961.0 {
        t.Fatal(fmt::Sprintf!("TPOT (0.26 name): sum=%v count=%v", s.TPOTSum, s.TPOTCount));
    }
    if math::Abs(s.E2ESum - 4348.180292844772) > 1e-6 || s.E2ECount != 961.0 {
        t.Fatal(fmt::Sprintf!("E2E: sum=%v count=%v", s.E2ESum, s.E2ECount));
    }
    // one series per finished_reason (stop 5 + length 956); a plain
    // assignment would keep only the last series
    if s.RequestSuccess != 961.0 {
        t.Fatal(fmt::Sprintf!("request_success must sum labeled series: got %v, want 961", s.RequestSuccess));
    }
    if s.GenerationTokens != 204020.0 || s.PromptTokens != 893811.0 {
        t.Fatal(fmt::Sprintf!("tokens: gen=%v prompt=%v", s.GenerationTokens, s.PromptTokens));
    }
    if s.PrefixCacheQueries != 893811.0 || s.PrefixCacheHits != 879872.0 {
        t.Fatal(fmt::Sprintf!("prefix cache: q=%v h=%v", s.PrefixCacheQueries, s.PrefixCacheHits));
    }
}

fn test_parse_real_before_snapshot(t: &mut testing::T) {
    let s = loadFixture(t, "metrics-before.txt");
    if s.RequestSuccess != 0.0 || s.TTFTCount != 0.0 || s.Running != 0.0 || s.Waiting != 0.0 {
        t.Fatal(fmt::Sprintf!(
            "before snapshot should be all zero: req=%v ttftc=%v run=%v wait=%v",
            s.RequestSuccess,
            s.TTFTCount,
            s.Running,
            s.Waiting
        ));
    }
}

fn test_derive_real_run(t: &mut testing::T) {
    let b = loadFixture(t, "metrics-before.txt");
    let a = loadFixture(t, "metrics-after.txt");
    let d = metrics::Derive(&b, &a, 180.0);
    if d.Requests != 961.0 {
        t.Fatal(fmt::Sprintf!("requests: got %v", d.Requests));
    }
    if !d.HasTTFT || math::Abs(d.TTFTMeanMs - 91.5) > 0.1 {
        t.Fatal(fmt::Sprintf!("TTFT mean: got %v, want 91.5 ms", d.TTFTMeanMs));
    }
    if !d.HasTPOT || math::Abs(d.TPOTMeanMs - 21.0) > 0.1 {
        t.Fatal(fmt::Sprintf!("TPOT mean: got %v, want 21.0 ms", d.TPOTMeanMs));
    }
    if !d.HasE2E || math::Abs(d.E2EMeanS - 4.52) > 0.01 {
        t.Fatal(fmt::Sprintf!("e2e mean: got %v, want 4.52 s", d.E2EMeanS));
    }
    if !d.HasPrefix || math::Abs(d.PrefixHitRate - 0.984) > 0.001 {
        t.Fatal(fmt::Sprintf!("prefix hit rate: got %v, want 0.984", d.PrefixHitRate));
    }
    if math::Abs(d.GenTokS - 1133.4) > 0.5 {
        t.Fatal(fmt::Sprintf!("gen rate: got %v, want ~1133.4 tok/s", d.GenTokS));
    }
    if math::Abs(d.PromptTokS - 4965.6) > 0.5 {
        t.Fatal(fmt::Sprintf!("prompt rate: got %v, want ~4965.6 tok/s", d.PromptTokS));
    }
    // the latency split: queue is negligible in this run (no capacity
    // pressure), prefill 65 ms, decode 4.43 s; together they explain
    // the 4.52 s e2e mean
    if !d.HasQueue || d.QueueMeanS > 0.001 {
        t.Fatal(fmt::Sprintf!("queue mean: got %v, want ~0 s", d.QueueMeanS));
    }
    if !d.HasPrefillT || math::Abs(d.PrefillMeanS - 0.0646) > 0.001 {
        t.Fatal(fmt::Sprintf!("prefill mean: got %v, want 0.0646 s", d.PrefillMeanS));
    }
    if !d.HasDecodeT || math::Abs(d.DecodeMeanS - 4.433) > 0.01 {
        t.Fatal(fmt::Sprintf!("decode mean: got %v, want 4.433 s", d.DecodeMeanS));
    }
    if d.Preemptions != 0.0 {
        t.Fatal(fmt::Sprintf!("preemptions: got %v, want 0", d.Preemptions));
    }
}

fn test_derive_no_traffic(t: &mut testing::T) {
    // Identical snapshots: every Has* is false and nothing divides by
    // zero or produces NaN.
    let a = loadFixture(t, "metrics-after.txt");
    let d = metrics::Derive(&a, &a, 30.0);
    if d.HasTTFT || d.HasTPOT || d.HasE2E || d.HasPrefix {
        t.Fatal(string("no-traffic window must report no data, not stale means"));
    }
    if d.Requests != 0.0 || d.GenTokS != 0.0 {
        t.Fatal(fmt::Sprintf!("no-traffic rates: req=%v gen=%v", d.Requests, d.GenTokS));
    }
}

#[goish::main]
fn main() {
    let tests: &[(&str, testing::TestFn)] = &[
        ("TestParseRealAfterSnapshot", test_parse_real_after_snapshot),
        ("TestParseRealBeforeSnapshot", test_parse_real_before_snapshot),
        ("TestDeriveRealRun", test_derive_real_run),
        ("TestDeriveNoTraffic", test_derive_no_traffic),
    ];
    let code = testing::Main(tests);
    syscall::Exit(int32(code));
}
