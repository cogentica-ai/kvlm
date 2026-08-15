// profile::propose regression tests: the flag-proposal arithmetic.
// Fixtures use the measured shape of a real pressure probe (peak
// running, KV percent, preemptions) and pin the computed values.
#![no_std]
#![no_main]
#![allow(non_snake_case)]

extern crate alloc;

use goish::strings;
use goish::syscall;
use goish::testing;
use goish::string;
use goish::{fmt, int32};

use kvlm::profile::propose;

// The cap pinned with KV headroom: propose doubling, bounded by the
// measured KV fit.
fn test_cap_bound_doubles(t: &mut testing::T) {
    let in_ = propose::Inputs {
        BoundBy: string("max-num-seqs"),
        PeakRunning: 32.0,
        PeakKvPct: 14.9,
        MaxNumSeqs: 32.0,
        WorstCaseFit: 9.35,
        KvCacheDtype: string("fp8"),
        ..Default::default()
    };
    let out = propose::Propose(&in_);
    if out.Len() != 1 {
        t.Fatal(fmt::Sprintf!("want 1 proposal, got %d", out.Len()));
    }
    let p = out[0usize].clone();
    if p.Flag != "max_num_seqs" || p.Value != "64" {
        t.Fatal(fmt::Sprintf!("want max_num_seqs=64, got %s=%s", p.Flag, p.Value));
    }
    // 14.9 / 32 = 0.47% per request, so 193 fit inside 90%
    if !strings::Contains(p.Why.clone(), "0.47% KV per request") || !strings::Contains(p.Why.clone(), "193 requests") {
        t.Fatal(fmt::Sprintf!("expectation arithmetic drifted: %q", p.Why));
    }
    // the worst-case fit (9.35 full-context requests) is below the
    // proposal, so the risk line must carry it
    if !strings::Contains(p.Risk.clone(), "9.4 full-context requests") {
        t.Fatal(fmt::Sprintf!("worst-case warning missing: %q", p.Risk));
    }
}

// The cap pinned but the measured KV fit is below doubling: the
// proposal clamps to the fit.
fn test_cap_bound_clamps_to_fit(t: &mut testing::T) {
    let in_ = propose::Inputs {
        BoundBy: string("max-num-seqs"),
        PeakRunning: 32.0,
        PeakKvPct: 64.0, // 2% per request: 45 fit inside 90%
        MaxNumSeqs: 32.0,
        KvCacheDtype: string("fp8"),
        ..Default::default()
    };
    let out = propose::Propose(&in_);
    if out.Len() != 1 {
        t.Fatal(fmt::Sprintf!("want 1 proposal, got %d", out.Len()));
    }
    let p = out[0usize].clone();
    if p.Flag != "max_num_seqs" || p.Value != "45" {
        t.Fatal(fmt::Sprintf!("want max_num_seqs=45, got %s=%s", p.Flag, p.Value));
    }
}

// The cap pinned with no KV room left at all: relief, not a raise.
fn test_cap_bound_no_room_shifts_to_kv(t: &mut testing::T) {
    let in_ = propose::Inputs {
        BoundBy: string("max-num-seqs"),
        PeakRunning: 32.0,
        PeakKvPct: 95.0, // 2.97% per request: only 30 fit, below the cap
        MaxNumSeqs: 32.0,
        KvCacheDtype: string("auto"),
        ..Default::default()
    };
    let out = propose::Propose(&in_);
    if out.Len() < 1 {
        t.Fatal(string("want kv relief proposals"));
    }
    let p = out[0usize].clone();
    if p.Flag != "kv_cache_dtype" || p.Value != "fp8" {
        t.Fatal(fmt::Sprintf!("want kv_cache_dtype=fp8 first, got %s=%s", p.Flag, p.Value));
    }
    if !strings::Contains(p.Risk.clone(), "lossy") {
        t.Fatal(fmt::Sprintf!("fp8 quality risk missing: %q", p.Risk));
    }
}

// KV-bound with fp8 already set and preemptions: cap admission at the
// measured fit.
fn test_kv_bound_fp8_set_caps_admission(t: &mut testing::T) {
    let in_ = propose::Inputs {
        BoundBy: string("kv"),
        PeakRunning: 64.0,
        PeakKvPct: 98.8,
        Preemptions: 31.0,
        MaxNumSeqs: 1024.0,
        KvCacheDtype: string("fp8"),
        ..Default::default()
    };
    let out = propose::Propose(&in_);
    if out.Len() != 1 {
        t.Fatal(fmt::Sprintf!("want 1 proposal, got %d", out.Len()));
    }
    let p = out[0usize].clone();
    // 98.8 / 64 = 1.54% per request: 58 fit inside 90%
    if p.Flag != "max_num_seqs" || p.Value != "58" {
        t.Fatal(fmt::Sprintf!("want max_num_seqs=58, got %s=%s", p.Flag, p.Value));
    }
    if !strings::Contains(p.Why.clone(), "31 preemptions") {
        t.Fatal(fmt::Sprintf!("preemption evidence missing: %q", p.Why));
    }
}

// KV-bound without fp8: dtype first, admission cap second.
fn test_kv_bound_proposes_fp8_first(t: &mut testing::T) {
    let in_ = propose::Inputs {
        BoundBy: string("kv"),
        PeakRunning: 18.0,
        PeakKvPct: 96.2,
        Preemptions: 4.0,
        MaxNumSeqs: 1024.0,
        KvCacheDtype: string("auto"),
        ..Default::default()
    };
    let out = propose::Propose(&in_);
    if out.Len() != 2 {
        t.Fatal(fmt::Sprintf!("want 2 proposals, got %d", out.Len()));
    }
    if out[0usize].Flag != "kv_cache_dtype" {
        t.Fatal(string("fp8 must come first: biggest pool payoff"));
    }
    if !strings::Contains(out[0usize].Why.clone(), "18 requests that filled it become ~36") {
        t.Fatal(fmt::Sprintf!("doubling arithmetic drifted: %q", out[0usize].Why));
    }
    if out[1usize].Flag != "max_num_seqs" {
        t.Fatal(string("admission cap must follow"));
    }
}

// Bound by neither: a report, not a flag.
fn test_unbound_reports_headroom(t: &mut testing::T) {
    let in_ = propose::Inputs {
        BoundBy: string("none"),
        PeakRunning: 32.0,
        PeakKvPct: 71.4,
        MaxNumSeqs: 1024.0,
        KvCacheDtype: string("fp8"),
        ..Default::default()
    };
    let out = propose::Propose(&in_);
    if out.Len() != 1 {
        t.Fatal(fmt::Sprintf!("want 1 entry, got %d", out.Len()));
    }
    let p = out[0usize].clone();
    if p.Flag != "" || p.Value != "" {
        t.Fatal(string("an unbound regime proposes no flag"));
    }
    if !strings::Contains(p.Why.clone(), "the offered load is the limit") {
        t.Fatal(fmt::Sprintf!("headroom wording drifted: %q", p.Why));
    }
}

#[goish::main]
fn main() {
    let tests: &[(&str, testing::TestFn)] = &[
        ("TestCapBoundDoubles", test_cap_bound_doubles),
        ("TestCapBoundClampsToFit", test_cap_bound_clamps_to_fit),
        ("TestCapBoundNoRoomShiftsToKV", test_cap_bound_no_room_shifts_to_kv),
        ("TestKVBoundFp8SetCapsAdmission", test_kv_bound_fp8_set_caps_admission),
        ("TestKVBoundProposesFp8First", test_kv_bound_proposes_fp8_first),
        ("TestUnboundReportsHeadroom", test_unbound_reports_headroom),
    ];
    let code = testing::Main(tests);
    syscall::Exit(int32(code));
}
