// propose: turn a measured pressure regime into concrete vLLM flag
// proposals with computed expectations. Pure functions over measured
// inputs; nothing here guesses. Every proposal names the numbers it
// was computed from, so the terminal line is also the evidence.
#![allow(non_snake_case)]

use goish::fmt;
use goish::math;
use goish::string;
use goish::goslice::slice;
use goish::{append, float64, make};

// Inputs are the measured facts a pressure probe and the run's own
// flag capture produce. Zero values mean unknown.
#[derive(Clone, Default)]
pub struct Inputs {
    pub BoundBy: string, // "max-num-seqs" | "kv" | "none"
    pub PeakRunning: float64,
    pub PeakKvPct: float64,   // 0..100
    pub Preemptions: float64, // count in the window
    pub MaxNumSeqs: float64,  // the effective flag value, 0 unknown
    // WorstCaseFit is vLLM's own "Maximum concurrency for N tokens
    // per request" startup line: how many full-context requests the
    // KV pool holds. 0 when the log was not captured.
    pub WorstCaseFit: float64,
    pub KvCacheDtype: string, // "fp8", "auto", ...
}

// Proposal is one flag change with its computed expectation.
#[derive(Clone, Default)]
pub struct Proposal {
    pub Flag: string,  // config key, underscores ("max_num_seqs")
    pub Value: string, // proposed value, "" for a report-only entry
    pub Why: string,   // the arithmetic, numbers included
    pub Risk: string,  // what to watch, "" when nothing specific
}

fn round1(x: float64) -> float64 {
    math::Round(x * 10.0) / 10.0
}

fn round2(x: float64) -> float64 {
    math::Round(x * 100.0) / 100.0
}

// kvPerRequestPct is the measured KV share one running request used.
fn kvPerRequestPct(in_: &Inputs) -> float64 {
    if in_.PeakRunning <= 0.0 {
        return 0.0;
    }
    in_.PeakKvPct / in_.PeakRunning
}

// fitUnder returns how many requests of the measured shape fit under
// the given KV budget (0 when unmeasurable).
fn fitUnder(in_: &Inputs, budgetPct: float64) -> float64 {
    let per = kvPerRequestPct(in_);
    if per <= 0.0 {
        return 0.0;
    }
    math::Floor(budgetPct / per)
}

// Propose maps a measured regime to flag proposals, strongest first.
pub fn Propose(in_: &Inputs) -> slice<Proposal> {
    let mut out: slice<Proposal> = make!([]Proposal, 0);

    if in_.BoundBy == "max-num-seqs" {
        // the scheduler cap was pinned; how far it can rise is set by
        // the measured KV appetite of the actual traffic
        let fit = fitUnder(in_, 90.0);
        let mut target = in_.MaxNumSeqs * 2.0;
        if fit > 0.0 && fit < target {
            target = fit;
        }
        if target > in_.MaxNumSeqs {
            let mut why = fmt::Sprintf!(
                "the cap %v was pinned while KV peaked at %v%%",
                in_.MaxNumSeqs,
                round1(in_.PeakKvPct)
            );
            if fit > 0.0 {
                why = fmt::Sprintf!(
                    "%s; at the measured %v%% KV per request, %v requests fit inside 90%%",
                    why,
                    round2(kvPerRequestPct(in_)),
                    fit
                );
            }
            let mut risk = string("per-stream speed drops as the batch grows; re-measure against your latency floor");
            if in_.WorstCaseFit > 0.0 && in_.WorstCaseFit < target {
                risk = fmt::Sprintf!(
                    "only %v full-context requests fit the pool (the startup line); long-prompt traffic hits KV before this cap",
                    round1(in_.WorstCaseFit)
                );
            }
            out = append!(
                out.clone(),
                Proposal {
                    Flag: string("max_num_seqs"),
                    Value: fmt::Sprintf!("%v", target),
                    Why: why,
                    Risk: risk,
                }
            );
        } else {
            // the KV pool cannot hold more of these requests: the cap
            // is right, the pool is the real ceiling
            out = kvRelief(in_, out);
        }
        return out;
    }

    if in_.BoundBy == "kv" {
        return kvRelief(in_, out);
    }

    // bound by neither: nothing server-side limited this pressure
    out = append!(
        out.clone(),
        Proposal {
            Flag: string(""),
            Value: string(""),
            Why: fmt::Sprintf!(
                "peak running %v stayed under the cap %v with KV at %v%%: the offered load is the limit; raise the pressure to find the next wall",
                in_.PeakRunning,
                in_.MaxNumSeqs,
                round1(in_.PeakKvPct)
            ),
            Risk: string(""),
        }
    );
    out
}

// kvRelief proposes the flags that shrink or cap KV appetite, in
// payoff order.
fn kvRelief(in_: &Inputs, mut out: slice<Proposal>) -> slice<Proposal> {
    if in_.KvCacheDtype != "fp8" && in_.KvCacheDtype != "fp8_e4m3" && in_.KvCacheDtype != "fp8_e5m2" {
        let mut expect = string("KV bytes per token halve, so the pool holds twice the tokens");
        if in_.PeakRunning > 0.0 {
            expect = fmt::Sprintf!(
                "%s; the %v requests that filled it become ~%v",
                expect,
                in_.PeakRunning,
                in_.PeakRunning * 2.0
            );
        }
        out = append!(
            out.clone(),
            Proposal {
                Flag: string("kv_cache_dtype"),
                Value: string("fp8"),
                Why: expect,
                Risk: string("fp8 KV is lossy; measure your eval quality, not just tok/s"),
            }
        );
    }
    if in_.Preemptions > 0.0 {
        let fit = fitUnder(in_, 90.0);
        if fit > 0.0 {
            out = append!(
                out.clone(),
                Proposal {
                    Flag: string("max_num_seqs"),
                    Value: fmt::Sprintf!("%v", fit),
                    Why: fmt::Sprintf!(
                        "%v preemptions in the window mean the pool was oversubscribed; capping admission at %v keeps KV under 90%% at the measured %v%% per request",
                        in_.Preemptions,
                        fit,
                        round2(kvPerRequestPct(in_))
                    ),
                    Risk: string("the waiting queue grows instead; that is deliberate backpressure, not loss"),
                }
            );
        }
    }
    out
}
