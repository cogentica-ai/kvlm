// vllmcfg: parsing vLLM's own statements of its configuration.
//
// vLLM logs two lines at startup that together carry the whole flag
// state: "non-default args: {'max_model_len': 32768, ...}" (what was
// set explicitly) and "Initializing a V1 LLM engine (v0.26.0) with
// config: model=..., max_num_seqs=32, ..." (the resolved config).
// The values are python reprs, so the tokenizer here splits on
// top-level commas while tracking bracket depth and quotes — nested
// ProfilerConfig(...), dicts, and lists stay intact.
#![allow(non_snake_case)]

use goish::strings;
use goish::string;
use goish::{append, int, make, range, slice};

// KV is one parsed key/value; the value keeps its python repr form.
#[derive(Clone, Default)]
pub struct KV {
    pub Key: string,
    pub Value: string,
}

// SplitTopLevel splits s on commas at bracket depth zero, respecting
// (), [], {} and single/double quotes.
pub fn SplitTopLevel(s: string) -> slice<string> {
    let mut out: slice<string> = make!([]string, 0);
    let mut depth: int = 0;
    let mut quote: u8 = 0;
    let mut start: int = 0;
    let mut i: int = 0;
    while i < s.Len() {
        let c = s[i as usize];
        if quote != 0 {
            if c == quote {
                quote = 0;
            }
        } else if c == b'\'' || c == b'"' {
            quote = c;
        } else if c == b'(' || c == b'[' || c == b'{' {
            depth += 1;
        } else if c == b')' || c == b']' || c == b'}' {
            depth -= 1;
        } else if c == b',' && depth == 0 {
            out = append!(out.clone(), strings::TrimSpace(s.slice(start, i)));
            start = i + 1;
        }
        i += 1;
    }
    let last = strings::TrimSpace(s.slice(start, s.Len()));
    if last != "" {
        out = append!(out.clone(), last);
    }
    out
}

fn unquote(v: string) -> string {
    let t = strings::TrimSpace(v);
    if t.Len() >= 2 {
        let f = t[0usize];
        let l = t[(t.Len() - 1) as usize];
        if (f == b'\'' && l == b'\'') || (f == b'"' && l == b'"') {
            return t.slice(1, t.Len() - 1);
        }
    }
    t
}

// ParseNonDefault extracts the explicit-args dict from the
// "non-default args: {...}" log line.
pub fn ParseNonDefault(line: string) -> slice<KV> {
    let mut out: slice<KV> = make!([]KV, 0);
    let idx = strings::Index(line.clone(), string("non-default args: {"));
    if idx < 0 {
        return out;
    }
    let rest = line.slice(idx + (string("non-default args: {")).Len(), line.Len());
    // the dict runs to the final closing brace on the line
    let end = strings::LastIndex(rest.clone(), string("}"));
    if end < 0 {
        return out;
    }
    for (_, part) in range!(SplitTopLevel(rest.slice(0, end))) {
        let ci = strings::Index(part.clone(), string(":"));
        if ci <= 0 {
            continue;
        }
        out = append!(
            out.clone(),
            KV {
                Key: unquote(part.slice(0, ci)),
                Value: strings::TrimSpace(part.slice(ci + 1, part.Len())),
            }
        );
    }
    out
}

// ParseResolved extracts key=value pairs from the "Initializing a V1
// LLM engine (vX.Y.Z) with config: ..." line.
pub fn ParseResolved(line: string) -> slice<KV> {
    let mut out: slice<KV> = make!([]KV, 0);
    let marker = string("with config: ");
    let idx = strings::Index(line.clone(), marker.clone());
    if idx < 0 {
        return out;
    }
    for (_, part) in range!(SplitTopLevel(line.slice(idx + marker.Len(), line.Len()))) {
        let ei = strings::Index(part.clone(), string("="));
        if ei <= 0 {
            continue;
        }
        let key = strings::TrimSpace(part.slice(0, ei));
        // keys are bare identifiers; anything else is a nested value
        // fragment that leaked through, skip it
        if strings::ContainsAny(key.clone(), " ({['\"") {
            continue;
        }
        out = append!(
            out.clone(),
            KV {
                Key: key,
                Value: strings::TrimSpace(part.slice(ei + 1, part.Len())),
            }
        );
    }
    out
}

// EngineVersion pulls "0.26.0" out of "Initializing a V1 LLM engine
// (v0.26.0) with config: ...".
pub fn EngineVersion(line: string) -> string {
    let idx = strings::Index(line.clone(), string("LLM engine (v"));
    if idx < 0 {
        return string("");
    }
    let rest = line.slice(idx + (string("LLM engine (v")).Len(), line.Len());
    let end = strings::Index(rest.clone(), string(")"));
    if end <= 0 {
        return string("");
    }
    rest.slice(0, end)
}

// Lookup returns the value for key in a parsed set.
pub fn Lookup(kvs: &slice<KV>, key: &'static str) -> (string, bool) {
    for (_, kv) in range!(kvs.clone()) {
        if kv.Key == key {
            return (kv.Value.clone(), true);
        }
    }
    (string(""), false)
}

// FlagFor maps a config key to the CLI flag that sets it. The general
// rule is underscores to dashes with a -- prefix; the exceptions are
// keys that are not flags.
pub fn FlagFor<S: Into<string>>(key: S) -> string {
    let k = key.into();
    if k == "model" || k == "model_tag" {
        return string("model (positional)");
    }
    ("--") + (strings::ReplaceAll(k, "_", "-"))
}

// RelevantKeys is the curated correlation set: every config key the
// lever registry references plus the scheduler, cache, and graph keys
// that decide the regimes kvlm diagnoses.
pub fn RelevantKeys() -> slice<string> {
    let mut out: slice<string> = make!([]string, 0);
    for k in [
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_seq_len",
        "max_model_len",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "enable_prefix_caching",
        "tensor_parallel_size",
        "quantization",
        "block_size",
        "enforce_eager",
        "speculative_config",
        "enable_chunked_prefill",
    ]
    .iter()
    {
        out = append!(out.clone(), string(*k));
    }
    out
}
