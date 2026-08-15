// vllmflags: the per-version vLLM flag catalog.
//
// One JSON file per vLLM version under vllm-flags/, holding every
// engine argument and its default for that release:
//   {"vllm": "0.26.0", "source": "...", "flags": {"max_num_seqs": "32", ...}}
// Collected once per version by introspecting the installed package
// on a target (dataclasses.fields(AsyncEngineArgs)); every run of
// that version then only needs its explicit args — the catalog
// supplies the rest. Values are python reprs, stored verbatim.
#![allow(non_snake_case)]

use goish::errors::error;
use goish::fmt;
use goish::os;
use goish::strings;
use goish::string;
use goish::{append, int, make, nil, range, slice};

use crate::profile::vllmcfg::KV;

pub const Dir: &str = "vllm-flags";

// DumpCommand is the introspection one-liner run on a machine that
// has the target vLLM version installed. Output: {"name": "repr", ...}
pub const DumpCommand: &str = "python3 -c \"import json,dataclasses; from vllm.engine.arg_utils import AsyncEngineArgs; a=AsyncEngineArgs(); print(json.dumps({f.name: repr(getattr(a, f.name)) for f in dataclasses.fields(AsyncEngineArgs)}))\"";

pub fn Path(version: string) -> string {
    fmt::Sprintf!("%s/%s.json", string(Dir), version)
}

fn esc(s: string) -> string {
    let mut e = strings::ReplaceAll(s, "\\", "\\\\");
    e = strings::ReplaceAll(e, "\"", "\\\"");
    e
}

fn unesc(s: string) -> string {
    let mut e = strings::ReplaceAll(s, "\\\"", "\"");
    e = strings::ReplaceAll(e, "\\\\", "\\");
    e
}

// Render builds the catalog JSON from the raw introspection dump
// (a single-line JSON object of name -> repr).
pub fn Render(version: string, source: string, dump: string) -> (string, error) {
    let flags = ParseFlags(dump);
    if flags.Len() == 0 {
        return (string(""), fmt::Errorf!("no flags parsed from the dump"));
    }
    let mut b = strings::Builder::new();
    let _ = b.WriteString(fmt::Sprintf!("{\n \"vllm\": \"%s\",\n \"source\": \"%s\",\n \"flags\": {\n", esc(version), esc(source)));
    for (i, kv) in range!(flags.clone()) {
        if i > 0 {
            let _ = b.WriteString(",\n");
        }
        let _ = b.WriteString(fmt::Sprintf!("  \"%s\": \"%s\"", esc(kv.Key.clone()), esc(kv.Value.clone())));
    }
    let _ = b.WriteString("\n }\n}\n");
    (string(b.String()), nil.into())
}

// ParseFlags reads a one-level JSON object of string keys and string
// values (both the introspection dump and the catalog's flags block).
pub fn ParseFlags(doc: string) -> slice<KV> {
    let mut out: slice<KV> = make!([]KV, 0);
    let mut i: int = 0;
    let n = doc.Len();
    while i < n {
        // find the next quoted key
        while i < n && doc[i as usize] != b'"' {
            i += 1;
        }
        if i >= n {
            break;
        }
        let (key, next) = readString(&doc, i + 1);
        i = next;
        // expect a colon then a quoted value
        while i < n && (doc[i as usize] == b' ' || doc[i as usize] == b':') {
            i += 1;
        }
        if i >= n || doc[i as usize] != b'"' {
            continue;
        }
        let (val, next) = readString(&doc, i + 1);
        i = next;
        if key != "" && key != "vllm" && key != "source" && key != "flags" {
            out = append!(
                out.clone(),
                KV {
                    Key: unesc(key),
                    Value: unesc(val),
                }
            );
        }
    }
    out
}

// readString scans a JSON string body starting after the opening
// quote; returns the raw body and the index after the closing quote.
fn readString(doc: &string, from: int) -> (string, int) {
    let mut i = from;
    let n = doc.Len();
    while i < n {
        let c = doc[i as usize];
        if c == b'\\' {
            i += 2;
            continue;
        }
        if c == b'"' {
            return (doc.slice(from, i), i + 1);
        }
        i += 1;
    }
    (doc.slice(from, n), n)
}

// Save writes the catalog file for a version.
pub fn Save(version: string, source: string, dump: string) -> error {
    let (doc, err) = Render(version.clone(), source, dump);
    if err != nil {
        return err;
    }
    let err = os::MkdirAll(string(Dir), 0o755);
    if err != nil {
        return err;
    }
    os::WriteFile(Path(version), doc, 0o644)
}

// Load reads a version's catalog; ok=false when the version has not
// been collected yet.
pub fn Load(version: string) -> (slice<KV>, bool) {
    let (data, err) = os::ReadFile(Path(version));
    if err != nil {
        return (make!([]KV, 0), false);
    }
    let flags = ParseFlags(string(data));
    (flags.clone(), flags.Len() > 0)
}

// Versions lists the collected catalog versions.
pub fn Versions() -> slice<string> {
    let mut out: slice<string> = make!([]string, 0);
    let (entries, err) = os::ReadDir(string(Dir));
    if err != nil {
        return out;
    }
    for e in entries.iter() {
        let n = e.Name();
        if strings::HasSuffix(n.clone(), ".json") {
            out = append!(out.clone(), strings::TrimSuffix(n, string(".json")));
        }
    }
    out
}
