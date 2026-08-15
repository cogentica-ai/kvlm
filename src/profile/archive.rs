// archive: the .kvlm revision archive format.
//
// A .kvlm file is one collected run, portable: gzip over tar, flat
// entries (no directories), with kvlm-manifest.json as the first
// entry. Import extracts it into profile-output/<run>/ and the
// dashboard renders it as a revision like any locally collected run.
//
// Manifest, schema version 1:
//   {
//    "kvlm": 1,
//    "run": "run1",
//    "provenance": "measured ...",
//    "env": ["vllm 0.26.0", ...],
//    "files": [{"path": "graph-structure.json", "role": "graph", "bytes": 123}, ...]
//   }
//
// Roles classify what each file is, by name: graph, metrics, env,
// server-log, folded, svg, capture (.nsys-rep/.sqlite), trace
// (torch profiler), other. Archives carry everything by default; the
// heavy roles (capture, trace) are what kvlm profile archive --basic
// leaves out when the file must travel light.
//
// Pack and Unpack are pure in-memory transforms so the round trip is
// pinned by tests; the archive/import commands only add file IO.
#![allow(non_snake_case)]

use goish::archive::tar;
use goish::bytes;
use goish::compress::gzip;
use goish::errors::error;
use goish::fmt;
use goish::io;
use goish::strconv;
use goish::strings;
use goish::string;
use goish::types;
use goish::{append, float64, int, make, nil, range, slice};

pub const ManifestName: &str = "kvlm-manifest.json";

// Entry is one file inside an archive.
#[derive(Clone, Default)]
pub struct Entry {
    pub Name: string,
    pub Data: slice<types::byte>,
}

// RoleOf classifies a run file by its name.
pub fn RoleOf<S: Into<string>>(name: S) -> string {
    let n = name.into();
    if n == "graph-structure.json" {
        return string("graph");
    }
    if n == "sweep.json" {
        return string("sweep");
    }
    if n == "vllm-args.txt" {
        return string("config");
    }
    if strings::HasPrefix(n.clone(), "vllm-flags-") && strings::HasSuffix(n.clone(), ".json") {
        return string("flags-catalog");
    }
    if strings::HasSuffix(n.clone(), ".nsys-rep") || strings::HasSuffix(n.clone(), ".sqlite") {
        return string("capture");
    }
    if strings::Contains(n.clone(), ".pt.trace.json") {
        return string("trace");
    }
    if strings::HasPrefix(n.clone(), "metrics-") && strings::HasSuffix(n.clone(), ".txt") {
        return string("metrics");
    }
    if n == "env.txt" {
        return string("env");
    }
    if n == "server-log.txt" {
        return string("server-log");
    }
    if strings::HasSuffix(n.clone(), "-folded.txt") {
        return string("folded");
    }
    if strings::HasSuffix(n.clone(), ".svg") {
        return string("svg");
    }
    string("other")
}

// CatalogEntryName is the flat archive name carrying the per-version
// flag catalog, so an archive reconstructs the full flag state on any
// machine (the run itself stores only its explicit args; the catalog
// supplies every default).
pub fn CatalogEntryName(version: string) -> string {
    ("vllm-flags-") + (version) + (".json")
}

// CatalogVersion inverts CatalogEntryName.
pub fn CatalogVersion<S: Into<string>>(name: S) -> (string, bool) {
    let n = name.into();
    if !strings::HasPrefix(n.clone(), "vllm-flags-") || !strings::HasSuffix(n.clone(), ".json") {
        return (string(""), false);
    }
    let v = strings::TrimSuffix(strings::TrimPrefix(n, string("vllm-flags-")), string(".json"));
    (v.clone(), v != "")
}

// IsHeavy reports whether a role is a raw capture excluded from
// archives by default.
pub fn IsHeavy<S: Into<string>>(role: S) -> bool {
    let r = role.into();
    r == "capture" || r == "trace"
}

fn esc(s: string) -> string {
    let mut e = strings::ReplaceAll(s, "\\", "\\\\");
    e = strings::ReplaceAll(e, "\"", "\\\"");
    e = strings::ReplaceAll(e, "\n", " ");
    e
}

// ManifestJSON renders the manifest for a run's entries.
pub fn ManifestJSON(run: string, provenance: string, env: slice<string>, entries: slice<Entry>) -> string {
    let mut b = strings::Builder::new();
    let _ = b.WriteString("{\n \"kvlm\": 1,\n");
    let _ = b.WriteString(fmt::Sprintf!(" \"run\": \"%s\",\n", esc(run)));
    let _ = b.WriteString(fmt::Sprintf!(" \"provenance\": \"%s\",\n", esc(provenance)));
    let _ = b.WriteString(" \"env\": [");
    for (i, e) in range!(env.clone()) {
        if i > 0 {
            let _ = b.WriteString(", ");
        }
        let _ = b.WriteString(fmt::Sprintf!("\"%s\"", esc(e.clone())));
    }
    let _ = b.WriteString("],\n \"files\": [");
    for (i, en) in range!(entries.clone()) {
        if i > 0 {
            let _ = b.WriteString(", ");
        }
        let _ = b.WriteString(fmt::Sprintf!(
            "{\"path\": \"%s\", \"role\": \"%s\", \"bytes\": %d}",
            esc(en.Name.clone()),
            RoleOf(en.Name.clone()),
            en.Data.Len()
        ));
    }
    let _ = b.WriteString("]\n}\n");
    string(b.String())
}

// jsonStringAfter scans doc for `"key":` (optional space, as in both
// kvlm's own pretty output and compact API responses) and returns the
// quoted value. Not a general JSON parser.
pub fn JSONStringField(doc: string, key: string) -> string {
    let marker = ("\"") + (key) + ("\":");
    let idx = strings::Index(doc.clone(), marker.clone());
    if idx < 0 {
        return string("");
    }
    let mut at = idx + marker.Len();
    while at < doc.Len() && (doc[at as usize] == b' ' || doc[at as usize] == b'\t') {
        at += 1;
    }
    if at >= doc.Len() || doc[at as usize] != b'"' {
        return string("");
    }
    let rest = doc.slice(at + 1, doc.Len());
    let mut endIdx: int = -1;
    let mut i: int = 0;
    while i < rest.Len() {
        if rest[i as usize] == b'"' && (i == 0 || rest[(i - 1) as usize] != b'\\') {
            endIdx = i;
            break;
        }
        i += 1;
    }
    if endIdx <= 0 {
        return string("");
    }
    rest.slice(0, endIdx)
}

// JSONNumberField scans doc for `"key":` and returns the number that
// follows. Same scope as JSONStringField.
pub fn JSONNumberField(doc: string, key: string) -> (float64, bool) {
    let marker = ("\"") + (key) + ("\":");
    let idx = strings::Index(doc.clone(), marker.clone());
    if idx < 0 {
        return (0.0, false);
    }
    let mut at = idx + marker.Len();
    while at < doc.Len() && (doc[at as usize] == b' ' || doc[at as usize] == b'\t') {
        at += 1;
    }
    let start = at;
    while at < doc.Len() {
        let c = doc[at as usize];
        if (c >= b'0' && c <= b'9') || c == b'.' || c == b'-' {
            at += 1;
        } else {
            break;
        }
    }
    if at == start {
        return (0.0, false);
    }
    let (v, err) = strconv::ParseFloat(doc.slice(start, at), 64);
    if err != nil {
        return (0.0, false);
    }
    (v, true)
}

// JSONStringArray scans doc for `"key": [` and returns the quoted
// strings inside the brackets. Same scope as JSONStringField.
pub fn JSONStringArray(doc: string, key: string) -> slice<string> {
    let mut out: slice<string> = make!([]string, 0);
    let marker = ("\"") + (key) + ("\": [");
    let idx = strings::Index(doc.clone(), marker.clone());
    if idx < 0 {
        return out;
    }
    let rest = doc.slice(idx + marker.Len(), doc.Len());
    let end = strings::Index(rest.clone(), string("]"));
    if end < 0 {
        return out;
    }
    let body = rest.slice(0, end);
    for (_, part) in range!(strings::Split(body, "\",")) {
        let mut p = strings::TrimSpace(part.clone());
        p = strings::TrimPrefix(p, string("\""));
        p = strings::TrimSuffix(p, string("\""));
        if p != "" {
            out = append!(out.clone(), p);
        }
    }
    out
}

// ManifestRun extracts the run name from a manifest.
pub fn ManifestRun(manifest: string) -> string {
    JSONStringField(manifest, string("run"))
}

// Pack builds a .kvlm archive: manifest first, then the entries.
pub fn Pack(run: string, provenance: string, env: slice<string>, entries: slice<Entry>) -> (slice<types::byte>, error) {
    let manifest = ManifestJSON(run, provenance, env, entries.clone());

    // tar into memory
    let tarBuf = bytes::NewBuffer(slice::new());
    let mut tw = tar::NewWriter(tarBuf);
    let mut all: slice<Entry> = make!([]Entry, 0);
    all = append!(
        all.clone(),
        Entry {
            Name: string(ManifestName),
            Data: bytes(manifest.clone()),
            ..Default::default()
        }
    );
    for (_, e) in range!(entries.clone()) {
        all = append!(all.clone(), e.clone());
    }
    for (_, e) in range!(all.clone()) {
        let mut h = tar::Header::new();
        h.Name = e.Name.clone();
        h.Size = e.Data.Len() as i64;
        h.Mode = 0o644;
        h.Typeflag = tar::TypeReg;
        let err = tw.WriteHeader(&h);
        if err != nil {
            return (slice::new(), err);
        }
        let (_, err) = tw.Write(e.Data.clone());
        if err != nil {
            return (slice::new(), err);
        }
    }
    let err = tw.Close();
    if err != nil {
        return (slice::new(), err);
    }
    let tarBytes = tw.into_writer().Bytes();

    // gzip the tar
    let gzBuf = bytes::NewBuffer(slice::new());
    let mut gz = gzip::NewWriter(gzBuf);
    let (_, err) = gz.Write(tarBytes);
    if err != nil {
        return (slice::new(), err);
    }
    let err = gz.Close();
    if err != nil {
        return (slice::new(), err);
    }
    (gz.into_writer().Bytes(), nil.into())
}

// Unpack reads a .kvlm archive back into its run name, manifest text,
// and entries (manifest excluded from the entry list). Entry names
// with path separators or dot-dot are rejected: archive entries are
// flat by construction and stay that way on import.
pub fn Unpack(data: slice<types::byte>) -> (string, string, slice<Entry>, error) {
    let none: slice<Entry> = slice::new();
    let (gzr, err) = gzip::NewReader(bytes::NewBuffer(data));
    if err != nil {
        return (string(""), string(""), none, fmt::Errorf!("not a .kvlm archive (gzip): %v", err));
    }
    let mut tr = tar::NewReader(alloc::boxed::Box::new(gzr));
    let mut run = string("");
    let mut manifest = string("");
    let mut out: slice<Entry> = make!([]Entry, 0);
    loop {
        let (h, err) = tr.Next();
        if err != nil {
            // io.EOF ends the archive
            break;
        }
        let name = h.Name.clone();
        if strings::Contains(name.clone(), "/") || strings::Contains(name.clone(), "..") {
            return (string(""), string(""), none, fmt::Errorf!("archive entry %q is not a flat file name", name));
        }
        let (data, err) = io::ReadAll(&mut tr);
        if err != nil {
            return (string(""), string(""), none, err);
        }
        if name == ManifestName {
            manifest = string(data.clone());
            run = ManifestRun(manifest.clone());
            continue;
        }
        out = append!(
            out.clone(),
            Entry {
                Name: name,
                Data: data,
                ..Default::default()
            }
        );
    }
    if manifest == "" {
        return (string(""), string(""), none, fmt::Errorf!("no %s in archive", string(ManifestName)));
    }
    (run, manifest, out, nil.into())
}
