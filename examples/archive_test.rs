// profile::archive regression tests: the .kvlm revision archive.
// Pins the role classifier, the manifest schema fields, the pure
// in-memory Pack/Unpack round trip, and the flat-entry safety check
// that keeps imports from writing outside the run directory.
#![no_std]
#![no_main]
#![allow(non_snake_case)]

extern crate alloc;

use goish::fmt;
use goish::strings;
use goish::syscall;
use goish::testing;
use goish::string;
use goish::{append, bytes, int32, make, slice};

use kvlm::profile::archive;

fn entry(name: &'static str, data: &'static str) -> archive::Entry {
    archive::Entry {
        Name: string(name),
        Data: bytes(string(data)),
        ..Default::default()
    }
}

fn test_role_of(t: &mut testing::T) {
    let cases: &[(&'static str, &'static str)] = &[
        ("graph-structure.json", "graph"),
        ("sweep.json", "sweep"),
        ("vllm-args.txt", "config"),
        ("vllm-flags-0.26.0.json", "flags-catalog"),
        ("win.nsys-rep", "capture"),
        ("win.sqlite", "capture"),
        ("rank0.1786.pt.trace.json.gz", "trace"),
        ("metrics-before.txt", "metrics"),
        ("metrics-after.txt", "metrics"),
        ("env.txt", "env"),
        ("server-log.txt", "server-log"),
        ("gpu-folded.txt", "folded"),
        ("gpu-time.svg", "svg"),
        ("notes.md", "other"),
    ];
    for (name, want) in cases.iter() {
        let got = archive::RoleOf(*name);
        if got != *want {
            t.Fatal(fmt::Sprintf!("%s: got %q want %q", string(*name), got, string(*want)));
        }
    }
    if !archive::IsHeavy("capture") || !archive::IsHeavy("trace") || archive::IsHeavy("graph") {
        t.Fatal(string("heavy roles must be exactly capture and trace"));
    }
    // the bundled catalog name round-trips so import can route it
    // back to the shared vllm-flags/ store
    let n = archive::CatalogEntryName(string("0.26.0"));
    if n != "vllm-flags-0.26.0.json" {
        t.Fatal(fmt::Sprintf!("catalog entry name: %q", n));
    }
    let (v, ok) = archive::CatalogVersion(n);
    if !ok || v != "0.26.0" {
        t.Fatal(fmt::Sprintf!("catalog version: %q %v", v, ok));
    }
    let (_, ok) = archive::CatalogVersion("graph-structure.json");
    if ok {
        t.Fatal(string("non-catalog name accepted"));
    }
}

fn test_manifest_fields(t: &mut testing::T) {
    let mut env: goish::slice<string> = make!([]string, 0);
    env = append!(env.clone(), string("vllm 0.26.0"));
    let mut entries: goish::slice<archive::Entry> = make!([]archive::Entry, 0);
    entries = append!(entries.clone(), entry("graph-structure.json", "{}"));
    let m = archive::ManifestJSON(string("run4"), string("measured 2026-08-08"), env, entries);
    for want in [
        "\"kvlm\": 1",
        "\"run\": \"run4\"",
        "\"provenance\": \"measured 2026-08-08\"",
        "\"vllm 0.26.0\"",
        "\"role\": \"graph\"",
    ]
    .iter()
    {
        if !strings::Contains(m.clone(), string(*want)) {
            t.Fatal(fmt::Sprintf!("manifest missing %q in %s", string(*want), m.clone()));
        }
    }
    if archive::ManifestRun(m.clone()) != "run4" {
        t.Fatal(fmt::Sprintf!("ManifestRun: got %q", archive::ManifestRun(m)));
    }
}

fn test_pack_unpack_roundtrip(t: &mut testing::T) {
    let mut env: goish::slice<string> = make!([]string, 0);
    env = append!(env.clone(), string("vllm 0.26.0"));
    env = append!(env.clone(), string("torch 2.11.0+cu130"));
    let mut entries: goish::slice<archive::Entry> = make!([]archive::Entry, 0);
    entries = append!(entries.clone(), entry("graph-structure.json", "{\"provenance\": \"p\"}"));
    entries = append!(entries.clone(), entry("env.txt", "vllm 0.26.0\n"));
    entries = append!(entries.clone(), entry("metrics-before.txt", "vllm:num_requests_running 3\n"));

    let (data, err) = archive::Pack(string("run9"), string("prov"), env, entries.clone());
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("Pack: %v", err));
    }
    if data.Len() == 0 {
        t.Fatal(string("Pack produced no bytes"));
    }
    // gzip magic
    if data[0usize] != 0x1f || data[1usize] != 0x8b {
        t.Fatal(string("archive is not gzip"));
    }

    let (run, manifest, got, err) = archive::Unpack(data);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("Unpack: %v", err));
    }
    if run != "run9" {
        t.Fatal(fmt::Sprintf!("run: got %q", run));
    }
    if !strings::Contains(manifest.clone(), string("\"kvlm\": 1")) {
        t.Fatal(string("manifest lost in round trip"));
    }
    if got.Len() != entries.Len() {
        t.Fatal(fmt::Sprintf!("entries: got %d want %d", got.Len(), entries.Len()));
    }
    let mut i: goish::int = 0;
    while i < got.Len() as goish::int {
        let a = got[i as usize].clone();
        let b = entries[i as usize].clone();
        if a.Name != b.Name || string(a.Data.clone()) != string(b.Data.clone()) {
            t.Fatal(fmt::Sprintf!("entry %d differs: %q", i, a.Name.clone()));
        }
        i += 1;
    }
}

fn test_unpack_rejects_paths(t: &mut testing::T) {
    // hand-build an archive whose entry name escapes the run dir
    let mut entries: goish::slice<archive::Entry> = make!([]archive::Entry, 0);
    entries = append!(entries.clone(), entry("../evil.txt", "x"));
    let (data, err) = archive::Pack(string("r"), string(""), make!([]string, 0), entries);
    if err != goish::nil {
        t.Fatal(fmt::Sprintf!("Pack: %v", err));
    }
    let (_, _, _, err) = archive::Unpack(data);
    if err == goish::nil {
        t.Fatal(string("Unpack must reject entry names with path separators"));
    }
}

fn test_json_scanners(t: &mut testing::T) {
    let doc = string("{\n \"provenance\": \"measured, with \\\"quotes\\\"\",\n \"env\": [\"a b\", \"c d\"],\n}");
    let p = archive::JSONStringField(doc.clone(), string("provenance"));
    if !strings::HasPrefix(p.clone(), string("measured, with")) {
        t.Fatal(fmt::Sprintf!("provenance scan: got %q", p));
    }
    let env = archive::JSONStringArray(doc, string("env"));
    if env.Len() != 2 || env[0usize] != "a b" || env[1usize] != "c d" {
        t.Fatal(fmt::Sprintf!("env scan: got %d entries", env.Len()));
    }
    // compact JSON (API responses have no space after the colon)
    let compact = string("{\"id\":\"abc123\",\"costPerHr\":6.58,\"publicPort\":19531}");
    if archive::JSONStringField(compact.clone(), string("id")) != "abc123" {
        t.Fatal(string("compact string field scan failed"));
    }
    let (v, ok) = archive::JSONNumberField(compact.clone(), string("costPerHr"));
    if !ok || v != 6.58 {
        t.Fatal(fmt::Sprintf!("compact number field: got %v %v", v, ok));
    }
    let (p, ok) = archive::JSONNumberField(compact, string("publicPort"));
    if !ok || p != 19531.0 {
        t.Fatal(fmt::Sprintf!("port number field: got %v", p));
    }
}

#[goish::main]
fn main() {
    let tests: &[(&str, testing::TestFn)] = &[
        ("TestRoleOf", test_role_of),
        ("TestManifestFields", test_manifest_fields),
        ("TestPackUnpackRoundtrip", test_pack_unpack_roundtrip),
        ("TestUnpackRejectsPaths", test_unpack_rejects_paths),
        ("TestJSONScanners", test_json_scanners),
    ];
    let code = testing::Main(tests);
    syscall::Exit(int32(code));
}
