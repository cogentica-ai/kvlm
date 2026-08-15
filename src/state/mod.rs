// state package: ~/.kvlm/state.json, the live-target cache.
//
// kvlm up records the pod it creates the moment the platform returns
// an id, before any wait loop, so an interrupted up can never orphan
// a billing pod with no record. kvlm down removes the entry, and
// kvlm ps reconciles the cache against the platform: the driver API
// is the truth, this file is only a cache of it.
//
//   {
//     "current": "runpod/abc123",
//     "targets": {
//       "runpod/abc123": { "driver": "runpod", "pod": "abc123", ... }
//     }
//   }
#![allow(non_snake_case)]

use goish::encoding::json;
use goish::os;
use goish::string;
use goish::gomap::map;
use goish::goslice::slice;
use goish::{append, float64, int, make, nil, range};

// Target is one live pod kvlm started: where it is, what it serves,
// and how to reach it. Structured facts only; display strings are
// computed by the commands that print them.
#[derive(Clone, Default)]
pub struct Target {
    pub Driver: string,
    pub Pod: string,
    pub SSH: string,      // user@host:port, "" until the port is published
    pub Endpoint: string, // https base URL, "" when the pod has none
    pub Model: string,
    pub Variant: string,
    pub Mode: string, // "production" or "profile"
    pub GPUType: string,
    pub GPUCount: int,
    pub VllmVersion: string,
    pub ServerLog: string,
    pub CostPerHr: float64,
    pub Created: string, // RFC3339
    // Applied accumulates the flag overrides kvlm apply has restarted
    // this server with, on top of the catalog recipe (dash-form keys,
    // e.g. "max-num-seqs" -> "64").
    pub Applied: map<string, string>,
}

// Key is the state-file identity of a target: driver/pod.
pub fn Key(t: &Target) -> string {
    (t.Driver.clone()) + ("/") + (t.Pod.clone())
}

fn path() -> string {
    let (home, err) = os::UserHomeDir();
    if err != nil {
        return string("");
    }
    (home) + ("/.kvlm/state.json")
}

fn str(v: &json::Value, key: &str) -> string {
    if let Some(obj) = v.AsObject() {
        let (val, _) = obj.Get(key);
        if let Some(s) = val.AsString() {
            return s.clone();
        }
    }
    string("")
}

fn num(v: &json::Value, key: &str) -> float64 {
    if let Some(obj) = v.AsObject() {
        let (val, _) = obj.Get(key);
        if let Some(n) = val.AsNumber() {
            return n;
        }
    }
    0.0
}

fn parseTarget(v: &json::Value) -> Target {
    let mut applied: map<string, string> = map::new_no_zero();
    if let Some(obj) = v.AsObject() {
        let (av, ok) = obj.Get("applied");
        if ok {
            if let Some(am) = av.AsObject() {
                for (_, k) in range!(am.Keys()) {
                    let (val, _) = am.Get(k.clone());
                    if let Some(s) = val.AsString() {
                        applied.Set(k.clone(), s.clone());
                    }
                }
            }
        }
    }
    Target {
        Applied: applied,
        Driver: str(v, "driver"),
        Pod: str(v, "pod"),
        SSH: str(v, "ssh"),
        Endpoint: str(v, "endpoint"),
        Model: str(v, "model"),
        Variant: str(v, "variant"),
        Mode: str(v, "mode"),
        GPUType: str(v, "gpu_type"),
        GPUCount: int(num(v, "gpu_count")),
        VllmVersion: str(v, "vllm_version"),
        ServerLog: str(v, "server_log"),
        CostPerHr: num(v, "cost_per_hr"),
        Created: str(v, "created"),
    }
}

fn targetValue(t: &Target) -> json::Value {
    let mut obj: map<string, json::Value> = map::new_no_zero();
    obj.Set(string("driver"), json::Value::String(t.Driver.clone()));
    obj.Set(string("pod"), json::Value::String(t.Pod.clone()));
    obj.Set(string("ssh"), json::Value::String(t.SSH.clone()));
    obj.Set(string("endpoint"), json::Value::String(t.Endpoint.clone()));
    obj.Set(string("model"), json::Value::String(t.Model.clone()));
    obj.Set(string("variant"), json::Value::String(t.Variant.clone()));
    obj.Set(string("mode"), json::Value::String(t.Mode.clone()));
    obj.Set(string("gpu_type"), json::Value::String(t.GPUType.clone()));
    obj.Set(string("gpu_count"), json::Value::Number(float64(t.GPUCount)));
    obj.Set(string("vllm_version"), json::Value::String(t.VllmVersion.clone()));
    obj.Set(string("server_log"), json::Value::String(t.ServerLog.clone()));
    obj.Set(string("cost_per_hr"), json::Value::Number(t.CostPerHr));
    obj.Set(string("created"), json::Value::String(t.Created.clone()));
    if t.Applied.Len() > 0 {
        let mut am: map<string, json::Value> = map::new_no_zero();
        for (_, k) in range!(t.Applied.Keys()) {
            let (v, _) = t.Applied.Get(k.clone());
            am.Set(k.clone(), json::Value::String(v));
        }
        obj.Set(string("applied"), json::Value::Object(am));
    }
    json::Value::Object(obj)
}

// Load returns every recorded target and the current key ("" when
// none). A missing or unreadable file is an empty state, not an error.
pub fn Load() -> (slice<Target>, string) {
    let mut targets: slice<Target> = make!([]Target, 0);
    let p = path();
    if p == "" {
        return (targets, string(""));
    }
    let (data, err) = os::ReadFile(p);
    if err != nil {
        return (targets, string(""));
    }
    let mut v = json::Value::Null;
    let perr = json::Unmarshal(data.as_ref(), &mut v);
    if perr != nil {
        return (targets, string(""));
    }
    let current = str(&v, "current");
    if let Some(obj) = v.AsObject() {
        let (tv, ok) = obj.Get("targets");
        if ok {
            if let Some(tm) = tv.AsObject() {
                for (_, k) in range!(tm.Keys()) {
                    let (entry, _) = tm.Get(k.clone());
                    targets = append!(targets.clone(), parseTarget(&entry));
                }
            }
        }
    }
    (targets, current)
}

fn save(targets: &slice<Target>, current: string) {
    let p = path();
    if p == "" {
        return;
    }
    let mut tm: map<string, json::Value> = map::new_no_zero();
    for (_, t) in range!(targets.clone()) {
        tm.Set(Key(&t), targetValue(&t));
    }
    let mut root: map<string, json::Value> = map::new_no_zero();
    root.Set(string("current"), json::Value::String(current));
    root.Set(string("targets"), json::Value::Object(tm));
    let v = json::Value::Object(root);
    let (data, err) = json::MarshalIndent(&v, "", "  ");
    if err != nil {
        return;
    }
    let _ = os::MkdirAll(strings_dir(&p), 0o700);
    let _ = os::WriteFile(p, data, 0o600);
}

fn strings_dir(p: &string) -> string {
    // path.Dir without importing path/filepath: cut at the last slash
    let mut last: int = -1;
    let bytes = p.as_bytes();
    let mut i: int = 0;
    while (i as usize) < bytes.len() {
        if bytes[i as usize] == b'/' {
            last = i;
        }
        i += 1;
    }
    if last <= 0 {
        return string("/");
    }
    p.slice(0, last)
}

// Record upserts a target and makes it current.
pub fn Record(t: &Target) {
    let (targets, _) = Load();
    let key = Key(t);
    let mut next: slice<Target> = make!([]Target, 0);
    for (_, old) in range!(targets) {
        if Key(&old) != key {
            next = append!(next.clone(), old.clone());
        }
    }
    next = append!(next.clone(), t.clone());
    save(&next, key);
}

// Update rewrites an existing target in place (matched by key) without
// touching which target is current. Records it when absent.
pub fn Update(t: &Target) {
    let (targets, current) = Load();
    let key = Key(t);
    let mut next: slice<Target> = make!([]Target, 0);
    let mut found = false;
    for (_, old) in range!(targets) {
        if Key(&old) == key {
            next = append!(next.clone(), t.clone());
            found = true;
        } else {
            next = append!(next.clone(), old.clone());
        }
    }
    if !found {
        next = append!(next.clone(), t.clone());
    }
    let mut cur = current;
    if cur == "" {
        cur = key;
    }
    save(&next, cur);
}

// Remove deletes a target; when it was current, the newest remaining
// target (by created) becomes current.
pub fn Remove(driverName: string, pod: string) {
    let (targets, current) = Load();
    let key = (driverName) + ("/") + (pod);
    let mut next: slice<Target> = make!([]Target, 0);
    for (_, old) in range!(targets) {
        if Key(&old) != key {
            next = append!(next.clone(), old.clone());
        }
    }
    let mut cur = current;
    if cur == key {
        cur = string("");
        let mut newest = string("");
        for (_, t) in range!(next.clone()) {
            if t.Created >= newest {
                newest = t.Created.clone();
                cur = Key(&t);
            }
        }
    }
    save(&next, cur);
}

// Current returns the current target.
pub fn Current() -> (Target, bool) {
    let (targets, current) = Load();
    if current == "" {
        return (Default::default(), false);
    }
    for (_, t) in range!(targets) {
        if Key(&t) == current {
            return (t.clone(), true);
        }
    }
    (Default::default(), false)
}

// Find resolves a user-typed reference: the full driver/pod key, a
// bare pod id, or a model name (newest match wins for models, since
// one model can have had several pods).
pub fn Find(reference: string) -> (Target, bool) {
    let (targets, _) = Load();
    let mut best: Target = Default::default();
    let mut found = false;
    for (_, t) in range!(targets) {
        if Key(&t) == reference || t.Pod == reference {
            return (t.clone(), true);
        }
        if t.Model == reference && t.Created >= best.Created {
            best = t.clone();
            found = true;
        }
    }
    (best, found)
}

// SetCurrent switches the current target to the given key; false when
// no target has that key.
pub fn SetCurrent(reference: string) -> bool {
    let (t, ok) = Find(reference);
    if !ok {
        return false;
    }
    let (targets, _) = Load();
    save(&targets, Key(&t));
    true
}
