// config package: read access to ~/.kvlm/config.json, shared by the
// driver and runtime packages.
//
//   {
//     "driver":  "runpod",
//     "runtime": "vllm",
//     "drivers": { "<name>": { "<key>": "<value>", ... } }
//   }
#![allow(non_snake_case)]

use goish::encoding::json;
use goish::os;
use goish::string;
use goish::{nil};

// load parses ~/.kvlm/config.json; returns Value::Null when the file is
// absent or unreadable (all lookups then fall through).
fn load() -> json::Value {
    let (home, err) = os::UserHomeDir();
    if err != nil {
        return json::Value::Null;
    }
    let path = (home) + ("/.kvlm/config.json");
    let (data, err) = os::ReadFile(path);
    if err != nil {
        return json::Value::Null;
    }
    let mut v = json::Value::Null;
    let perr = json::Unmarshal(data.as_ref(), &mut v);
    if perr != nil {
        return json::Value::Null;
    }
    v
}

// objGet returns v[key] for a JSON object, Value::Null otherwise.
fn objGet(v: &json::Value, key: &str) -> json::Value {
    if let Some(obj) = v.AsObject() {
        if obj.Has(key) {
            let (val, _) = obj.Get(key);
            return val;
        }
    }
    json::Value::Null
}

// TopLevel reads a top-level string key ("" if absent).
pub fn TopLevel<S: Into<string>>(key: S) -> string {
    let key = key.into();
    match objGet(&load(), key.as_ref()).AsString() {
        Some(s) => s.clone(),
        None => string(""),
    }
}

// Value reads config[section][name][key] as a string ("" if absent) —
// e.g. Value("drivers", "runpod", "api_key").
pub fn Value<S1: Into<string>, S2: Into<string>, S3: Into<string>>(
    section: S1,
    name: S2,
    key: S3,
) -> string {
    let section = section.into();
    let name = name.into();
    let key = key.into();
    let section_obj = objGet(&load(), section.as_ref());
    let name_obj = objGet(&section_obj, name.as_ref());
    match objGet(&name_obj, key.as_ref()).AsString() {
        Some(s) => s.clone(),
        None => string(""),
    }
}
