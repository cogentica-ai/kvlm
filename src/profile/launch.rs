// launch: composing the vLLM serving launch for a profiling target.
//
// kvlm up <model> launches vLLM on the pod it deployed: install the
// pinned vLLM and Nsight Systems, then launch the catalog's serve
// command under the nsys node-mode session with the torch profiler
// enabled. Everything here composes from the existing sources of truth
// — the model catalog's ServeSpec and the profile registry's Install
// and Setup contracts — so what launches is exactly what `model show`
// and `profile show` print.
#![allow(non_snake_case)]

use goish::fmt;
use goish::strings;
use goish::string;
use goish::{append, int, make, range, slice};

use crate::model;
use crate::profile;

// ProfilerConfigValue is the measured working torch-profiler launch flag
// (vLLM 0.26.0: the flag parses even though serve --help omits it;
// trust the ProfilerConfig line the server logs at startup).
pub const ProfilerConfigValue: &str = "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"/tmp/kvlm-profile/torch\"}";

pub const ServeScriptPath: &str = "/workspace/serve-kvlm.sh";
pub const SetupScriptPath: &str = "/workspace/kvlm-setup.sh";
pub const ServerLogPath: &str = "/workspace/vllm.log";
pub const RestartLogPath: &str = "/workspace/restart.log";

fn shQuote(v: string) -> string {
    if strings::Contains(v.clone(), " ") || strings::Contains(v.clone(), "\"") || strings::Contains(v.clone(), "{") {
        return ("'") + (v) + ("'");
    }
    v
}

// ServeLine renders the exec form of a catalog serve spec, tensor
// parallelism sized to the pod's GPU count (appended only when the
// spec does not set it already). profiled adds the torch profiler
// flag; production serving omits it so no profiling endpoints exist.
pub fn ServeLine(spec: &model::ServeSpec, gpuCount: int, profiled: bool) -> string {
    let mut b = strings::Builder::new();
    let _ = b.WriteString("exec");
    for (_, a) in range!(ServeArgv(spec, gpuCount)) {
        let _ = b.WriteString((" ") + (shQuote(a.clone())));
    }
    if profiled {
        let _ = b.WriteString((" --profiler-config ") + (shQuote(string(ProfilerConfigValue))));
    }
    string(b.String())
}

// ServeArgv is the catalog serve command with tensor parallelism
// sized to the pod (appended only when the recipe does not pin it) —
// the container command production deploys run natively.
pub fn ServeArgv(spec: &model::ServeSpec, gpuCount: int) -> slice<string> {
    let mut argv = spec.Argv();
    let (_, hasTP) = spec.FlagValue("--tensor-parallel-size");
    if gpuCount > 1 && !hasTP {
        argv = append!(argv.clone(), string("--tensor-parallel-size"));
        argv = append!(argv.clone(), fmt::Sprintf!("%d", gpuCount));
    }
    argv
}

// WithFlag returns the spec with one flag overridden (or appended when
// the recipe does not carry it). Flag names are the dash CLI form
// ("--max-num-seqs").
pub fn WithFlag(spec: &model::ServeSpec, name: string, value: string) -> model::ServeSpec {
    let mut out = spec.clone();
    let mut found = false;
    let mut flags: slice<model::Flag> = make!([]model::Flag, 0);
    for (_, f) in range!(out.Flags.clone()) {
        if f.Name == name {
            flags = append!(
                flags.clone(),
                model::Flag {
                    Name: f.Name.clone(),
                    Value: value.clone(),
                    ..Default::default()
                }
            );
            found = true;
        } else {
            flags = append!(flags.clone(), f.clone());
        }
    }
    if !found {
        flags = append!(
            flags.clone(),
            model::Flag {
                Name: name,
                Value: value,
                ..Default::default()
            }
        );
    }
    out.Flags = flags;
    out
}

// writeServeScript writes the serve-kvlm.sh heredoc: HF cache on the
// volume, and the pip nvidia lib dirs on the loader path (torchcodec
// and friends dlopen CUDA libs that live in site-packages, measured
// failing without this on a fresh pod).
fn writeServeScript(b: &mut strings::Builder, serveLine: string) {
    let _ = b.WriteString(fmt::Sprintf!(
        "cat > %s <<'KVLMEOF'\n#!/bin/bash\nexport HF_HOME=/workspace/hf\nexport LD_LIBRARY_PATH=$(ls -d /usr/local/lib/python*/dist-packages/nvidia/*/lib 2>/dev/null | tr '\\n' ':')$LD_LIBRARY_PATH\n%s\nKVLMEOF\n",
        string(ServeScriptPath),
        serveLine
    ));
    let _ = b.WriteString(fmt::Sprintf!("chmod +x %s\n", string(ServeScriptPath)));
}

// writePreflight writes the import check that runs before the server
// starts: seconds to fail with the real traceback, instead of a dead
// server discovered by a 30-minute wait. Vision models add torchcodec
// (its native lib needs ffmpeg and the nvidia lib path).
fn writePreflight(b: &mut strings::Builder, vision: bool) {
    let mut imports = string("torch, vllm");
    if vision {
        imports = string("torch, vllm, torchcodec");
    }
    let _ = b.WriteString("NVLIBS=$(ls -d /usr/local/lib/python*/dist-packages/nvidia/*/lib 2>/dev/null | tr '\\n' ':')\n");
    let _ = b.WriteString(fmt::Sprintf!(
        "if ! LD_LIBRARY_PATH=$NVLIBS$LD_LIBRARY_PATH python3 -c 'import %s' 2>&1; then echo PREFLIGHT_FAILED; exit 1; fi\n",
        imports
    ));
    let _ = b.WriteString("echo PREFLIGHT_OK\n");
}

// RestartScript composes the warm-restart script kvlm apply uploads:
// stop the serving process, rewrite the serve script with the new
// line, and relaunch it under a fresh nsys node-mode session. Weights
// stay in the volume cache, so this is a ~90 second cycle, not a
// redeploy.
pub fn RestartScript(serveLine: string) -> string {
    let (nsys, _) = profile::Find("nsys");
    let mut b = strings::Builder::new();
    let _ = b.WriteString(fmt::Sprintf!(
        "#!/bin/bash\nset -x\nexec > %s 2>&1\nexport HF_HOME=/workspace/hf\n\n",
        string(RestartLogPath)
    ));
    // the [v] character class keeps pkill from matching this script
    let _ = b.WriteString("pkill -f '[v]llm serve' || true\nsleep 3\n");
    // the relaunch uses --session-new, which refuses a name that still
    // exists; shut the old session down and verify it is really gone
    // (a lingering session was a silent way for the relaunch to die)
    let _ = b.WriteString("for i in 1 2 3 4 5; do\n  nsys shutdown --session=kvlm >/dev/null 2>&1 || true\n  nsys sessions list 2>/dev/null | grep -q kvlm || break\n  sleep 2\ndone\n");
    let _ = b.WriteString("mkdir -p /tmp/kvlm-profile/torch\n");
    writeServeScript(&mut b, serveLine);
    let vars = slice!([]profile::Var{
        profile::Var{ Key: string("session"), Value: string("kvlm"), ..Default::default() },
        profile::Var{ Key: string("serve"), Value: string(ServeScriptPath), ..Default::default() },
    });
    let setup = profile::Expand(nsys.Setup.clone(), vars);
    for (_, c) in range!(setup.clone()) {
        let _ = b.WriteString(fmt::Sprintf!("%s > %s 2>&1 &\n", profile::RenderCmd(&c), string(ServerLogPath)));
    }
    let _ = b.WriteString("echo RESTARTED $!\n");
    string(b.String())
}

// Script composes the one-shot install-and-launch script: pinned vLLM
// install, the serve script, and — in profile mode — the nsys install
// steps and the node-mode launch from the registry's Setup contract
// (measured cost: about 4x cudaGraphLaunch overhead, a few percent of
// a decode step, which is why production mode launches bare).
pub fn Script(serveLine: string, vllmVersion: string, profiled: bool, vision: bool) -> string {
    let (nsys, _) = profile::Find("nsys");
    let mut b = strings::Builder::new();
    let _ = b.WriteString("#!/bin/bash\nset -x\nexec > /workspace/setup.log 2>&1\nexport DEBIAN_FRONTEND=noninteractive\nexport HF_HOME=/workspace/hf\n\n");
    let _ = b.WriteString(fmt::Sprintf!("pip install -q vllm==%s 2>&1 | tail -2\n\n", vllmVersion));
    if vision {
        // vision models import torchcodec, whose native lib dlopens
        // ffmpeg (measured failing without it on a fresh pod)
        let _ = b.WriteString("apt-get install -y -qq ffmpeg >/dev/null 2>&1\n");
    }
    if profiled {
        for (_, c) in range!(nsys.Install.clone()) {
            let _ = b.WriteString(profile::RenderCmd(&c));
            let _ = b.WriteString("\n");
        }
    }
    let _ = b.WriteString("\nmkdir -p /tmp/kvlm-profile/torch\n");
    writeServeScript(&mut b, serveLine);
    writePreflight(&mut b, vision);
    let _ = b.WriteString("echo SETUP_DONE\n");
    if profiled {
        // the registry's launch contract, with the placeholders bound
        let vars = slice!([]profile::Var{
            profile::Var{ Key: string("session"), Value: string("kvlm"), ..Default::default() },
            profile::Var{ Key: string("serve"), Value: string(ServeScriptPath), ..Default::default() },
        });
        let setup = profile::Expand(nsys.Setup.clone(), vars);
        for (_, c) in range!(setup.clone()) {
            let _ = b.WriteString(fmt::Sprintf!("%s > %s 2>&1 &\n", profile::RenderCmd(&c), string(ServerLogPath)));
        }
    } else {
        let _ = b.WriteString(fmt::Sprintf!("%s > %s 2>&1 &\n", string(ServeScriptPath), string(ServerLogPath)));
    }
    let _ = b.WriteString("echo LAUNCHED $!\n");
    string(b.String())
}
