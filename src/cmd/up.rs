// cmd/up.go equivalent: the up command, self-registered via init().
//
// `kvlm up` starts a pod. `kvlm up <model>` also launches vLLM on it:
// the catalog's serve recipe, pinned vLLM version, Nsight Systems
// node-mode session, and the torch profiler flag — everything a full
// `kvlm profile run` collects from. The launch script is composed
// from the same registries `model show` and `profile show` print.
#![allow(non_snake_case)]

use goish::encoding::base64;
use goish::fmt;
use goish::string;
use goish::strings;
use goish::time;
use goish::goslice::slice;
use goish::errors::error;
use goish::{append, int, make, nil, range};

use spf13_cobra as cobra;

use crate::cmd::profile as profcmd;
use crate::cmd::rootCmd;
use crate::driver;
use crate::model;
use crate::profile;
use crate::runtime;
use crate::state;

// launchServer installs and launches vLLM on the pod, then waits for
// the endpoint (model download plus load; several minutes).
fn launchServer(dest: string, identity: string, spec: &model::ServeSpec, gpuCount: int, vllmVersion: string, profiled: bool) -> error {
    let serveLine = profile::launch::ServeLine(spec, gpuCount, profiled);
    let script = profile::launch::Script(serveLine, vllmVersion.clone(), profiled);
    let tr = profcmd::sshTransport(dest.clone(), identity);

    fmt::Printf!("kvlm: waiting for sshd on %s...\n", dest.clone());
    let mut sshUp = false;
    let mut i: int = 0;
    // image pulls on a cold host can take 10-15 minutes; wait 25
    while i < 100 {
        let (_, err) = tr.exec(string("true"));
        if err == nil {
            sshUp = true;
            break;
        }
        time::Sleep(time::Seconds(15));
        i += 1;
    }
    if !sshUp {
        return fmt::Errorf!(
            "sshd did not answer on %s after 25 minutes; the pod may still be pulling its image. It is still running and billing: reconnect later with kvlm profile run, or stop the charge with kvlm down",
            dest
        );
    }

    let enc = base64::StdEncoding.EncodeToString(script.as_bytes());
    let (_, err) = tr.exec(fmt::Sprintf!(
        "echo %s | base64 -d > %s && chmod +x %s && nohup %s >/dev/null 2>&1 & echo launched",
        enc,
        string(profile::launch::SetupScriptPath),
        string(profile::launch::SetupScriptPath),
        string(profile::launch::SetupScriptPath)
    ));
    if err != nil {
        return fmt::Errorf!("upload launch script: %v", err);
    }
    if profiled {
        fmt::Printf!(
            "kvlm: installing vLLM %s and launching %s under the profilers (model load takes a few minutes)...\n",
            vllmVersion,
            spec.Model.clone()
        );
    } else {
        fmt::Printf!(
            "kvlm: installing vLLM %s and launching %s for production serving, no profiler overhead (model load takes a few minutes)...\n",
            vllmVersion,
            spec.Model.clone()
        );
    }

    let mut i: int = 0;
    while i < 60 {
        time::Sleep(time::Seconds(30));
        let (code, _) = tr.exec(string("curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null"));
        if strings::TrimSpace(code.clone()) == "200" {
            let _ = profiled;
            fmt::Printf!("kvlm: vLLM is serving; collect with: kvlm profile run\n");
            return nil.into();
        }
        let (fails, err) = tr.exec(fmt::Sprintf!(
            "grep -c 'Engine core initialization failed' %s 2>/dev/null; true",
            string(profile::launch::ServerLogPath)
        ));
        if err == nil {
            let f = strings::TrimSpace(fails);
            if f != "" && f != "0" {
                return fmt::Errorf!(
                    "vLLM engine failed to initialize. Read the cause with: kvlm logs (a host driver older than torch's CUDA build is the common one). The pod is still running and billing: kvlm down stops it"
                );
            }
        }
        i += 1;
    }
    fmt::Errorf!("server not answering after 30 minutes; check /workspace/setup.log and %s on the pod", string(profile::launch::ServerLogPath))
}

// variantGPURef renders a variant's production GPU reference (falling
// back to the minimum) for the alternatives note.
fn variantGPURef(v: &model::Variant) -> string {
    if v.ProdGPU != "" {
        return v.ProdGPU.clone();
    }
    v.MinGPU.clone()
}

// upCmd represents the up command. (Go: var upCmd = &cobra.Command{...})
fn upCmd() -> cobra::Command {
    cobra::Command {
        Use: string("up [model]"),
        Short: string("Start a kvlm pod, and launch vLLM on it when a model is named"),
        Long: string(
            "Start a kvlm pod on the platform selected with --driver/-d\n\
             (runpod, vastai, or k8s). Credentials are resolved from\n\
             environment variables, then ~/.kvlm/config.json.\n\
             \n\
             With a model name, vLLM serves it: hardware is sized from the\n\
             catalog's production figure for the chosen --quantization, and\n\
             the platform runs the catalog serve command natively (Docker\n\
             command on RunPod). --mode profile instead launches over ssh\n\
             under Nsight Systems with the torch profiler flag, ready for\n\
             kvlm profile run. Without a model, the pod comes up bare.",
        ),
        Args: Some(cobra::MaximumNArgs(1)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: slice<string>| -> error {
                let (mode, _) = cmd.Flags().GetString("mode");
                if mode != "profile" && mode != "production" {
                    return fmt::Errorf!("--mode must be production or profile, got %q", mode);
                }
                let production = mode == "production";

                // model + variant selection (--quantization picks the
                // weight format; default: the first with a recipe)
                let mut modelName = string("");
                let mut spec: model::ServeSpec = Default::default();
                let mut variant: model::Variant = Default::default();
                let mut haveSpec = false;
                let (quant, _) = cmd.Flags().GetString("quantization");
                if args.Len() > 0 {
                    modelName = args[0usize].clone();
                    let (_, m, ok) = model::Find(modelName.clone());
                    if !ok {
                        return fmt::Errorf!("unknown model %q (see 'kvlm model ls')", modelName);
                    }
                    // without --quantization, serve the variant that
                    // needs the fewest GPUs (catalog order breaks ties);
                    // the alternatives are named so the choice is visible
                    let mut bestGPUs: int = 0;
                    let mut others: slice<string> = make!([]string, 0);
                    for (_, v) in range!(m.Variants.clone()) {
                        if quant != "" && v.Name != quant {
                            continue;
                        }
                        let (s, ok) = model::DefaultServe(&v);
                        if !ok {
                            continue;
                        }
                        let mut refStr = v.ProdGPU.clone();
                        if refStr == "" {
                            refStr = v.MinGPU.clone();
                        }
                        let (n, _, refOk) = model::ParseGPURef(refStr.clone());
                        let mut gpus: int = 1;
                        if refOk {
                            gpus = n;
                        }
                        if !haveSpec || gpus < bestGPUs {
                            if haveSpec {
                                others = append!(others.clone(), fmt::Sprintf!("%s %s", variant.Name.clone(), variantGPURef(&variant)));
                            }
                            spec = s;
                            variant = v.clone();
                            bestGPUs = gpus;
                            haveSpec = true;
                        } else {
                            others = append!(others.clone(), fmt::Sprintf!("%s %s", v.Name.clone(), refStr));
                        }
                    }
                    if !haveSpec {
                        return fmt::Errorf!(
                            "no serve recipe for %q (quantization %q); kvlm model show %s lists what exists",
                            modelName,
                            quant,
                            modelName
                        );
                    }
                    if quant == "" && others.Len() > 0 {
                        fmt::Printf!(
                            "kvlm: serving the %s variant (fewest GPUs); also in the catalog: %s (pick with --quantization)\n",
                            variant.Name.clone(),
                            strings::Join(others, ", ")
                        );
                    }
                }

                let (d, creds, err) = driver::FromCommand(cmd);
                if err != nil {
                    return err;
                }
                let (rt, err) = runtime::FromCommand(cmd);
                if err != nil {
                    return err;
                }
                let rt = rt.unwrap();

                // hardware: estimated from the catalog's production
                // sizing for the variant; flags override explicitly
                let (mut gpuType, _) = cmd.Flags().GetString("gpu-type");
                let (mut gpuCount, _) = cmd.Flags().GetInt("gpu-count");
                if haveSpec {
                    let mut refStr = variant.ProdGPU.clone();
                    if refStr == "" {
                        refStr = variant.MinGPU.clone();
                    }
                    let (n, short, ok) = model::ParseGPURef(refStr.clone());
                    if ok {
                        if !cmd.Flags().Changed("gpu-count") {
                            gpuCount = n;
                        }
                        if !cmd.Flags().Changed("gpu-type") {
                            gpuType = short.clone();
                        }
                        fmt::Printf!("kvlm: %s %s sized from the catalog: %s\n", modelName.clone(), variant.Name.clone(), refStr);
                    }
                }
                if gpuCount < 1 {
                    gpuCount = 1;
                }

                // image: the variant's runtime pin in production, the
                // platform base image (with sshd) in profile mode
                let (mut image, _) = cmd.Flags().GetString("image");
                if image == "" {
                    if production && haveSpec {
                        image = variant.Image.clone();
                    } else if production {
                        image = rt.Image();
                    } else {
                        image = string("runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04");
                    }
                }

                let (cuda, _) = cmd.Flags().GetString("cuda");
                let mut cudas: slice<string> = make!([]string, 0);
                for (_, v) in range!(strings::Split(cuda, ",")) {
                    let t = strings::TrimSpace(v.clone());
                    if t != "" {
                        cudas = append!(cudas.clone(), t);
                    }
                }
                let (volumeID, _) = cmd.Flags().GetString("volume");
                let mut serveArgv: slice<string> = make!([]string, 0);
                if production && haveSpec {
                    serveArgv = profile::launch::ServeArgv(&spec, gpuCount);
                }
                let (vllmVersion, _) = cmd.Flags().GetString("vllm-version");
                let opts = driver::Options {
                    Model: modelName.clone(),
                    Runtime: rt.Name(),
                    Image: image,
                    GPUType: gpuType,
                    GPUCount: gpuCount,
                    CudaVersions: cudas,
                    ServeArgv: serveArgv,
                    VolumeID: volumeID,
                    Variant: variant.Name.clone(),
                    Mode: mode,
                    VllmVersion: vllmVersion.clone(),
                    ..Default::default()
                };

                let (dryRun, _) = cmd.Flags().GetBool("dry-run");
                if dryRun {
                    fmt::Printf!("kvlm: would deploy %dx %s on %s, image %s, mode %s\n",
                        opts.GPUCount, opts.GPUType.clone(), d.as_ref().unwrap().Name(), opts.Image.clone(), opts.Mode.clone());
                    if opts.VolumeID != "" {
                        fmt::Printf!("kvlm: shared volume %s mounted at /workspace\n", opts.VolumeID.clone());
                    }
                    if opts.ServeArgv.Len() > 0 {
                        fmt::Printf!("kvlm: container command: %s\n", strings::Join(opts.ServeArgv.clone(), " "));
                    } else if haveSpec {
                        fmt::Printf!("kvlm: ssh launch: %s\n", profile::launch::ServeLine(&spec, gpuCount, true));
                    }
                    return nil.into();
                }

                let (dest, err) = d.unwrap().Up(&creds, &opts);
                if err != nil {
                    return err;
                }
                if production || !haveSpec || dest == "" {
                    return nil.into();
                }
                let (identity, _) = cmd.Flags().GetString("identity");
                let err = launchServer(dest, identity, &spec, gpuCount, vllmVersion, true);
                if err != nil {
                    return err;
                }
                // the launch script's server log location is a kvlm
                // constant; remember it so profile run and logs need
                // no --server-log
                let (mut t, ok) = state::Current();
                if ok {
                    t.ServerLog = string(profile::launch::ServerLogPath);
                    state::Update(&t);
                }
                nil.into()
            },
        )),
        ..Default::default()
    }
}

// Go: func init() { rootCmd.AddCommand(upCmd) }
#[goish::init]
fn init() {
    let mut c = upCmd();
    let _ = c.Flags().String_flag(
        string("gpu-type"),
        string("H100"),
        string("GPU type (H100, H200, L40S, ...); default: sized from the model catalog"),
    );
    let _ = c.Flags().Int_flag(string("gpu-count"), 1, string("number of GPUs (tensor parallel size follows); default: sized from the model catalog"));
    let _ = c.Flags().String_flag(string("quantization"), string(""), string("weight format to serve (fp8, bf16, ...); default: the first variant with a recipe"));
    let _ = c.Flags().String_flag(string("volume"), string(""), string("shared network volume id: model weights and HF cache persist and are shared across pods"));
    let _ = c.Flags().Bool_flag(string("dry-run"), false, string("print the resolved shape and serve command without deploying"));
    let _ = c.Flags().String_flag(
        string("cuda"),
        string("13.0"),
        string("allowed CUDA versions, comma separated; guards against hosts whose driver cannot run the runtime's torch build"),
    );

    let _ = c.Flags().String_flag(string("vllm-version"), string("0.26.0"), string("vLLM version to pin on the pod"));
    let _ = c.Flags().String_flag(
        string("image"),
        string(""),
        string("container image; default: the variant's runtime pin in production, the platform base image (with sshd) in profile mode"),
    );
    let _ = c.Flags().String_flag(string("mode"), string("production"), string("production: the platform runs the serve command natively (no ssh, no profiler overhead); profile: ssh launch under nsys with the torch profiler flag for kvlm profile run"));
    c.Flags().StringP(string("identity"), string("I"), string(""), string("ssh identity file for the launch (defaults to your ssh config)"));
    rootCmd.Lock().AddCommand(c);
}
