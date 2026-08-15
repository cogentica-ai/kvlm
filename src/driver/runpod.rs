// RunPod driver. Credentials: RUNPOD_API_KEY env var, or
// drivers.runpod.api_key in ~/.kvlm/config.json.
//
// Up and Down talk to the GraphQL API through the native HTTP client,
// so the key travels in a header and never appears on a command line.
// The deploy constrains allowedCudaVersions because hosts are a
// lottery: a host driver older than the runtime's torch CUDA build
// fails engine init with "driver too old". The public ssh key is
// injected so kvlm profile run can connect the moment the port is up.
#![allow(non_snake_case)]

use goish::encoding::json;
use goish::fmt;
use goish::io;
use goish::net::http;
use goish::os;
use goish::strings;
use goish::time;
use goish::errors::error;
use goish::string;
use goish::goslice::slice;
use goish::{append, bytes, float64, int, make, nil, range};

use crate::driver::{modelOrDash, resolveString, Credentials, Driver, Mask, Options, PodInfo, Register};
use crate::profile::archive::{JSONNumberField, JSONStringField};
use crate::state;

const graphqlURL: &str = "https://api.runpod.io/graphql";

struct RunPod {}

fn esc(s: string) -> string {
    let mut e = strings::ReplaceAll(s, "\\", "\\\\");
    e = strings::ReplaceAll(e, "\"", "\\\"");
    e
}

// gql posts one GraphQL document and returns the response body.
// GraphQL transports errors in the body, so callers get the body even
// alongside an error and any "errors" array is surfaced as one.
fn gql(apiKey: string, body: string) -> (string, error) {
    let (mut req, err) = http::NewRequest(string("POST"), string(graphqlURL), bytes(body));
    if err != nil {
        return (string(""), err);
    }
    req.Header.Set(string("Content-Type"), string("application/json"));
    req.Header.Set(string("Authorization"), ("Bearer ") + (apiKey));
    let client = http::Client::default();
    let (resp, err) = client.Do(&req);
    if err != nil {
        return (string(""), fmt::Errorf!("runpod api: %v", err));
    }
    let mut respBody = resp.Body;
    let (data, err) = io::ReadAll(&mut respBody);
    if err != nil {
        return (string(""), err);
    }
    let text = string(data);
    if strings::Contains(text.clone(), "\"errors\"") {
        let msg = JSONStringField(text.clone(), string("message"));
        return (text.clone(), fmt::Errorf!("runpod api error: %s", msg));
    }
    (text, nil.into())
}

// publicKey resolves the ssh public key injected into the pod:
// RUNPOD_PUBLIC_KEY / config first, then the usual key files.
fn publicKey() -> string {
    let cfg = resolveString("RUNPOD_PUBLIC_KEY", "runpod", "public_key");
    if cfg != "" {
        return cfg;
    }
    let (home, err) = os::UserHomeDir();
    if err != nil {
        return string("");
    }
    for rel in ["/.runpod/id_ed25519.pub", "/.ssh/id_ed25519.pub", "/.ssh/id_rsa.pub"].iter() {
        let (data, err) = os::ReadFile((home.clone()) + (string(*rel)));
        if err == nil {
            return strings::TrimSpace(string(data));
        }
    }
    string("")
}

// gpuTypeID maps the catalog's short GPU names (H100, L40S, ...) to
// RunPod gpuTypeId strings; unknown names pass through verbatim so a
// full platform id keeps working.
fn gpuTypeID(short: string) -> string {
    let pairs: &[(&str, &str)] = &[
        ("H200", "NVIDIA H200"),
        ("H100", "NVIDIA H100 80GB HBM3"),
        ("B200", "NVIDIA B200"),
        ("A100", "NVIDIA A100 80GB PCIe"),
        ("L40S", "NVIDIA L40S"),
        ("L4", "NVIDIA L4"),
        ("RTX 4090", "NVIDIA GeForce RTX 4090"),
    ];
    for (k, v) in pairs.iter() {
        if short == *k {
            return string(*v);
        }
    }
    short
}

// dockerCommand renders the container command override for the vLLM
// runtime image: everything after the image's `vllm serve` entrypoint
// (the model and its flags), space-joined with single quotes around
// values that need them.
fn dockerCommand(serveArgv: &goish::slice<string>) -> string {
    let mut b = strings::Builder::new();
    let mut wrote: int = 0;
    for (i, a) in range!(serveArgv.clone()) {
        // drop the ["vllm", "serve"] prefix: the image entrypoint is it
        if i < 2 && (a == "vllm" || a == "serve") {
            continue;
        }
        if wrote > 0 {
            let _ = b.WriteString(" ");
        }
        if strings::Contains(a.clone(), " ") || strings::Contains(a.clone(), "{") {
            let _ = b.WriteString(("'") + (a.clone()) + ("'"));
        } else {
            let _ = b.WriteString(a.clone());
        }
        wrote += 1;
    }
    string(b.String())
}

// walk descends a parsed JSON value through object keys, Null when a
// step is missing.
fn walk(v: &json::Value, keys: &[&str]) -> json::Value {
    let mut cur = v.clone();
    for k in keys.iter() {
        let mut next = json::Value::Null;
        if let Some(obj) = cur.AsObject() {
            let (val, _) = obj.Get(*k);
            next = val;
        }
        cur = next;
    }
    cur
}

fn jstr(v: &json::Value, key: &str) -> string {
    if let Some(s) = walk(v, &[key]).AsString() {
        return s.clone();
    }
    string("")
}

fn jnum(v: &json::Value, key: &str) -> float64 {
    if let Some(n) = walk(v, &[key]).AsNumber() {
        return n;
    }
    0.0
}

// proxyURL is the platform's http proxy for a pod's port 8000.
fn proxyURL(podID: string) -> string {
    fmt::Sprintf!("https://%s-8000.proxy.runpod.net", podID)
}

// securePrice asks the platform for the on-demand secure-cloud price
// of one GPU type, per GPU per hour. Best effort: 0 when the API or
// the field is unavailable, and callers omit the price line then.
fn securePrice(apiKey: string, gpuTypeId: string) -> float64 {
    let body = fmt::Sprintf!(
        "{\"query\": \"query GpuTypes($id: String!) { gpuTypes(input: {id: $id}) { id securePrice } }\", \"variables\": {\"id\": \"%s\"}}",
        esc(gpuTypeId)
    );
    let (resp, err) = gql(apiKey, body);
    if err != nil {
        return 0.0;
    }
    let (price, ok) = JSONNumberField(resp, string("securePrice"));
    if !ok {
        return 0.0;
    }
    price
}

impl Driver for RunPod {
    fn Name(&self) -> string {
        string("runpod")
    }

    fn ResolveCredentials(&self) -> (Credentials, error) {
        let apiKey = resolveString("RUNPOD_API_KEY", self.Name(), "api_key");
        if apiKey == "" {
            return (
                Default::default(),
                fmt::Errorf!(
                    "runpod: no API key found: set RUNPOD_API_KEY or \"drivers\": {\"runpod\": {\"api_key\": ...}} in ~/.kvlm/config.json"
                ),
            );
        }
        (
            Credentials {
                APIKey: apiKey,
                ..Default::default()
            },
            nil.into(),
        )
    }

    fn Up(&self, creds: &Credentials, opts: &Options) -> (string, error) {
        let pubkey = publicKey();
        if pubkey == "" {
            return (
                string(""),
                fmt::Errorf!(
                    "runpod: no ssh public key: set RUNPOD_PUBLIC_KEY, drivers.runpod.public_key in ~/.kvlm/config.json, or have ~/.ssh/id_ed25519.pub"
                ),
            );
        }
        let mut name = ("kvlm-") + (opts.Model.clone());
        if opts.Model == "" {
            name = string("kvlm-pod");
        }
        let production = opts.ServeArgv.len() > 0;
        let mut cudas = string("");
        for (i, v) in range!(opts.CudaVersions.clone()) {
            if i > 0 {
                cudas = (cudas) + (", ");
            }
            cudas = (cudas) + ("\"") + (esc(v.clone())) + ("\"");
        }
        let typeId = gpuTypeID(opts.GPUType.clone());
        let price = securePrice(creds.APIKey.clone(), typeId.clone());
        let mut priceNote = string("");
        if price > 0.0 {
            priceNote = fmt::Sprintf!(", ~$%v/hr", price * float64(opts.GPUCount));
        }
        fmt::Printf!(
            "kvlm: [runpod] deploying %dx %s, image %s%s (CUDA %s, api key %s)\n",
            opts.GPUCount,
            typeId.clone(),
            opts.Image.clone(),
            priceNote,
            cudas.clone(),
            Mask(&creds.APIKey)
        );
        // production serves through the platform: the runtime image
        // runs the catalog command natively, no ssh in the loop;
        // profile mode keeps ssh for the capture transport
        let mut ports = string("22/tcp,8000/http");
        let mut dockerArgs = string("");
        if production {
            ports = string("8000/http");
            dockerArgs = dockerCommand(&opts.ServeArgv);
            fmt::Printf!("kvlm: [runpod] container command: %s\n", dockerArgs.clone());
        }
        let mut volume = string("\"volumeInGb\": 60");
        if opts.VolumeID != "" {
            volume = fmt::Sprintf!("\"networkVolumeId\": \"%s\"", esc(opts.VolumeID.clone()));
        }
        let input = fmt::Sprintf!(
            "{\"cloudType\": \"SECURE\", \"gpuCount\": %d, \"gpuTypeId\": \"%s\", \"name\": \"%s\", \"imageName\": \"%s\", \"dockerArgs\": \"%s\", \"ports\": \"%s\", %s, \"containerDiskInGb\": 40, \"volumeMountPath\": \"/workspace\", \"supportPublicIp\": true, \"allowedCudaVersions\": [%s], \"env\": [{\"key\": \"PUBLIC_KEY\", \"value\": \"%s\"}, {\"key\": \"HF_HOME\", \"value\": \"/workspace/hf\"}]}",
            opts.GPUCount,
            esc(gpuTypeID(opts.GPUType.clone())),
            esc(name),
            esc(opts.Image.clone()),
            esc(dockerArgs),
            ports,
            volume,
            cudas,
            esc(pubkey)
        );
        let body = fmt::Sprintf!(
            "{\"query\": \"mutation Deploy($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id costPerHr machineId } }\", \"variables\": {\"input\": %s}}",
            input
        );
        let (resp, err) = gql(creds.APIKey.clone(), body);
        if err != nil {
            return (string(""), err);
        }
        let podID = JSONStringField(resp.clone(), string("id"));
        if podID == "" {
            return (string(""), fmt::Errorf!("runpod: no pod id in response: %s", resp));
        }
        let (cost, _) = JSONNumberField(resp, string("costPerHr"));
        fmt::Printf!("kvlm: [runpod] pod %s deployed at $%v/hr\n", podID.clone(), cost);
        // record the pod the moment it exists, before any wait loop:
        // an interrupted up must never orphan a billing pod
        let mut endpoint = string("");
        if production {
            endpoint = proxyURL(podID.clone());
        }
        state::Record(&state::Target {
            Driver: self.Name(),
            Pod: podID.clone(),
            Endpoint: endpoint,
            Model: opts.Model.clone(),
            Variant: opts.Variant.clone(),
            Mode: opts.Mode.clone(),
            GPUType: opts.GPUType.clone(),
            GPUCount: opts.GPUCount,
            VllmVersion: opts.VllmVersion.clone(),
            CostPerHr: cost,
            Created: time::Now().Format(string(time::RFC3339)),
            ..Default::default()
        });

        if production {
            // the serving endpoint is the platform's http proxy; the
            // model pull and load happen inside the container, so the
            // wait can be long on the first run of a model
            let url = fmt::Sprintf!("https://%s-8000.proxy.runpod.net/v1/models", podID.clone());
            fmt::Printf!("kvlm: [runpod] waiting for vLLM at %s (image pull + model load; several minutes)...\n", url.clone());
            let client = http::Client::default();
            let mut tries: int = 0;
            while tries < 90 {
                time::Sleep(time::Seconds(20));
                tries += 1;
                let (resp, err) = client.Get(url.clone());
                if err != nil {
                    continue;
                }
                let code = resp.StatusCode;
                let mut respBody = resp.Body;
                let _ = io::ReadAll(&mut respBody);
                if code == 200 {
                    fmt::Printf!("kvlm: [runpod] vLLM is serving: %s/v1\n", proxyURL(podID.clone()));
                    fmt::Printf!("kvlm: [runpod] tear down with: kvlm down\n");
                    return (string(""), nil.into());
                }
            }
            fmt::Printf!("kvlm: [runpod] endpoint not answering after 30 minutes; the pod is still running and billing. Check the logs in the console, or stop the charge with: kvlm down --pod %s\n", podID.clone());
            return (string(""), nil.into());
        }

        fmt::Printf!("kvlm: [runpod] waiting for ssh...\n");
        // poll until the ssh port is published (the pod boots and
        // pulls the image first; a few minutes is normal)
        let query = fmt::Sprintf!(
            "{\"query\": \"query Pod($id: String!) { pod(input: {podId: $id}) { id desiredStatus runtime { ports { ip publicPort privatePort } } } }\", \"variables\": {\"id\": \"%s\"}}",
            podID.clone()
        );
        let mut tries: int = 0;
        while tries < 40 {
            time::Sleep(time::Seconds(15));
            tries += 1;
            let (resp, err) = gql(creds.APIKey.clone(), query.clone());
            if err != nil {
                continue;
            }
            for (_, chunk) in range!(strings::Split(resp.clone(), "{")) {
                if !strings::Contains(chunk.clone(), "\"privatePort\":22") {
                    continue;
                }
                let ip = JSONStringField(chunk.clone(), string("ip"));
                let (port, ok) = JSONNumberField(chunk.clone(), string("publicPort"));
                if ip != "" && ok {
                    let dest = fmt::Sprintf!("root@%s:%d", ip.clone(), port as int);
                    let (mut t, found) = state::Find(podID.clone());
                    if found {
                        t.SSH = dest.clone();
                        state::Update(&t);
                    }
                    fmt::Printf!("kvlm: [runpod] ssh port published: %s (sshd may need another minute while the image finishes pulling)\n", dest.clone());
                    fmt::Printf!("kvlm: [runpod] verify the driver before launching vLLM: ssh -p %d root@%s nvidia-smi --query-gpu=driver_version --format=csv,noheader\n", port as int, ip.clone());
                    fmt::Printf!("kvlm: [runpod] tear down with: kvlm down\n");
                    return (dest, nil.into());
                }
            }
        }
        fmt::Printf!("kvlm: [runpod] pod %s has no ssh port yet and is still billing; watch it with kvlm ps, or stop the charge with: kvlm down --pod %s\n", podID.clone(), podID);
        (string(""), nil.into())
    }

    fn Down(&self, creds: &Credentials, opts: &Options) -> error {
        if opts.PodID == "" {
            return fmt::Errorf!("runpod: no pod id. List what is running with kvlm ps, then: kvlm down <pod-id>");
        }
        fmt::Printf!(
            "kvlm: [runpod] terminating pod %s (model %s, api key %s)\n",
            opts.PodID.clone(),
            modelOrDash(opts),
            Mask(&creds.APIKey)
        );
        let body = fmt::Sprintf!(
            "{\"query\": \"mutation { podTerminate(input: {podId: \\\"%s\\\"}) }\"}",
            esc(opts.PodID.clone())
        );
        let (resp, err) = gql(creds.APIKey.clone(), body);
        if err != nil {
            return err;
        }
        // a null podTerminate in data is the API's success shape
        if !strings::Contains(resp.clone(), "podTerminate") {
            return fmt::Errorf!("runpod: unexpected response: %s", resp);
        }
        state::Remove(self.Name(), opts.PodID.clone());
        fmt::Println!("kvlm: [runpod] terminated");
        nil.into()
    }

    fn List(&self, creds: &Credentials) -> (slice<PodInfo>, error) {
        let empty: slice<PodInfo> = make!([]PodInfo, 0);
        let body = string(
            "{\"query\": \"query { myself { pods { id name desiredStatus costPerHr gpuCount imageName machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip publicPort privatePort } } } } }\"}",
        );
        let (resp, err) = gql(creds.APIKey.clone(), body);
        if err != nil {
            return (empty, err);
        }
        let mut v = json::Value::Null;
        let perr = json::Unmarshal(resp.as_bytes(), &mut v);
        if perr != nil {
            return (empty, fmt::Errorf!("runpod: unparseable pod list: %v", perr));
        }
        let mut pods: slice<PodInfo> = make!([]PodInfo, 0);
        let arr = walk(&v, &["data", "myself", "pods"]);
        if let Some(list) = arr.AsArray() {
            for (_, p) in range!(list.clone()) {
                let mut info = PodInfo {
                    ID: jstr(&p, "id"),
                    Name: jstr(&p, "name"),
                    Status: jstr(&p, "desiredStatus"),
                    GPUCount: int(jnum(&p, "gpuCount")),
                    Image: jstr(&p, "imageName"),
                    CostPerHr: jnum(&p, "costPerHr"),
                    ..Default::default()
                };
                let machine = walk(&p, &["machine"]);
                info.GPUType = jstr(&machine, "gpuDisplayName");
                let rt = walk(&p, &["runtime"]);
                info.UptimeSeconds = int(jnum(&rt, "uptimeInSeconds"));
                if let Some(ports) = walk(&rt, &["ports"]).AsArray() {
                    for (_, port) in range!(ports.clone()) {
                        if int(jnum(&port, "privatePort")) == 22 {
                            let ip = jstr(&port, "ip");
                            let public = int(jnum(&port, "publicPort"));
                            if ip != "" && public > 0 {
                                info.SSH = fmt::Sprintf!("root@%s:%d", ip, public);
                            }
                        }
                        if int(jnum(&port, "privatePort")) == 8000 {
                            info.Endpoint = proxyURL(info.ID.clone());
                        }
                    }
                }
                pods = append!(pods.clone(), info);
            }
        }
        (pods, nil.into())
    }

    fn Exec(&self, _creds: &Credentials, podId: string, cmd: string) -> (string, error) {
        // RunPod has no exec API; kvlm profile run reaches pods over
        // ssh (--ssh root@ip:port), which Up prints when the pod is up.
        let _ = podId;
        let _ = cmd;
        (
            string(""),
            fmt::Errorf!("runpod: no exec API; use kvlm profile run --ssh root@<ip>:<port> (printed by kvlm up)"),
        )
    }

    fn Download(
        &self,
        _creds: &Credentials,
        podId: string,
        remotePath: string,
        localPath: string,
    ) -> error {
        let _ = podId;
        fmt::Errorf!(
            "runpod: no file API; kvlm profile run fetches artifacts over scp (%s -> %s)",
            remotePath,
            localPath
        )
    }
}

// Go: func init() { driver.Register("runpod", &RunPod{}) }
#[goish::init]
fn init() {
    Register("runpod", alloc::sync::Arc::new(RunPod {}));
}
