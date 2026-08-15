// Kubernetes driver. Credentials: KUBECONFIG env var, or
// drivers.k8s.kubeconfig in ~/.kvlm/config.json, or ~/.kube/config.
// Context/namespace: the conventional KUBE_CONTEXT / KUBE_NAMESPACE env
// vars, or drivers.k8s.{context,namespace}; namespace defaults to
// "default".
#![allow(non_snake_case)]

use goish::bytes;
use goish::fmt;
use goish::io::Reader as _;
use goish::os;
use goish::os::exec;
use goish::errors::error;
use goish::string;
use goish::goslice::slice;
use goish::types;
use goish::{append, make, nil};

use crate::driver::{modelOrDash, resolveString, Credentials, Driver, Options, Register};

struct K8s {}

impl K8s {
    // kubectlArgs builds the common kubectl flags (--kubeconfig, --context, -n).
    fn kubectlArgs(&self, creds: &Credentials) -> slice<string> {
        let mut args: slice<string> = make!([]string, 0);
        if creds.Kubeconfig != "" {
            args = append!(args.clone(), string("--kubeconfig"));
            args = append!(args.clone(), creds.Kubeconfig.clone());
        }
        if creds.Context != "" {
            args = append!(args.clone(), string("--context"));
            args = append!(args.clone(), creds.Context.clone());
        }
        if creds.Namespace != "" {
            args = append!(args.clone(), string("-n"));
            args = append!(args.clone(), creds.Namespace.clone());
        }
        args
    }
}

impl Driver for K8s {
    fn Name(&self) -> string {
        string("k8s")
    }

    fn ResolveCredentials(&self) -> (Credentials, error) {
        let mut kubeconfig = resolveString("KUBECONFIG", self.Name(), "kubeconfig");
        if kubeconfig == "" {
            let (home, err) = os::UserHomeDir();
            if err != nil {
                return (Default::default(), err);
            }
            kubeconfig = (home) + ("/.kube/config");
        }
        let (_, err) = os::Stat(kubeconfig.clone());
        if err != nil {
            return (
                Default::default(),
                fmt::Errorf!(
                    "k8s: kubeconfig not found at %q: set KUBECONFIG or \"drivers\": {\"k8s\": {\"kubeconfig\": ...}} in ~/.kvlm/config.json",
                    kubeconfig
                ),
            );
        }

        let context = resolveString("KUBE_CONTEXT", self.Name(), "context");
        let mut namespace = resolveString("KUBE_NAMESPACE", self.Name(), "namespace");
        if namespace == "" {
            namespace = string("default");
        }

        (
            Credentials {
                Kubeconfig: kubeconfig,
                Context: context,
                Namespace: namespace,
                ..Default::default()
            },
            nil.into(),
        )
    }

    fn Up(&self, creds: &Credentials, opts: &Options) -> (string, error) {
        let mut context = creds.Context.clone();
        if context == "" {
            context = string("<current-context>");
        }
        let _ = modelOrDash(opts);
        let _ = context;
        // TODO: POST the Pod manifest to the API server (goish net/http + TLS).
        (
            string(""),
            fmt::Errorf!("k8s: pod creation not implemented; kvlm profile run --k8s-pod <name> works against pods you created yourself"),
        )
    }

    fn Down(&self, creds: &Credentials, opts: &Options) -> error {
        let mut context = creds.Context.clone();
        if context == "" {
            context = string("<current-context>");
        }
        let _ = modelOrDash(opts);
        let _ = context;
        // TODO: DELETE the Pod via the API server.
        fmt::Errorf!("k8s: pod deletion not implemented; delete the pod with kubectl")
    }

    fn Exec(&self, creds: &Credentials, podId: string, cmd: string) -> (string, error) {
        // Build: kubectl [common flags] exec <pod> -- <cmd>
        let mut args = self.kubectlArgs(creds);
        args = append!(args.clone(), string("exec"));
        args = append!(args.clone(), podId);
        args = append!(args.clone(), string("--"));
        args = append!(args.clone(), string("sh"));
        args = append!(args.clone(), string("-c"));
        args = append!(args.clone(), cmd);

        let mut command = exec::Command(string("kubectl"), args);
        
        // Use pipes to capture output
        let (mut stdoutReader, err) = command.StdoutPipe();
        if err != nil {
            return (string(""), fmt::Errorf!("k8s exec: failed to create stdout pipe: %v", err));
        }
        let (mut stderrReader, err) = command.StderrPipe();
        if err != nil {
            return (string(""), fmt::Errorf!("k8s exec: failed to create stderr pipe: %v", err));
        }

        let err = command.Start();
        if err != nil {
            return (string(""), fmt::Errorf!("k8s exec: failed to start: %v", err));
        }

        // Drain stderr in a goroutine while this thread drains stdout:
        // sequential drains deadlock once the child fills the unread
        // pipe (~64KB) while we are still blocked on the other one.
        let stderrCh = make!(chan string, 1);
        {
            let ch = stderrCh.clone();
            goish::go!(move || {
                let mut b = bytes::Buffer::new();
                let mut buf = make!([]types::byte, 4096);
                loop {
                    let (n, err) = stderrReader.Read(&mut buf);
                    if n > 0 {
                        let slice = buf.slice(0, n);
                        b.Write(slice);
                    }
                    if err != nil {
                        break;
                    }
                }
                ch.Send(b.String());
            });
        }

        // Read stdout
        let mut stdoutBuf = bytes::Buffer::new();
        let mut buf = make!([]types::byte, 4096);
        loop {
            let (n, err) = stdoutReader.Read(&mut buf);
            if n > 0 {
                let slice = buf.slice(0, n);
                stdoutBuf.Write(slice);
            }
            if err != nil {
                break;
            }
        }

        let (stderrStr, _) = stderrCh.Recv();

        let err = command.Wait();
        if err != nil {
            return (
                string(""),
                fmt::Errorf!("k8s exec failed: %v\nstderr: %s", err, stderrStr),
            );
        }
        (stdoutBuf.String(), nil.into())
    }

    fn Download(
        &self,
        creds: &Credentials,
        podId: string,
        remotePath: string,
        localPath: string,
    ) -> error {
        // Build: kubectl [common flags] cp <pod>:<remotePath> <localPath>
        let mut args = self.kubectlArgs(creds);
        args = append!(args.clone(), string("cp"));
        let src = (podId.clone()) + (":") + (remotePath);
        args = append!(args.clone(), src);
        args = append!(args.clone(), localPath);

        let mut command = exec::Command(string("kubectl"), args);
        
        // Use pipe to capture stderr
        let (mut stderrReader, err) = command.StderrPipe();
        if err != nil {
            return fmt::Errorf!("k8s cp: failed to create stderr pipe: %v", err);
        }

        let err = command.Start();
        if err != nil {
            return fmt::Errorf!("k8s cp: failed to start: %v", err);
        }

        // Read stderr
        let mut stderrBuf = bytes::Buffer::new();
        let mut buf = make!([]types::byte, 4096);
        loop {
            let (n, err) = stderrReader.Read(&mut buf);
            if n > 0 {
                let slice = buf.slice(0, n);
                stderrBuf.Write(slice);
            }
            if err != nil {
                break;
            }
        }

        let err = command.Wait();
        if err != nil {
            let stderrStr = stderrBuf.String();
            return fmt::Errorf!("k8s cp failed: %v\nstderr: %s", err, stderrStr);
        }
        nil.into()
    }
}

// Go: func init() { driver.Register("k8s", &K8s{}) }
#[goish::init]
fn init() {
    Register("k8s", alloc::sync::Arc::new(K8s {}));
}
