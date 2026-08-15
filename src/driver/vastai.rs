// Vast.ai driver. Credentials: VAST_API_KEY env var, or
// drivers.vastai.api_key in ~/.kvlm/config.json.
#![allow(non_snake_case)]

use goish::fmt;
use goish::errors::error;
use goish::string;
use goish::{nil};

use crate::driver::{resolveString, Credentials, Driver, Options, Register};

struct VastAI {}

impl Driver for VastAI {
    fn Name(&self) -> string {
        string("vastai")
    }

    fn ResolveCredentials(&self) -> (Credentials, error) {
        let apiKey = resolveString("VAST_API_KEY", self.Name(), "api_key");
        if apiKey == "" {
            return (
                Default::default(),
                fmt::Errorf!(
                    "vastai: no API key found: set VAST_API_KEY or \"drivers\": {\"vastai\": {\"api_key\": ...}} in ~/.kvlm/config.json"
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
        let _ = creds;
        let _ = opts;
        // TODO: call the Vast.ai REST API (PUT /api/v0/asks/<id>/).
        (
            string(""),
            fmt::Errorf!("vastai: driver not implemented; use --driver runpod or --driver k8s"),
        )
    }

    fn Down(&self, creds: &Credentials, opts: &Options) -> error {
        let _ = creds;
        let _ = opts;
        // TODO: call the Vast.ai REST API (DELETE /api/v0/instances/<id>/).
        fmt::Errorf!("vastai: driver not implemented; use --driver runpod or --driver k8s")
    }

    fn Exec(&self, _creds: &Credentials, _podId: string, _cmd: string) -> (string, error) {
        // TODO: implement via vastai exec or SSH.
        (string(""), fmt::Errorf!("vastai: exec not implemented yet"))
    }

    fn Download(
        &self,
        _creds: &Credentials,
        _podId: string,
        _remotePath: string,
        _localPath: string,
    ) -> error {
        // TODO: implement via vastai scp or SSH.
        fmt::Errorf!("vastai: download not implemented yet")
    }
}

// Go: func init() { driver.Register("vastai", &VastAI{}) }
#[goish::init]
fn init() {
    Register("vastai", alloc::sync::Arc::new(VastAI {}));
}
