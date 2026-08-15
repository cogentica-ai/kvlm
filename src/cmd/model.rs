// cmd/model.go equivalent: the model command group, self-registered via
// init(). `kvlm model ls` lists the catalog docker-style.
#![allow(non_snake_case)]

use goish::string;
use goish::slice;
use goish::goslice::slice as goslice;

use spf13_cobra as cobra;

use crate::cmd::rootCmd;
use crate::model;

// modelCmd represents the model command group (not runnable — shows help).
fn modelCmd() -> cobra::Command {
    cobra::Command {
        Use: string("model"),
        Short: string("Manage models kvlm can serve"),
        Long: string("List and inspect the model families kvlm knows how to run."),
        ..Default::default()
    }
}

// lsCmd represents `kvlm model ls` (alias: list).
fn lsCmd() -> cobra::Command {
    cobra::Command {
        Use: string("ls"),
        Aliases: slice!([]string{"list"}),
        Short: string("List models"),
        Run: Some(alloc::sync::Arc::new(
            |_cmd: &mut cobra::Command, _args: goslice<string>| {
                model::List();
            },
        )),
        ..Default::default()
    }
}

// showCmd represents `kvlm model show <model> [variant]`.
fn showCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("show <model> [variant]"),
        Short: string("Show the serving recipe for a model"),
        Long: string(
            "Show the serving recipe for one model. Bare show prints the\n\
             overview: variants, quantizations, and a hardware support\n\
             matrix. Name a variant to get its serve commands and hardware\n\
             table, or pass --gpu to resolve every variant against one GPU\n\
             type (image, floor, and the serve profile that applies there).",
        ),
        Args: Some(cobra::RangeArgs(1, 2)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: goslice<string>| -> goish::errors::error {
                let (gpu, _) = cmd.Flags().GetString("gpu");
                let mut variant = string("");
                if args.Len() > 1 {
                    variant = args[1usize].clone();
                }
                model::Show(args[0usize].clone(), variant, gpu)
            },
        )),
        ..Default::default()
    };
    c.Flags().StringP(
        string("gpu"),
        string("g"),
        string(""),
        string("resolve recipes for one GPU type (h100, b200, mi300x, ...)"),
    );
    c
}

// vramCmd represents `kvlm model vram <model> [variant]` — the
// context → KV cache → VRAM pipeline calculator.
fn vramCmd() -> cobra::Command {
    let mut c = cobra::Command {
        Use: string("vram <model> [variant]"),
        Short: string("Estimate KV cache and VRAM for a context size"),
        Long: string(
            "Compute the memory pipeline for a model + weight variant:\n\
             context size -> KV cache per sequence -> total VRAM\n\
             (weights + KV pool + 10% overhead), using the model's\n\
             attention architecture (GQA/MLA/sliding-window).",
        ),
        Args: Some(cobra::RangeArgs(1, 2)),
        RunE: Some(alloc::sync::Arc::new(
            |cmd: &mut cobra::Command, args: goslice<string>| -> goish::errors::error {
                let (ctx, _) = cmd.Flags().GetInt("ctx");
                let (seqs, _) = cmd.Flags().GetInt("seqs");
                let mut variant = string("");
                if args.Len() > 1 {
                    variant = args[1usize].clone();
                }
                model::VRAM(args[0usize].clone(), variant, ctx, seqs)
            },
        )),
        ..Default::default()
    };
    c.Flags().IntP(
        string("ctx"),
        string("c"),
        32768,
        string("context length in tokens per sequence"),
    );
    c.Flags().IntP(
        string("seqs"),
        string("s"),
        1,
        string("concurrent sequences sharing the KV pool"),
    );
    c
}

// Go: func init() { modelCmd.AddCommand(lsCmd, showCmd, vramCmd); rootCmd.AddCommand(modelCmd) }
#[goish::init]
fn init() {
    let mut m = modelCmd();
    m.AddCommand(lsCmd());
    m.AddCommand(showCmd());
    m.AddCommand(vramCmd());
    rootCmd.Lock().AddCommand(m);
}
