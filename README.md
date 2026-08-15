# kvlm

Deploy, profile, and tune vLLM serving from one CLI. Every feature
ends in a vLLM flag or a kernel pointer: kvlm measures a server,
names what bounds it, proposes a concrete flag change with the
arithmetic shown, applies it, and measures again.

Written in Goish Rust (Go semantics on a freestanding Rust runtime),
on the [goish](https://github.com/cogentica-ai/goish) runtime and the
[spf13-cobra port](https://github.com/cogentica-ai/ported-crates).

## The loop

```
kvlm up qwen3-32b --mode profile      # pod with vLLM under the profilers
kvlm tune --goal total@32             # collect, verdict, recommendation
kvlm apply max-num-seqs=64            # restart with the change, ~90 s
kvlm tune                             # measure the change against its parent
kvlm ship --up                        # deploy the flags the loop earned
kvlm down
```

Each `tune` collects a full revision under `profile-output/runN`: an
nsys node-mode window, a torch profiler window, a concurrency sweep, a
KV pressure probe, live metrics, the server's complete flag state, and
a CUDA-graph analysis with per-node verdicts. Revisions chain: run2
records that it is run1 plus `max-num-seqs=64`, and `kvlm run diff`
shows flags first, then what they changed.

## Commands

- `up` / `down` / `ps` / `status` / `logs` / `ssh` - pod lifecycle on
  RunPod (Kubernetes driver is a stub). `ps` shows cost per hour and
  total burn; `up` records its pod so nothing needs retyping.
- `tune` / `apply` / `why` / `ship` - the tuning loop. `why` prints
  the evidence behind a verdict, number by number.
- `run ls|show|diff|graph|flamegraph|pack|import` - collected
  revisions as durable objects; `.kvlm` archives are self-contained.
- `model ls|show|vram` - the serving catalog: recipes, quantizations,
  KV-cache arithmetic per attention architecture (GQA, MLA, sliding
  window, linear-attention hybrids).
- `profile run [tool]` / `profiler ls|show` - the collector and the
  tool registry behind it.
- `dash` - local dashboard over the collected revisions.

## Build

```
cargo build --release        # or: make install
make test                    # builds and runs every suite
```

Credentials live in `~/.kvlm/config.json` (`drivers.runpod.api_key`)
or `RUNPOD_API_KEY`; secrets never travel on command lines. Profile
mode needs an ssh keypair the pod can accept (`~/.runpod/id_ed25519`
or `~/.ssh/id_ed25519.pub`).
