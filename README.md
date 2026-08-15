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
kvlm up qwen3.8-27b --mode profile    # pod with vLLM under the profilers
kvlm tune --goal total@32             # collect, verdict, recommendation
kvlm apply max-num-seqs=64            # restart with the change, a few minutes
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

## The catalog carries measurements, not guesses

`kvlm model show qwen3.8-27b` prints what the catalog knows about
Qwen 3.8 27B, the model the loop above was proven on:

```
Size:     27B (dense, multimodal)
Context:  256K native, 976.6K max (YaRN)
KV/seq:   8.7 GB at 256K (KDA, fp8 KV)

VARIANT   WEIGHTS   PROD GPU   KV POOL   TOK/S   SEQS @ 32K/64K/128K/256K
fp8       28 GB     1xH100     47 GB     79      32 / ~21 / ~10 / ~5
bf16      55 GB     1xH100     20 GB     ~31     16 / ~9 / ~4 / -
nvfp4     25 GB     1xB200     159 GB    ~120    32 / 32 / 32 / ~18
```

A tilde marks an estimate from the KV arithmetic. A bare number was
measured: fp8 on a RunPod 1xH100 under vLLM 0.26.0 gives 79 tok/s
single-stream decode and 1,442 tok/s aggregate at 32 streams, and the
catalog says so with the date and conditions attached.

That number is 40% above what a dense 27B's memory bandwidth predicts,
because 48 of the model's 64 layers are linear attention and read no
per-token KV. The catalog models that layout (`kvlm model vram`), so
KV pool and resident-sequence counts follow the architecture instead
of a dense approximation.

Live loop result on this model: run1 measured 1,226 tok/s at 32
streams and named `--max-num-seqs` as the bound; `kvlm apply
max-num-seqs=64` and a second run cleared it, taking the pressure
probe from 32 running with 15 queued to 48 running with none, and
mean time-to-first-token from 1,255 ms to 221 ms.

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
