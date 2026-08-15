// probe: the batch-size probe `kvlm profile run --probe` drives during
// a capture window. The batch-ratio verdicts in `profile graph` need
// the same executable graph replayed at two different batch sizes
// inside one window; a steady load pins the decode batch to a single
// capture bucket, so the probe holds two deliberate concurrency
// plateaus (low, then high) to force two FULL graphs into the trace.
//
// The script is a constant: every parameter arrives as a positional
// argument, so nothing is interpolated into it and it survives any
// ssh quoting (it travels base64-encoded anyway).
#![allow(non_snake_case)]

use goish::string;

// ProbeScript returns the POSIX sh probe. Usage on the target:
//   sh probe.sh <addr> <model> <concurrency> <seconds> <max_tokens>
// It holds <concurrency> completion loops against the OpenAI-style
// endpoint for <seconds>, each request a unique prompt (no prefix-
// cache shortcut), then waits for stragglers.
pub fn ProbeScript() -> string {
    string(
        "#!/bin/sh\n\
         ADDR=\"$1\"; MODEL=\"$2\"; CONC=\"$3\"; SECS=\"$4\"; MAXTOK=\"$5\"\n\
         END=$(( $(date +%s) + SECS ))\n\
         i=0\n\
         while [ \"$i\" -lt \"$CONC\" ]; do\n\
           (\n\
             while [ \"$(date +%s)\" -lt \"$END\" ]; do\n\
               curl -s -m 120 -X POST -H \"Content-Type: application/json\" \\\n\
                 -d \"{\\\"model\\\":\\\"$MODEL\\\",\\\"prompt\\\":\\\"kvlm probe $$ $i $(date +%s%N)\\\",\\\"max_tokens\\\":$MAXTOK,\\\"temperature\\\":0.8}\" \\\n\
                 \"http://$ADDR/v1/completions\" > /dev/null 2>&1\n\
             done\n\
           ) &\n\
           i=$((i+1))\n\
         done\n\
         wait\n",
    )
}
