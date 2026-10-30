# Welvet sweep → SNAT

Apache-2.0 Go producer for Welvet 1.1.3 and the SNAT viewer inside Fable.
TCP manages sessions; TANHI UDP carries native timings and measured neural data.
No screen recording is performed.

## Run

1. Launch Fable, open **Neural Wave / SNAT**, select **Start TCP**, and leave
   **Follow producer** enabled. This starts TCP `127.0.0.1:9731` and TANHI UDP
   `127.0.0.1:17481`. Pin a session to keep it in the main view.
2. In this folder run `./run.sh`.
3. Open <http://127.0.0.1:9732> for each configuration's name, explanation,
   percentage, outcomes and pause/resume control. Ctrl+C stops the producer.

**The default is the full exhaustive run**, as requested. It is not a quick demo.
No catalogue cases run when you use `./run.sh --plan`.
For an explicitly bounded first check:

```sh
./run.sh --data runs/trial --layers dense --dtypes float32 --modes NormalBP --max-grid 1 --limit 2
```

Checkpoints and JSONL outcomes are saved under `runs/`; Fable independently saves
session folders under its configured data root. Restart with the same arguments
to resume. Use `--new-run` for a fresh run. A different catalogue/recipe needs a
new run or data folder. Results distinguish completed, unsupported and failed
configurations. The producer waits for Fable to start, and waits for an active
following viewer to consume each session before advancing. A pinned viewer
allows the producer to continue.

## Finite exhaustive catalogue

The current product is **603,733,716 named cases**, generated lazily:

- 34 native weight dtypes × 23 fixed tiny layer recipes × 34 training modes.
- All 27 rectangular volume shapes with each axis 1..3, up to 3×3×3.
- 841 topologies: single, plus 20 cameral policies × 7 combine modes ×
  1..3 branches × nesting depths 1..2.

An initial 841-case coverage pass rotates across the axes before visiting every
remaining product member exactly once. `--profile showcase` limits execution to
that opening pass; the default `exhaustive` continues through the product.
Use `--plan` to list actual axes without connecting or running any models.
This covers this finite recipe catalogue, not every possible graph, dimension,
hyperparameter or heterogeneous branch composition. Examples use CPU tiled
execution, FormatNone and fixed small tensors; unsupported Welvet combinations
are recorded honestly rather than silently substituted.

## Trace meaning

Native Welvet TANHI reports runtime timings and shapes. Additional diagnostic
forwards record actual measured activations. Backward-coloured sampled weight
updates are labelled parameter deltas, not gradients. Diagnostic branch forwards
are additional computations, rather than a claimed training tape. Setup geometry
travels over TCP without sampled weights; sampled tensors travel over UDP.

UDP samples are sequenced, deduplicated and repaired via UDP retransmission.
TCP flush acknowledgements contain only management counts. Native timing gaps
are reported in trace-quality statistics; missing native packets are not
invented. The dashboard exposes sample/native gap counts and partial traces.

`SNAT_TCP_TOKEN` supplies authentication for an explicitly configured remote TCP
listener; configure Fable's matching token. Loopback defaults need no token.
Ports can be changed using `--tcp` and `--web`; UDP is discovered from SNAT.

## Development

`go test ./...` runs bounded model and transport checks, not the full sweep.
`go test -race ./...` additionally checks concurrent dashboard/runner access.
The sibling SNAT checkout is used during development via the Go module replace.
