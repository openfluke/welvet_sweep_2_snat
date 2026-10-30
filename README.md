# Welvet → Neural Replay

Apache-2.0 Go showcase for Welvet 1.1.3 and Fable's Neural Replay panel,
backed by SNAT. TCP manages sessions; TANHI UDP carries neural telemetry.

## Run

1. Start Fable, open **Neural Replay**, and press **Start server**. Leave
   **Follow producer** enabled.
2. In `/home/openfluke/gs/welvet_sweep_2_snat`, run:

   ```sh
   go run .
   ```

3. Open <http://127.0.0.1:9732>. Enter Fable's IP address and TCP port
   (`127.0.0.1`, `9731` on the same machine), then press **Start showcase**.

Nothing runs before Start. The website shows the current chapter's name,
explanation, percentage, outcomes and pause/resume control. Ctrl+C stops it.
`./run.sh` and `make run` provide the same defaults.

## Default: five sessions, 123 named chapters

| Saved session | Examples |
| --- | ---: |
| Number formats | 34 |
| Network layers | 23 |
| Learning modes | 34 |
| Parallel networks | 12 |
| Cameral coordination | 20 |
| Total | 123 |

Each chapter demonstrates one feature with other settings fixed. Parallel
examples cover seven combiners, branch counts 1–3 and nesting depths 1–2.
One CNN3 example uses the largest requested volume (3×3×3 by default).
This is feature coverage rather than every interaction between features.
Unsupported combinations are labelled honestly, with the underlying error.

The five sessions are created as their groups begin. Fable's chapter selector
jumps to the corresponding network geometry and recorded timeline. Playback
switches geometry at chapter boundaries. Pin a session for manual exploration;
Follow producer watches the live tour. A group marked complete means its tour
finished; each chapter retains its own completed/unsupported/failed outcome.

Producer checkpoints and JSONL outcomes live under `runs/showcase/`; Fable
independently saves its session folders under its configured data root. Restart
with the same options, then press Start to resume. Use `--new-run` for a fresh
tour, or `--data` for an independent checkpoint. An unfinished last chapter is
retried from its saved boundary without duplicating earlier chapters. Changing
the catalogue/recipe requires a separate run. `--limit` caps examples per launch;
a group stopped midway remains running/resumable until its remaining chapters
finish. The producer waits for an active following viewer to consume each chapter
before advancing. Pinning the viewer permits the producer to continue.

## Optional commands

```sh
# Counts only: no models, listeners or session writes.
go run . --plan

# Tiny bounded check, still started from the website.
go run . --data runs/trial --layers dense --modes NormalBP --max-grid 1 --limit 2

# Explicit automatic execution for scripts (skips the Start form).
go run . --autostart --limit 2 --data runs/script-trial

# Inspect the exhaustive plan without executing it.
go run . --profile exhaustive --plan
```

`--profile exhaustive` explicitly opts into the **603,733,716-case** Cartesian
catalogue: 34 dtypes × 23 layer recipes × 34 training modes × 27 grid shapes ×
841 topologies. It saves one session per case and uses a separate
`runs/exhaustive/` checkpoint by default. Its initial coverage pass changes order,
not membership. This mode is never selected by an ordinary `go run .`.

The finite recipe catalogue uses CPU tiled execution, FormatNone, small fixed
native layer examples, homogeneous branch leaves and nesting up to depth 2.
It does not enumerate every imaginable graph or hyperparameter.

## Transport and trace meaning

Fable starts TCP `127.0.0.1:9731` for session/chapter management and TANHI UDP
`127.0.0.1:17481` for native timings and measured tensor samples. UDP is discovered
from TCP; setup geometry has no sampled weights. For LAN use, configure Fable's
Server settings and shared token, then enter the matching IP/port/token on the
website. `--tcp` pre-fills the website address; `--web` changes its listening port.
The form's token can be omitted when using an existing `SNAT_TCP_TOKEN` variable.

Native Welvet TANHI reports timings/shapes. Additional diagnostic forwards record
measured activations. Sampled parameter deltas are labelled as weight updates,
not gradients; diagnostic branch forwards are extra computations, not a claimed
training tape. UDP samples are sequenced, deduplicated and repaired over UDP.
TCP flush receipts carry counts and missing sequence numbers. Native timing loss
is reported in trace-quality statistics; packets are never invented. No screen
recording is performed.

## Development

`go test ./...` and `make check` run bounded model, transport, chapter and website
checks. They do not execute the exhaustive sweep. The local SNAT sibling checkout
is currently required through the module's `replace` while chapters are being
developed. The renderer and Flutter controls remain in Fable.
