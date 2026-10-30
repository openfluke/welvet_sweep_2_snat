// SPDX-License-Identifier: Apache-2.0
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	r := &Runner{}
	flag.StringVar(&r.tcp, "tcp", "127.0.0.1:9731", "SNAT listener started inside Fable")
	web := flag.String("web", "127.0.0.1:9732", "Local progress website address")
	flag.StringVar(&r.data, "data", "runs", "Checkpoint and JSONL result directory")
	flag.StringVar(&r.profile, "profile", "exhaustive", "exhaustive or showcase (coverage prelude only)")
	flag.IntVar(&r.samples, "samples", 3, "Training samples per case (Step fill ticks are added)")
	flag.Float64Var(&r.lr, "lr", .001, "Tiny example learning rate")
	flag.DurationVar(&r.hold, "hold", time.Second, "Display hold between configurations")
	flag.DurationVar(&r.delay, "event-delay", 20*time.Millisecond, "Delay per acknowledged event batch")
	flag.Int64Var(&r.limit, "limit", 0, "Maximum cases this invocation; 0 means every combination")
	flag.Int64Var(&r.start, "start", 0, "Start at a zero-based catalogue index")
	maxGrid := flag.Int("max-grid", 3, "Maximum size of each grid axis (1..3)")
	dtypes := flag.String("dtypes", "", "Optional comma-separated dtype filter")
	layers := flag.String("layers", "", "Optional comma-separated layer filter")
	modes := flag.String("modes", "", "Optional comma-separated training-mode filter")
	plan := flag.Bool("plan", false, "Print finite catalogue and counts, without running or connecting")
	newRun := flag.Bool("new-run", false, "Start an independent run instead of resuming checkpoint")
	exitOnComplete := flag.Bool("exit-on-complete", false, "Exit instead of retaining the final progress page")
	flag.Parse()
	r.token = os.Getenv("SNAT_TCP_TOKEN")
	fatal := func(e error) { fmt.Fprintln(os.Stderr, e); os.Exit(1) }
	if *maxGrid < 1 || *maxGrid > 3 || r.samples < 1 || r.samples > 100 || r.lr <= 0 || r.limit < 0 || r.start < 0 || r.hold < 0 || r.delay < 0 {
		fatal(fmt.Errorf("invalid sweep bounds"))
	}
	if r.profile != "exhaustive" && r.profile != "showcase" {
		fatal(fmt.Errorf("profile must be exhaustive or showcase"))
	}
	r.catalogue = NewCatalogue(*maxGrid)
	if err := r.catalogue.Filter(*dtypes, *layers, *modes); err != nil {
		fatal(err)
	}
	if r.start >= r.catalogue.Total {
		fatal(fmt.Errorf("start is beyond catalogue"))
	}
	if *plan {
		dt, tm := []string{}, []string{}
		for _, d := range r.catalogue.DTypes {
			dt = append(dt, d.String())
		}
		for _, m := range r.catalogue.Modes {
			tm = append(tm, m.String())
		}
		_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"total": r.catalogue.Total, "dtypes": dt, "layers": r.catalogue.Layers, "modes": tm, "grid_shapes": (*maxGrid) * (*maxGrid) * (*maxGrid), "topologies": len(r.catalogue.Topologies), "early_coverage_cases": len(r.catalogue.Prelude), "recipe": "CPU tiled, native weight dtype, FormatNone, fixed tiny layer examples, homogeneous branch leaf kind, nesting <= 2"})
		return
	}
	if err := r.initialize(*newRun); err != nil {
		fatal(err)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	listener, err := net.Listen("tcp", *web)
	if err != nil {
		fatal(err)
	}
	server := &http.Server{Handler: dashboardHandler(r), ReadHeaderTimeout: 5 * time.Second}
	go func() {
		if e := server.Serve(listener); e != nil && e != http.ErrServerClosed {
			fmt.Fprintln(os.Stderr, e)
			cancel()
		}
	}()
	fmt.Printf("Progress: http://%s\nSNAT TCP: %s\nCatalogue: %d cases; checkpoint: %s/checkpoint.json\n", listener.Addr(), r.tcp, r.catalogue.Total, r.data)
	done := make(chan error, 1)
	go func() { done <- r.run(ctx) }()
	select {
	case err = <-done:
		if err != nil && ctx.Err() == nil {
			r.mu.Lock()
			r.state.Progress.Status = "error"
			r.state.Progress.Error = err.Error()
			r.mu.Unlock()
			fmt.Fprintln(os.Stderr, err)
		}
		if !*exitOnComplete && ctx.Err() == nil {
			<-ctx.Done()
		}
	case <-ctx.Done():
		err = <-done
	}
	shutdown, stop := context.WithTimeout(context.Background(), 3*time.Second)
	defer stop()
	_ = server.Shutdown(shutdown)
	if err != nil && err != context.Canceled {
		fatal(err)
	}
}
