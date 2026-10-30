// SPDX-License-Identifier: Apache-2.0
package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/openfluke/snat"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/systems/tanhi"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Result struct {
	Telemetry snat.UDPStats `json:"telemetry"`
	Index     int64         `json:"index"`
	Name      string        `json:"name"`
	ID        string        `json:"id"`
	SessionID string        `json:"session_id"`
	Status    string        `json:"status"`
	Error     string        `json:"error,omitempty"`
	Duration  float64       `json:"duration_seconds"`
}
type Dashboard struct {
	Telemetry      snat.UDPStats  `json:"telemetry"`
	Progress       snat.Progress  `json:"progress"`
	Configuration  *Configuration `json:"configuration"`
	DType          string         `json:"dtype"`
	Mode           string         `json:"mode"`
	TCP            string         `json:"tcp"`
	Connected      bool           `json:"connected"`
	Paused         bool           `json:"paused"`
	RunID          string         `json:"run_id"`
	CatalogueTotal int64          `json:"catalogue_total"`
	OverallPercent float64        `json:"overall_percent"`
	Recent         []Result       `json:"recent"`
	Started        time.Time      `json:"started"`
}
type checkpoint struct {
	Fingerprint string `json:"fingerprint"`
	RunID       string `json:"run_id"`
	Next        int64  `json:"next"`
	Succeeded   int64  `json:"succeeded"`
	Unsupported int64  `json:"unsupported"`
	Failed      int64  `json:"failed"`
}
type Runner struct {
	mu               sync.Mutex
	state            Dashboard
	paused           atomic.Bool
	catalogue        *Catalogue
	cp               checkpoint
	tcp, token, data string
	samples          int
	lr               float64
	hold, delay      time.Duration
	limit, start     int64
	profile          string
	client           *snat.Client
}

func writeAtomic(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".checkpoint-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err = f.Write(b); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}
func (r *Runner) initialize(newRun bool) error {
	b, _ := json.Marshal(struct {
		Catalogue *Catalogue
		Samples   int
		LR        float64
	}{r.catalogue, r.samples, r.lr})
	fingerprint := fmt.Sprintf("%x", sha256.Sum256(b))
	path := filepath.Join(r.data, "checkpoint.json")
	if !newRun {
		raw, err := os.ReadFile(path)
		if err == nil {
			if err = json.Unmarshal(raw, &r.cp); err != nil {
				return err
			}
			if r.cp.Fingerprint != fingerprint {
				return fmt.Errorf("saved run uses a different catalogue/recipe; use --new-run or restore its options")
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	if r.cp.RunID == "" {
		var id [12]byte
		if _, err := rand.Read(id[:]); err != nil {
			return err
		}
		r.cp = checkpoint{Fingerprint: fingerprint, RunID: hex.EncodeToString(id[:]), Next: r.start}
	}
	if r.start > r.cp.Next {
		r.cp.Next = r.start
	}
	if err := writeAtomic(path, r.cp); err != nil {
		return err
	}
	r.state = Dashboard{RunID: r.cp.RunID, TCP: r.tcp, CatalogueTotal: r.catalogue.Total, Recent: []Result{}, Started: time.Now().UTC()}
	return nil
}
func (r *Runner) snapshot() Dashboard {
	r.mu.Lock()
	defer r.mu.Unlock()
	s := r.state
	s.Paused = r.paused.Load()
	s.Recent = append([]Result(nil), s.Recent...)
	return s
}
func (r *Runner) setPhase(phase string) { r.mu.Lock(); r.state.Progress.Status = phase; r.mu.Unlock() }
func (r *Runner) wait(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
func (r *Runner) waitPause(ctx context.Context) error {
	for r.paused.Load() {
		if err := r.wait(ctx, 100*time.Millisecond); err != nil {
			return err
		}
	}
	return ctx.Err()
}
func (r *Runner) send(ctx context.Context, m snat.WireMessage) (snat.WireReply, error) {
	for {
		if err := r.waitPause(ctx); err != nil {
			return snat.WireReply{}, err
		}
		if r.client == nil {
			dialCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
			c, err := snat.Dial(dialCtx, r.tcp, r.token)
			cancel()
			if err != nil {
				if strings.HasPrefix(err.Error(), "SNAT:") {
					return snat.WireReply{}, err
				}
				r.mu.Lock()
				r.state.Connected = false
				r.state.Progress.Status = "waiting_for_snat"
				r.mu.Unlock()
				if err = r.wait(ctx, 500*time.Millisecond); err != nil {
					return snat.WireReply{}, err
				}
				continue
			}
			r.client = c
		}
		reply, err := r.client.Send(ctx, m)
		if err == nil {
			r.mu.Lock()
			r.state.Connected = true
			r.mu.Unlock()
			return reply, nil
		}
		if strings.HasPrefix(err.Error(), "SNAT:") {
			return reply, err
		}
		r.client.Close()
		r.client = nil
		r.mu.Lock()
		r.state.Connected = false
		r.mu.Unlock()
		if err = r.wait(ctx, 300*time.Millisecond); err != nil {
			return reply, err
		}
	}
}
func (r *Runner) run(ctx context.Context) error {
	defer func() {
		if r.client != nil {
			r.client.Close()
		}
	}()
	end := r.catalogue.Total
	if r.profile == "showcase" {
		end = int64(len(r.catalogue.Prelude))
	}
	if r.limit > 0 {
		end = min(end, r.cp.Next+r.limit)
	}
	report, err := os.OpenFile(filepath.Join(r.data, r.cp.RunID+"-results.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer report.Close()
	for index := r.cp.Next; index < end; index++ {
		c := r.catalogue.At(index)
		p := snat.Progress{RunID: r.cp.RunID, Name: c.Name(), Description: c.Description(), Index: index + 1, Total: end, Completed: index, Succeeded: r.cp.Succeeded, Unsupported: r.cp.Unsupported, Failed: r.cp.Failed, Status: "preparing"}
		r.mu.Lock()
		r.state.Progress = p
		r.state.Configuration = &c
		r.state.DType = c.DType.String()
		r.state.Mode = c.Mode.String()
		r.mu.Unlock()
		if _, err = r.send(ctx, snat.WireMessage{Op: "progress", Progress: &p}); err != nil {
			return err
		}
		started := time.Now()
		m, buildErr := safeBuild(c)
		layers := []snat.Layer{}
		if m != nil {
			layers = m.layers()
		}
		if len(layers) == 0 {
			layers = declaration(c)
		}
		session := &snat.Session{Title: c.Name(), Model: "Welvet " + c.Layer, Source: "welvet-sweep", Description: c.Description(), Layers: layers, Metadata: map[string]any{"configuration": c, "configuration_id": c.ID(), "welvet": "1.1.3", "trace_contract": "TCP management; native tanhi1 UDP timing plus session-tagged TANHI UDP measured activation and parameter-delta events; real shape adapters"}}
		created, err := r.send(ctx, snat.WireMessage{Op: "create", Key: r.cp.RunID + "/" + c.ID(), Session: session, Progress: &p})
		if err != nil {
			return err
		}
		sender, err := r.client.TANHI(created.ID, created.StreamToken)
		if err != nil {
			return err
		}
		host, port, e := net.SplitHostPort(sender.Address())
		if e != nil {
			sender.Close()
			return e
		}
		udpPort, _ := strconv.Atoi(port)
		nativeConfig := &tanhi.UDPConfig{Enabled: true, Host: host, Port: udpPort, SendShape: true}
		if buildErr == nil {
			m.grid.Tanhi = nativeConfig
			parallel.SyncGridTanhi(m.grid)
			tanhi.EmitSweep(nativeConfig, sender.StartMarker())
		}
		eventCount := 0
		pending := []snat.Event{}
		flush := func() error {
			if len(pending) == 0 {
				return nil
			}
			for _, event := range pending {
				if e := sender.SendEvent(event); e != nil {
					return e
				}
			}
			receipt, e := sender.Flush(ctx, false)
			if e != nil {
				return e
			}
			if _, e = r.send(ctx, snat.WireMessage{Op: "progress", Progress: &p}); e != nil {
				return e
			}
			r.mu.Lock()
			r.state.Telemetry = receipt.Telemetry
			r.mu.Unlock()
			eventCount += len(pending)
			pending = nil
			return r.wait(ctx, r.delay)
		}
		emit := func(events []snat.Event) error {
			for _, e := range events {
				pending = append(pending, e)
				if len(pending) >= 32 {
					if err := flush(); err != nil {
						return err
					}
				}
			}
			return nil
		}
		var ioErr error
		progress := func(percent float64) {
			p.Percent = percent
			p.Status = "streaming"
			r.mu.Lock()
			r.state.Progress = p
			r.mu.Unlock()
			if ioErr == nil {
				ioErr = flush()
			}
		}
		runErr := buildErr
		if runErr == nil {
			runErr = emit(initialWeightEvents(m, c))
		}
		if runErr == nil {
			runErr = safeExample(ctx, c, m, r.samples, r.lr, emit, progress)
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if ioErr != nil {
			return ioErr
		}
		if err = flush(); err != nil {
			return err
		}
		if buildErr == nil {
			tanhi.EmitSweep(nativeConfig, sender.EndMarker())
		}
		receipt, err := sender.Flush(ctx, buildErr == nil)
		if err != nil {
			sender.Close()
			return err
		}
		sender.Close()
		eventCount = receipt.EventCount
		r.mu.Lock()
		r.state.Telemetry = receipt.Telemetry
		r.mu.Unlock()
		status := classifyFailure(runErr)
		p.Status = status
		p.Percent = 100
		if runErr != nil {
			p.Error = runErr.Error()
		}
		switch status {
		case "complete":
			r.cp.Succeeded++
		case "unsupported":
			r.cp.Unsupported++
		default:
			r.cp.Failed++
		}
		p.Succeeded = r.cp.Succeeded
		p.Unsupported = r.cp.Unsupported
		p.Failed = r.cp.Failed
		p.Completed = index + 1
		if _, err = r.send(ctx, snat.WireMessage{Op: "finish", ID: created.ID, ExpectedPackets: receipt.Telemetry.Samples + receipt.Telemetry.SampleMissing, Status: status, Progress: &p}); err != nil {
			return err
		}
		r.mu.Lock()
		r.state.Progress = p
		r.mu.Unlock()
		// Do not skip a visible session while Fable is still consuming its events.
		for eventCount > 0 {
			reply, e := r.send(ctx, snat.WireMessage{Op: "view"})
			if e != nil {
				return e
			}
			if !reply.ViewerActive || !reply.Following || (reply.ViewedID == created.ID && reply.ViewedEvents >= eventCount) {
				break
			}
			r.setPhase("waiting_for_viewer")
			if e = r.wait(ctx, 100*time.Millisecond); e != nil {
				return e
			}
		}
		result := Result{Telemetry: receipt.Telemetry, Index: index + 1, Name: c.Name(), ID: c.ID(), SessionID: created.ID, Status: status, Duration: time.Since(started).Seconds()}
		if runErr != nil {
			result.Error = runErr.Error()
		}
		if err = json.NewEncoder(report).Encode(result); err != nil {
			return err
		}
		if err = report.Sync(); err != nil {
			return err
		}
		r.mu.Lock()
		r.state.Recent = append([]Result{result}, r.state.Recent...)
		if len(r.state.Recent) > 20 {
			r.state.Recent = r.state.Recent[:20]
		}
		r.state.OverallPercent = float64(index+1) / float64(end) * 100
		r.state.Progress = p
		r.mu.Unlock()
		fmt.Printf("[%d/%d] %s %s\n", index+1, end, status, c.Name())
		r.cp.Next = index + 1
		if err = writeAtomic(filepath.Join(r.data, "checkpoint.json"), r.cp); err != nil {
			return err
		}
		if err = r.wait(ctx, r.hold); err != nil {
			return err
		}
	}
	r.setPhase("finished")
	return nil
}
func safeBuild(c Configuration) (m *model, err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("construction panic: %v", p)
		}
	}()
	return buildModel(c)
}
func safeExample(ctx context.Context, c Configuration, m *model, samples int, lr float64, emit func([]snat.Event) error, progress func(float64)) (err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("runtime panic: %v", p)
		}
	}()
	return runExample(ctx, c, m, samples, lr, emit, progress)
}
func declaration(c Configuration) []snat.Layer {
	out := []snat.Layer{}
	for z := 0; z < c.Z; z++ {
		for y := 0; y < c.Y; y++ {
			for x := 0; x < c.X; x++ {
				out = append(out, snat.Layer{ID: fmt.Sprintf("grid/%d/%d/%d/0", z, y, x), Name: c.Layer, Type: c.Layer, GridDepth: z, GridRow: y, GridCol: x})
			}
		}
	}
	return out
}
