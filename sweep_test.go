// SPDX-License-Identifier: Apache-2.0
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/openfluke/snat"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/systems/dna"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func baseCase() Configuration {
	return Configuration{Recipe: 1, DType: core.DTypeFloat32, Layer: "dense", Mode: parallel.ModeNormalBP, Z: 1, Y: 1, X: 1, Topology: Topology{Kind: "single", Combine: "none", Branches: 1, Nesting: 1, Cameral: "none"}}
}
func exerciseTest(t *testing.T, c Configuration) (int, error) {
	t.Helper()
	m, err := safeBuild(c)
	if err != nil {
		return 0, err
	}
	count := 0
	err = safeExample(context.Background(), c, m, 1, .001, func(events []snat.Event) error {
		count += len(events)
		for _, e := range events {
			if e["layer_id"] == nil || e["values_kind"] == nil {
				t.Fatal("unlabelled trace")
			}
		}
		return nil
	}, func(float64) {})
	return count, err
}
func TestEveryLayerRecipeFloat32(t *testing.T) {
	c := NewCatalogue(1)
	for _, kind := range c.Layers {
		t.Run(kind, func(t *testing.T) {
			cfg := baseCase()
			cfg.Layer = kind
			n, err := exerciseTest(t, cfg)
			if err != nil {
				if kind == "gdn" && classifyFailure(err) == "unsupported" {
					t.Log(err)
					return
				}
				t.Fatal(err)
			}
			if n == 0 {
				t.Fatal("no measured events")
			}
		})
	}
}
func TestAllNativeDTypeStorage(t *testing.T) {
	for _, dt := range core.AllDTypes {
		t.Run(dt.String(), func(t *testing.T) {
			c := baseCase()
			c.DType = dt
			m, err := safeBuild(c)
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range dna.CollectStores(m.stacks[0]) {
				if s.DType != dt {
					t.Fatalf("silently used %s instead of %s", s.DType, dt)
				}
			}
			if _, err = exerciseTest(t, c); err != nil {
				t.Fatal(err)
			}
		})
	}
}
func TestAllTrainingModesAndCameralPolicies(t *testing.T) {
	for _, mode := range parallel.AllTrainModes() {
		t.Run(mode.String(), func(t *testing.T) {
			c := baseCase()
			c.Mode = mode
			if _, err := exerciseTest(t, c); err != nil {
				t.Fatal(err)
			}
		})
	}
	policies := map[string]bool{}
	for _, top := range NewCatalogue(1).Topologies {
		if top.Kind == "cameral" {
			policies[top.Cameral] = true
		}
	}
	for policy := range policies {
		t.Run("cam_"+policy, func(t *testing.T) {
			c := baseCase()
			c.Topology = Topology{Kind: "cameral", Combine: "add", Branches: 2, Nesting: 1, Cameral: policy}
			if policy == "action" {
				c.Mode = parallel.ModeStepBP
			}
			if _, err := exerciseTest(t, c); err != nil {
				t.Fatal(err)
			}
		})
	}
}
func TestAllParallelCombinesAndVolumetricCoordinates(t *testing.T) {
	for _, combine := range []string{"concat", "add", "avg", "filter", "max", "sparsek", "disagree"} {
		for branches := 1; branches <= 3; branches++ {
			t.Run(fmt.Sprintf("%s_%d", combine, branches), func(t *testing.T) {
				c := baseCase()
				c.Topology = Topology{Kind: "parallel", Combine: combine, Branches: branches, Nesting: 2, Cameral: "none"}
				if _, err := exerciseTest(t, c); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
	c := baseCase()
	c.Z = 3
	c.Y = 3
	c.X = 3
	c.Mode = parallel.ModeMeshBP
	m, err := safeBuild(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.nodes) != 27 {
		t.Fatal("missing volume cells")
	}
	seen := map[string]bool{}
	for _, l := range m.layers() {
		if seen[l.ID] {
			t.Fatal("duplicate spatial ID")
		}
		seen[l.ID] = true
		if l.GridDepth < 0 || l.GridDepth > 2 {
			t.Fatal("bad depth")
		}
	}
	if _, err = exerciseTest(t, c); err != nil {
		t.Fatal(err)
	}
}
func TestCatalogueCoveragePreludeDoesNotChangeCartesianMembership(t *testing.T) {
	c := NewCatalogue(1)
	c.DTypes = c.DTypes[:2]
	c.Layers = c.Layers[:2]
	c.Modes = c.Modes[:2]
	c.Topologies = c.Topologies[:3]
	c.rebuild()
	if c.Total != 24 {
		t.Fatal(c.Total)
	}
	seen := map[string]bool{}
	for i := int64(0); i < c.Total; i++ {
		id := c.At(i).ID()
		if seen[id] {
			t.Fatal("repeated configuration")
		}
		seen[id] = true
	}
	full := NewCatalogue(3)
	if full.Total != 603733716 {
		t.Fatal(full.Total)
	}
	dtypes, layers, modes, grids, topologies := map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}
	for i := range full.Prelude {
		v := full.At(int64(i))
		dtypes[v.DType.String()] = true
		layers[v.Layer] = true
		modes[v.Mode.String()] = true
		grids[fmt.Sprintf("%d%d%d", v.Z, v.Y, v.X)] = true
		b, _ := json.Marshal(v.Topology)
		topologies[string(b)] = true
	}
	if len(dtypes) != 34 || len(layers) != 23 || len(modes) != 34 || len(grids) != 27 || len(topologies) != 841 {
		t.Fatal("prelude fails to cover a catalogue axis")
	}
}
func TestRunnerTCPAndCheckpointResume(t *testing.T) {
	root := t.TempDir()
	b, _ := json.Marshal(snat.Request{Action: "server_start", Address: "127.0.0.1:0"})
	v, err := snat.Execute(root, b)
	if err != nil {
		t.Fatal(err)
	}
	defer snat.StopLive(root)
	catalogue := NewCatalogue(1)
	if err = catalogue.Filter("float32", "dense", "NormalBP"); err != nil {
		t.Fatal(err)
	}
	catalogue.Topologies = catalogue.Topologies[:1]
	catalogue.rebuild()
	r := &Runner{catalogue: catalogue, tcp: v.(snat.LiveStatus).Address, data: t.TempDir(), samples: 1, lr: .001, profile: "exhaustive"}
	if err = r.initialize(false); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err = r.run(ctx); err != nil {
		t.Fatal(err)
	}
	if r.cp.Next != 1 || r.cp.Succeeded != 1 {
		t.Fatalf("result %+v", r.cp)
	}
	b, _ = json.Marshal(snat.Request{Action: "live_poll"})
	v, err = snat.Execute(root, b)
	if err != nil {
		t.Fatal(err)
	}
	session := v.(snat.LiveStatus).Session
	if session == nil || session.Status != "complete" || len(session.Events) == 0 {
		t.Fatal("missing saved live session")
	}
	resumed := &Runner{catalogue: catalogue, tcp: r.tcp, data: r.data, samples: 1, lr: .001, profile: "exhaustive"}
	if err = resumed.initialize(false); err != nil {
		t.Fatal(err)
	}
	if resumed.cp.Next != 1 || resumed.cp.RunID != r.cp.RunID {
		t.Fatal("checkpoint lost")
	}
	if _, err = os.Stat(filepath.Join(r.data, r.cp.RunID+"-results.jsonl")); err != nil {
		t.Fatal(err)
	}
}
func TestDashboardAndPause(t *testing.T) {
	r := &Runner{state: Dashboard{RunID: "test", Recent: []Result{}}}
	handler := dashboardHandler(r)
	for _, path := range []string{"/", "/api/state"} {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
		if w.Code != 200 {
			t.Fatal(path, w.Code)
		}
	}
	w := httptest.NewRecorder()
	q := httptest.NewRequest("POST", "/api/pause", strings.NewReader(""))
	handler.ServeHTTP(w, q)
	if w.Code != http.StatusNoContent || !r.paused.Load() {
		t.Fatal("pause failed")
	}
	w = httptest.NewRecorder()
	q = httptest.NewRequest("POST", "/api/resume", nil)
	q.Header.Set("Origin", "https://unrelated.example")
	handler.ServeHTTP(w, q)
	if w.Code != http.StatusForbidden {
		t.Fatal("accepted cross-origin control")
	}
}

func TestCompactShowcasePlan(t *testing.T) {
	c := NewCatalogue(3)
	c.UseShowcase()
	if c.Total != 123 || c.SessionCount() != 5 {
		t.Fatalf("plan: %d examples / %d sessions", c.Total, c.SessionCount())
	}
	counts := map[string]int{}
	ids := map[string]bool{}
	dtypes, layers, modes := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for i := int64(0); i < c.Total; i++ {
		v := c.At(i)
		counts[v.Group]++
		if ids[v.ID()] {
			t.Fatal("duplicate chapter id")
		}
		ids[v.ID()] = true
		if v.Group == "dtypes" {
			dtypes[v.DType.String()] = true
		}
		if v.Group == "layers" {
			layers[v.Layer] = true
		}
		if v.Group == "training" {
			modes[v.Mode.String()] = true
		}
	}
	if counts["dtypes"] != 34 || counts["layers"] != 23 || counts["training"] != 34 || counts["parallel"] != 12 || counts["cameral"] != 20 || len(dtypes) != 34 || len(layers) != 23 || len(modes) != 34 {
		t.Fatal(counts)
	}
}
func TestGroupedShowcaseSavesFiveSessions(t *testing.T) {
	root := t.TempDir()
	raw, _ := json.Marshal(snat.Request{Action: "server_start", Address: "127.0.0.1:0"})
	value, err := snat.Execute(root, raw)
	if err != nil {
		t.Fatal(err)
	}
	defer snat.StopLive(root)
	c := NewCatalogue(1)
	c.UseShowcase()
	// Two cheap examples in each group exercise every transition without running
	// the entire catalogue or the entire showcase during this integration check.
	picks := []Configuration{}
	seen := map[string]int{}
	for _, cfg := range c.Showcase {
		if seen[cfg.Group] < 2 {
			v := cfg
			v.Layer = "dense"
			v.DType = core.DTypeFloat32
			v.Mode = parallel.ModeNormalBP
			picks = append(picks, v)
			seen[cfg.Group]++
		}
	}
	c.Showcase = picks
	c.Total = int64(len(picks))
	r := &Runner{catalogue: c, tcp: value.(snat.LiveStatus).Address, data: t.TempDir(), samples: 1, lr: .001, profile: "showcase", limit: 1}
	if err = r.initialize(false); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err = r.run(ctx); err != nil {
		t.Fatal(err)
	}
	if r.cp.Next != 1 {
		t.Fatal("bounded first launch did not stop")
	}
	resumed := &Runner{catalogue: c, tcp: r.tcp, data: r.data, samples: 1, lr: .001, profile: "showcase"}
	if err = resumed.initialize(false); err != nil {
		t.Fatal(err)
	}
	if err = resumed.run(ctx); err != nil {
		t.Fatal(err)
	}
	if resumed.cp.Succeeded != 10 {
		t.Fatalf("resume counts: %+v", resumed.cp)
	}
	raw, _ = json.Marshal(snat.Request{Action: "library"})
	value, err = snat.Execute(root, raw)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(value)
	var library struct {
		Sessions []snat.Summary `json:"sessions"`
	}
	json.Unmarshal(encoded, &library)
	if len(library.Sessions) != 5 {
		t.Fatalf("got %d sessions", len(library.Sessions))
	}
	for _, entry := range library.Sessions {
		raw, _ = json.Marshal(snat.Request{Action: "open", ID: entry.ID})
		v, e := snat.Execute(root, raw)
		if e != nil {
			t.Fatal(e)
		}
		saved := v.(*snat.Session)
		if saved.Status != "complete" || len(saved.Chapters) != 2 {
			t.Fatalf("bad grouped recording: %+v", saved)
		}
		for _, ch := range saved.Chapters {
			if ch.EndEvent <= ch.StartEvent || ch.Status != "complete" {
				t.Fatalf("empty/incomplete chapter: %+v", ch)
			}
		}
	}
}
func TestDashboardStartValidatesAddressAndWaits(t *testing.T) {
	r := &Runner{state: Dashboard{Profile: "showcase", CatalogueTotal: 123, SessionCount: 5}}
	called := 0
	handler := dashboardHandler(r, func(address, token string) error {
		called++
		if address != "127.0.0.1:9731" || token != "test" {
			t.Fatal(address, token)
		}
		return nil
	})
	for _, test := range []struct {
		body string
		code int
	}{
		{`{"host":"not an IP","port":9731}`, 400},
		{`{"host":"127.0.0.1","port":0}`, 400},
		{`{"host":"127.0.0.1","port":9731,"token":"test"}`, 202},
	} {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, httptest.NewRequest("POST", "/api/start", strings.NewReader(test.body)))
		if w.Code != test.code {
			t.Fatalf("%d: %s", w.Code, w.Body.String())
		}
	}
	if called != 1 {
		t.Fatal("invalid requests started a run")
	}
}
