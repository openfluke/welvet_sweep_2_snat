// SPDX-License-Identifier: Apache-2.0
package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"sort"
	"strings"
)

type Topology struct {
	Kind     string `json:"kind"`
	Combine  string `json:"combine"`
	Branches int    `json:"branches"`
	Nesting  int    `json:"nesting"`
	Cameral  string `json:"cameral"`
}
type Configuration struct {
	Recipe   int                `json:"recipe"`
	DType    core.DType         `json:"dtype"`
	Layer    string             `json:"layer"`
	Mode     parallel.TrainMode `json:"mode"`
	Z        int                `json:"depth"`
	Y        int                `json:"rows"`
	X        int                `json:"cols"`
	Topology Topology           `json:"topology"`
}

func (c Configuration) ID() string {
	b, _ := json.Marshal(c)
	return fmt.Sprintf("%x", sha256.Sum256(b))[:20]
}
func (c Configuration) Name() string {
	return fmt.Sprintf("%s · %s · %s · %dx%dx%d · %s/%s/%d branches/depth %d · %s", c.Layer, c.DType, c.Mode, c.Z, c.Y, c.X, c.Topology.Kind, c.Topology.Combine, c.Topology.Branches, c.Topology.Nesting, c.Topology.Cameral)
}
func (c Configuration) Description() string {
	return fmt.Sprintf("Tiny CPU example of %s with native %s weight storage; %s credit on a %dx%dx%d grid. %s routing, %d branch(es), nesting %d, cameral policy %s. Float32 activation carrier; measured outputs and weight deltas are streamed separately.", c.Layer, c.DType, c.Mode, c.Z, c.Y, c.X, c.Topology.Combine, c.Topology.Branches, c.Topology.Nesting, c.Topology.Cameral)
}

type Catalogue struct {
	DTypes     []core.DType         `json:"dtypes"`
	Layers     []string             `json:"layers"`
	Modes      []parallel.TrainMode `json:"modes"`
	Topologies []Topology           `json:"topologies"`
	MaxGrid    int                  `json:"max_grid"`
	Total      int64                `json:"total"`
	Prelude    []int64              `json:"coverage_prelude"`
	sorted     []int64
}

func NewCatalogue(maxGrid int) *Catalogue {
	c := &Catalogue{DTypes: append([]core.DType(nil), core.AllDTypes...), Modes: parallel.AllTrainModes(), MaxGrid: maxGrid}
	for _, k := range polyKinds() {
		c.Layers = append(c.Layers, k.name)
	}
	c.Layers = append(c.Layers, "parallel", "stack")
	c.Topologies = []Topology{{Kind: "single", Combine: "none", Branches: 1, Nesting: 1, Cameral: "none"}}
	policies := []string{"none", "independent", "freeze", "shadow", "adversarial", "memory", "dna_attract", "dna_diversify", "dream", "rotate", "soft_sync", "hard_sync", "group_sync", "cross_sync", "asymmetric_sync", "sync_after_sample", "sync_after_step", "sync_after_pulse", "per_cam_lr", "action"}
	for _, policy := range policies {
		for _, combine := range []string{"concat", "add", "avg", "filter", "max", "sparsek", "disagree"} {
			for b := 1; b <= 3; b++ {
				for n := 1; n <= 2; n++ {
					kind := "parallel"
					if policy != "none" {
						kind = "cameral"
					}
					c.Topologies = append(c.Topologies, Topology{Kind: kind, Combine: combine, Branches: b, Nesting: n, Cameral: policy})
				}
			}
		}
	}
	c.rebuild()
	return c
}
func (c *Catalogue) rebuild() {
	dims := []int{len(c.Layers), len(c.DTypes), len(c.Modes), c.MaxGrid * c.MaxGrid * c.MaxGrid, len(c.Topologies)}
	c.Total = 1
	maxDim := 0
	for _, n := range dims {
		c.Total *= int64(n)
		if n > maxDim {
			maxDim = n
		}
	}
	// A coverage-first traversal changes order, not membership. Afterwards visit
	// every remaining Cartesian entry exactly once, without allocating the matrix.
	seen := map[int64]bool{}
	c.Prelude = nil
	for i := 0; i < maxDim; i++ {
		var raw, stride int64 = 0, 1
		for _, n := range dims {
			raw += int64(i%n) * stride
			stride *= int64(n)
		}
		if !seen[raw] {
			c.Prelude = append(c.Prelude, raw)
			seen[raw] = true
		}
	}
	c.sorted = append([]int64(nil), c.Prelude...)
	sort.Slice(c.sorted, func(i, j int) bool { return c.sorted[i] < c.sorted[j] })
}
func (c *Catalogue) At(index int64) Configuration {
	if index < 0 || index >= c.Total {
		panic("catalogue index out of range")
	}
	raw := index
	if index < int64(len(c.Prelude)) {
		raw = c.Prelude[index]
	} else {
		raw = index - int64(len(c.Prelude))
		for _, skip := range c.sorted {
			if skip <= raw {
				raw++
			} else {
				break
			}
		}
	}
	pick := func(n int) int { v := int(raw % int64(n)); raw /= int64(n); return v }
	l := pick(len(c.Layers))
	dt := pick(len(c.DTypes))
	m := pick(len(c.Modes))
	g := pick(c.MaxGrid * c.MaxGrid * c.MaxGrid)
	top := pick(len(c.Topologies))
	return Configuration{Recipe: 1, Layer: c.Layers[l], DType: c.DTypes[dt], Mode: c.Modes[m], X: g%c.MaxGrid + 1, Y: (g/c.MaxGrid)%c.MaxGrid + 1, Z: g/(c.MaxGrid*c.MaxGrid) + 1, Topology: c.Topologies[top]}
}
func (c *Catalogue) Filter(dtypes, layers, modes string) error {
	filter := func(spec string, values []string) (map[string]bool, error) {
		m := map[string]bool{}
		if spec == "" {
			return m, nil
		}
		for _, v := range strings.Split(spec, ",") {
			v = strings.TrimSpace(v)
			ok := false
			for _, s := range values {
				if strings.EqualFold(s, v) {
					m[s] = true
					ok = true
				}
			}
			if !ok {
				return nil, fmt.Errorf("unknown filter %q; choices: %s", v, strings.Join(values, ","))
			}
		}
		return m, nil
	}
	names := []string{}
	for _, v := range c.DTypes {
		names = append(names, v.String())
	}
	set, e := filter(dtypes, names)
	if e != nil {
		return e
	}
	if len(set) > 0 {
		out := c.DTypes[:0]
		for _, v := range c.DTypes {
			if set[v.String()] {
				out = append(out, v)
			}
		}
		c.DTypes = out
	}
	set, e = filter(layers, c.Layers)
	if e != nil {
		return e
	}
	if len(set) > 0 {
		out := []string{}
		for _, v := range c.Layers {
			if set[v] {
				out = append(out, v)
			}
		}
		c.Layers = out
	}
	names = nil
	for _, v := range c.Modes {
		names = append(names, v.String())
	}
	set, e = filter(modes, names)
	if e != nil {
		return e
	}
	if len(set) > 0 {
		out := c.Modes[:0]
		for _, v := range c.Modes {
			if set[v.String()] {
				out = append(out, v)
			}
		}
		c.Modes = out
	}
	c.rebuild()
	return nil
}
