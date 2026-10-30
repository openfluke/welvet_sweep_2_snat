// SPDX-License-Identifier: Apache-2.0
package main

import (
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

var groupTitles = map[string]string{
	"dtypes": "Number formats", "layers": "Network layers", "training": "Learning modes", "parallel": "Parallel networks", "cameral": "Cameral coordination",
}

func (c *Catalogue) UseShowcase() {
	base := Configuration{Recipe: 2, DType: c.DTypes[0], Layer: c.Layers[0], Mode: c.Modes[0], Z: 1, Y: 1, X: 1, Topology: Topology{Kind: "single", Combine: "none", Branches: 1, Nesting: 1, Cameral: "none"}}
	for _, dt := range c.DTypes {
		if dt == core.DTypeFloat32 {
			base.DType = dt
		}
	}
	for _, kind := range c.Layers {
		if kind == "dense" {
			base.Layer = kind
		}
	}
	for _, mode := range c.Modes {
		if mode == parallel.ModeNormalBP {
			base.Mode = mode
		}
	}
	c.Showcase = []Configuration{}
	add := func(group, name string, v Configuration) {
		v.Group = group
		v.Example = name
		c.Showcase = append(c.Showcase, v)
	}
	for _, dt := range c.DTypes {
		v := base
		v.DType = dt
		add("dtypes", dt.String(), v)
	}
	for _, kind := range c.Layers {
		v := base
		v.Layer = kind
		// One volumetric example shows the full 3D range without multiplying the plan.
		if kind == "cnn3" {
			v.Z = c.MaxGrid
			v.Y = c.MaxGrid
			v.X = c.MaxGrid
		}
		add("layers", kind, v)
	}
	for _, mode := range c.Modes {
		v := base
		v.Mode = mode
		add("training", mode.String(), v)
	}
	parallelBase := base
	parallelBase.Topology = Topology{Kind: "parallel", Combine: "add", Branches: 2, Nesting: 1, Cameral: "none"}
	for _, combine := range []string{"concat", "add", "avg", "filter", "max", "sparsek", "disagree"} {
		v := parallelBase
		v.Topology.Combine = combine
		add("parallel", "Combine: "+combine, v)
	}
	for branches := 1; branches <= 3; branches++ {
		v := parallelBase
		v.Topology.Branches = branches
		add("parallel", fmt.Sprintf("Branches: %d", branches), v)
	}
	for nesting := 1; nesting <= 2; nesting++ {
		v := parallelBase
		v.Topology.Nesting = nesting
		add("parallel", fmt.Sprintf("Nesting: %d", nesting), v)
	}
	seen := map[string]bool{}
	for _, top := range c.Topologies {
		if seen[top.Cameral] {
			continue
		}
		seen[top.Cameral] = true
		v := parallelBase
		v.Topology.Kind = "cameral"
		v.Topology.Cameral = top.Cameral
		// Native action policies require a step tape.
		if top.Cameral == "action" {
			v.Mode = parallel.ModeStepBP
		}
		add("cameral", top.Cameral, v)
	}
	c.Total = int64(len(c.Showcase))
	c.Prelude = nil
	c.sorted = nil
}
func (c *Catalogue) SessionCount() int {
	if c.Showcase == nil {
		return int(c.Total)
	}
	seen := map[string]bool{}
	for _, v := range c.Showcase {
		seen[v.Group] = true
	}
	return len(seen)
}
func (c Configuration) ExampleName() string {
	if c.Example != "" {
		return c.Example + " · " + c.Name()
	}
	return c.Name()
}
