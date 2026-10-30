// SPDX-License-Identifier: Apache-2.0
package main

import (
	"fmt"
	"github.com/openfluke/snat"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/systems/dna"
	"math"
	"reflect"
	"strings"
)

const carrierWidth = 16

type node struct {
	op              any
	layer           snat.Layer
	children        []*node
	lastIn, lastOut *core.Tensor[float32]
}
type model struct {
	grid   *architecture.Grid
	nodes  []*node
	stacks []*parallel.Stack
	cams   []*parallel.Layer
}

func meta(op any) core.Layer {
	if v, ok := op.(interface{ CoreMeta() core.Layer }); ok {
		return v.CoreMeta()
	}
	v := reflect.ValueOf(op)
	if v.Kind() == reflect.Pointer && !v.IsNil() {
		f := v.Elem().FieldByName("Core")
		if f.IsValid() && f.CanInterface() {
			if c, ok := f.Interface().(core.Layer); ok {
				return c
			}
		}
	}
	return core.Layer{Type: core.LayerDense, DType: core.DTypeFloat32}
}
func newNode(op any, name string) *node {
	n := &node{op: op, layer: snat.Layer{Name: name, Type: meta(op).Type.String(), InputShape: []int{carrierWidth}, OutputShape: []int{carrierWidth}}}
	if _, ok := op.(*parallel.View); ok {
		n.layer.Type = "View"
	}
	for _, s := range dna.CollectStores(op) {
		n.layer.Params += int64(s.Rows * s.Cols)
	}
	return n
}
func initWeights(op any) error {
	for _, store := range dna.CollectStores(op) {
		old, err := store.FlattenF32()
		if err != nil {
			return err
		}
		nonzero := false
		for _, v := range old {
			if v != 0 {
				nonzero = true
				break
			}
		}
		if nonzero {
			continue
		}
		data := make([]float32, store.Rows*store.Cols)
		for r := 0; r < store.Rows; r++ {
			for col := 0; col < store.Cols; col++ {
				if r%store.Cols == col {
					data[r*store.Cols+col] = .4
				} else {
					data[r*store.Cols+col] = float32((r+col)%3-1) * .01
				}
			}
		}
		if err = store.SetFromF32(data); err != nil {
			return err
		}
	}
	return nil
}
func projection(in, out int) (*dense.Layer, error) {
	return dense.NewConfigured[float32](in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, nil)
}
func nativeRecipe(kind string) (any, *core.Tensor[float32], error) {
	for _, k := range polyKinds() {
		if k.name == kind {
			branches, _, input, _, err := k.make()
			if err != nil {
				return nil, nil, err
			}
			return branches[0], input, nil
		}
	}
	if kind == "parallel" {
		p, e := parallel.NewConfigured[float32](parallel.Config{Dim: 4, OutFeat: 4, Branches: 2, Combine: parallel.CombineAvg}, core.DTypeFloat32, quant.FormatNone, nil, nil)
		return p, core.NewTensor[float32](1, 4), e
	}
	if kind == "stack" {
		a, e := projection(4, 4)
		if e != nil {
			return nil, nil, e
		}
		b, e := projection(4, 4)
		if e != nil {
			return nil, nil, e
		}
		s, e := parallel.NewStack(a, b)
		return s, core.NewTensor[float32](1, 4), e
	}
	return nil, nil, fmt.Errorf("unregistered layer kind %s", kind)
}
func leafAdapter(kind string) (*node, error) {
	op, input, err := nativeRecipe(kind)
	if err != nil {
		return nil, err
	}
	if err = initWeights(op); err != nil {
		return nil, err
	}
	// Discover the real native output shape before adding explicit carrier
	// projections. All adaptations are real Welvet Ops and appear in metadata.
	_, out, err := forward.Cell(&architecture.Cell{Layer: meta(op), Op: op}, input)
	if err != nil {
		return nil, fmt.Errorf("recipe probe %s: %w", kind, err)
	}
	stem, err := projection(carrierWidth, input.Len())
	if err != nil {
		return nil, err
	}
	view, _ := parallel.NewView(input.Shape...)
	flat, _ := parallel.NewView(1, out.Len())
	head, err := projection(out.Len(), carrierWidth)
	if err != nil {
		return nil, err
	}
	stack, err := parallel.NewStack(stem, view, op, flat, head)
	if err != nil {
		return nil, err
	}
	native := newNode(op, kind)
	native.layer.InputShape = append([]int(nil), input.Shape...)
	native.layer.OutputShape = append([]int(nil), out.Shape...)
	// Preserve real container children for native Sequential/Parallel/Stack recipes.
	decorateChildren(native)
	n := newNode(stack, kind+" + shape adapters")
	n.layer.CombineMode = "sequence"
	n.children = []*node{newNode(stem, "Input projection"), newNode(view, "Input shape"), native, newNode(flat, "Output shape"), newNode(head, "Output projection")}
	return n, nil
}
func decorateChildren(n *node) {
	var ops []any
	switch p := n.op.(type) {
	case *parallel.Stack:
		ops = p.Children
		n.layer.CombineMode = "sequence"
	case *parallel.Layer:
		ops = p.Branches
		n.layer.CombineMode = string(p.Cfg.Combine)
	}
	for i, op := range ops {
		child := newNode(op, fmt.Sprintf("%s %d", meta(op).Type, i+1))
		decorateChildren(child)
		n.children = append(n.children, child)
	}
}
func buildModel(c Configuration) (*model, error) {
	m := &model{grid: architecture.NewGrid(c.Z, c.Y, c.X, 1)}
	for _, coord := range m.grid.HopOrder() {
		var build func(int) (*node, error)
		build = func(depth int) (*node, error) {
			if c.Topology.Kind == "single" || depth == 0 {
				return leafAdapter(c.Layer)
			}
			children := []*node{}
			ops := []any{}
			for b := 0; b < c.Topology.Branches; b++ {
				child, e := build(depth - 1)
				if e != nil {
					return nil, e
				}
				children = append(children, child)
				ops = append(ops, child.op)
			}
			var gate *dense.Layer
			var err error
			if c.Topology.Combine == "filter" {
				gate, err = projection(carrierWidth, len(ops))
				if err != nil {
					return nil, err
				}
			}
			p, err := parallel.NewFromBranches(parallel.Config{Dim: carrierWidth, OutFeat: carrierWidth, Branches: len(ops), Combine: parallel.CombineMode(c.Topology.Combine)}, ops, gate)
			if err != nil {
				return nil, err
			}
			configureCam(p, c)
			m.cams = append(m.cams, p)
			pn := newNode(p, fmt.Sprintf("%s · %s", c.Topology.Kind, c.Topology.Cameral))
			pn.layer.CombineMode = c.Topology.Combine
			pn.children = children
			stem, err := projection(carrierWidth, carrierWidth)
			if err != nil {
				return nil, err
			}
			head, err := projection(p.Cfg.OutDim(), carrierWidth)
			if err != nil {
				return nil, err
			}
			stack, err := parallel.NewStack(stem, p, head)
			if err != nil {
				return nil, err
			}
			if c.Topology.Cameral == "cross_sync" {
				stack.CamSync = &parallel.CamSyncConfig{Enabled: true, Alpha: .1, When: parallel.SyncAfterSample, Cross: []parallel.SyncPair{{A: parallel.SyncEndpoint{StackIdx: 0, Store: 0}, B: parallel.SyncEndpoint{StackIdx: 2, Store: 0}}}}
			}
			n := newNode(stack, "Cameral sandwich")
			n.layer.CombineMode = "sequence"
			n.children = []*node{newNode(stem, "Stem"), pn, newNode(head, "Head")}
			return n, nil
		}
		n, err := build(c.Topology.Nesting)
		if err != nil {
			return m, err
		}
		stack, ok := n.op.(*parallel.Stack)
		if !ok {
			return m, fmt.Errorf("recipe root must be a stack")
		}
		if err = initWeights(stack); err != nil {
			return m, err
		}
		if err = stack.SetDType(c.DType); err != nil {
			return m, fmt.Errorf("unsupported native dtype %s: %w", c.DType, err)
		}
		n.layer.Name = fmt.Sprintf("%s [%d,%d,%d]", c.Layer, coord.Z, coord.Y, coord.X)
		stamp(n, fmt.Sprintf("grid/%d/%d/%d/0", coord.Z, coord.Y, coord.X), coord)
		if err = parallel.PlaceStack(m.grid, coord.Z, coord.Y, coord.X, 0, stack); err != nil {
			return m, err
		}
		m.nodes = append(m.nodes, n)
		m.stacks = append(m.stacks, stack)
	}
	return m, nil
}
func configureCam(p *parallel.Layer, c Configuration) {
	profile := c.Topology.Cameral
	modes := make([]parallel.TrainMode, len(p.Branches))
	for i := range modes {
		modes[i] = parallel.ModeInherit
	}
	kit := parallel.CamKit{}
	switch profile {
	case "independent":
		for i := range modes {
			modes[i] = c.Mode
		}
	case "freeze":
		for i := range modes {
			modes[i] = parallel.ModeFreeze
		}
	case "shadow":
		modes[0] = parallel.ModeShadow
		kit.ShadowCoef = .2
	case "adversarial":
		modes[0] = parallel.ModeAdversarial
	case "memory":
		modes[0] = parallel.ModeMemory
		kit.SurpriseThresh = .01
	case "dna_attract":
		kit.DNAReg = -.02
	case "dna_diversify":
		kit.DNAReg = .02
	case "dream":
		kit.Dream = &parallel.DreamBuffer{Cap: 4}
	case "rotate":
		other := append([]parallel.TrainMode(nil), modes...)
		other[0] = parallel.ModeFreeze
		p.SetRotateSchedule([][]parallel.TrainMode{modes, other}, 1)
	case "per_cam_lr":
		scales := make([]float64, len(modes))
		for i := range scales {
			scales[i] = 1 / float64(i+1)
		}
		p.SetBranchLRs(scales...)
	}
	if strings.Contains(profile, "sync") {
		cfg := parallel.CamSyncConfig{Enabled: true, Alpha: .1, When: parallel.SyncManual}
		switch profile {
		case "sync_after_sample":
			cfg.When = parallel.SyncAfterSample
		case "sync_after_step":
			cfg.When = parallel.SyncAfterStep
		case "sync_after_pulse":
			cfg.When = parallel.SyncAfterPulse
		}
		if profile == "hard_sync" {
			cfg.Alpha = 1
		}
		if profile == "asymmetric_sync" {
			cfg.BranchAlpha = make([]float64, len(modes))
			for i := range cfg.BranchAlpha {
				cfg.BranchAlpha[i] = .1
			}
			cfg.BranchAlpha[0] = 0
		}
		if profile == "group_sync" {
			cfg.Groups = [][]int{{0, len(modes) - 1}}
		}
		if profile != "cross_sync" {
			p.CamSync = &cfg
		}
	}
	p.SetBranchModes(modes...)
	p.SetCamKit(kit)
}
func stamp(n *node, id string, coord architecture.Coord) {
	n.layer.ID = id
	n.layer.DType = meta(n.op).DType.String()
	if att, ok := n.op.(*mha.Layer); ok {
		n.layer.Heads = att.Cfg.NumHeads
	}
	n.layer.GridDepth = coord.Z
	n.layer.GridRow = coord.Y
	n.layer.GridCol = coord.X
	n.layer.CellLayer = coord.L
	if values, e := dna.FlattenOp(n.op); e == nil {
		n.layer.Samples = sample(values, 64)
		n.layer.Params = int64(len(values))
	}
	n.layer.Branches = nil
	for i, ch := range n.children {
		stamp(ch, fmt.Sprintf("%s/%d", id, i), coord)
		n.layer.Branches = append(n.layer.Branches, ch.layer)
	}
}
func sample(values []float32, limit int) []float32 {
	out := []float32{}
	for i := 0; i < len(values); i += max(1, (len(values)+limit-1)/limit) {
		out = append(out, values[i])
	}
	return out
}
func finite(values []float32) error {
	for _, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite runtime output")
		}
	}
	return nil
}
func tensorInput() *core.Tensor[float32] {
	x := core.NewTensor[float32](1, carrierWidth)
	for i := range x.Data {
		x.Data[i] = float32(i%7+1) * .02
	}
	return x
}
func (m *model) layers() []snat.Layer {
	layers := []snat.Layer{}
	for _, n := range m.nodes {
		layers = append(layers, n.layer)
	}
	return layers
}
