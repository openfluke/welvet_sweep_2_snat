// SPDX-License-Identifier: Apache-2.0
package main

import (
	"context"
	"fmt"
	"github.com/openfluke/snat"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/step"
	"github.com/openfluke/welvet/systems/dna"
	"math"
	"strings"
)

type unsupportedError struct{ reason string }

func (e unsupportedError) Error() string { return e.reason }
func eventFor(n *node, values []float32, direction, phase, kind string, tick int, c Configuration) snat.Event {
	mode := "normal"
	if c.Mode.IsLineStep() || c.Mode.RequiresGrid() {
		mode = "step"
	}
	return snat.Event{"Type": direction, "mode": mode, "step_count": tick, "layer_id": n.layer.ID, "grid_depth": n.layer.GridDepth, "grid_row": n.layer.GridRow, "grid_col": n.layer.GridCol, "cell_layer": n.layer.CellLayer, "values": sample(values, 32), "values_kind": kind, "phase": phase, "train_mode": c.Mode.String(), "dtype": c.DType.String(), "dtype_id": int(c.DType), "Stats": map[string]any{"LayerType": n.layer.Type}}
}
func runOp(n *node, input *core.Tensor[float32]) (*core.Tensor[float32], error) {
	switch op := n.op.(type) {
	case *parallel.View:
		_, out, e := parallel.ForwardView(op, input)
		return out, e
	}
	_, out, err := forward.Cell(&architecture.Cell{Layer: meta(n.op), Op: n.op}, input)
	return out, err
}
func probeNode(ctx context.Context, n *node, input *core.Tensor[float32], tick int, c Configuration, emit func([]snat.Event) error) (*core.Tensor[float32], error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	n.lastIn = input
	var out *core.Tensor[float32]
	var err error
	if _, ok := n.op.(*parallel.Stack); ok && len(n.children) > 0 {
		out = input
		for _, child := range n.children {
			out, err = probeNode(ctx, child, out, tick, c, emit)
			if err != nil {
				return nil, err
			}
		}
	} else {
		out, err = runOp(n, input)
		if err != nil {
			return nil, err
		}
		if _, ok := n.op.(*parallel.Layer); ok {
			for _, child := range n.children {
				if _, err = probeNode(ctx, child, input, tick, c, emit); err != nil {
					return nil, err
				}
			}
		}
	}
	if out == nil {
		return nil, fmt.Errorf("nil output at %s", n.layer.Name)
	}
	if err = finite(out.Data); err != nil {
		return nil, err
	}
	n.lastOut = out
	if err = emit([]snat.Event{activationEvent(n, out, tick, c)}); err != nil {
		return nil, err
	}
	return out, nil
}
func snapshotWeights(nodes []*node) map[string][]float32 {
	values := map[string][]float32{}
	var walk func(*node)
	walk = func(n *node) {
		if v, e := dna.FlattenOp(n.op); e == nil {
			values[n.layer.ID] = sample(v, 32)
		}
		for _, ch := range n.children {
			walk(ch)
		}
	}
	for _, n := range nodes {
		walk(n)
	}
	return values
}
func weightEvents(nodes []*node, before map[string][]float32, tick int, c Configuration) ([]snat.Event, error) {
	out := []snat.Event{}
	var walk func(*node) error
	walk = func(n *node) error {
		after, e := dna.FlattenOp(n.op)
		if e != nil {
			return e
		}
		if e = finite(after); e != nil {
			return e
		}
		after = sample(after, 32)
		old := before[n.layer.ID]
		if len(old) == len(after) && len(old) > 0 {
			delta := make([]float32, len(old))
			for i := range old {
				delta[i] = after[i] - old[i]
			}
			out = append(out, eventFor(n, delta, "backward", "parameter_update", "sampled_weight_delta", tick, c))
		}
		for _, ch := range n.children {
			if e = walk(ch); e != nil {
				return e
			}
		}
		return nil
	}
	for _, n := range nodes {
		if e := walk(n); e != nil {
			return nil, e
		}
	}
	return out, nil
}
func runExample(ctx context.Context, c Configuration, m *model, samples int, lr float64, emit func([]snat.Event) error, progress func(float64)) error {
	if c.Topology.Cameral == "action" && !c.Mode.IsLineStep() {
		return unsupportedError{"action role requires a Step training mode"}
	}
	iterations := samples
	if c.Mode.IsLineStep() {
		depth := 1
		for _, s := range m.stacks {
			depth = max(depth, parallel.StackLineDepth(s))
		}
		iterations += depth
	}
	var mesh *step.State[float32]
	if c.Mode.RequiresGrid() {
		mesh = step.New[float32](m.grid)
		mesh.SetInput(tensorInput())
		for i := 0; i < len(m.nodes); i++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			if _, err := step.Forward(m.grid, mesh, true); err != nil {
				return err
			}
		}
	}
	for tick := 0; tick < iterations; tick++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		before := snapshotWeights(m.nodes)
		inputs := make([]*core.Tensor[float32], len(m.nodes))
		outputs := make([]*core.Tensor[float32], len(m.nodes))
		if mesh != nil {
			mesh.SetInput(tensorInput())
			if _, err := step.Forward(m.grid, mesh, true); err != nil {
				return err
			}
			for i, n := range m.nodes {
				inputs[i] = mesh.HistoryIn[len(mesh.HistoryIn)-1][i]
				outputs[i] = mesh.LayerData[i]
				if inputs[i] == nil || outputs[i] == nil {
					return fmt.Errorf("unfilled mesh cell %d", i)
				}
				if err := emit([]snat.Event{eventFor(n, outputs[i].Data, "forward", "mesh_tick", "activation", tick, c)}); err != nil {
					return err
				}
				// Branch diagnostics are extra real forwards, not invented training tape.
				if _, err := probeNode(ctx, n, inputs[i], tick, c, emit); err != nil {
					return err
				}
			}
		} else {
			current := tensorInput()
			for i, n := range m.nodes {
				inputs[i] = current
				out, err := probeNode(ctx, n, current, tick, c, emit)
				if err != nil {
					return err
				}
				outputs[i] = out
				current = out
			}
		}
		var totalLoss float64
		for i, stack := range m.stacks {
			target := core.NewTensor[float32](outputs[i].Shape...)
			for j := range target.Data {
				target.Data[j] = .15
			}
			var loss float64
			var err error
			if c.Topology.Cameral == "action" {
				loss, _, err = parallel.TickStackLine(stack, inputs[i], target, parallel.RoleAction, c.Mode, lr)
			} else {
				loss, err = parallel.TrainStackMSE(stack, inputs[i], target, c.Mode, lr)
			}
			if err != nil {
				return err
			}
			if math.IsNaN(loss) || math.IsInf(loss, 0) {
				return fmt.Errorf("non-finite loss")
			}
			totalLoss += loss
		}
		for _, cam := range m.cams {
			cam.NoteLoss(totalLoss / float64(len(m.stacks)))
			if cam.CamSync != nil {
				var syncErr error
				if cam.CamSync.When == parallel.SyncAfterPulse {
					syncErr = cam.Pulse()
				} else if cam.CamSync.When == parallel.SyncManual {
					syncErr = cam.SyncNow()
				} else {
					syncErr = cam.MaybeSync(cam.CamSync.When)
				}
				if err := syncErr; err != nil {
					return err
				}
			}
			if c.Topology.Cameral == "rotate" {
				cam.AdvanceRotate()
			}
			if c.Topology.Cameral == "dream" {
				// Dream uses the parallel merge's real native input and output shapes.
				var find func(*node) *node
				find = func(n *node) *node {
					if n.op == cam {
						return n
					}
					for _, ch := range n.children {
						if v := find(ch); v != nil {
							return v
						}
					}
					return nil
				}
				for _, root := range m.nodes {
					if n := find(root); n != nil && n.lastIn != nil && n.lastOut != nil {
						target := core.NewTensor[float32](n.lastOut.Shape...)
						for i := range target.Data {
							target.Data[i] = .15
						}
						cam.Remember(n.lastIn, target)
						if _, err := cam.DreamPulse(1, parallel.ModeNormalBP, lr); err != nil {
							return err
						}
						break
					}
				}
			}
			_ = cam.RefreshMetrics()
		}
		events, err := weightEvents(m.nodes, before, tick, c)
		if err != nil {
			return err
		}
		for i := range events {
			events[i]["loss"] = totalLoss / float64(len(m.stacks))
		}
		if len(events) > 0 {
			if err = emit(events); err != nil {
				return err
			}
		}
		progress(float64(tick+1) / float64(iterations) * 100)
	}
	return nil
}
func classifyFailure(err error) string {
	if err == nil {
		return "complete"
	}
	if _, ok := err.(unsupportedError); ok {
		return "unsupported"
	}
	message := strings.ToLower(err.Error())
	for _, word := range []string{"unsupported", "not implemented", "not supported", "only supports", "only float", "requires float"} {
		if strings.Contains(message, word) {
			return "unsupported"
		}
	}
	if err == context.Canceled {
		return "cancelled"
	}
	return "failed"
}

func activationEvent(n *node, out *core.Tensor[float32], tick int, c Configuration) snat.Event {
	e := eventFor(n, out.Data, "forward", "diagnostic_forward", "activation", tick, c)
	e["shape"] = out.Shape
	return e
}
