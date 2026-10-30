// SPDX-License-Identifier: Apache-2.0
// Tiny layer recipes adapted from OpenFluke w2a suites/parallel/poly_ops.go.
package main

import (
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
)

type polyKind struct {
	name string
	make func() (branches []any, cfg parallel.Config, x *core.Tensor[float32], trainable bool, err error)
}

func polyKinds() []polyKind {
	const dim, seq, batch = 4, 2, 1
	return []polyKind{
		{name: "dense", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			a, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, dim), true, nil
		}},
		{name: "mha", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			mcfg := mha.Config{DModel: dim, NumHeads: 4, MaxSeqLen: seq, Causal: true}
			a, err := mha.New(mcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := mha.New(mcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, seq, dim), true, nil
		}},
		{name: "swiglu", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			scfg := swiglu.Config{InputDim: dim, IntermediateDim: dim * 2}
			a, err := swiglu.New(scfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := swiglu.New(scfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, dim), true, nil
		}},
		{name: "rmsnorm", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			a, err := rmsnorm.New(rmsnorm.Config{Dim: dim})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := rmsnorm.New(rmsnorm.Config{Dim: dim})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, dim), true, nil
		}},
		{name: "layernorm", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			a, err := layernorm.New(layernorm.Config{Dim: dim})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := layernorm.New(layernorm.Config{Dim: dim})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, dim), true, nil
		}},
		{name: "softmax", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			a, err := softmax.New(softmax.Config{Dim: dim, SeqLen: 1})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := softmax.New(softmax.Config{Dim: dim, SeqLen: 1})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, dim), false, nil
		}},
		{name: "sequential", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			a, err := sequential.New(sequential.Config{Dim: dim, Depth: 2})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := sequential.New(sequential.Config{Dim: dim, Depth: 2})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, dim), true, nil
		}},
		{name: "residual", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			a, err := residual.New(residual.Config{Dim: dim, Depth: 1})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := residual.New(residual.Config{Dim: dim, Depth: 1})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, dim), true, nil
		}},
		{name: "cnn1", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			ccfg := cnn1.Config{InChannels: 4, Filters: 4, SeqLen: 8, Kernel: 3, Padding: 1}
			a, err := cnn1.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := cnn1.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: 8, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, 4, 8), true, nil
		}},
		{name: "cnn2", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			ccfg := cnn2.Config{InChannels: 2, Filters: 2, Height: 4, Width: 4, Kernel: 3, Padding: 1}
			a, err := cnn2.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := cnn2.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: 4, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, 2, 4, 4), true, nil
		}},
		{name: "cnn3", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			ccfg := cnn3.Config{InChannels: 2, Filters: 2, Depth: 2, Height: 4, Width: 4, Kernel: 3, Padding: 1}
			a, err := cnn3.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := cnn3.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: 4, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, 2, 2, 4, 4), true, nil
		}},
		{name: "convt1", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			ccfg := convt1.Config{InChannels: 4, Filters: 4, SeqLen: 8, Kernel: 3, Padding: 1}
			a, err := convt1.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := convt1.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: 8, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, 4, 8), true, nil
		}},
		{name: "convt2", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			ccfg := convt2.Config{InChannels: 2, Filters: 2, Height: 4, Width: 4, Kernel: 3, Padding: 1}
			a, err := convt2.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := convt2.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: 4, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, 2, 4, 4), true, nil
		}},
		{name: "convt3", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			ccfg := convt3.Config{InChannels: 2, Filters: 2, Depth: 2, Height: 4, Width: 4, Kernel: 3, Padding: 1}
			a, err := convt3.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := convt3.New(ccfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: 4, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, 2, 2, 4, 4), true, nil
		}},
		{name: "rnn", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			rcfg := rnn.Config{InputSize: dim, HiddenSize: dim, SeqLen: seq}
			a, err := rnn.New(rcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := rnn.New(rcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, seq, dim), true, nil
		}},
		{name: "lstm", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			lcfg := lstm.Config{InputSize: dim, HiddenSize: dim, SeqLen: seq}
			a, err := lstm.New(lcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := lstm.New(lcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, seq, dim), true, nil
		}},
		{name: "embedding", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			ecfg := embedding.Config{VocabSize: 16, EmbeddingDim: dim, SeqLen: seq}
			a, err := embedding.New(ecfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := embedding.New(ecfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			// Token ids [batch, seq]; Parallel uses polymorphic shape fallback.
			cfg := parallel.Config{Dim: dim, Branches: 2, Combine: parallel.CombineAdd}
			x := core.NewTensor[float32](batch, seq)
			for i := range x.Data {
				x.Data[i] = float32(i % 16)
			}
			return []any{a, b}, cfg, x, true, nil
		}},
		{name: "kmeans", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			kcfg := kmeans.Config{NumClusters: 4, FeatureDim: dim, OutputMode: kmeans.OutputFeatures}
			a, err := kmeans.New(kcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := kmeans.New(kcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, dim), true, nil
		}},
		{name: "mamba", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			mcfg := mamba.Config{DModel: dim, DState: 8, SeqLen: seq}
			a, err := mamba.New(mcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := mamba.New(mcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, seq, dim), true, nil
		}},
		{name: "metacognition", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			a, err := metacognition.New(metacognition.Config{Dim: dim})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := metacognition.New(metacognition.Config{Dim: dim})
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, dim), true, nil
		}},
		{name: "gdn", make: func() ([]any, parallel.Config, *core.Tensor[float32], bool, error) {
			gcfg := gdn.Config{
				HiddenSize: dim, NumKeyHeads: 2, NumValueHeads: 2,
				KeyHeadDim: 8, ValueHeadDim: 8, ConvKernel: 2, Eps: 1e-6,
			}
			a, err := gdn.New(gcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			b, err := gdn.New(gcfg)
			if err != nil {
				return nil, parallel.Config{}, nil, false, err
			}
			cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq}
			return []any{a, b}, cfg, core.NewTensor[float32](batch, seq, dim), true, nil
		}},
	}
}
