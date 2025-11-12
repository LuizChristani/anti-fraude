package models

import (
    "math/rand"
)

type Bagging struct {
    NEstimators int
    MaxDepth    int
    MinSamples  int
    MaxThresholdsPerFe int
    Trees       []*DecisionTree
}

func NewBagging() *Bagging {
    return &Bagging{NEstimators: 30, MaxDepth: 6, MinSamples: 100, MaxThresholdsPerFe: 32, Trees: []*DecisionTree{}}
}

func (bg *Bagging) Name() string { return "Bagging" }

func (bg *Bagging) Fit(X [][]float64, y []int) error {
    if bg.NEstimators <= 0 { bg.NEstimators = 30 }
    n := len(X)
    bg.Trees = make([]*DecisionTree, 0, bg.NEstimators)
    for k := 0; k < bg.NEstimators; k++ {
        idx := make([]int, n)
        for i := 0; i < n; i++ { idx[i] = rand.Intn(n) }
        Xb := make([][]float64, n)
        yb := make([]int, n)
        for i := 0; i < n; i++ { Xb[i] = X[idx[i]]; yb[i] = y[idx[i]] }
        dt := NewDecisionTree()
        dt.MaxDepth = bg.MaxDepth
        dt.MinSamplesSplit = bg.MinSamples
        dt.MaxThresholdsPerFe = bg.MaxThresholdsPerFe
        dt.MaxFeatures = 0
        if err := dt.Fit(Xb, yb); err != nil { return err }
        bg.Trees = append(bg.Trees, dt)
    }
    return nil
}

func (bg *Bagging) Predict(X [][]float64) []int {
    ps := bg.PredictProba(X)
    out := make([]int, len(ps))
    for i := range ps { if ps[i] >= 0.5 { out[i] = 1 } }
    return out
}

func (bg *Bagging) PredictProba(X [][]float64) []float64 {
    n := len(X)
    if len(bg.Trees) == 0 { out := make([]float64, n); for i := range out { out[i] = 0.5 }; return out }
    out := make([]float64, n)
    for _, dt := range bg.Trees {
        p := dt.PredictProba(X)
        for i := 0; i < n; i++ { out[i] += p[i] }
    }
    m := float64(len(bg.Trees))
    for i := 0; i < n; i++ { out[i] /= m }
    return out
}