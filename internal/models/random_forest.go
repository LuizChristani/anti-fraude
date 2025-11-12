package models

import (
    "math"
    "math/rand"
)

type RandomForest struct {
    NEstimators int
    MaxDepth    int
    MinSamples  int
    MaxThresholdsPerFe int
    MaxFeatures int
    Trees       []*DecisionTree
}

func NewRandomForest() *RandomForest {
    return &RandomForest{NEstimators: 30, MaxDepth: 6, MinSamples: 100, MaxThresholdsPerFe: 32, MaxFeatures: 0, Trees: []*DecisionTree{}}
}

func (rf *RandomForest) Name() string { return "RandomForest" }

func (rf *RandomForest) Fit(X [][]float64, y []int) error {
    if rf.NEstimators <= 0 { rf.NEstimators = 30 }
    n := len(X)
    nFeats := len(X[0])
    if rf.MaxFeatures <= 0 {
        rf.MaxFeatures = int(math.Max(1, math.Min(float64(nFeats), math.Sqrt(float64(nFeats)))))
    }
    rf.Trees = make([]*DecisionTree, 0, rf.NEstimators)
    for k := 0; k < rf.NEstimators; k++ {
        idx := make([]int, n)
        for i := 0; i < n; i++ { idx[i] = rand.Intn(n) }
        Xb := make([][]float64, n)
        yb := make([]int, n)
        for i := 0; i < n; i++ { Xb[i] = X[idx[i]]; yb[i] = y[idx[i]] }
        dt := NewDecisionTree()
        dt.MaxDepth = rf.MaxDepth
        dt.MinSamplesSplit = rf.MinSamples
        dt.MaxThresholdsPerFe = rf.MaxThresholdsPerFe
        dt.MaxFeatures = rf.MaxFeatures
        if err := dt.Fit(Xb, yb); err != nil { return err }
        rf.Trees = append(rf.Trees, dt)
    }
    return nil
}

func (rf *RandomForest) Predict(X [][]float64) []int {
    ps := rf.PredictProba(X)
    out := make([]int, len(ps))
    for i := range ps { if ps[i] >= 0.5 { out[i] = 1 } }
    return out
}

func (rf *RandomForest) PredictProba(X [][]float64) []float64 {
    n := len(X)
    if len(rf.Trees) == 0 { out := make([]float64, n); for i := range out { out[i] = 0.5 }; return out }
    out := make([]float64, n)
    for _, dt := range rf.Trees {
        p := dt.PredictProba(X)
        for i := 0; i < n; i++ { out[i] += p[i] }
    }
    m := float64(len(rf.Trees))
    for i := 0; i < n; i++ { out[i] /= m }
    return out
}