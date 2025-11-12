package models

import (
    "math"
    "math/rand"
)

type DTNode struct {
    Feature   int
    Threshold float64
    Left      *DTNode
    Right     *DTNode
    IsLeaf    bool
    ProbaLeaf float64
}

type DecisionTree struct {
    MaxDepth           int
    MinSamplesSplit    int
    MaxThresholdsPerFe int
    MaxFeatures        int
    Root               *DTNode
}

func NewDecisionTree() *DecisionTree {
    return &DecisionTree{MaxDepth: 6, MinSamplesSplit: 100, MaxThresholdsPerFe: 64}
}

func (dt *DecisionTree) Name() string { return "DecisionTree" }

func (dt *DecisionTree) Fit(X [][]float64, y []int) error {
    idx := make([]int, len(X))
    for i := range idx { idx[i] = i }
    dt.Root = dt.build(X, y, idx, 0)
    return nil
}

func (dt *DecisionTree) Predict(X [][]float64) []int {
    out := make([]int, len(X))
    for i := range X {
        p := dt.predictProbaOne(X[i])
        if p >= 0.5 { out[i] = 1 } else { out[i] = 0 }
    }
    return out
}

func (dt *DecisionTree) PredictProba(X [][]float64) []float64 {
    out := make([]float64, len(X))
    for i := range X { out[i] = dt.predictProbaOne(X[i]) }
    return out
}

func (dt *DecisionTree) predictProbaOne(x []float64) float64 {
    n := dt.Root
    if n == nil { return 0.5 }
    for !n.IsLeaf {
        if x[n.Feature] <= n.Threshold { n = n.Left } else { n = n.Right }
        if n == nil { return 0.5 }
    }
    return n.ProbaLeaf
}

func (dt *DecisionTree) build(X [][]float64, y []int, idx []int, depth int) *DTNode {
    node := &DTNode{}
    if len(idx) < dt.MinSamplesSplit || depth >= dt.MaxDepth {
        node.IsLeaf = true
        node.ProbaLeaf = classProba(y, idx)
        return node
    }
    p := classProba(y, idx)
    if p == 0 || p == 1 {
        node.IsLeaf = true
        node.ProbaLeaf = p
        return node
    }
    bestFeature := -1
    bestThr := 0.0
    bestImp := math.MaxFloat64
    leftIdxBest := []int{}
    rightIdxBest := []int{}

    nFeats := len(X[0])
    feats := pickFeatures(nFeats, dt.MaxFeatures)
    for _, f := range feats {
        cand := candidateThresholds(X, idx, f, dt.MaxThresholdsPerFe)
        for _, thr := range cand {
            lIdx, rIdx := splitIdx(X, idx, f, thr)
            if len(lIdx) == 0 || len(rIdx) == 0 { continue }
            imp := giniImpurity(y, lIdx, rIdx)
            if imp < bestImp {
                bestImp = imp
                bestFeature = f
                bestThr = thr
                leftIdxBest = lIdx
                rightIdxBest = rIdx
            }
        }
    }

    if bestFeature == -1 {
        node.IsLeaf = true
        node.ProbaLeaf = p
        return node
    }
    node.Feature = bestFeature
    node.Threshold = bestThr
    node.Left = dt.build(X, y, leftIdxBest, depth+1)
    node.Right = dt.build(X, y, rightIdxBest, depth+1)
    return node
}

func classProba(y []int, idx []int) float64 {
    sum := 0
    for _, i := range idx { sum += y[i] }
    return float64(sum)/float64(len(idx))
}

func splitIdx(X [][]float64, idx []int, f int, thr float64) ([]int, []int) {
    l := make([]int, 0, len(idx))
    r := make([]int, 0, len(idx))
    for _, i := range idx {
        if X[i][f] <= thr { l = append(l, i) } else { r = append(r, i) }
    }
    return l, r
}

func giniImpurity(y []int, lIdx, rIdx []int) float64 {
    g := func(ids []int) float64 {
        if len(ids) == 0 { return 0 }
        p := 0.0
        for _, i := range ids { p += float64(y[i]) }
        p = p/float64(len(ids))
        return p*(1-p)
    }
    gl := g(lIdx)
    gr := g(rIdx)
    wl := float64(len(lIdx))
    wr := float64(len(rIdx))
    n := wl+wr
    return (wl/n)*gl + (wr/n)*gr
}

func candidateThresholds(X [][]float64, idx []int, f int, maxC int) []float64 {
    values := make([]float64, len(idx))
    for j, i := range idx { values[j] = X[i][f] }
    for i := range values {
        j := rand.Intn(len(values))
        values[i], values[j] = values[j], values[i]
    }
    m := int(math.Min(float64(maxC), float64(len(values))))
    out := make([]float64, 0, m)
    for i := 0; i < m; i++ { out = append(out, values[i]) }
    return out
}

func pickFeatures(nFeats int, maxFeats int) []int {
    if maxFeats <= 0 || maxFeats >= nFeats {
        out := make([]int, nFeats)
        for i := 0; i < nFeats; i++ { out[i] = i }
        return out
    }
    idx := make([]int, nFeats)
    for i := 0; i < nFeats; i++ { idx[i] = i }
    for i := range idx {
        j := rand.Intn(nFeats)
        idx[i], idx[j] = idx[j], idx[i]
    }
    out := make([]int, maxFeats)
    copy(out, idx[:maxFeats])
    return out
}