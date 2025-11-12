package models

import (
    "math"
    "sort"
)

type gbTree struct {
    Feature   int
    Threshold float64
    LeftVal   float64
    RightVal  float64
}

type GradientBoosting struct {
    NEstimators  int
    LearningRate float64
    MaxDepth     int
    MinSamples   int
    MaxThresholdsPerFe int
    Trees        []gbTree
}

func NewGradientBoosting() *GradientBoosting {
    return &GradientBoosting{NEstimators: 50, LearningRate: 0.1, MaxDepth: 1, MaxThresholdsPerFe: 32}
}

func (gb *GradientBoosting) Name() string { return "GradientBoosting" }

func sigmoid(z float64) float64 { return 1.0 / (1.0 + math.Exp(-z)) }

func (gb *GradientBoosting) Fit(X [][]float64, y []int) error {
    n := len(X)
    if n == 0 { return nil }
    pos := 0
    for i := 0; i < n; i++ { if y[i] == 1 { pos++ } }
    base := float64(pos) / float64(n)
    if base <= 1e-3 { base = 1e-3 }
    if base >= 1-1e-3 { base = 1 - 1e-3 }
    init := math.Log(base / (1.0 - base))
    F := make([]float64, n)
    for i := 0; i < n; i++ { F[i] = init }

    for m := 0; m < gb.NEstimators; m++ {
        r := make([]float64, n)
        for i := 0; i < n; i++ {
            p := sigmoid(F[i])
            r[i] = float64(y[i]) - p
        }

        best := gbTree{Feature: -1}
        bestSSE := math.MaxFloat64
        nFeats := len(X[0])
        for j := 0; j < nFeats; j++ {
            cands := gbCandidateThresholds(X, j, gb.MaxThresholdsPerFe)
            for _, thr := range cands {
                leftSum, leftCount := 0.0, 0.0
                rightSum, rightCount := 0.0, 0.0
                for i := 0; i < n; i++ {
                    if X[i][j] <= thr { leftSum += r[i]; leftCount++ } else { rightSum += r[i]; rightCount++ }
                }
                if int(leftCount) < gb.MinSamples || int(rightCount) < gb.MinSamples { continue }
                if leftCount == 0 || rightCount == 0 { continue }
                leftAvg := leftSum / leftCount
                rightAvg := rightSum / rightCount

                leftSS, rightSS := 0.0, 0.0
                for i := 0; i < n; i++ {
                    if X[i][j] <= thr {
                        d := r[i] - leftAvg
                        leftSS += d * d
                    } else {
                        d := r[i] - rightAvg
                        rightSS += d * d
                    }
                }
                sse := leftSS + rightSS
                if sse < bestSSE {
                    bestSSE = sse
                    best.Feature = j
                    best.Threshold = thr
                    best.LeftVal = leftAvg
                    best.RightVal = rightAvg
                }
            }
        }
        if best.Feature == -1 { break }
        gb.Trees = append(gb.Trees, best)
        for i := 0; i < n; i++ {
            inc := best.LeftVal
            if X[i][best.Feature] > best.Threshold { inc = best.RightVal }
            F[i] += gb.LearningRate * inc
        }
    }
    return nil
}

func (gb *GradientBoosting) PredictProba(X [][]float64) []float64 {
    out := make([]float64, len(X))
    for i := range X {
        f := 0.0
        for _, t := range gb.Trees {
            inc := t.LeftVal
            if X[i][t.Feature] > t.Threshold { inc = t.RightVal }
            f += gb.LearningRate * inc
        }
        out[i] = sigmoid(f)
    }
    return out
}

func (gb *GradientBoosting) Predict(X [][]float64) []int {
    out := make([]int, len(X))
    p := gb.PredictProba(X)
    for i := range p {
        if p[i] >= 0.5 { out[i] = 1 } else { out[i] = 0 }
    }
    return out
}

func gbCandidateThresholds(X [][]float64, j int, nCand int) []float64 {
    if nCand <= 0 { nCand = 16 }
    n := len(X)
    vals := make([]float64, n)
    for i := 0; i < n; i++ { vals[i] = X[i][j] }
    sort.Float64s(vals)
    out := make([]float64, 0, nCand)
    for k := 1; k < nCand; k++ {
        idx := int(math.Round(float64(k) / float64(nCand) * float64(n-1)))
        if idx <= 0 || idx >= n { continue }
        thr := vals[idx]
        if len(out) == 0 || thr != out[len(out)-1] {
            out = append(out, thr)
        }
    }
    if len(out) == 0 {
        sum := 0.0
        for i := 0; i < n; i++ { sum += vals[i] }
        out = append(out, sum/float64(n))
    }
    return out
}