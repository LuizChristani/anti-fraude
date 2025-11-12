package models

type Model interface {
    Fit(X [][]float64, y []int) error
    Predict(X [][]float64) []int
    PredictProba(X [][]float64) []float64
    Name() string
}