package main

import (
    "encoding/csv"
    "flag"
    "fmt"
    "math"
    "os"
    "strconv"
    "time"

    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/plotutil"
    "gonum.org/v1/plot/vg"

    "antifraude/internal/features"
    "antifraude/internal/models"
)

func main() {
    algo := flag.String("algo", "dt", "Algoritmo: dt|rf|bagging|gb")
    estimators := flag.Int("estimators", 30, "Número de estimadores (rf/bagging/gb)")
    maxDepth := flag.Int("max_depth", 6, "Profundidade máxima da árvore")
    minSamples := flag.Int("min_samples", 100, "Mínimo de amostras para split")
    lr := flag.Float64("lr", 0.1, "Learning rate para GradientBoosting")
    points := flag.Int("points", 8, "Quantidade de pontos na curva")
    dataPath := flag.String("data", "data/synthetic.csv", "CSV de entrada")
    outImg := flag.String("out_img", "cmd/api/static/learning_curve.png", "PNG de saída")
    outCsv := flag.String("out_csv", "data/learning_curve.csv", "CSV de saída")
    flag.Parse()

    X, y := loadXY(*dataPath)
    if len(X) == 0 { fmt.Println("Dataset vazio"); return }

    split := int(0.8 * float64(len(X)))
    Xtrain, ytrain := X[:split], y[:split]
    Xtest, ytest := X[split:], y[split:]

    sizes := make([]int, 0, *points)
    for i := 1; i <= *points; i++ {
        frac := float64(i) / float64(*points)
        s := int(math.Max(100, frac*float64(len(Xtrain))))
        if s > len(Xtrain) { s = len(Xtrain) }
        sizes = append(sizes, s)
    }

    trainAcc := make([]float64, len(sizes))
    testAcc := make([]float64, len(sizes))

    for k, s := range sizes {
        subX := Xtrain[:s]
        subY := ytrain[:s]
        mdl := buildModel(*algo, *estimators, *maxDepth, *minSamples, *lr)
        if err := mdl.Fit(subX, subY); err != nil { fmt.Println("Falha treino:", err); return }
        pTrain := mdl.Predict(subX)
        pTest := mdl.Predict(Xtest)
        trainAcc[k] = accuracy(subY, pTrain)
        testAcc[k] = accuracy(ytest, pTest)
        fmt.Printf("%s | size=%d | train=%.3f | test=%.3f\n", mdl.Name(), s, trainAcc[k], testAcc[k])
    }

    if err := writeCSV(*outCsv, sizes, trainAcc, testAcc); err != nil {
        fmt.Println("Erro ao salvar CSV:", err)
    } else {
        fmt.Println("Curva salva em:", *outCsv)
    }

    if err := plotCurve(*outImg, sizes, trainAcc, testAcc); err != nil {
        fmt.Println("Erro ao salvar PNG:", err)
    } else {
        fmt.Println("Gráfico salvo em:", *outImg)
    }
}

func loadXY(path string) ([][]float64, []int) {
    f, err := os.Open(path)
    if err != nil { fmt.Println("Falha ao abrir CSV:", err); return nil, nil }
    defer f.Close()
    r := csv.NewReader(f)
    rows, err := r.ReadAll()
    if err != nil || len(rows) < 2 { fmt.Println("CSV inválido"); return nil, nil }
    X := make([][]float64, 0, len(rows)-1)
    y := make([]int, 0, len(rows)-1)
    for i := 1; i < len(rows); i++ {
        row := rows[i]
        reqDate, _ := time.Parse("2006-01-02", row[5])
        travelDate, _ := time.Parse("2006-01-02", row[6])
        amount, _ := strconv.ParseFloat(row[9], 64)
        fraud, _ := strconv.Atoi(row[14])
        e := features.BuildExpense(
            row[0], row[1], row[2], row[3], row[4],
            reqDate, travelDate,
            row[7], row[8],
            amount,
            row[10], row[11], row[12], row[13],
        )
        v, _ := features.Vectorize(e)
        X = append(X, v)
        y = append(y, fraud)
    }
    return X, y
}

func buildModel(algo string, estimators, maxDepth, minSamples int, lr float64) models.Model {
    switch algo {
    case "rf":
        rf := models.NewRandomForest()
        rf.NEstimators = estimators
        rf.MaxDepth = maxDepth
        rf.MinSamples = minSamples
        return rf
    case "bagging":
        bg := models.NewBagging()
        bg.NEstimators = estimators
        bg.MaxDepth = maxDepth
        bg.MinSamples = minSamples
        return bg
    case "gb":
        gb := models.NewGradientBoosting()
        gb.NEstimators = estimators
        gb.LearningRate = lr
        return gb
    default:
        dt := models.NewDecisionTree()
        dt.MaxDepth = maxDepth
        dt.MinSamplesSplit = minSamples
        return dt
    }
}

func accuracy(y, p []int) float64 {
    if len(y) == 0 { return 0 }
    c := 0
    for i := range y { if y[i] == p[i] { c++ } }
    return float64(c)/float64(len(y))
}

func writeCSV(path string, sizes []int, trainAcc, testAcc []float64) error {
    if err := os.MkdirAll("data", 0o755); err != nil { return err }
    f, err := os.Create(path)
    if err != nil { return err }
    defer f.Close()
    w := csv.NewWriter(f)
    defer w.Flush()
    if err := w.Write([]string{"size", "train_acc", "test_acc"}); err != nil { return err }
    for i := range sizes {
        rec := []string{strconv.Itoa(sizes[i]), fmt.Sprintf("%.6f", trainAcc[i]), fmt.Sprintf("%.6f", testAcc[i])}
        if err := w.Write(rec); err != nil { return err }
    }
    return nil
}

func plotCurve(path string, sizes []int, trainAcc, testAcc []float64) error {
    p := plot.New()
    p.Title.Text = "Curva de Aprendizagem"
    p.X.Label.Text = "Amostras de treino"
    p.Y.Label.Text = "Acurácia"
    p.Y.Min = 0
    p.Y.Max = 1

    toXY := func(xs []int, ys []float64) plotter.XYs {
        pts := make(plotter.XYs, len(xs))
        for i := range xs { pts[i].X = float64(xs[i]); pts[i].Y = ys[i] }
        return pts
    }
    trPts := toXY(sizes, trainAcc)
    tePts := toXY(sizes, testAcc)

    if err := plotutil.AddLinePoints(p, "Treino", trPts, "Teste", tePts); err != nil { return err }
    if err := os.MkdirAll("cmd/api/static", 0o755); err != nil { return err }
    return p.Save(8*vg.Inch, 4*vg.Inch, path)
}