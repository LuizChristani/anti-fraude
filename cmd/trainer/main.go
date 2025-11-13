package main

import (
	"encoding/csv"
	"encoding/gob"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"

	"go.uber.org/zap"

	"antifraude/internal/data"
	"antifraude/internal/features"
	"antifraude/internal/models"
	"antifraude/pkg/utils"
)

func main() {
    logger := utils.Logger()
    defer logger.Sync()

    regen := flag.Bool("regen", true, "Regenerar dataset sintético")
    n := flag.Int("n", 260000, "Número de registros sintéticos")
    out := flag.String("out", "data/synthetic.csv", "Caminho do CSV de saída")
    algo := flag.String("algo", "dt", "Algoritmo: dt|rf|bagging|gb|lgbm")
    estimators := flag.Int("estimators", 30, "Número de estimadores no ensemble (rf/bagging)")
    maxDepth := flag.Int("max_depth", 6, "Profundidade máxima da árvore")
    minSamples := flag.Int("min_samples", 100, "Mínimo de amostras para split")
    lr := flag.Float64("lr", 0.1, "Learning rate para GradientBoosting")
    curve := flag.Bool("curve", true, "Gerar curva de aprendizagem (PNG e CSV)")
    curvePoints := flag.Int("curve_points", 10, "Quantidade de pontos na curva")
    curveImg := flag.String("curve_out_img", "cmd/api/static/learning_curve.png", "PNG da curva")
    curveCsv := flag.String("curve_out_csv", "data/learning_curve.csv", "CSV da curva")
    curveMin := flag.Int("curve_min", 500, "Tamanho mínimo inicial da curva")
    curveLog := flag.Bool("curve_log", true, "Usar escala logarítmica para os tamanhos")
    threshold := flag.Float64("threshold", 0.5, "Threshold para classificação (métricas F1/precisão/recall)")
    thresholdAuto := flag.Bool("threshold_auto", true, "Escolher automaticamente o threshold que maximiza F1 no holdout")
    thresholdMetric := flag.String("threshold_metric", "f1", "Métrica para escolher threshold: f1|acc")
    thrMin := flag.Float64("threshold_min", 0.05, "Limite inferior para threshold automático")
    thrMax := flag.Float64("threshold_max", 0.95, "Limite superior para threshold automático")
    flag.Parse()

    if *regen {
        logger.Info("Gerando dataset sintético", zap.Int("n", *n), zap.String("out", *out))
        if err := data.GenerateSyntheticExpenses(*n, 0.08, *out); err != nil {
            logger.Fatal("Falha ao gerar dataset", zap.Error(err))
        }
    }

    f, err := os.Open(*out)
    if err != nil { logger.Fatal("Falha ao abrir CSV", zap.Error(err)) }
    defer f.Close()

    r := csv.NewReader(f)
    rows, err := r.ReadAll()
    if err != nil { logger.Fatal("Falha ao ler CSV", zap.Error(err)) }
    if len(rows) < 2 { logger.Fatal("CSV vazio") }

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

    rand.Seed(time.Now().UnixNano())
    idx := rand.Perm(len(X))
    shX := make([][]float64, len(X))
    shY := make([]int, len(y))
    for i, j := range idx { shX[i] = X[j]; shY[i] = y[j] }
    X, y = shX, shY

    var pos, neg int
    for i := range y { if y[i] == 1 { pos++ } else { neg++ } }
    logger.Info("Distribuição da classe", zap.Int("positivos", pos), zap.Int("negativos", neg))

    var posIdx, negIdx []int
    for i := range y { if y[i] == 1 { posIdx = append(posIdx, i) } else { negIdx = append(negIdx, i) } }
    rp := rand.Perm(len(posIdx))
    rn := rand.Perm(len(negIdx))
    pTrain := int(0.8 * float64(len(posIdx)))
    nTrain := int(0.8 * float64(len(negIdx)))
    trainIdx := make([]int, 0, pTrain+nTrain)
    testIdx := make([]int, 0, len(posIdx)-pTrain+len(negIdx)-nTrain)
    for i := 0; i < len(posIdx); i++ { if i < pTrain { trainIdx = append(trainIdx, posIdx[rp[i]]) } else { testIdx = append(testIdx, posIdx[rp[i]]) } }
    for i := 0; i < len(negIdx); i++ { if i < nTrain { trainIdx = append(trainIdx, negIdx[rn[i]]) } else { testIdx = append(testIdx, negIdx[rn[i]]) } }
    rTrain := rand.Perm(len(trainIdx))
    rTest := rand.Perm(len(testIdx))
    var Xtrain [][]float64
    var ytrain []int
    var Xtest [][]float64
    var ytest []int
    Xtrain, ytrain = make([][]float64, len(trainIdx)), make([]int, len(trainIdx))
    Xtest, ytest = make([][]float64, len(testIdx)), make([]int, len(testIdx))
    for i := range rTrain { idx := trainIdx[rTrain[i]]; Xtrain[i] = X[idx]; ytrain[i] = y[idx] }
    for i := range rTest { idx := testIdx[rTest[i]]; Xtest[i] = X[idx]; ytest[i] = y[idx] }

    var mdl models.Model
    var path string
    switch *algo {
    case "rf":
        rf := models.NewRandomForest()
        rf.NEstimators = *estimators
        rf.MaxDepth = *maxDepth
        rf.MinSamples = *minSamples
        if err := rf.Fit(Xtrain, ytrain); err != nil {
            logger.Fatal("Falha ao treinar RF", zap.Error(err))
        }
        mdl = rf
        path = "models/rf_model.gob"
    case "bagging":
        bg := models.NewBagging()
        bg.NEstimators = *estimators
        bg.MaxDepth = *maxDepth
        bg.MinSamples = *minSamples
        if err := bg.Fit(Xtrain, ytrain); err != nil {
            logger.Fatal("Falha ao treinar Bagging", zap.Error(err))
        }
        mdl = bg
        path = "models/bag_model.gob"
    case "gb":
        gb := models.NewGradientBoosting()
        gb.NEstimators = *estimators
        gb.LearningRate = *lr
        gb.MinSamples = *minSamples
        if err := gb.Fit(Xtrain, ytrain); err != nil {
            logger.Fatal("Falha ao treinar GradientBoosting", zap.Error(err))
        }
        mdl = gb
        path = "models/gb_model.gob"
    case "lgbm":
        lgbm := models.NewLightGBMCLI()
        if *maxDepth > 0 { lgbm.MaxDepth = *maxDepth; lgbm.NumLeaves = int(math.Pow(2, float64(*maxDepth))) }
        lgbm.MinDataInLeaf = *minSamples
        lgbm.NumIterations = *estimators
        lgbm.LearningRate = *lr
        lgbm.Device = "gpu"
        if err := lgbm.Fit(Xtrain, ytrain); err != nil {
            logger.Fatal("Falha ao treinar LightGBM", zap.Error(err))
        }
        mdl = lgbm
        path = "models/lgbm_model.gob"
    default:
        dt := models.NewDecisionTree()
        dt.MaxDepth = *maxDepth
        dt.MinSamplesSplit = *minSamples
        if err := dt.Fit(Xtrain, ytrain); err != nil {
            logger.Fatal("Falha ao treinar DT", zap.Error(err))
        }
        mdl = dt
        path = "models/dt_model.gob"
    }

    probaTest := mdl.PredictProba(Xtest)
    valSize := int(0.1 * float64(len(Xtrain)))
    if valSize < 100 { valSize = 100 }
    if valSize > len(Xtrain) { valSize = len(Xtrain) }
    valX := Xtrain[len(Xtrain)-valSize:]
    valY := ytrain[len(ytrain)-valSize:]
    probaVal := mdl.PredictProba(valX)
    thrUsed := *threshold
    if *thresholdAuto {
        if *thresholdMetric == "acc" { thrUsed, _ = bestThresholdAcc(valY, probaVal) } else { thrUsed, _ = bestThresholdF1(valY, probaVal) }
    }
    if thrUsed < *thrMin { thrUsed = *thrMin }
    if thrUsed > *thrMax { thrUsed = *thrMax }
    preds := probaToPred(probaTest, thrUsed)
    acc := accuracy(ytest, preds)
    prec, rec, f1 := prf1(ytest, probaTest, thrUsed)
    roc := rocAUC(ytest, probaTest)
    pr := prAUC(ytest, probaTest)
    logger.Info("Métricas holdout",
        zap.String("model", mdl.Name()),
        zap.Float64("accuracy", acc),
        zap.Float64("f1", f1),
        zap.Float64("precision", prec),
        zap.Float64("recall", rec),
        zap.Float64("roc_auc", roc),
        zap.Float64("pr_auc", pr),
        zap.Float64("threshold", thrUsed),
    )

    if err := os.MkdirAll("models", 0o755); err != nil { logger.Fatal("mkdir models", zap.Error(err)) }
    mf, err := os.Create(path)
    if err != nil { logger.Fatal("criar modelo", zap.Error(err)) }
    defer mf.Close()
    enc := gob.NewEncoder(mf)
    if err := enc.Encode(mdl); err != nil { logger.Fatal("serializar modelo", zap.Error(err)) }
    logger.Info("Modelo salvo", zap.String("path", path))
    fmt.Println("Modelo:", mdl.Name())

    if *curve {
        sizes := computeCurveSizes(len(Xtrain), *curvePoints, *curveMin, *curveLog)
        trainAcc := make([]float64, len(sizes))
        testAcc := make([]float64, len(sizes))
        trainF1 := make([]float64, len(sizes))
        testF1 := make([]float64, len(sizes))
        trainROC := make([]float64, len(sizes))
        testROC := make([]float64, len(sizes))
        trainPR := make([]float64, len(sizes))
        testPR := make([]float64, len(sizes))
        for k, s := range sizes {
            subX := Xtrain[:s]
            subY := ytrain[:s]
            cm := constructModel(*algo, *estimators, *maxDepth, *minSamples, *lr)
            if err := cm.Fit(subX, subY); err != nil { logger.Fatal("Falha ao treinar no ponto da curva", zap.Error(err)) }
            probaTrain := cm.PredictProba(subX)
            probaTest := cm.PredictProba(Xtest)
            vs := int(0.1 * float64(len(subX)))
            if vs < 50 { vs = 50 }
            if vs > len(subX) { vs = len(subX) }
            vX := subX[len(subX)-vs:]
            vY := subY[len(subY)-vs:]
            probaV := cm.PredictProba(vX)
            thrCurve := *threshold
            if *thresholdAuto {
                if *thresholdMetric == "acc" { thrCurve, _ = bestThresholdAcc(vY, probaV) } else { thrCurve, _ = bestThresholdF1(vY, probaV) }
            }
            if thrCurve < *thrMin { thrCurve = *thrMin }
            if thrCurve > *thrMax { thrCurve = *thrMax }
            pTrain := probaToPred(probaTrain, thrCurve)
            pTest := probaToPred(probaTest, thrCurve)
            trainAcc[k] = accuracy(subY, pTrain)
            testAcc[k] = accuracy(ytest, pTest)
            _, _, f1 := prf1(subY, probaTrain, thrCurve)
            trainF1[k] = f1
            trainROC[k] = rocAUC(subY, probaTrain)
            trainPR[k] = prAUC(subY, probaTrain)
            _, _, f1 = prf1(ytest, probaTest, thrCurve)
            testF1[k] = f1
            testROC[k] = rocAUC(ytest, probaTest)
            testPR[k] = prAUC(ytest, probaTest)
        }
        if err := writeCurveCSV(*curveCsv, sizes, trainAcc, testAcc, trainF1, testF1, trainROC, testROC, trainPR, testPR); err != nil {
            logger.Warn("Falha ao salvar CSV da curva", zap.Error(err))
        }
        if err := plotCurvePNG(*curveImg, sizes, trainAcc, testAcc, trainF1, testF1); err != nil {
            logger.Warn("Falha ao salvar PNG da curva", zap.Error(err))
        } else {
            logger.Info("Curva de aprendizagem gerada", zap.String("png", *curveImg), zap.String("csv", *curveCsv))
        }
    }
}

func accuracy(y, p []int) float64 {
    if len(y) == 0 { return 0 }
    c := 0
    for i := range y { if y[i] == p[i] { c++ } }
    return float64(c)/float64(len(y))
}

func constructModel(algo string, estimators, maxDepth, minSamples int, lr float64) models.Model {
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
    case "lgbm":
        lgbm := models.NewLightGBMCLI()
        if maxDepth > 0 { lgbm.MaxDepth = maxDepth; lgbm.NumLeaves = int(math.Pow(2, float64(maxDepth))) }
        lgbm.MinDataInLeaf = minSamples
        lgbm.NumIterations = estimators
        lgbm.LearningRate = lr
        lgbm.Device = "gpu"
        return lgbm
    default:
        dt := models.NewDecisionTree()
        dt.MaxDepth = maxDepth
        dt.MinSamplesSplit = minSamples
        return dt
    }
}

func writeCurveCSV(path string, sizes []int, trainAcc, testAcc, trainF1, testF1, trainROC, testROC, trainPR, testPR []float64) error {
    if err := os.MkdirAll("data", 0o755); err != nil { return err }
    f, err := os.Create(path)
    if err != nil { return err }
    defer f.Close()
    w := csv.NewWriter(f)
    defer w.Flush()
    if err := w.Write([]string{"size", "train_acc", "test_acc", "train_f1", "test_f1", "train_roc_auc", "test_roc_auc", "train_pr_auc", "test_pr_auc"}); err != nil { return err }
    for i := range sizes {
        rec := []string{strconv.Itoa(sizes[i]), fmt.Sprintf("%.6f", trainAcc[i]), fmt.Sprintf("%.6f", testAcc[i]),
            fmt.Sprintf("%.6f", trainF1[i]), fmt.Sprintf("%.6f", testF1[i]),
            fmt.Sprintf("%.6f", trainROC[i]), fmt.Sprintf("%.6f", testROC[i]),
            fmt.Sprintf("%.6f", trainPR[i]), fmt.Sprintf("%.6f", testPR[i]),
        }
        if err := w.Write(rec); err != nil { return err }
    }
    return nil
}

func plotCurvePNG(path string, sizes []int, trainAcc, testAcc, trainF1, testF1 []float64) error {
    p := plot.New()
    p.Title.Text = "Curva de Aprendizagem"
    p.X.Label.Text = "Amostras de treino"
    p.Y.Label.Text = "Métrica"
    p.Y.Min = 0
    p.Y.Max = 1

    toXY := func(xs []int, ys []float64) plotter.XYs {
        pts := make(plotter.XYs, len(xs))
        for i := range xs { pts[i].X = float64(xs[i]); pts[i].Y = ys[i] }
        return pts
    }
    trAcc := toXY(sizes, trainAcc)
    teAcc := toXY(sizes, testAcc)
    trF1Pts := toXY(sizes, trainF1)
    teF1Pts := toXY(sizes, testF1)
    if err := plotutil.AddLinePoints(p, "Treino (Acc)", trAcc, "Teste (Acc)", teAcc, "Treino (F1)", trF1Pts, "Teste (F1)", teF1Pts); err != nil { return err }
    if err := os.MkdirAll("cmd/api/static", 0o755); err != nil { return err }
    return p.Save(8*vg.Inch, 4*vg.Inch, path)
}
func computeCurveSizes(totalTrain, points, min int, useLog bool) []int {
    if points <= 1 { points = 2 }
    if min < 10 { min = 10 }
    if min > totalTrain { min = int(math.Max(10, float64(totalTrain)/2)) }
    sizes := make([]int, 0, points)
    if useLog {
        ratio := math.Pow(float64(totalTrain)/float64(min), 1.0/float64(points-1))
        for i := 0; i < points; i++ {
            s := int(math.Round(float64(min) * math.Pow(ratio, float64(i))))
            if s > totalTrain { s = totalTrain }
            sizes = append(sizes, s)
        }
    } else {
        step := float64(totalTrain-min) / float64(points-1)
        for i := 0; i < points; i++ {
            s := int(math.Round(float64(min) + float64(i)*step))
            if s > totalTrain { s = totalTrain }
            sizes = append(sizes, s)
        }
    }
    cleaned := make([]int, 0, len(sizes))
    last := -1
    for _, s := range sizes {
        if s <= last { s = last + 1 }
        if s > totalTrain { s = totalTrain }
        if s != last { cleaned = append(cleaned, s); last = s }
    }
    if cleaned[len(cleaned)-1] != totalTrain {
        cleaned[len(cleaned)-1] = totalTrain
    }
    return cleaned
}

func probaToPred(ps []float64, thr float64) []int {
    out := make([]int, len(ps))
    for i := range ps { if ps[i] >= thr { out[i] = 1 } }
    return out
}

func confusion(y []int, ps []float64, thr float64) (tp, fp, tn, fn int) {
    for i := range y {
        pred := 0
        if ps[i] >= thr { pred = 1 }
        if pred == 1 && y[i] == 1 { tp++ } else if pred == 1 && y[i] == 0 { fp++ } else if pred == 0 && y[i] == 0 { tn++ } else if pred == 0 && y[i] == 1 { fn++ }
    }
    return
}

func prf1(y []int, ps []float64, thr float64) (precision, recall, f1 float64) {
    tp, fp, _, fn := confusion(y, ps, thr)
    if tp+fp > 0 { precision = float64(tp) / float64(tp+fp) }
    if tp+fn > 0 { recall = float64(tp) / float64(tp+fn) }
    if precision+recall > 0 { f1 = 2 * precision * recall / (precision + recall) }
    return
}

func rocAUC(y []int, ps []float64) float64 {
    type pair struct{ s float64; y int }
    n := len(y)
    pairs := make([]pair, n)
    for i := 0; i < n; i++ { pairs[i] = pair{ps[i], y[i]} }
    sort.Slice(pairs, func(i, j int) bool { return pairs[i].s > pairs[j].s })
    var pos, neg int
    for _, p := range pairs { if p.y == 1 { pos++ } else { neg++ } }
    if pos == 0 || neg == 0 { return 0 }
    tp, fp := 0, 0
    prevS := math.Inf(1)
    var auc float64
    prevTPR, prevFPR := 0.0, 0.0
    for i := 0; i < n; i++ {
        if pairs[i].s != prevS {
            tpr := float64(tp) / float64(pos)
            fpr := float64(fp) / float64(neg)
            auc += (fpr - prevFPR) * (tpr + prevTPR) / 2.0
            prevTPR, prevFPR = tpr, fpr
            prevS = pairs[i].s
        }
        if pairs[i].y == 1 { tp++ } else { fp++ }
    }
    tpr := float64(tp) / float64(pos)
    fpr := float64(fp) / float64(neg)
    auc += (fpr - prevFPR) * (tpr + prevTPR) / 2.0
    return auc
}

func prAUC(y []int, ps []float64) float64 {
    type pair struct{ s float64; y int }
    n := len(y)
    pairs := make([]pair, n)
    for i := 0; i < n; i++ { pairs[i] = pair{ps[i], y[i]} }
    sort.Slice(pairs, func(i, j int) bool { return pairs[i].s > pairs[j].s })
    var tp, fp, fn int
    for _, p := range pairs { if p.y == 1 { fn++ } }
    var prevRec, auc float64
    for i := 0; i < n; i++ {
        if pairs[i].y == 1 { tp++; fn-- } else { fp++ }
        var prec, rec float64
        if tp+fp > 0 { prec = float64(tp) / float64(tp+fp) }
        if tp+fn > 0 { rec = float64(tp) / float64(tp+fn) }
        auc += (rec - prevRec) * prec
        prevRec = rec
    }
    return auc
}

func bestThresholdF1(y []int, ps []float64) (thr float64, best float64) {
    if len(ps) == 0 { return 0.5, 0 }
    steps := 200
    best = -1
    thr = 0.5
    for i := 0; i <= steps; i++ {
        t := float64(i) / float64(steps)
        _, _, f1 := prf1(y, ps, t)
        if f1 > best { best = f1; thr = t }
    }
    return
}

func bestThresholdAcc(y []int, ps []float64) (thr float64, best float64) {
    if len(ps) == 0 { return 0.5, 0 }
    steps := 200
    best = -1
    thr = 0.5
    for i := 0; i <= steps; i++ {
        t := float64(i) / float64(steps)
        p := probaToPred(ps, t)
        a := accuracy(y, p)
        if a > best { best = a; thr = t }
    }
    return
}
