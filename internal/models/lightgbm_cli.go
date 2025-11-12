package models

import (
    "bufio"
    "errors"
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
)

type LightGBMCLI struct {
    ExecPath       string
    NumLeaves      int
    MaxDepth       int
    MinDataInLeaf  int
    NumIterations  int
    LearningRate   float64
    Device         string
    ModelPath      string
}

func NewLightGBMCLI() *LightGBMCLI {
    return &LightGBMCLI{
        ExecPath:     "lightgbm",
        NumLeaves:    31,
        MaxDepth:     -1,
        MinDataInLeaf: 100,
        NumIterations: 200,
        LearningRate: 0.1,
        Device:       "gpu",
        ModelPath:    filepath.Join("models", "lgbm_model.txt"),
    }
}

func (l *LightGBMCLI) Name() string {
    if l.Device == "gpu" { return "LightGBM(GPU)" }
    return "LightGBM(CPU)"
}

func (l *LightGBMCLI) Fit(X [][]float64, y []int) error {
    if len(X) == 0 { return nil }
    if err := os.MkdirAll("models", 0o755); err != nil { return err }
    if err := os.MkdirAll("data", 0o755); err != nil { return err }

    trainCSV := filepath.Join("data", "lgbm_train.csv")
    if err := writeCSVLabelFirst(trainCSV, X, y); err != nil { return err }

    conf := filepath.Join("data", "lgbm_train.conf")
    device := l.Device
    if device == "" { device = "gpu" }

    cfg := fmt.Sprintf("task=train\nboosting=gbdt\nobjective=binary\nmetric=auc\n"+
        "data=%s\nheader=false\nlabel_column=0\n"+
        "num_leaves=%d\nmax_depth=%d\nmin_data_in_leaf=%d\n"+
        "num_iterations=%d\nlearning_rate=%f\n"+
        "device=%s\ntree_learner=%s\noutput_model=%s\n",
        trainCSV, l.NumLeaves, l.MaxDepth, l.MinDataInLeaf, l.NumIterations, l.LearningRate,
        device, ternary(device == "gpu", "gpu", "serial"), l.ModelPath,
    )
    if err := os.WriteFile(conf, []byte(cfg), 0o644); err != nil { return err }

    cmd := exec.Command(l.ExecPath, fmt.Sprintf("config=%s", conf))
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    if err := cmd.Run(); err != nil {
        return errors.New("falha ao executar LightGBM CLI (verifique se 'lightgbm' está instalado e no PATH)")
    }
    if _, err := os.Stat(l.ModelPath); err != nil {
        return errors.New("modelo do LightGBM não encontrado após treinamento")
    }
    return nil
}

func (l *LightGBMCLI) Predict(X [][]float64) []int {
    ps := l.PredictProba(X)
    out := make([]int, len(ps))
    for i := range ps { if ps[i] >= 0.5 { out[i] = 1 } }
    return out
}

func (l *LightGBMCLI) PredictProba(X [][]float64) []float64 {
    if len(X) == 0 { return []float64{} }
    predCSV := filepath.Join("data", "lgbm_pred.csv")
    zeros := make([]int, len(X))
    if err := writeCSVLabelFirst(predCSV, X, zeros); err != nil { return []float64{} }

    conf := filepath.Join("data", "lgbm_predict.conf")
    outPath := filepath.Join("data", "lgbm_preds.txt")
    cfg := fmt.Sprintf("task=predict\ninput_model=%s\ndata=%s\nheader=false\nlabel_column=0\noutput_result=%s\n",
        l.ModelPath, predCSV, outPath,
    )
    if err := os.WriteFile(conf, []byte(cfg), 0o644); err != nil { return []float64{} }

    cmd := exec.Command(l.ExecPath, fmt.Sprintf("config=%s", conf))
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    if err := cmd.Run(); err != nil { return []float64{} }

    f, err := os.Open(outPath)
    if err != nil { return []float64{} }
    defer f.Close()
    sc := bufio.NewScanner(f)
    ps := make([]float64, 0, len(X))
    for sc.Scan() {
        var v float64
        if _, err := fmt.Sscan(sc.Text(), &v); err == nil { ps = append(ps, v) }
    }
    return ps
}

func writeCSVLabelFirst(path string, X [][]float64, y []int) error {
    f, err := os.Create(path)
    if err != nil { return err }
    defer f.Close()
    w := bufio.NewWriter(f)
    for i := range X {
        fmt.Fprintf(w, "%d", y[i])
        for j := range X[i] {
            fmt.Fprintf(w, ",%g", X[i][j])
        }
        fmt.Fprintln(w)
    }
    return w.Flush()
}

func ternary[T any](cond bool, a, b T) T { if cond { return a } ; return b }