package main

import (
    "encoding/csv"
    "encoding/gob"
    "net/http"
    "os"
    "path/filepath"
    "strconv"
    "strings"
    "time"

    "github.com/gin-gonic/gin"

    "antifraude/internal/features"
    "antifraude/internal/models"
    "antifraude/pkg/utils"
)

type ruleModel struct{}

func (r *ruleModel) Fit(X [][]float64, y []int) error { return nil }
func (r *ruleModel) Predict(X [][]float64) []int {
    out := make([]int, len(X))
    for i, v := range X {
        p := r.score(v)
        if p >= 0.5 { out[i] = 1 }
    }
    return out
}
func (r *ruleModel) PredictProba(X [][]float64) []float64 {
    out := make([]float64, len(X))
    for i, v := range X { out[i] = r.score(v) }
    return out
}
func (r *ruleModel) Name() string { return "RuleModel" }
func (r *ruleModel) score(v []float64) float64 {
    s := 0.05
    if v[4] == 1 { s += 0.35 }
    if v[5] == 1 { s += 0.1 }
    if v[6] == 1 { s += 0.15 }
    if v[7] == 1 { s += 0.15 }
    if v[len(v)-3] == 1 && v[0] > 200 { s += 0.2 }
    if v[1] < 0 { s += 0.3 }
    if s > 0.95 { s = 0.95 }
    return s
}

var model models.Model

type catRule struct { Min float64; Max float64; HardMax float64 }
var categoryRules = map[string]catRule{
    "alimentação": {Min: 5, Max: 300, HardMax: 1500},
    "transporte":  {Min: 10, Max: 800, HardMax: 5000},
    "taxi":        {Min: 10, Max: 300, HardMax: 2000},
    "pedágio":     {Min: 2, Max: 200, HardMax: 5000},
    "hospedagem":  {Min: 80, Max: 600, HardMax: 5000},
}

func validateAmount(category string, amount float64) (bool, string) {
    if amount <= 0 { return false, "valor deve ser maior que zero" }
    return true, ""
}

func detectAnomalies(category string, amount float64, reqDate, travelDate time.Time) []string {
    flags := []string{}
    r, ok := categoryRules[strings.ToLower(category)]
    if amount <= 0 {
        flags = append(flags, "valor não positivo")
    }
    if ok {
        if amount > r.HardMax {
            flags = append(flags, "valor acima do máximo permitido para a categoria")
        } else if amount > r.Max {
            flags = append(flags, "valor acima da faixa típica da categoria")
        } else if amount < r.Min {
            flags = append(flags, "valor abaixo da faixa típica da categoria")
        }
    }
    if travelDate.Before(reqDate) {
        flags = append(flags, "data de viagem anterior à solicitação")
    }
    return flags
}

func riskWithAnomalies(p float64, category string, amount float64, reqDate, travelDate time.Time, flags []string) string {
    base := riskBandWithCategory(p, category, amount)
    critical := false
    r, ok := categoryRules[strings.ToLower(category)]
    if amount <= 0 || travelDate.Before(reqDate) {
        critical = true
    }
    if ok && amount > r.HardMax {
        critical = true
    }
    if critical { return "alto" }
    if ok && amount > r.Max && base == "muito_baixo" {
        return "medio"
    }
    return base
}

func main() {
    logger := utils.Logger()
    defer logger.Sync()

    algo := strings.ToLower(os.Getenv("MODEL_ALGO"))
    if algo == "" { algo = "dt" }
    var path string
    switch algo {
    case "rf":
        path = filepath.Join("models", "rf_model.gob")
        if f, err := os.Open(path); err == nil {
            defer f.Close()
            dec := gob.NewDecoder(f)
            var rf models.RandomForest
            if err := dec.Decode(&rf); err == nil && len(rf.Trees) > 0 {
                model = &rf
            }
        }
    case "bagging":
        path = filepath.Join("models", "bag_model.gob")
        if f, err := os.Open(path); err == nil {
            defer f.Close()
            dec := gob.NewDecoder(f)
            var bg models.Bagging
            if err := dec.Decode(&bg); err == nil && len(bg.Trees) > 0 {
                model = &bg
            }
        }
    case "gb":
        path = filepath.Join("models", "gb_model.gob")
        if f, err := os.Open(path); err == nil {
            defer f.Close()
            dec := gob.NewDecoder(f)
            var gb models.GradientBoosting
            if err := dec.Decode(&gb); err == nil && len(gb.Trees) > 0 {
                model = &gb
            }
        }
    default:
        path = filepath.Join("models", "dt_model.gob")
        if f, err := os.Open(path); err == nil {
            defer f.Close()
            dec := gob.NewDecoder(f)
            var dt models.DecisionTree
            if err := dec.Decode(&dt); err == nil && dt.Root != nil {
                model = &dt
            }
        }
    }
    if model == nil { model = &ruleModel{} }

    r := gin.Default()

    r.Static("/static", "cmd/api/static")
    r.GET("/dashboard", func(c *gin.Context) {
        c.File("cmd/api/static/index.html")
    })
    r.GET("/dashboard/data", dashboardData)
    r.GET("/dashboard/metrics", dashboardMetrics)

    api := r.Group("/")
    api.Use(apiKeyMiddleware)
    api.POST("/predict", handlePredict)
    api.POST("/batch", handleBatch)

    port := os.Getenv("PORT")
    if port == "" { port = "8080" }
    r.Run(":" + port)
}

func apiKeyMiddleware(c *gin.Context) {
    key := os.Getenv("API_KEY")
    if key == "" { c.Next(); return }
    got := c.GetHeader("X-API-Key")
    if got != key { c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"}); return }
    c.Next()
}

type predictReq struct {
    ExpenseID      string `json:"expense_id"`
    RequestID      string `json:"request_id"`
    RequesterID    string `json:"requester_id"`
    TravellerID    string `json:"traveller_id"`
    ApproverID     string `json:"approver_id"`
    RequestDate    string `json:"request_date"`
    TravelDate     string `json:"travel_date"`
    Category       string `json:"category"`
    Description    string `json:"description"`
    Amount         float64 `json:"amount"`
    Currency       string `json:"currency"`
    JobTitle       string `json:"job_title"`
    Department     string `json:"department"`
    ApprovalStatus string `json:"approval_status"`
}

func handlePredict(c *gin.Context) {
    var req predictReq
    if err := c.BindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "invalid json"}); return
    }
    rd, _ := time.Parse("2006-01-02", req.RequestDate)
    td, _ := time.Parse("2006-01-02", req.TravelDate)
    e := features.BuildExpense(req.ExpenseID, req.RequestID, req.RequesterID, req.TravellerID, req.ApproverID,
        rd, td, req.Category, req.Description, req.Amount, req.Currency, req.JobTitle, req.Department, req.ApprovalStatus)
    v, _ := features.Vectorize(e)
    p := model.PredictProba([][]float64{v})[0]
    flags := detectAnomalies(req.Category, req.Amount, rd, td)
    risk := riskWithAnomalies(p, req.Category, req.Amount, rd, td, flags)
    c.JSON(http.StatusOK, gin.H{"score": p, "risk": risk, "model": model.Name(), "flags": flags})
}

func handleBatch(c *gin.Context) {
    var items []predictReq
    if err := c.BindJSON(&items); err != nil { c.JSON(http.StatusBadRequest, gin.H{"error": "invalid json"}); return }
    X := make([][]float64, 0, len(items))
    for _, it := range items {
        rd, _ := time.Parse("2006-01-02", it.RequestDate)
        td, _ := time.Parse("2006-01-02", it.TravelDate)
        e := features.BuildExpense(it.ExpenseID, it.RequestID, it.RequesterID, it.TravellerID, it.ApproverID,
            rd, td, it.Category, it.Description, it.Amount, it.Currency, it.JobTitle, it.Department, it.ApprovalStatus)
        v, _ := features.Vectorize(e)
        X = append(X, v)
    }
    ps := model.PredictProba(X)
    out := make([]gin.H, len(items))
    for i := range items {
        rd, _ := time.Parse("2006-01-02", items[i].RequestDate)
        td, _ := time.Parse("2006-01-02", items[i].TravelDate)
        flags := detectAnomalies(items[i].Category, items[i].Amount, rd, td)
        out[i] = gin.H{
            "score": ps[i],
            "risk": riskWithAnomalies(ps[i], items[i].Category, items[i].Amount, rd, td, flags),
            "flags": flags,
        }
    }
    c.JSON(http.StatusOK, out)
}

func riskBand(p float64) string {
    switch {
    case p >= 0.95:
        return "alto"
    case p >= 0.7:
        return "medio"
    case p >= 0.5:
        return "baixo"
    default:
        return "muito_baixo"
    }
}

func riskBandWithCategory(p float64, category string, amount float64) string {
    base := riskBand(p)
    r, ok := categoryRules[strings.ToLower(category)]
    if !ok { return base }
    if amount > r.Max {
        if p < 0.7 { return "medio" }
        return "alto"
    }
    if amount < r.Min && p < 0.7 {
        return "baixo"
    }
    return base
}

func dashboardData(c *gin.Context) {
    path := "data/synthetic.csv"
    f, err := os.Open(path)
    if err != nil { c.JSON(http.StatusOK, gin.H{"items": []gin.H{}}); return }
    defer f.Close()
    r := csv.NewReader(f)
    rows, err := r.ReadAll()
    if err != nil || len(rows) < 2 { c.JSON(http.StatusOK, gin.H{"items": []gin.H{}}); return }
    max := 200
    items := make([]gin.H, 0, max)
    for i := 1; i < len(rows) && len(items) < max; i++ {
        row := rows[i]
        rd, _ := time.Parse("2006-01-02", row[5])
        td, _ := time.Parse("2006-01-02", row[6])
        amt, _ := strconv.ParseFloat(row[9], 64)
        e := features.BuildExpense(row[0], row[1], row[2], row[3], row[4], rd, td, row[7], row[8], amt, row[10], row[11], row[12], row[13])
        v, _ := features.Vectorize(e)
        p := model.PredictProba([][]float64{v})[0]
        items = append(items, gin.H{
            "expense_id": row[0],
            "category": row[7],
            "amount": amt,
            "department": row[12],
            "date": row[5],
            "score": p,
            "risk": riskBand(p),
            "model": model.Name(),
        })
    }
    q := strings.ToLower(c.Query("category"))
    if q != "" {
        filt := make([]gin.H, 0, len(items))
        for _, it := range items { if strings.ToLower(it["category"].(string)) == q { filt = append(filt, it) } }
        items = filt
    }
    c.JSON(http.StatusOK, gin.H{"items": items})
}

func dashboardMetrics(c *gin.Context) {
    path := "data/learning_curve.csv"
    f, err := os.Open(path)
    if err != nil { c.JSON(http.StatusOK, gin.H{"metrics": gin.H{}}); return }
    defer f.Close()
    r := csv.NewReader(f)
    rows, err := r.ReadAll()
    if err != nil || len(rows) < 2 { c.JSON(http.StatusOK, gin.H{"metrics": gin.H{}}); return }
    hdr := rows[0]
    last := rows[len(rows)-1]
    vals := map[string]string{}
    for i := range hdr {
        if i < len(last) { vals[hdr[i]] = last[i] }
    }
    out := gin.H{}
    for _, k := range []string{"size","train_acc","test_acc","train_f1","test_f1","train_roc_auc","test_roc_auc","train_pr_auc","test_pr_auc"} {
        if v, ok := vals[k]; ok { out[k] = v }
    }
    c.JSON(http.StatusOK, gin.H{"metrics": out})
}