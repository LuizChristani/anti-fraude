package features

import (
    "strings"
    "time"

    "antifraude/internal/data"
)

var cats = []string{"Alimentação", "Transporte", "Taxi", "Pedágio", "Hospedagem"}

func Vectorize(e data.Expense) ([]float64, []string) {
    names := []string{}
    vec := []float64{}

    names = append(names, "Amount")
    vec = append(vec, e.Amount)

    intervalDays := float64(int(e.TravelDate.Sub(e.RequestDate).Hours() / 24))
    names = append(names, "IntervaloSolicitante")
    vec = append(vec, intervalDays)

    names = append(names, "DiaSemana")
    vec = append(vec, float64(int(e.RequestDate.Weekday())))
    names = append(names, "Mes")
    vec = append(vec, float64(int(e.RequestDate.Month())))

    sameApprover := boolToFloat(e.ApproverID == e.RequesterID)
    reqIsTraveller := boolToFloat(e.RequesterID == e.TravellerID)
    valorInteiro := boolToFloat(e.Amount == float64(int(e.Amount)))
    valorMultiplo5 := boolToFloat(int(e.Amount)%5 == 0)
    names = append(names, "MesmoAprovador", "SolicitanteViajante", "ValorInteiro", "ValorMultiplo5")
    vec = append(vec, sameApprover, reqIsTraveller, valorInteiro, valorMultiplo5)

    catLower := strings.ToLower(e.Category)
    for _, c := range cats {
        names = append(names, "Cat_"+c)
        if strings.ToLower(c) == catLower {
            vec = append(vec, 1.0)
        } else {
            vec = append(vec, 0.0)
        }
    }

    return vec, names
}

func boolToFloat(b bool) float64 { if b { return 1.0 } ; return 0.0 }

func BuildExpense(
    expenseID, requestID, requesterID, travellerID, approverID string,
    requestDate, travelDate time.Time,
    category, description string,
    amount float64,
    currency, jobTitle, department, approvalStatus string,
) data.Expense {
    return data.Expense{
        ExpenseID:      expenseID,
        RequestID:      requestID,
        RequesterID:    requesterID,
        TravellerID:    travellerID,
        ApproverID:     approverID,
        RequestDate:    requestDate,
        TravelDate:     travelDate,
        Category:       category,
        Description:    description,
        Amount:         amount,
        Currency:       currency,
        JobTitle:       jobTitle,
        Department:     department,
        ApprovalStatus: approvalStatus,
    }
}