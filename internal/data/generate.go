package data

import (
	"encoding/csv"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

var categories = []string{"Alimentação", "Transporte", "Taxi", "Pedágio", "Hospedagem"}
var departments = []string{"Financeiro", "Comercial", "Operações", "Tecnologia", "RH"}
var jobTitles = []string{"Analista", "Coordenador", "Gerente", "Especialista", "Diretor"}

func GenerateSyntheticExpenses(n int, fraudRate float64, outPath string) error {
    if err := os.MkdirAll("data", 0o755); err != nil {
        return err
    }
    f, err := os.Create(outPath)
    if err != nil {
        return err
    }
    defer f.Close()

    w := csv.NewWriter(f)
    defer w.Flush()

    header := []string{"expense_id", "request_id", "requester_id", "traveller_id", "approver_id", "request_date", "travel_date", "category", "description", "amount", "currency", "job_title", "department", "approval_status", "fraud"}
    if err := w.Write(header); err != nil {
        return err
    }

    rand.Seed(time.Now().UnixNano())
    baseDate := time.Now().AddDate(-1, 0, 0)

    for i := 0; i < n; i++ {
        expenseID := "E" + strconv.Itoa(1000000+i)
        requestID := "R" + strconv.Itoa(500000+i)
        requesterID := "U" + strconv.Itoa(rand.Intn(5000))
        travellerID := requesterID
        if rand.Float64() < 0.2 {
            travellerID = "U" + strconv.Itoa(rand.Intn(5000))
        }
        approverID := "A" + strconv.Itoa(rand.Intn(800))
        if rand.Float64() < 0.03 {
            approverID = requesterID
        }

        reqOffset := rand.Intn(300)
        travelOffset := reqOffset + rand.Intn(30)
        if rand.Float64() < 0.02 {
            travelOffset = reqOffset - rand.Intn(5)
        }
        reqDate := baseDate.AddDate(0, 0, reqOffset)
        travelDate := baseDate.AddDate(0, 0, travelOffset)

        cat := categories[rand.Intn(len(categories))]
        words := []string{"almoço", "viagem", "hotel", "uber", "táxi", "pedágio", "combustível", "reunião", "cliente", "evento"}
        desc := cat + " " + words[rand.Intn(len(words))] + " " + words[rand.Intn(len(words))]

        currency := "BRL"
        amount := rand.Float64()*450 + 10
        round := rand.Float64() < 0.25
        multiple5 := rand.Float64() < 0.25
        if round {
            amount = float64(int(amount))
        }
        if multiple5 {
            amount = float64(5 * int(amount/5))
        }

        job := jobTitles[rand.Intn(len(jobTitles))]
        dept := departments[rand.Intn(len(departments))]

        status := "Aprovado"
        if rand.Float64() < 0.1 {
            status = "Reprovado"
        } else if rand.Float64() < 0.1 {
            status = "Pendente"
        }

        fraud := 0
        score := 0.0
        flags := 0
        if requesterID == approverID {
            score += 0.35
            flags++
        }
        if requesterID == travellerID {
            score += 0.1
            flags++
        }
        if round {
            score += 0.15
            flags++
        }
        if multiple5 {
            score += 0.15
            flags++
        }
        if travelDate.Before(reqDate) {
            score += 0.3
            flags++
        }
        if cat == "Taxi" && amount > 200 {
            score += 0.2
            flags++
        }
        base := fraudRate
        if flags >= 2 || travelDate.Before(reqDate) {
            fraud = 1
        } else if rand.Float64() < base+score {
            fraud = 1
        }

        rec := []string{
            expenseID,
            requestID,
            requesterID,
            travellerID,
            approverID,
            reqDate.Format("2006-01-02"),
            travelDate.Format("2006-01-02"),
            cat,
            strings.ToLower(desc),
            strconv.FormatFloat(amount, 'f', 2, 64),
            currency,
            job,
            dept,
            status,
            strconv.Itoa(fraud),
        }
        if err := w.Write(rec); err != nil {
            return err
        }
    }
    return nil
}
