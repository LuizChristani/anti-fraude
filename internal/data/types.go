package data

import "time"

type Expense struct {
    ExpenseID      string    `json:"expense_id"`
    RequestID      string    `json:"request_id"`
    RequesterID    string    `json:"requester_id"`
    TravellerID    string    `json:"traveller_id"`
    ApproverID     string    `json:"approver_id"`
    RequestDate    time.Time `json:"request_date"`
    TravelDate     time.Time `json:"travel_date"`
    Category       string    `json:"category"`
    Description    string    `json:"description"`
    Amount         float64   `json:"amount"`
    Currency       string    `json:"currency"`
    JobTitle       string    `json:"job_title"`
    Department     string    `json:"department"`
    ApprovalStatus string    `json:"approval_status"`
    Fraud          int       `json:"fraud"`
}