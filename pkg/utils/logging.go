package utils

import (
    "os"
    "path/filepath"
    "go.uber.org/zap"
    "go.uber.org/zap/zapcore"
)

var logger *zap.Logger

func Logger() *zap.Logger {
    if logger != nil { return logger }
    logFile := os.Getenv("LOG_FILE")
    if logFile == "" {
        l, _ := zap.NewProduction()
        logger = l
        return logger
    }
    _ = os.MkdirAll(filepath.Dir(logFile), 0o755)
    f, err := os.OpenFile(logFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
    if err != nil {
        l, _ := zap.NewProduction()
        logger = l
        return logger
    }
    encCfg := zap.NewProductionEncoderConfig()
    enc := zapcore.NewJSONEncoder(encCfg)
    lvl := zapcore.InfoLevel
    fileCore := zapcore.NewCore(enc, zapcore.AddSync(f), lvl)
    consoleCore := zapcore.NewCore(enc, zapcore.AddSync(os.Stdout), lvl)
    logger = zap.New(zapcore.NewTee(fileCore, consoleCore))
    return logger
}