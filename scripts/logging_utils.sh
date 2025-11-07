#!/bin/bash
# Simple logging utility

setup_log() {
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_DIR="logs/$1"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
    echo "Log file: $LOG_FILE"
}

