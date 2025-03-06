#!/bin/bash
LAST_RUN_FILE="$HOME/.last_run_script
TODAY=$(date +%s)

if [ -f "$LAST_RUN_FILE" ]; then
    LAST_RUN=$(cat "$LAST_RUN_FILE")
    DIFF=$(( (TODAY - LAST_RUN) / 86400 ))
    if [ $DIFF -lt 15 ]; then
        exit 0
    fi
fi

/usr/bin/python3 $HOME/insar/vinsar.py

date +%s > "$LAST_RUN_FILE"