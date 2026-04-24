#!/bin/bash
# Weekend trainer: relaunches run.py on any crash, resuming from the
# atomic latest_model.pt checkpoint that run.py writes every ~60 s.
# Exits only on clean completion or a second consecutive failure within
# the same minute (to prevent a tight crash loop if something is really
# broken).

set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1     # stream stdout/stderr in real time through tee

LOG="train.log"
FAIL_WINDOW=60            # seconds
MAX_FAST_FAILS=3          # fail this many times in a row inside FAIL_WINDOW => bail

echo "=== run_weekend.sh starting at $(date -Is) ===" | tee -a "$LOG"
fails=0
last_fail_ts=0

while true; do
    start_ts=$(date +%s)
    python3 run.py 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    end_ts=$(date +%s)

    if [ "$status" -eq 0 ]; then
        echo "=== run.py completed cleanly at $(date -Is) ===" | tee -a "$LOG"
        break
    fi

    # Distinguish transient failures from a tight crash loop.
    if [ $(( end_ts - start_ts )) -lt $FAIL_WINDOW ]; then
        if [ $(( end_ts - last_fail_ts )) -lt $FAIL_WINDOW ]; then
            fails=$(( fails + 1 ))
        else
            fails=1
        fi
        last_fail_ts=$end_ts
        if [ $fails -ge $MAX_FAST_FAILS ]; then
            echo "=== $fails fast failures in a row, giving up at $(date -Is) ===" \
                | tee -a "$LOG"
            exit 1
        fi
    else
        fails=0
    fi

    echo "=== run.py exited $status at $(date -Is); restarting from latest checkpoint in 10s ===" \
        | tee -a "$LOG"
    sleep 10
done
