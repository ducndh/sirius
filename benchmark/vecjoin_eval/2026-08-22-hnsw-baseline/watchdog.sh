#!/usr/bin/env bash
# Abort a benchmark if its RSS crosses a ceiling, instead of letting the box OOM.
# Usage: watchdog.sh <pid> <ceiling_gb>   -- runs until the pid exits.
pid=$1; ceil=${2:-40}
while kill -0 "$pid" 2>/dev/null; do
  rss=$(awk '/VmRSS/{print int($2/1048576)}' /proc/$pid/status 2>/dev/null || echo 0)
  if [ "${rss:-0}" -ge "$ceil" ]; then
    echo "WATCHDOG: pid $pid RSS ${rss}GB >= ${ceil}GB -- killing" >&2
    kill -9 "$pid"; exit 1
  fi
  sleep 2
done
