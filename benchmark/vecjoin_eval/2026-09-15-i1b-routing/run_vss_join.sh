#!/usr/bin/env bash
# CPU-only; waits for any Sirius build to finish so the 64 cores are not shared with ninja.
set -u
E="$(cd "$(dirname "$0")" && pwd)"
until ! pgrep -x ninja >/dev/null; do sleep 20; done
for v in 1.4.4 1.5.4; do
  rm -f /var/tmp/vj/vssjoin/v$v.db
  echo "##### duckdb v$v $(date -u +%FT%TZ)"
  timeout 3600 /var/tmp/ddb_old/v$v/duckdb /var/tmp/vj/vssjoin/v$v.db < "$E/vss_join.sql" 2>&1
  echo "##### rc=$?"
done
