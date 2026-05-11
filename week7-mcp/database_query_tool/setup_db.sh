#!/usr/bin/env bash
# One-shot helper: download the Chinook PostgreSQL script and start the container.
# Idempotent — safe to re-run.

set -euo pipefail

cd "$(dirname "$0")"

SQL_FILE="Chinook_PostgreSql.sql"
SQL_URL="https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_PostgreSql.sql"

if [[ ! -f "$SQL_FILE" ]]; then
  echo "[setup] downloading Chinook SQL from lerocha/chinook-database ..."
  curl -fsSL "$SQL_URL" -o "$SQL_FILE"
fi

echo "[setup] starting Postgres + Chinook ..."
docker compose up -d

echo "[setup] waiting for the database to finish loading Chinook ..."
# We poll for the `album` table because the entrypoint reports the server as
# `pg_isready` long before `Chinook_PostgreSql.sql` has finished running. The
# table is the canonical "init complete" signal.
for i in {1..60}; do
  if docker compose exec -T chinook \
       psql -U chinook -d chinook -tAc "SELECT to_regclass('public.album')" 2>/dev/null \
       | grep -q '^album$'; then
    echo "[setup] ready — Chinook loaded."
    exit 0
  fi
  sleep 1
done

echo "[setup] timed out waiting for Chinook tables to appear."
echo "[setup] inspect with:  docker logs chinook-db | tail -40"
exit 1
