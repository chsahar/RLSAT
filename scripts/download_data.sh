#!/usr/bin/env bash
# Fetch SATLIB uf20-91 (1000 satisfiable 3-SAT instances, 20 vars / 91 clauses)
# into data/uf20-91/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/data/uf20-91"
URL="https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz"

mkdir -p "$TARGET"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

curl -fL "$URL" -o "$tmp"
tar -xzf "$tmp" -C "$TARGET" --strip-components=1 2>/dev/null \
  || tar -xzf "$tmp" -C "$TARGET"

count=$(find "$TARGET" -maxdepth 1 -name "*.cnf" | wc -l | tr -d ' ')
echo "extracted $count cnf files to $TARGET"

if [ "$count" -ne 1000 ]; then
  echo "warning: expected 1000 .cnf files, got $count" >&2
  exit 1
fi
