#!/usr/bin/env bash
set -euo pipefail
MSG="${1:-chore: add week3 baselines}"
git add -A
git commit -m "$MSG" || true
git push
