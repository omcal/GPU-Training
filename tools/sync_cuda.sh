#!/usr/bin/env bash
# Copies source changes from this Mac to the cuda SSH host. No remote deletion.
# Preview: bash tools/sync_cuda.sh --dry-run
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec rsync -azv "$@" \
  --exclude='.venv/' --exclude='.uv-cache/' --exclude='.swiftcache/' \
  --exclude='.pytest_cache/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.git/' --exclude='build/' --exclude='dist/' --exclude='out/' \
  "$root/" cuda:GPU-Training/
