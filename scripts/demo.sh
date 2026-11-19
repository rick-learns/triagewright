#!/usr/bin/env bash
# Offline demo: five steps of the real stack with a scripted model, for a 3-5 minute walkthrough
# (presenter notes in docs/demo.md). No API key, no network, nothing written outside a temp dir.
#
# 1. A grounded answer with verbatim KB citations (eval case happy-007).
# 2. A refund held for approval, then approved through the HTTP API's decision endpoint.
# 3. A prompt-injection ticket routed to the security queue, its refund blocked (inject-001).
# 4. A tool that keeps timing out, escalated to tier1 (toolfail-003).
# 5. The refund request's trace: tokens, cost, latency, and the span tree.
#
# Usage: scripts/demo.sh   (after `uv sync`)
# Exits non-zero if any step fails or the run takes longer than 60 seconds.
set -euo pipefail

cd "$(dirname "$0")/.."

# Keys and TW_* settings never reach the demo; demo_run.py drops them as well.
unset GEMINI_API_KEY OPENROUTER_API_KEY
while IFS= read -r var; do unset "$var"; done < <(compgen -v | grep -i '^tw_' || true)

start=$SECONDS
timeout 60 uv run --offline python docs/assets/demo_run.py
echo
echo "demo finished in $((SECONDS - start)) s, offline, with a scripted model"
