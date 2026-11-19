#!/usr/bin/env bash
# Compose smoke test: build and start the stack, wait for health, submit a ticket, check the
# approval endpoint refuses a missing or wrong token, then tear everything down.
#
# No model is called. The API runs the gemini adapter with a placeholder key against an address
# nothing listens on (127.0.0.1:9 inside the api container), so every model call fails at once
# and the agent takes its safe fallback: code escalates the ticket over MCP. The ticket must come
# back `escalated` with stop `error`, and its escalation must be in the MCP server's database.
# The smoke ignores .env (its env_file is reset), so a real key there is never used.
#
# Usage: scripts/compose_smoke.sh
#   TW_COMPOSE        compose command, e.g. "docker compose" or "podman-compose"
#                     (default: the first that works of podman compose, podman-compose,
#                     docker compose)
#   TW_SMOKE_PORT     host port for the API (default 18080, apart from a dev stack on 8080)
#   TW_SMOKE_TIMEOUT  seconds to wait for health (default 180)
#   TW_SMOKE_PROJECT  compose project name (default tw-smoke-<pid>); its volume is removed at the end
# Exits non-zero on any failure.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${TW_COMPOSE:-}" ]]; then
    read -r -a compose <<<"$TW_COMPOSE"
elif podman compose version >/dev/null 2>&1; then
    compose=(podman compose)
elif command -v podman-compose >/dev/null 2>&1; then
    compose=(podman-compose)
elif docker compose version >/dev/null 2>&1; then
    compose=(docker compose)
else
    echo "compose_smoke: no compose found; set TW_COMPOSE" >&2
    exit 2
fi

port="${TW_SMOKE_PORT:-18080}"
timeout_s="${TW_SMOKE_TIMEOUT:-180}"
project="${TW_SMOKE_PROJECT:-tw-smoke-$$}"
base="http://127.0.0.1:${port}"
ticket_id="tkt_smoke$(date +%s)"

tmp="$(mktemp -d)"
override="$tmp/compose.smoke.yaml"
token="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
# Interpolated by compose: the host port, and a throwaway approval key for this run only.
export TW_API_PORT="$port"
TW_APPROVAL_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export TW_APPROVAL_KEY

cat >"$override" <<EOF
services:
  mcp:
    env_file: !reset []
  api:
    env_file: !reset []
    environment:
      TW_LLM_PROVIDER: gemini
      GEMINI_API_KEY: offline-smoke-placeholder
      TW_GEMINI_BASE_URL: http://127.0.0.1:9/v1beta/openai/
      TW_LLM_REQUESTS_PER_MINUTE: "0"
      TW_RECORD: "false"
      TW_APPROVER_TOKEN: "$token"
EOF

run() { "${compose[@]}" -p "$project" -f compose.yaml -f "$override" "$@"; }
curl() { command curl --noproxy '*' "$@"; }

fail() {
    echo "compose_smoke: FAIL: $*" >&2
    exit 1
}

down() {
    run down -v --remove-orphans >/dev/null 2>&1 || {
        echo "compose_smoke: FAIL: teardown (${compose[*]} -p $project down -v)" >&2
        return 1
    }
}

on_exit() {
    local status=$?
    trap - EXIT
    if ((status != 0)); then
        echo "--- logs (last 40 lines per service) ---" >&2
        run logs --tail 40 mcp >&2 || true
        run logs --tail 40 api >&2 || true
        down || true
    fi
    rm -rf "$tmp"
    exit "$status"
}
trap on_exit EXIT

echo "compose_smoke: ${compose[*]} (project $project, API on $base)"
started=$SECONDS
run up -d --build

# ---- health ----
deadline=$((SECONDS + timeout_s))
until curl -fsS --max-time 2 "$base/healthz" >/dev/null 2>&1; do
    ((SECONDS < deadline)) || fail "API not healthy within ${timeout_s}s"
    sleep 2
done
until run exec -T mcp python -c "import socket; socket.create_connection(('127.0.0.1', 8765), 2)" \
    >/dev/null 2>&1; do
    ((SECONDS < deadline)) || fail "MCP server not listening within ${timeout_s}s"
    sleep 2
done
echo "compose_smoke: healthy after $((SECONDS - started))s"

# ---- a ticket ----
body="$tmp/ticket.json"
cat >"$tmp/request.json" <<EOF
{"ticket_id": "$ticket_id", "customer_id": "cus_ava01", "channel": "chat",
 "subject": "Has my cable order shipped?",
 "body": "Hi! Has my order ord_a1002 shipped yet? Could you send me the tracking number?",
 "created_at": "2026-06-15T09:12:00Z"}
EOF
code="$(curl -sS --max-time 90 -o "$body" -w '%{http_code}' -H 'Content-Type: application/json' \
    --data @"$tmp/request.json" "$base/tickets")"
[[ "$code" == 200 ]] || fail "POST /tickets returned $code: $(cat "$body")"
python3 -I - "$body" "$ticket_id" <<'EOF' || fail "unexpected ticket response: $(cat "$body")"
import json, sys

r = json.load(open(sys.argv[1]))
assert r["ticket_id"] == sys.argv[2], r["ticket_id"]
assert r["request_id"].startswith("req_"), r["request_id"]
assert r["reply"].strip(), "empty reply"
# Offline: every model call fails, so code takes the safe fallback and hands off.
assert r["status"] == "escalated", r["status"]
assert r["stop"] == "error", r["stop"]
assert r["escalation"]["escalate"] is True, r["escalation"]
assert r["escalation"]["queue"] == "tier1", r["escalation"]
assert r["actions"] == [], r["actions"]
assert r["metrics"]["model_calls"] == [] and r["metrics"]["cost_usd"] == 0, r["metrics"]
print(f"compose_smoke: ticket -> {r['status']} ({r['stop']}), queue {r['escalation']['queue']}")
EOF
# The fallback's escalation went through MCP into its database (proves api -> mcp).
escalations="$(run exec -T mcp python -c "
import sqlite3, sys
db = sqlite3.connect('var/triagewright.sqlite')
print(db.execute('SELECT count(*) FROM escalations WHERE ticket_id = ?', (sys.argv[1],)).fetchone()[0])
" "$ticket_id" | tr -d '\r')"
[[ "$escalations" == 1 ]] || fail "expected 1 escalation for $ticket_id in MCP, got '$escalations'"

# ---- approvals: listing is open, decisions need the approver token ----
code="$(curl -sS --max-time 10 -o "$body" -w '%{http_code}' "$base/approvals")"
[[ "$code" == 200 ]] || fail "GET /approvals returned $code"
for auth in "" "Authorization: Bearer wrong-token"; do
    code="$(curl -sS --max-time 10 -o "$body" -D "$tmp/headers" -w '%{http_code}' \
        -H 'Content-Type: application/json' ${auth:+-H "$auth"} \
        --data '{"approved": true}' "$base/approvals/apr_smoke0000/decision")"
    [[ "$code" == 401 ]] || fail "decision with '${auth:-no token}' returned $code, not 401"
    python3 -I -c 'import json, sys; assert json.load(open(sys.argv[1]))["error"]["code"] == "unauthorized"' \
        "$body" || fail "401 body: $(cat "$body")"
    grep -qi '^www-authenticate: bearer' "$tmp/headers" || fail "401 without WWW-Authenticate"
done
# With the token the request gets past auth: the approval doesn't exist.
code="$(curl -sS --max-time 10 -o "$body" -w '%{http_code}' -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $token" --data '{"approved": true}' \
    "$base/approvals/apr_smoke0000/decision")"
[[ "$code" == 404 ]] || fail "decision with the token returned $code, not 404"
echo "compose_smoke: approval decisions refuse a missing or wrong token (401)"

down
echo "compose_smoke: PASS in $((SECONDS - started))s"
