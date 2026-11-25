# Security

## Reporting a vulnerability

Please report it privately through GitHub: **Security → Report a vulnerability** on this
repository. Don't open a public issue for it.

## What triagewright is built for

triagewright is a local-first demo of a support agent. It runs on one machine, over synthetic
data, on free-tier models. It is not hardened for the internet, and these are deliberate:

- **The HTTP API trusts localhost.** Compose publishes it on `127.0.0.1:8080` only.
  `POST /tickets` and `GET /approvals` need no credentials. Only approval decisions need one:
  `TW_APPROVER_TOKEN`, sent as a bearer token and compared in constant time. Unset, decisions
  are refused.
- **The MCP server has no client authentication.** Under Compose it isn't published outside
  the Compose network. Refunds still need an approval token signed with `TW_APPROVAL_KEY`
  (HMAC-SHA256), which the server verifies, re-applying the refund policy in the same
  transaction as the write.
- **No real data.** Customers, orders, and tickets are synthetic. Gemini free-tier prompts may
  be used for training, so nothing real belongs in a ticket.

Putting the API or the MCP server on a network others can reach needs authentication in front
of every endpoint, and is out of scope here.

## What's checked

CI runs ruff's security rules (bandit's `S` set), mypy, the test suite (Python 3.12 and 3.13),
`pip-audit` against `uv.lock`, `zizmor` on `.github/workflows/`, `scripts/compose_smoke.sh`, and
gitleaks over the full history on every pull request and on a weekly schedule. Secrets come from
the environment; only `.env.example` is committed.
