"""Lane H: command-line interface over the same ports as the API.

Exit codes: 0 ok, 1 the decided action failed or was blocked, 2 bad usage or
input, 3 unavailable (decisions disabled or agent not wired), 4 unauthorized,
5 approval not found, 6 conflicting decision.

``approvals decide`` reads the presented token from the ``TW_APPROVER_TOKEN``
environment variable (never a flag, which would show up in ``ps``) and compares
it in constant time with the expected token from ``settings_loader``. By default
that is ``Settings``, which reads the same variable, so in-process the check
only enforces "unset disables decisions": whoever runs the CLI already holds the
environment, the database, and ``TW_APPROVAL_KEY``. The API's bearer check is
the authentication boundary between a reviewer and the agent.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
import yaml
from pydantic import ValidationError

from triagewright.api.services import (
    ApprovalNotFoundError,
    DecisionConflictError,
    Services,
    WiringUnavailableError,
    decide_approval,
    load_services,
    token_matches,
)
from triagewright.config import Settings, get_settings
from triagewright.contracts.agents import ActionStatus
from triagewright.contracts.tickets import Ticket

APPROVER_TOKEN_ENV = "TW_APPROVER_TOKEN"  # noqa: S105 - a variable name, not a secret


class ExitCode(IntEnum):
    OK = 0
    FAILED = 1
    USAGE = 2
    UNAVAILABLE = 3
    UNAUTHORIZED = 4
    NOT_FOUND = 5
    CONFLICT = 6


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _fail(code: ExitCode, message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(int(code))


def _read_ticket_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            return json.loads(text)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError):
        _fail(ExitCode.USAGE, f"{path} is not valid {path.suffix.lstrip('.').upper()}")
    _fail(ExitCode.USAGE, "the ticket file must end in .json, .yaml, or .yml")


def _validation_message(exc: ValidationError) -> str:
    # Never echo the rejected input: it can hold customer PII.
    return "invalid ticket: " + "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'ticket'}: {e['msg']}" for e in exc.errors()
    )


def create_cli(
    services_loader: Callable[[Settings], Services] = load_services,
    settings_loader: Callable[[], Settings] = get_settings,
    clock: Callable[[], datetime] = _utcnow,
) -> typer.Typer:
    """Build the CLI on injected dependencies. ``main`` uses the real settings and wiring."""
    app = typer.Typer(no_args_is_help=True, add_completion=False, help="triagewright support agent")
    approvals = typer.Typer(no_args_is_help=True, help="Human approvals for irreversible actions.")
    app.add_typer(approvals, name="approvals")

    def services() -> Services:
        try:
            return services_loader(settings_loader())
        except WiringUnavailableError as exc:
            _fail(ExitCode.UNAVAILABLE, str(exc))

    @app.command()
    def resolve(
        file: Annotated[
            Path | None,
            typer.Argument(help="Ticket as JSON or YAML.", exists=True, dir_okay=False),
        ] = None,
        ticket_id: Annotated[str | None, typer.Option(help="e.g. tkt_0001")] = None,
        customer_id: Annotated[str | None, typer.Option(help="e.g. cus_ava01")] = None,
        channel: Annotated[str | None, typer.Option(help="email, chat, or web")] = None,
        subject: Annotated[str | None, typer.Option()] = None,
        body: Annotated[str | None, typer.Option()] = None,
        created_at: Annotated[str | None, typer.Option(help="ISO 8601; default now")] = None,
    ) -> None:
        """Resolve one ticket and print the FinalResponse as JSON."""
        flags = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "channel": channel,
            "subject": subject,
            "body": body,
            "created_at": created_at,
        }
        given = {k: v for k, v in flags.items() if v is not None}
        if file is not None:
            if given:
                _fail(ExitCode.USAGE, "give a ticket file or ticket flags, not both")
            data = _read_ticket_file(file)
        else:
            if "ticket_id" not in given or "body" not in given:
                _fail(ExitCode.USAGE, "without a ticket file, --ticket-id and --body are required")
            data = {"created_at": clock().isoformat(), **given}
        try:
            ticket = Ticket.model_validate(data)
        except ValidationError as exc:
            _fail(ExitCode.USAGE, _validation_message(exc))
        response = asyncio.run(services().resolver.resolve(ticket))
        typer.echo(response.model_dump_json(indent=2))

    @approvals.command("list")
    def list_approvals() -> None:
        """Print pending approvals as JSON."""
        pending = services().approvals.list_pending()
        typer.echo(json.dumps([a.model_dump(mode="json") for a in pending], indent=2))

    @approvals.command("decide")
    def decide(
        approval_id: Annotated[str, typer.Argument(help="e.g. apr_3f9c0a1b2c4d")],
        approve: Annotated[bool, typer.Option("--approve", help="Approve the action.")] = False,
        reject: Annotated[bool, typer.Option("--reject", help="Reject the action.")] = False,
        note: Annotated[str, typer.Option(help="Optional note for the record.")] = "",
    ) -> None:
        """Approve or reject a pending action. Needs TW_APPROVER_TOKEN in the environment."""
        if approve == reject:
            _fail(ExitCode.USAGE, "give exactly one of --approve or --reject")
        expected = settings_loader().approver_token
        if expected is None:
            _fail(ExitCode.UNAVAILABLE, "approval decisions are disabled (TW_APPROVER_TOKEN)")
        if not token_matches(os.environ.get(APPROVER_TOKEN_ENV), expected):
            _fail(ExitCode.UNAUTHORIZED, f"set {APPROVER_TOKEN_ENV} to the approver token")
        svc = services()
        try:
            outcome = asyncio.run(
                decide_approval(
                    svc, approval_id, approved=approve, reviewer="cli", note=note, now=clock()
                )
            )
        except ApprovalNotFoundError:
            _fail(ExitCode.NOT_FOUND, "approval not found")
        except DecisionConflictError as exc:
            _fail(ExitCode.CONFLICT, str(exc))
        typer.echo(outcome.model_dump_json(indent=2))
        if outcome.status in {ActionStatus.FAILED, ActionStatus.BLOCKED}:
            raise typer.Exit(int(ExitCode.FAILED))

    return app


def main() -> None:
    create_cli()()
