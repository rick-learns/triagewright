"""Replay and record model calls (see ``contracts.llm.Recording``).

A :class:`RecordingFile` holds one case's recordings (``<case_id>.jsonl`` for
the agent, ``<case_id>.judge.jsonl`` for the judge) and is shared by every role
that reads it, so identical fingerprints are served once each, in ``sequence``
order, whichever role asks.

Record mode is resumable: it serves an unserved matching recording first and
calls the live client only on a miss, appending the new call with the next
``sequence``. An interrupted run (say, out of free quota) picks up where it
stopped; delete the file to re-record a case from scratch.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import ValidationError

from triagewright.config import is_free_model
from triagewright.contracts.common import canonical_json
from triagewright.contracts.llm import ModelRequest, ModelResponse, Provider, Recording
from triagewright.contracts.ports import ModelClient
from triagewright.llm.errors import ModelConfigError, ReplayMissError

_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def recording_path(recordings_dir: Path, case_id: str, *, judge: bool = False) -> Path:
    if not _CASE_ID.fullmatch(case_id):
        raise ModelConfigError(f"invalid case id for a recording file: {case_id!r}")
    return recordings_dir / f"{case_id}{'.judge' if judge else ''}.jsonl"


class RecordingFile:
    """The recordings of one file, with a per-fingerprint cursor."""

    def __init__(self, path: Path, case_id: str | None = None) -> None:
        self.path = path
        self.case_id = case_id
        self._by_fingerprint: dict[str, list[Recording]] = {}
        self._served: dict[str, int] = {}
        self._next_sequence = 0
        if path.exists():
            self._load()

    def _load(self) -> None:
        seen: set[int] = set()
        # Split on "\n" only: canonical JSON keeps U+2028 and friends, which splitlines() splits on.
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                rec = Recording.model_validate_json(line)
            except ValidationError as exc:
                raise ModelConfigError(f"{self.path}:{lineno}: invalid recording") from exc
            if rec.fingerprint != rec.request.fingerprint():
                raise ModelConfigError(f"{self.path}:{lineno}: fingerprint does not match request")
            if rec.sequence in seen:
                raise ModelConfigError(f"{self.path}:{lineno}: duplicate sequence {rec.sequence}")
            seen.add(rec.sequence)
            self._by_fingerprint.setdefault(rec.fingerprint, []).append(rec)
            self._next_sequence = max(self._next_sequence, rec.sequence + 1)
        for recs in self._by_fingerprint.values():
            recs.sort(key=lambda r: r.sequence)

    def take(self, request: ModelRequest) -> ModelResponse | None:
        """The next unserved response for ``request``'s fingerprint, if any."""
        fp = request.fingerprint()
        recs = self._by_fingerprint.get(fp, [])
        i = self._served.get(fp, 0)
        if i >= len(recs):
            return None
        self._served[fp] = i + 1
        return recs[i].response

    def append(self, request: ModelRequest, response: ModelResponse) -> Recording:
        """Append one call as canonical JSON and count it as served."""
        rec = Recording(
            case_id=self.case_id,
            sequence=self._next_sequence,
            fingerprint=request.fingerprint(),
            request=request,
            response=response,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as f:
            f.seek(0, 2)
            if f.tell() > 0:
                f.seek(-1, 2)
                lead = b"" if f.read(1) == b"\n" else b"\n"
            else:
                lead = b""
            f.write(lead + (canonical_json(rec.model_dump(mode="json")) + "\n").encode())
        self._next_sequence += 1
        recs = self._by_fingerprint.setdefault(rec.fingerprint, [])
        recs.append(rec)
        self._served[rec.fingerprint] = len(recs)
        return rec


class ReplayClient:
    """Serves recorded responses unchanged (recorded latency, provider, and usage)."""

    def __init__(self, recordings: RecordingFile, *, model: str) -> None:
        if not is_free_model(Provider.REPLAY, model):
            raise ModelConfigError(f"{model!r} is not a free model")
        self.model = model
        self._recordings = recordings

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.model != self.model:
            raise ModelConfigError(f"client for {self.model!r} got a request for {request.model!r}")
        response = self._recordings.take(request)
        if response is None:
            raise ReplayMissError(
                f"no unserved recording for fingerprint {request.fingerprint()[:16]} "
                f"(purpose {request.purpose!r}) in {self._recordings.path}"
            )
        return response


class RecordingClient:
    """Wraps a live client: replays what's already recorded, records the rest."""

    def __init__(self, live: ModelClient, recordings: RecordingFile, *, model: str) -> None:
        self.model = model
        self._live = live
        self._recordings = recordings

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.model != self.model:
            raise ModelConfigError(f"client for {self.model!r} got a request for {request.model!r}")
        recorded = self._recordings.take(request)
        if recorded is not None:
            return recorded
        response = await self._live.complete(request)
        self._recordings.append(request, response)
        return response
