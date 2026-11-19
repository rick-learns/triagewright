from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from triagewright.config import Settings
from triagewright.contracts.common import canonical_json
from triagewright.contracts.llm import Message, ModelRequest, Recording, TextBlock
from triagewright.llm import (
    GeminiClient,
    ModelConfigError,
    RecordingClient,
    RecordingFile,
    ReplayClient,
    ReplayMissError,
    build_model_clients,
    recording_path,
)
from triagewright.testing.fixtures import model_response, text_block

pytestmark = pytest.mark.anyio

GEMINI = "gemini-3.5-flash-lite"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def request(text: str = "hello", model: str = GEMINI) -> ModelRequest:
    return ModelRequest(
        model=model,
        system="sys",
        messages=[Message(role="user", content=[TextBlock(text=text)])],
        purpose="triage",
    )


def write(path: Path, *recs: Recording) -> None:
    path.write_text("".join(canonical_json(r.model_dump(mode="json")) + "\n" for r in recs))


def rec(seq: int, req: ModelRequest, reply: str, latency_ms: float = 1.0) -> Recording:
    resp = model_response(text_block(reply), model=req.model).model_copy(
        update={"latency_ms": latency_ms}
    )
    return Recording(
        case_id="happy-001", sequence=seq, fingerprint=req.fingerprint(), request=req, response=resp
    )


async def test_replay_hit_serves_the_recorded_response(tmp_path: Path) -> None:
    path = tmp_path / "happy-001.jsonl"
    write(path, rec(0, request(), "hi there", latency_ms=432.0))
    client = ReplayClient(RecordingFile(path), model=GEMINI)
    resp = await client.complete(request())
    assert resp.content == [TextBlock(text="hi there")]
    assert resp.latency_ms == 432.0  # replay latency is the recorded latency


async def test_identical_fingerprints_are_served_in_sequence_order(tmp_path: Path) -> None:
    path = tmp_path / "happy-001.jsonl"
    # Written out of order: sequence, not file position, decides.
    write(path, rec(2, request(), "second"), rec(0, request(), "first"), rec(1, request("x"), "x"))
    client = ReplayClient(RecordingFile(path), model=GEMINI)
    assert (await client.complete(request())).content == [TextBlock(text="first")]
    assert (await client.complete(request("x"))).content == [TextBlock(text="x")]
    assert (await client.complete(request())).content == [TextBlock(text="second")]
    with pytest.raises(ReplayMissError, match="no unserved recording"):
        await client.complete(request())  # exhausted


async def test_a_miss_is_a_hard_error(tmp_path: Path) -> None:
    client = ReplayClient(RecordingFile(tmp_path / "missing.jsonl"), model=GEMINI)
    with pytest.raises(ReplayMissError, match="purpose 'triage'"):
        await client.complete(request())
    with pytest.raises(ModelConfigError, match="got a request for"):
        await client.complete(request(model="gemini-3.8-flash"))
    with pytest.raises(ModelConfigError, match="not a free"):
        ReplayClient(RecordingFile(tmp_path / "x.jsonl"), model="gemini-3.5-pro")


def test_corrupt_files_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    tampered = rec(0, request(), "x").model_copy(update={"fingerprint": "0" * 64})
    write(path, tampered)
    with pytest.raises(ModelConfigError, match="fingerprint does not match"):
        RecordingFile(path)
    write(path, rec(0, request(), "a"), rec(0, request("b"), "b"))
    with pytest.raises(ModelConfigError, match="duplicate sequence"):
        RecordingFile(path)
    path.write_text("{not json}\n")
    with pytest.raises(ModelConfigError, match="invalid recording"):
        RecordingFile(path)


def test_recording_paths() -> None:
    assert recording_path(Path("r"), "happy-001") == Path("r/happy-001.jsonl")
    assert recording_path(Path("r"), "happy-001", judge=True) == Path("r/happy-001.judge.jsonl")
    for bad in ("../etc", "a/b", "", ".hidden"):
        with pytest.raises(ModelConfigError, match="invalid case id"):
            recording_path(Path("r"), bad)


async def test_record_then_replay_round_trip_and_resume(tmp_path: Path) -> None:
    replies = iter(["one", "two", "three"])
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        message = {"role": "assistant", "content": next(replies)}
        body = {"id": f"c{calls}", "choices": [{"message": message, "finish_reason": "stop"}]}
        return httpx.Response(200, json=body)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    live = GeminiClient(model=GEMINI, api_key="k", base_url="https://g.test", http_client=http)
    path = tmp_path / "rec" / "happy-001.jsonl"

    recorder = RecordingClient(live, RecordingFile(path, case_id="happy-001"), model=GEMINI)
    first = [await recorder.complete(request()), await recorder.complete(request())]
    assert calls == 2

    lines = path.read_text().splitlines()
    assert [json.loads(line)["sequence"] for line in lines] == [0, 1]
    for line in lines:  # canonical JSON: sorted keys, compact
        assert line == canonical_json(json.loads(line))
        assert Recording.model_validate_json(line).case_id == "happy-001"

    # Resume: the recorded calls are served from the file, only the new one goes live.
    resumed = RecordingClient(live, RecordingFile(path, case_id="happy-001"), model=GEMINI)
    assert await resumed.complete(request()) == first[0]
    assert await resumed.complete(request()) == first[1]
    third = await resumed.complete(request())
    assert calls == 3 and third.content == [TextBlock(text="three")]
    assert [json.loads(line)["sequence"] for line in path.read_text().splitlines()] == [0, 1, 2]

    # Replay serves all three in order, offline.
    replay = ReplayClient(RecordingFile(path), model=GEMINI)
    assert [await replay.complete(request()) for _ in range(3)] == [*first, third]
    with pytest.raises(ModelConfigError, match="got a request for"):
        await resumed.complete(request(model="gemini-3.8-flash"))
    await http.aclose()


async def test_factory_shares_one_file_across_roles(tmp_path: Path) -> None:
    write(tmp_path / "happy-001.jsonl", rec(0, request(), "first"), rec(1, request(), "second"))
    settings = Settings(_env_file=None, recordings_dir=tmp_path)
    clients = build_model_clients(settings, case_id="happy-001")
    a = await clients.for_role("triage").complete(request())
    b = await clients.for_role("answer").complete(request())
    assert (a.content, b.content) == ([TextBlock(text="first")], [TextBlock(text="second")])
    with pytest.raises(ReplayMissError, match=r"happy-001\.judge\.jsonl"):
        await clients.for_role("judge").complete(request(model="gemini-3.8-flash"))
    await clients.aclose()
