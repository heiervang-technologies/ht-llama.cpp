import os
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_ctx = 512
    server.n_predict = 16
    server.temperature = 0.0
    server.server_slots = True
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


CHAT_TURN_1 = [
    {"role": "user", "content": "Once upon a time"},
]
CHAT_TURN_2 = CHAT_TURN_1 + [
    {"role": "assistant", "content": "there was a small wooden boat that floated down a calm river under tall pine trees."},
    {"role": "user", "content": "Continue the story for several more sentences."},
]
CHAT_TURN_3 = CHAT_TURN_2 + [
    {"role": "assistant", "content": "The boat passed a sleeping fox and a curious owl perched on a low branch above the water."},
    {"role": "user", "content": "Now describe what the fox was dreaming about."},
]


def _post_chat(messages):
    return server.make_request("POST", "/v1/chat/completions", data={
        "messages": messages,
        "max_tokens": 16,
        "cache_prompt": True,
    })


def test_default_args_dont_break_checkpoints():
    # No new flag set — defaults must keep the existing count-only behavior.
    global server
    server.start()
    log = LogReader(server.log_path)

    # warm + multi-turn so at least one checkpoint may be created
    for messages in (CHAT_TURN_1, CHAT_TURN_2, CHAT_TURN_3):
        res = _post_chat(messages)
        assert res.status_code == 200, res.body

    # If any checkpoint is created, the success log mentions the per-slot footprint marker
    # introduced alongside the byte cap. If no checkpoint was created in this short run,
    # the test still passes — we only assert that the server didn't error.
    drained = log.drain()
    if "created context checkpoint" in drained:
        assert "MiB cap" in drained, "byte-cap footprint marker missing from create_checkpoint log line"


def test_byte_cap_disabled_when_zero():
    # Setting --ctx-checkpoints-max-mib 0 must keep the legacy count-only behavior
    # without erroring at startup.
    global server
    server.ctx_checkpoints_max_mib = 0
    server.start()

    res = _post_chat(CHAT_TURN_1)
    assert res.status_code == 200, res.body


def test_negative_byte_cap_rejected():
    # arg parser must reject negative values.
    global server
    server.ctx_checkpoints_max_mib = -1
    with pytest.raises(Exception):
        server.start()


def test_byte_cap_eviction_reason_bytes():
    # Force the byte cap to bite first (count cap large, byte cap tiny) and verify
    # the eviction reason is reported as `bytes` when create_checkpoint fires.
    # This test is best-effort: if tinyllama2 doesn't accumulate any checkpoints
    # at the chosen ctx/min-step in this short conversation, we skip rather than
    # flake.
    global server
    server.n_ctx = 1024
    server.n_ctx_checkpoints = 100      # count cap effectively disabled
    server.checkpoint_min_step = 16     # easier to trigger multiple checkpoints
    server.ctx_checkpoints_max_mib = 1  # tiny budget, expect bytes-eviction
    server.start()
    log = LogReader(server.log_path)

    for messages in (CHAT_TURN_1, CHAT_TURN_2, CHAT_TURN_3):
        res = _post_chat(messages)
        assert res.status_code == 200, res.body

    drained = log.drain()
    if "context checkpoint" not in drained:
        pytest.skip("no checkpoints created in this short conversation — cap path not exercised")
    if "erasing old context checkpoint" not in drained and "created context checkpoint" not in drained:
        pytest.skip("no eviction happened — single checkpoint stayed under the cap")

    assert "reason=bytes" in drained, (
        f"eviction occurred but reason!=bytes; full log: {drained[-2000:]}"
    )
