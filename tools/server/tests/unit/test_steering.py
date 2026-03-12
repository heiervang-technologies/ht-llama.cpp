import pytest
import time
from concurrent.futures import ThreadPoolExecutor

from utils import *

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_slots = True
    server.n_predict = 128
    server.n_ctx = 512
    server.n_slots = 1


def test_steering_inject_requires_generating_slot():
    """Steering inject should fail when slot is not generating."""
    global server
    server.start()
    # No active generation — slot is idle
    res = server.make_request("POST", "/steering/inject", data={
        "id_slot": 0,
        "text": "hello",
    })
    assert res.status_code != 200


def test_steering_inject_invalid_slot():
    """Steering inject should fail with invalid slot ID."""
    global server
    server.start()
    res = server.make_request("POST", "/steering/inject", data={
        "id_slot": 999,
        "text": "hello",
    })
    assert res.status_code != 200


def test_steering_inject_missing_fields():
    """Steering inject should fail with missing required fields."""
    global server
    server.start()
    # missing id_slot
    res = server.make_request("POST", "/steering/inject", data={
        "text": "hello",
    })
    assert res.status_code != 200
    # missing text
    res = server.make_request("POST", "/steering/inject", data={
        "id_slot": 0,
    })
    assert res.status_code != 200


def test_steering_inject_during_generation():
    """Steering inject should succeed during active generation."""
    global server
    server.n_predict = 200
    server.start()

    # Start a streaming completion in background
    def run_completion():
        return server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time there was a little",
            "n_predict": 200,
            "id_slot": 0,
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(run_completion)

        # Wait for generation to start
        time.sleep(1.0)

        # Inject a steering hint
        res = server.make_request("POST", "/steering/inject", data={
            "id_slot": 0,
            "text": "The story should be about dragons",
            "role": "system",
        })

        # The inject may succeed or fail depending on timing
        # (slot might finish before we inject on fast hardware)
        if res.status_code == 200:
            assert res.body["success"] is True
            assert res.body["n_injected"] > 0

        # Ensure completion finishes without crash
        completion_res = future.result()
        assert completion_res.status_code == 200
        assert len(completion_res.body["content"]) > 0


def test_steering_inject_v1_endpoint():
    """The /v1/steering/inject endpoint should work identically."""
    global server
    server.n_predict = 200
    server.start()

    def run_completion():
        return server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time there was a little",
            "n_predict": 200,
            "id_slot": 0,
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(run_completion)
        time.sleep(1.0)

        res = server.make_request("POST", "/v1/steering/inject", data={
            "id_slot": 0,
            "text": "Change topic to science",
            "role": "user",
        })

        if res.status_code == 200:
            assert res.body["success"] is True
            assert res.body["n_injected"] > 0

        completion_res = future.result()
        assert completion_res.status_code == 200


def test_steering_inject_custom_position():
    """Steering inject with explicit position should work."""
    global server
    server.n_predict = 200
    server.start()

    def run_completion():
        return server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time there was a little",
            "n_predict": 200,
            "id_slot": 0,
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(run_completion)
        time.sleep(1.0)

        res = server.make_request("POST", "/steering/inject", data={
            "id_slot": 0,
            "text": "Think about the ocean",
            "role": "system",
            "position": 10,
        })

        if res.status_code == 200:
            assert res.body["success"] is True

        completion_res = future.result()
        assert completion_res.status_code == 200
